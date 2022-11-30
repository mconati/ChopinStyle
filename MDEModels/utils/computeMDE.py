import numpy as np
import matplotlib.pyplot as plt
from mido import MidiFile, tick2second
from pretty_midi import PrettyMIDI
import pickle
import os
from os import path
import time
import pathlib
import copy

def visualizeBootlegScore(bs, lines):
    showImage(1 - bs, (10,10))
    for l in range(1, bs.shape[0], 2):
        plt.axhline(l, c = 'b')
    for l in lines:
        plt.axhline(l, c = 'r')
        
def showImage(X, sz = (6,6)):
    plt.figure(figsize = sz)
    plt.imshow(X, cmap = 'gray', origin = 'lower', aspect='auto')
    
def getNoteEvents(midifile, quant = 10):
    ### Given a midi file, return a list of (t_tick, t_sec, notes) tuples for simultaneous note events
    
    # get note onset info
    mid = MidiFile(midifile)
    noteEvents = []
    checkForDuplicates = {}
    for i, track in enumerate(mid.tracks):
        t = 0 
        for msg in track:
            t += msg.time # ticks since last event
            if msg.type == 'note_on' and msg.velocity > 0:
                key = '{},{}'.format(t,msg.note)
                if key not in checkForDuplicates:
                    noteEvents.append((t, msg.note))
                    checkForDuplicates[key] = 0
    noteEvents = sorted(noteEvents) # merge note events from all tracks, sort by time
    pm = PrettyMIDI(midifile)
    noteOnsets = [(t_ticks, pm.tick_to_time(t_ticks), note) for (t_ticks, note) in noteEvents]
    
    # collapse simultaneous notes
    d = {}
    ticks_quant = [n[0]//quant for n in noteOnsets] # quantized time units (ticks)
    for n, t_quant in zip(noteOnsets, ticks_quant):
        if t_quant not in d:
            d[t_quant] = {}
            d[t_quant]['ticks'] = []
            d[t_quant]['secs'] = []
            d[t_quant]['notes'] = []
        d[t_quant]['ticks'].append(n[0])
        d[t_quant]['secs'].append(n[1])
        d[t_quant]['notes'].append(n[2])
        
    result = [(d[key]['ticks'][0], d[key]['secs'][0], d[key]['notes']) for key in sorted(d.keys())]
    
    return result, d # return d for debugging

def generateBootlegScore(noteEvents, repeatNotes = 1, filler = 0):
    rh_dim = 88 # E3 to C8 (inclusive)
    lh_dim = 0 # A1 to G4 (inclusive)
    rh = [] # list of arrays of size rh_dim
    lh = [] # list of arrays of size lh_dim
    numNotes = [] # number of simultaneous notes
    times = [] # list of (tsec, ttick) tuples indicating the time in ticks and seconds
    mapN = getNoteheadPlacementMapping() # maps midi numbers to locations on right and left hand staves
    
    for i, (ttick, tsec, notes) in enumerate(noteEvents):
        
        # insert empty filler columns between note events
        if i > 0:
            for j in range(filler):
                rh.append(np.zeros((rh_dim,1)))
                lh.append(np.zeros((lh_dim,1)))
                numNotes.append(0)
            # get corresponding times using linear interpolation
            interp_ticks = np.interp(np.arange(1, filler+1), [0, filler+1], [noteEvents[i-1][0], ttick])
            interp_secs = np.interp(np.arange(1, filler+1), [0, filler+1], [noteEvents[i-1][1], tsec])
            for tup in zip(interp_secs, interp_ticks):
                times.append((tup[0], tup[1]))

        # insert note events columns
        rhvec = np.zeros((rh_dim, 1))
        lhvec = np.zeros((lh_dim, 1))
        for midinum in notes:
            rhvec += getNoteheadPlacement(midinum, mapN, rh_dim)
        for j in range(repeatNotes):
            rh.append(rhvec)
            lh.append(lhvec)
            numNotes.append(len(notes))
            times.append((tsec, ttick))
    rh = np.clip(np.squeeze(np.array(rh)).T, 0, 1) # clip in case e.g. E and F played simultaneously
    lh = np.clip(np.squeeze(np.array(lh)).T, 0, 1) 
    try:
        both = np.vstack((lh, rh))
    except:
        both = lh
    staffLinesRH = [7,9,11,13,15]
    staffLinesLH = [13,15,17,19,21]
    staffLinesBoth = [13,15,17,19,21,35,37,39,41,43]
    return both, times, numNotes, staffLinesBoth, (rh, staffLinesRH), (lh, staffLinesLH)


def editBscoreByShifts(bscore, shifts):
    # positive = shift right (higher)
    augmented_bscores = []
    for shift in shifts:
        bscore1 = copy.copy(bscore)
        if bscore.shape[0] != 88 or bscore.shape[1] == 0:
            continue
        LH = bscore1
        shift_LH = np.zeros((LH.shape[0]+abs(shift),LH.shape[1]))
        if shift < 0:
            shift_LH[:-abs(shift),:] = LH
            new_LH = shift_LH[-LH.shape[0]:,:]
        else:
            shift_LH[abs(shift):,:] = LH
            new_LH = shift_LH[:LH.shape[0],:]
        bscore1 = new_LH
        augmented_bscores.append(bscore1)
    return augmented_bscores


def calculateLeftHandQanons(midiFiles, QANONDir):  
    shifts = [-3,-2,-1,0,1,2,3]

    count = 0
    tempLH = './temp'
    for i in range(len(midiFiles)):
        mid = MidiFile(midiFiles[i])

        #Generate a left hand only midiFile
        delete = []
        for x in reversed(range(len(mid.tracks))):
            if 'left' not in mid.tracks[x].name.lower() and 'links' not in mid.tracks[x].name.lower():
                delete.append(x)
        for y in delete:
            del mid.tracks[y]
        mid.save(tempLH)

        note_events, _ = getNoteEvents(tempLH)
        path = pathlib.PurePath(midiFiles[i])
        name = str(i)
        bscore, times, num_notes, stafflines, _, _ = generateBootlegScore(note_events, 1, 0)
        bscores = editBscoreByShifts(bscore, shifts)

        
        #Save all of the files
        for n in range(len(bscores)):
            idx = np.argwhere(np.all(bscores[n][..., :] == 0, axis=0))
            fixed = np.delete(bscores[n], idx, axis=1)
            bscores[n] = fixed
            np.save(QANONDir + '/' + name + '_' + str(n), fixed)



        #CHECK THAT THE QANON REP HAS NO COLS OF ZEROS
        for bs in bscores:
            for c in range(bs.shape[1]):
                column = bs[:, c]
                if np.sum(column)==0:
                    print(c)
                    print(midiFiles[i])
                    print("ERROR: there is a column of zeros")
                    break

    os.remove(tempLH)

    
    
def calculateBothHandQanons(midiFiles, QANONDir, visualize= False, output=False):  
    count = 0
    for i in range(len(midiFiles)):

        note_events, _ = getNoteEvents(midiFiles[i])

        name = format(count, '07d')
        bscore, times, num_notes, stafflines, _, _ = generateBootlegScore(note_events, 1, 0)


        count+=1

        if output:
            return bscore
        
        #CHECK THAT THE QANON REP HAS NO COLS OF ZEROS
        #Note that a QANON rep with no note onsets will fail this check as well
        
        for c in range(bscore.shape[1]):
            column = bscore[:, c]

            if np.sum(column)==0:
                print(midiFiles[i])
                print(c)
                bscore = bscore[:]
                visualizeBootlegScore(bscore[:,c-5:c+5], stafflines)
                print("ERROR: there is a column of zeros")
                np.delete(bscore, c, 1)
                print(np.sum(column))
                break
                
        if visualize:
            visualizeBootlegScore(bscore, stafflines)

        else:
            np.save(QANONDir + '/' + name, bscore)
   
    
    
def calculateQANONForCompat(mid, visualize=False):  

    note_events, _ = getNoteEvents(mid)

    bscore, times, num_notes, stafflines, _, _ = generateBootlegScore(note_events, 1, 0)
    if visualize:
        visualizeBootlegScore(bscore, stafflines)

    return bscore



def getNoteheadPlacementMapping():
    r = getNoteheadPlacementMapping()
    return noteMap

def getNoteheadPlacementMapping():
    d = {}
    for x in range(21, 109):
        d[x] = [x-21]
    return d

def getNoteheadPlacement(midinum, midi2loc, dim):
    r = np.zeros((dim, 1))
    if midinum in midi2loc:
        for idx in midi2loc[midinum]:
            r[idx,0] = 1
    return r

def getMDE(column, c):
    indexes = np.nonzero(column)[0]
    lowest = min(indexes)
    octave = int(np.floor((lowest+21)/12)-2)+1
    pitch = (lowest+21)%12
    hand = list("0"*(max(indexes-lowest)+1))
    fingerPosns = indexes-lowest
    for x in fingerPosns: hand[x] = "1" 
    hand = ''.join(hand)
    return octave, pitch, hand, c

def computeMDE(mid, hands):
    midiFiles = [mid]
    q = calculateBothHandQanons(midiFiles, '', output=True)
    piece = []
    txt = ''
    try:
        for c in range(q.shape[1]):
            column = q[:, c]
            o,p,h,c = getMDE(column, c)

            try: hc = hands[h] + 1
            except: hc = 0

            MDE = [int(o),'s',int(p),'s',int(hc)]
            listCol = [str(x) for x in MDE]
            strCol = ''.join(listCol)
            txt = txt + strCol + ' '
    except:
        return txt
    return txt[:-1]
    
    
def computeMDE_from_QANON(q, hands):
    piece = []
    txt = ''
    try:
        for c in range(q.shape[1]):
            column = q[:, c]
            o,p,h,c = getMDE(column, c)

            try: hc = hands[h] + 1
            except: hc = 0

            MDE = [int(o),'s',int(p),'s',int(hc)]
            listCol = [str(x) for x in MDE]
            strCol = ''.join(listCol)
            txt = txt + strCol + ' '
    except:
        return txt
    return txt[:-1]

    
    