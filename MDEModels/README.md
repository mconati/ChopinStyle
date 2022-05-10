This folder contains all of the current code for training the MDE-based models for both NSP and Compatibility. 



**Current status:**
There is code in place for generating representations, pretraining models on MLM, and finetuning on the compatibility and NSP tasks. There is code in place for checking the MDE representations and pretraining LM portions of the project, so that code is most likely solid. However, at the point of this handoff, there are no checks in place to validate the performance of the finetuned model on NSP and compatibility. 


**Notebooks:**

The notebooks are meant to be run in numerical order:

01: Calculates the MDE representations for the desired dataset
02: Pretrains an LM on the MDE representations from above with the MLM task
03a: Finetunes the pretrained model on compatibility task
03b: Finetunes the pretrained model on NSP



**Folder contents:**



**The Datasets directory contains:**

1. Chopin43: 43 left hand/right hand separated Chopin midi files. The MIDI files don't have a set number of tracks, but their labeled tracks each have 'left' or 'right' in their names. 

2. ChopinAndHannds: 167 left/right hand separated midi files. This is the Chopin43 along with labeled samples from the hannds paper https://github.com/cemfi/hannds

3. Maestro: The Maestro dataset

4. Compat: The compatibility dataset. Half of the samples are two handed audio separated into measures(positive samples). The other half are incompatible left and right hand parts that are time stretched/shrunk to coincide in one measure(negative samples). There is currently a bug in this dataset where some samples are longer than a measure.

5. NSP: The Next Sentence Prediction dataset. This is a text file along with a folder of midi measures. The text file outlines consecutive and random pairings of measures




**The Extracted Representations directory contains:**

MDE and QANON folders for the various datasets, along with tokenizers and models. The final character is a key for which dataset the representations were generated from:

_M is from Maestro, _C is from Chopin43, and _CH is ChopinAndHannds

The contents of the folder is laid out at the top of the 01_Process_Data_For_LM_Task notebook. This directory is populated as the notebooks are run. Pretrained models are available at the MIR google drive



**The utils directory contains:**
Unmodified source from huggingface, along with helper functions for the custom tokenizer and computing MDE representations.







