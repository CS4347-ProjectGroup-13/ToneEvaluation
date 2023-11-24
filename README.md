# ToneEvaluation


# Stripping output from Ipynb files
- We're probably gonna have alot of output in ipynb files
- The embedded output will send git into a fit.
- pip install nbstripout
- This tool strips the outputs from ipynbs

# Instructions for dataset download
Data not included in repository. Must be Downloaded.
1. Download the Michigan one from telegram link (not a public dataset).
2. Download this dataset [aidatatang_200zh](https://openslr.org/62/) ~18GB 
3. Download this other dataset [aishell](https://us.openslr.org/33/) ~15GB (Not used yet)
4. Folder structure:
```
📦ToneEvaluation
 ┗ 📂data_full
 ┃ ┣ 📂aidatatang
 ┃ ┃ ┣ 📂corpus
 ┃ ┃ ┃ ┣ 📂dev
 ┃ ┃ ┃ ┣ 📂test
 ┃ ┃ ┃ ┗ 📂train
 ┃ ┃ ┗ 📂transcript
 ┃ ┗ 📂michigan
 ┃ ┃ ┣ 📂tone_perfect_all_mp3
 ┃ ┃ ┗ 📂tone_perfect_all_xml
```

# Install
Run the following commands. Yes, we mixed package managers. Its research Code. 
Pip Freeze/requirements files are a lot less reliable across wildly different machines. Especially with pytorch and cuda versions
Installing from command line seem to be much less of a headache than deconflicting versions, since it resolves dependencies locally and in order
Assumes you have both mamba and pip installed. If you are missing a module later on, just pip or mamba install it. Change for cuda versions as needed

- mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia <br>
- mamba install pandas tqdm scikit-learn matplotlib flask transformers chardet<br>
- pip install pinyin<br>


# Definition of Tone in Mandarin (for this project + baseline):
- Four Main Tone classes ā,á,ǎ,à
- One Neutral Tone class (unused) a 


# Baseline Objectives: Tone Classification
- Input: segmented audio clip. Fixed(padded) length. Contains only one word
- Output: Tone class for that word.

# Enhanced Objectives: Segmentation + Tone Classification
- Input: segmented audio clip. variable length. Contains a phrase
- Output: Tone class for that each word in phrase.


# System Architecture and Files of interest:
Our System consists of two separate components
- Tone Classifier Module
- Segmentation Module
These components can be run separately and do not have to interact with each other. <br>

The following list files that are of note and their function:
- model.py --> Contains the Model definition of for our Tone Classifier CNN
- hparams.py --> Contains some configuration details for model running. You can change or create new configurations following the Hparams class
- dataset.py --> VERY important file. Contains most of the functional code for this project. All functions are documented
- train.py --> run this script to train and evaluate the Tone Classifier Model
- evaluate_advanced.py --> run this script to evaluate the ENHANCED case (segmentation + Tone Classification)

Other Source Files are scratch pads and/or legacy code during the developmental process

# Training and Evaluation of the Tone Classifier Model BASELINE:
The Tone Classifier is trained and test on the Tone Perfect Michigan (see dataset.py DatasetMichigan).

- The Tone classifier for the baseline task can be trained by running the train.py script.
- Trained models, Training and evaluation logs results will be put in the ./results folder

In the current configuration, train.py will perform leave-one-out cross-validation training on the Michigan Dataset for each speaker.

# Evaluation of the ENHANCED case (Segmentation + Tone Classification):
The evaluation of the Enchanced pipeline used a Synthetic Dataset Built from Tone Perfect Michigan (see dataset.py FusedSentenceMichigan).  

- Full Segementation + Tone classification pipeline can be evaluated by running the evaluate_advanced.py script.
- No Training required in this step. We used a process the output of a pre-trained model calculate segmentations.
- Requires trained models trained in BASELINE.
- Evaluation logs results and results will be put in the ./results_segmentation folder
- Detailed results are stored in the .pkl file. They are a dictionary pandas dataframes. 

In the current configuration, evaluate_advanced.py will evaluate speaker MV1's set of models (Trained on all except MV1) for different overlaps (see dataset.py PhomemeLibrary) 


# Running Flask Webapp:
The Flask application is a small web application that will evaluate your spoken mandarin tones. It uses our ENCHANCED pipeline to perform Segmentation+ToneClassification+ASR to evaluate user utterance in real time.

- Navigate to the Flask folder
- Run the flaskmain.py script. Change the port as needed. 
- Webapp will be hosted on localhost:port
- Please use Chrome access the webapp. Other browsers might not work.






