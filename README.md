# ToneEvaluation


# Stripping output from Ipynb files
- We're probably gonna have alot of output in ipynb files
- The embedded output will send git into a fit.
- pip install nbstripout
- This tool strips the outputs from ipynbs

# Instructions for dataset download
Data not included in repository. Must be Downloaded.
1. Download the Michigan one from telegram link (not a public dataset).
2. Download this dataset [aidatatang_200zh](https://openslr.org/62/) ~18GB (Not used yet)
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
Assumes you have both mamba and pip installed. If you are missing a module later on, just pip or mamba install it.

-mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
-mamba install pandas tqdm scikit-learn matplotlib flask transformers chardet
-pip install pinyin

# Definition of Tone in Mandarin (for this project + baseline):
- Four Main Tone classes ā,á,ǎ,à
- One Neutral Tone class (unused) a 


# Baseline Model: Tone Classification:
- Input: segmented audio clip. Fixed(padded) length. Contains only one word
- Output: Tone class for that word.
- Basic convolutional network on melspectrogram should work
- 1-D convolution on pyin or yin should also work
- use Michigan's tone perfect

# Enhanced Model:
- Input: segmented audio clip. variable length. Contains a phrase
- Output: Tone class for that each word in phrase.