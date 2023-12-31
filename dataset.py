import math
import json
import torch
import librosa
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import scipy.signal as signal
import pandas as pd
import os
import matplotlib.pyplot as plt


from utils import read_json, save_json, ls, jpath
import sklearn.model_selection
import librosa
import tarfile

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa.display
import pinyin


## OLD STUFF DEPRECATED BEGIN##
def move_data_to_device(data, device):
    ret = []
    for i in data:
        if isinstance(i, torch.Tensor):
            ret.append(i.to(device))
    return ret

def get_data_loader(split, args, fns=None):
    dataset = MyDataset(
        dataset_root=args['dataset_root'],
        split=split,
        sampling_rate=args['sampling_rate'],
        sample_length=args['sample_length'],
        frame_size=args['frame_size'],
        song_fns=fns,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return data_loader

def collate_fn(batch):
    '''
    This function help to
    1. Group different components into separate tensors.
    2. Pad samples in the maximum length in the batch.
    '''

    mel_spectrogram = []
    yin = []
    pyin = []
    max_frame_num = 0
    for sample in batch:
        max_frame_num = max(max_frame_num, sample[0].shape[0], sample[1].shape[0], sample[2].shape[0])

    for sample in batch:
        mel_spectrogram.append(
            torch.nn.functional.pad(sample[0], (0, 0, 0, max_frame_num - sample[0].shape[0]), mode='constant', value=0))
        yin.append(
            torch.nn.functional.pad(sample[1], (0, max_frame_num - sample[1].shape[0]), mode='constant', value=0))
        pyin.append(
            torch.nn.functional.pad(sample[2], (0, max_frame_num - sample[2].shape[0]), mode='constant', value=0))

    mel_spectrogram = torch.stack(mel_spectrogram)
    yin = torch.stack(yin)
    pyin = torch.stack(pyin)

    return mel_spectrogram, yin, pyin

class MyDataset(Dataset):
    def __init__(self, dataset_root, split, sampling_rate, sample_length, frame_size,
                 song_fns=None):
        '''
        This dataset return an audio clip in a specific duration in the training loop, with its "__getitem__" function.
        '''
        self.dataset_root = dataset_root
        self.split = split
        self.dataset_path = jpath(self.dataset_root, self.split)
        self.sampling_rate = sampling_rate
        self.duration = {} # dictionary. key: audio sample idx(eg.60), value: length of audio file in seconds
        if song_fns == None:
            self.song_fns = ls(self.dataset_path) #type list
            self.song_fns.sort()
        else:
            self.song_fns = song_fns
        #self.all_audio = {} # dictionary. key: audio sample idx(eg.60), value: loaded audio
        self.all_audio_segments = []
        self.sample_length = sample_length
        
        self.index = self.load_index_data(sample_length) # list of starting time of each sample

        #self.frame_size = frame_size #0.02
        #self.frame_per_sec = int(1 / self.frame_size) # 50
        
    def load_index_data(self, sample_length):
        '''
        Prepare the index for the dataset, i.e., the audio file name and starting time of each sample
        '''
        index = []
        for song_fn in self.song_fns:
            if song_fn.startswith('.'):  # Ignore any hidden file
                continue
            audio_fp = jpath(self.dataset_path, song_fn, 'audio.mp3')  
            #y, sr = librosa.load(audio_fp, sr=self.sampling_rate)
            #self.all_audio[song_fn] = y
            duration = librosa.get_duration(filename=audio_fp)
            num_seg = math.ceil(duration / sample_length) # number of segments for the audio file
            for i in range(num_seg):
                index.append([song_fn, i * sample_length]) # starting time of each sample for every music file
                y, sr = librosa.load(audio_fp, sr=self.sampling_rate, offset=i*sample_length, duration=sample_length)
                self.all_audio_segments.append([song_fn, y])
            self.duration[song_fn] = duration # length of audio file in seconds
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        '''
        Return spectrogram and 4 labels of an audio clip
        '''
        audio_fn, start_sec = self.index[idx]
        audio_fn, y = self.all_audio_segments[idx]

        #mel_spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=self.sampling_rate, hop_length=321, n_fft=1024, n_mels=256) #sr=sr
        #mel_spectrogram = np.transpose(mel_spectrogram)
        mel_spectrogram = torch.from_numpy(mel_spectrogram)

        #YIN
        yin = librosa.yin(y=y, fmin=1, fmax=self.sampling_rate/2, sr=self.sampling_rate, frame_length=200 , hop_length=200)
        yin = torch.from_numpy(yin)
        #print('yin type: ', type(yin))
        #print('yin shape: ', yin)

        #PYIN
        pyin=librosa.pyin(y=y, fmin=1, fmax=self.sampling_rate/2, sr=self.sampling_rate, frame_length=200 , hop_length=200)
        pyin = torch.from_numpy(pyin[0])
        #print('pyin type: ', type(pyin))
        #print('pyin shape: ', pyin[0].shape)
        
        return mel_spectrogram, yin, pyin
## OLD STUFF DEPRECATED END##


# For Interacting with Michigan Dataset
def read_michigan_dataset_index(data_root=os.path.join(os.getcwd(),"data_full")):
    """
    Reads the Michigan dataset index and returns a DataFrame containing the audio and transcript information.

    Args:
        data_root (str): The root directory of the dataset. Defaults to the 'data_full' directory in the current working directory.

    Returns:
        pandas.DataFrame: A DataFrame containing the participant ID, word, tone class, audio filename, and XML filename.

    """
    # handling transcripts
    audio = os.path.join(data_root, 'michigan', 'tone_perfect_all_mp3', 'tone_perfect')
    transcripts= os.path.join(data_root, 'michigan', 'tone_perfect_all_xml', 'tone_perfect')
    
    audioIndex = os.listdir(audio)
    transcriptIndex = os.listdir(transcripts)
    # ignoreing the metadata for now
    
    def parseAudioIndex(filename):
        elem = filename.split("_")
        word = elem[0]
        word_tone_class = int(word[-1]) 
        particpantID = elem[1]
        xml_fn = filename.replace(".mp3", ".xml").replace("MP3", "CUSTOM")
        return (particpantID, word, word_tone_class, filename,xml_fn)
    
    audioData = [parseAudioIndex(filename) for filename in audioIndex]
    return pd.DataFrame.from_records(data=audioData, columns=["participantID", "word", "toneclass", "filename",'xml_fn'])

def read_michigan_dataset_audio(filename, 
                                data_root=os.path.join(os.getcwd(),"data_full"),
                                sr=16000,
                                mono=True):
    """
    Reads an audio file from the Michigan dataset.

    Args:
        filename (str): The name of the audio file to read.
        data_root (str, optional): The root directory of the dataset. Defaults to the 'data_full' directory in the current working directory.
        sr (int, optional): The target sampling rate of the audio file. Defaults to 16000.
        mono (bool, optional): Whether to convert the audio to mono. Defaults to True.

    Returns:
        numpy.ndarray: The audio data as a 1D numpy array.
    """
    filepath = os.path.join(data_root, 'michigan', 'tone_perfect_all_mp3', 'tone_perfect', filename)
    return librosa.load(filepath, sr=sr, mono=mono)[0]

def load_presegmented_audio_index(
        data_root=os.path.join(os.getcwd(),"data_full"),
        cache_name = "saved",sr=16000,mono=True
        ):
    """
    Reads an audio file from the Michigan dataset.

    Args:
        filename (str): The name of the audio file to read.
        data_root (str, optional): The root directory of the dataset. Defaults to the 'data_full' directory in the current working directory.
        sr (int, optional): The target sampling rate of the audio file. Defaults to 16000.
        mono (bool, optional): Whether to convert the audio to mono. Defaults to True.

    Returns:
        numpy.ndarray: The audio data as a 1D numpy array.
    """
    audio_index = os.path.join(data_root, 'saved_segmentations', cache_name, "segementations.pkl")
    meta_data = os.path.join(data_root, 'saved_segmentations', cache_name, "meta_data.json")
    return audio_index,meta_data

class DatasetMichigan(Dataset):
    """
    A class representing a dataset of audio files.

    Args:
        dataset_root (str): The root directory of the dataset.
        dataset_index (pandas.DataFrame): A DataFrame containing metadata about the audio files in the dataset.
        sampling_rate (int): The sampling rate to use when loading audio files.
        preload_audio (bool): Whether to preload all audio files into memory.
        pad_audio (bool): Whether to pad audio files to a fixed length.
        sample_length (float): The length of audio files to pad to, in seconds.
        feature_options (dict): Additional options for the data processing pipeline.

    Raises:
        ValueError: If `dataset_index` is `None`.

    Attributes:
        dataset_root (str): The root directory of the dataset.
        sampling_rate (int): The sampling rate to use when loading audio files.
        pad_samples (int): The number of samples to pad audio files to.
        preload_audio (bool): Whether to preload all audio files into memory.
        pad_audio (bool): Whether to pad audio files to a fixed length.
        indexData (pandas.DataFrame): A DataFrame containing metadata about the audio files in the dataset.

    Methods:
        read_michigan_dataset_index: Reads the dataset index from disk.
        read_michigan_dataset_audio: Reads an audio file from disk.

    Examples:
        >>> dataset = Dataset(dataset_root="./data_full", dataset_index=index_df, preload_audio=True)
    """
    def __init__(self, 
                    dataset_root = os.path.join(os.getcwd(),"data_full"),
                    dataset_index = None,
                    sampling_rate = 16000,
                    preload_audio = True,
                    pad_audio = True,
                    sample_length = 1,
                    feature_options = {}, #for future iterations
                    load_presegmented_audio = False,
                    path_to_segmented_audio = None,
                    ):
        
        self.dataset_root = dataset_root
        self.sampling_rate = sampling_rate
        self.pad_samples = librosa.time_to_samples(sample_length, sr=sampling_rate)
        self.preload_audio = preload_audio
        self.pad_audio = pad_audio
        self.features = feature_options

        self.load_presegmented_audio = load_presegmented_audio
        self.path_to_segmented_audio = path_to_segmented_audio

        if dataset_index is None:
            raise ValueError("dataset_index must be specified. Call read_michigan_dataset_index()")
        else:
            self.indexData = dataset_index
            
        if preload_audio:
            self.indexData["audio"] =  self.indexData.apply(lambda x: read_michigan_dataset_audio(
                x["filename"], sr=self.sampling_rate, mono=True
                ), axis=1)

    def __len__(self):
        return len(self.indexData)
    
    @staticmethod
    def preproccessData(audio_data,features={}, pad_audio = True, pad_samples = 16000, sampling_rate = 16000):
        """
        Preprocesses the audio data by applying various feature extraction techniques.

        Args:
            audio_data (numpy.ndarray): The input audio data.
            features (dict): A dictionary specifying the features to extract. Possible keys are "mel_spectrogram", "yin", and "pyin".
            pad_audio (bool): Whether to pad the audio data to a fixed length.
            pad_samples (int): The length to which the audio data should be padded.
            sampling_rate (int): The sampling rate of the audio data.

        Returns:
            tuple: A tuple containing the preprocessed audio data and the extracted features.
                - padded_audio_data (numpy.ndarray): The padded audio data.
                - mel_spectrogram_normalised_log_scale (numpy.ndarray): The normalized log-scaled mel spectrogram.
                - yin_normalised (numpy.ndarray): The normalized YIN pitch estimation.
                - pyin_fundamental_normalised (numpy.ndarray): The normalized PYIN pitch estimation.
        """
        
        if pad_audio:
            padded_audio_data = np.pad(audio_data, (0, max(pad_samples - len(audio_data),0)), 'constant')
        else:
            padded_audio_data = audio_data

        # pipeline parameters. refactor later
        mel_spectrogram_hop_length = 321 #@sijin any special reason for this btw
        mel_spectrogram_window_length = 1024
        mel_spectrogram_n_fft = 1024
        mel_spectrogram_n_mels = 128

        yinPyinMaxFreq = 300 #fundamental human about 300Hz. Rest is harmonics #self.sampling_rate/2 #nyquist
        yinPyinMinFreq = 20 #Voide lower limit
        yinPyinFrameLength = 200
        yinPyinHopLength = 50
        yinEffectivePostSR = sampling_rate/yinPyinHopLength

        pYinDecimationFactor = 4
        pYinSampleRate = sampling_rate/pYinDecimationFactor
        pYinMaxFreq = min(300,pYinSampleRate/2)
        pYinFrameLength = int(200/4)
        pYinHopLength = int(50/4)
        pYinEffectivePostSR = pYinSampleRate/pYinHopLength 


        # Thanks @Sijin for getting us started
        results = {}

        #mel_spectrogram
        if "mel_spectrogram" in features:
            mel_spectrogram = librosa.feature.melspectrogram(
                y=padded_audio_data, sr=sampling_rate, 
                hop_length=mel_spectrogram_hop_length, 
                n_fft=mel_spectrogram_n_fft,
                win_length=mel_spectrogram_window_length,
                n_mels=mel_spectrogram_n_mels,
                ) 
            mel_spectrogram_normalised_log_scale = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)
            results["mel_spectrogram"] = mel_spectrogram_normalised_log_scale
        else:
            # mel_spectrogram is mandatory
            raise ValueError("Features: mel_spectrogram is mandatory")
        
        #YIN
        if "yin" in features:
            yin = librosa.yin(y=padded_audio_data, fmin=yinPyinMinFreq, fmax=yinPyinMaxFreq, sr=sampling_rate, frame_length=yinPyinFrameLength , hop_length=yinPyinHopLength)
            yin_normalised = (yin - yinPyinMinFreq) / (yinPyinMaxFreq - yinPyinMinFreq)

            # lowpass filter for yin
            # sosfilt = signal.butter(2, 50, 'lp', fs=yinEffectivePostSR, output='sos')
            # yin_normalised = signal.sosfilt(sosfilt, yin_normalised)
            results["yin"] = yin_normalised
        else:
            yin_normalised = np.zeros(1)

        if "pyin" in features:
            #PYIN
            # decimated for speed. Pyin is slow
            resampled_forPyin = signal.decimate(padded_audio_data, pYinDecimationFactor, ftype='fir', axis=-1, zero_phase=True)
            pyin_fundamental ,pyin_voiced, pyin_probability, =librosa.pyin(y=resampled_forPyin, fmin=yinPyinMinFreq, fmax=pYinMaxFreq, sr=pYinSampleRate, frame_length=pYinFrameLength , hop_length=pYinHopLength)
            pyin_fundamental_no_nan = np.nan_to_num(pyin_fundamental,nan=yinPyinMinFreq) # nan_values go to minfreq
            pyin_fundamental_normalised = (pyin_fundamental_no_nan - yinPyinMinFreq) / (pYinMaxFreq - yinPyinMinFreq)

            # lowpass filter for yin
            # sosfilt = signal.butter(2, 50, 'lp', fs=pYinEffectivePostSR, output='sos')
            # pyin_fundamental_normalised = signal.sosfilt(sosfilt, pyin_fundamental_normalised)
            results["pyin"] = pyin_fundamental_normalised
        else:
            pyin_fundamental_normalised = np.zeros(1)

        return padded_audio_data, mel_spectrogram_normalised_log_scale, yin_normalised, pyin_fundamental_normalised


    def __getitem__(self, idx, for_plot=False, features_override=None):
        '''
        Return spectrogram and 4 labels of an audio clip

        Parameters:
        -----------
        idx : int
            Index of the audio clip to retrieve.
        Returns:
        --------
            mel_spectrogram_normalised_log_scale_torch : torch.Tensor
                The normalized log-scale mel spectrogram of the audio clip, as a PyTorch tensor.
            yin_normalised_torch : torch.Tensor
                The normalized YIN pitch estimate of the audio clip, as a PyTorch tensor.
            pyin_normalised_torch : torch.Tensor
                The normalized fundamental frequency estimate of the audio clip, obtained using the PYIN algorithm, as a PyTorch tensor.
            word : str
                The word spoken in the audio clip.
            toneclass : str
                The tone class of the word spoken in the audio clip.
        '''

        features = self.features if features_override is None else features_override

        entry = self.indexData.iloc[idx]

        if self.preload_audio:
            audio_data = entry["audio"]
        else:
            audio_data = read_michigan_dataset_audio(entry["filename"], sr=self.sampling_rate, mono=True)

        res = self.preproccessData(audio_data,features=features, pad_audio = self.pad_audio, pad_samples = self.pad_samples, sampling_rate = self.sampling_rate)
        padded_audio_data,mel_spectrogram_normalised_log_scale, yin_normalised, pyin_fundamental_normalised = res

        
        # labels 
        word = entry["word"]
        toneclass = entry["toneclass"]

        if for_plot:
            return padded_audio_data,mel_spectrogram_normalised_log_scale, yin_normalised, pyin_fundamental_normalised, word, toneclass
        else:
            mel_spectrogram_normalised_log_scale_torch = torch.from_numpy(mel_spectrogram_normalised_log_scale)
            yin_normalised_torch = torch.from_numpy(yin_normalised)
            pyin_normalised_torch = torch.from_numpy(pyin_fundamental_normalised)

            return mel_spectrogram_normalised_log_scale_torch, yin_normalised_torch, pyin_normalised_torch, word, toneclass

        
    def plot_item(self, idx):
        """
        Plots the audio data, YIN pitch estimation, PYIN pitch estimation, and mel spectrogram for a given index in the dataset.

        Args:
            idx (int): The index of the item to plot.

        Returns:
            None
        """
        padded_audio_data,mel_spectrogram_normalised_log_scale, yin_normalised, pyin_fundamental_normalised, word, toneclass = self.__getitem__(idx, for_plot=True)
        
        print(f"Word {word} Toneclass {toneclass} ")

        fig, axs = plt.subplots(nrows=4, figsize=(15, 20))
        ax,ax2, ax3, ax4 =axs

        librosa.display.waveshow(padded_audio_data, sr=16000, ax=ax)
        librosa.display.waveshow(yin_normalised, sr=int(16000/50), ax=ax2)
        librosa.display.waveshow(pyin_fundamental_normalised, sr=int((16000/4)/(50/4)), ax=ax3)
        librosa.display.specshow(mel_spectrogram_normalised_log_scale, sr=16000, hop_length=321, ax=ax4)
        plt.show()


def get_data_loader_michigan(args, test_size = 0.2, split_individual = False, test_speakers = {"FV1","MV1"}):
    """
    Returns train and test data loaders for the Michigan dataset.

    Args:
        args (dict): A dictionary containing the following keys:
            - dataset_root (str): The root directory of the dataset.
            - sampling_rate (int): The sampling rate of the audio files.
            - preload_audio (bool): Whether to preload audio files into memory.
            - sample_length (int): The length of each audio sample.
            - pad_audio (bool): Whether to pad audio samples to the specified length.
            - batch_size (int): The batch size for the data loaders.
            - num_workers (int): The number of worker threads for the data loaders.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        split_individual (bool, optional): Whether to split the dataset individually for each tone class. Defaults to False.
        test_speakers (set, optional): A set of speaker IDs to include in the test split. Defaults to {"FV1","MV1"}.

    Returns:
        tuple: A tuple containing the following elements:
            - train_ds (DatasetMichigan): The training dataset.
            - test_ds (DatasetMichigan): The testing dataset.
            - data_loader_train (DataLoader): The training data loader.
            - data_loader_test (DataLoader): The testing data loader.
    """
    index = read_michigan_dataset_index()

    if split_individual:
        tone_classes = index["toneclass"].values
        ids = list(range(len(index)))
        train_ids, test_ids= sklearn.model_selection.train_test_split(ids, test_size=test_size, random_state=42, shuffle=True, stratify=tone_classes)

        train_index = index.iloc[train_ids]
        test_index = index.iloc[test_ids]
    else:
        filter_by_speaker = index["participantID"].isin(test_speakers)
        test_index = index[filter_by_speaker]
        train_index =  index[~filter_by_speaker]
        trainSpeakers = train_index["participantID"].unique()

        print(f"Splitting by speaker Test:{test_speakers} Train:{trainSpeakers}")


    train_ds = DatasetMichigan(
        dataset_index=train_index, 
        dataset_root=args['dataset_root'], 
        sampling_rate=args['sampling_rate'], 
        preload_audio=args['preload_audio'],
        sample_length=args['sample_length'],
        pad_audio=args['pad_audio'],
        feature_options=args['features'],
        )
    
    test_ds = DatasetMichigan(
        dataset_index=test_index, 
        dataset_root=args['dataset_root'], 
        sampling_rate=args['sampling_rate'], 
        preload_audio=args['preload_audio'],
        sample_length=args['sample_length'],
        pad_audio=args['pad_audio'],
        feature_options=args['features'],
        )
    
    data_loader_train = DataLoader(
        train_ds,
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        pin_memory=True,
        shuffle=True,
    )
    data_loader_test = DataLoader(
        test_ds,
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        pin_memory=True,
        shuffle=True,
    )
    return train_ds, test_ds , data_loader_train, data_loader_test

def get_data_loader_miget_sentence_loaderchigan(args, test_size = 0.2, split_individual = False):
    """
    Returns train and test data loaders for the Michigan dataset.

    Args:
        args (dict): A dictionary containing the following keys:
            - dataset_root (str): The root directory of the dataset.
            - sampling_rate (int): The sampling rate of the audio files.
            - preload_audio (bool): Whether to preload audio files into memory.
            - sample_length (int): The length of each audio sample.
            - pad_audio (bool): Whether to pad audio samples to the specified length.
            - batch_size (int): The batch size for the data loaders.
            - num_workers (int): The number of worker threads for the data loaders.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.

    Returns:
        tuple: A tuple containing the following elements:
            - train_ds (DatasetMichigan): The training dataset.
            - test_ds (DatasetMichigan): The testing dataset.
            - data_loader_train (DataLoader): The training data loader.
            - data_loader_test (DataLoader): The testing data loader.
    """
    index = read_michigan_dataset_index()

    if split_individual:
        tone_classes = index["toneclass"].values
        ids = list(range(len(index)))
        train_ids, test_ids= sklearn.model_selection.train_test_split(ids, test_size=test_size, random_state=42, shuffle=True, stratify=tone_classes)

        train_index = index.iloc[train_ids]
        test_index = index.iloc[test_ids]
    else:
        filter_by_speaker = index["participantID"].isin({"FV1","MV1"})
        test_index = index[filter_by_speaker]
        train_index =  index[~filter_by_speaker]

    train_ds = DatasetMichigan(
        dataset_index=train_index, 
        dataset_root=args['dataset_root'], 
        sampling_rate=args['sampling_rate'], 
        preload_audio=args['preload_audio'],
        sample_length=args['sample_length'],
        pad_audio=args['pad_audio'],
        feature_options=args['features'],
        )
    
    test_ds = DatasetMichigan(
        dataset_index=test_index, 
        dataset_root=args['dataset_root'], 
        sampling_rate=args['sampling_rate'], 
        preload_audio=args['preload_audio'],
        sample_length=args['sample_length'],
        pad_audio=args['pad_audio'],
        feature_options=args['features'],
        )
    
    data_loader_train = DataLoader(
        train_ds,
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        pin_memory=True,
        shuffle=True,
    )
    data_loader_test = DataLoader(
        test_ds,
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        pin_memory=True,
        shuffle=True,
    )
    return train_ds, test_ds , data_loader_train, data_loader_test


# For Segementation Pipeline
def extract_pinyin(sentence_word_list):
    """
    Extracts the pinyin representation of each word in a sentence.

    Args:
        sentence_word_list (list): A list of words in the sentence.

    Returns:
        tuple: A tuple containing two elements:
            - pinyin_word_list (tuple): A tuple of pinyin representations of each word.
            - pinyin_word_list_tone_class (tuple): A tuple of tone classes for each pinyin representation.
    """
    pinyin_word_list = tuple([pinyin.get(x, format="numerical", delimiter=" ") for x in sentence_word_list])
    pinyin_word_list_tone_class = tuple([int(x[-1]) if len(x)>1 else 0 for x in pinyin_word_list])
    return pinyin_word_list, pinyin_word_list_tone_class

def read_aidatatang_index(data_root=os.path.join(os.getcwd(),"data_full")):
    """
    Reads the AIDATATANG index file and returns a DataFrame containing the participant ID, sentence ID,
    transcript, tone class, and folder information.

    Parameters:
    - data_root (str): The root directory of the AIDATATANG dataset (default: current working directory + '/data_full')

    Returns:
    - df (pandas.DataFrame): DataFrame containing the following columns:
        - participantID (str): Participant ID
        - sentenceID (str): Sentence ID
        - transcript (tuple): Transcript text
        - toneclass (tuple): Tone class IDs
        - folder (str): Folder category (dev, test, or train)
    """

    # handling transcripts
    transcript_path = os.path.join(data_root, 'aidatatang', 'transcript', 'aidatatang_200_zh_transcript.txt')
    with open(transcript_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    def parseLine(line):
        split = line.split(' ')
        transcriptIDX = split[0]
        # T0055 G0002 S0002
        prefix = transcriptIDX[:5]
        participant_id = transcriptIDX[5:10]
        sentence_id = transcriptIDX[10:]
        
        text = split[1:]
        text,text_ids = extract_pinyin(list(' '.join(text).strip()))
        text = tuple([x for x in text if (x != ' ' and x != '\n')])
        text_ids = tuple([x for x in text_ids if x != 0])
        return participant_id,sentence_id, text,text_ids
    
    lines=[parseLine(line) for line in lines]

    # handling actual index
    def read_subfolder(subfolder_path):
        files = os.listdir(subfolder_path)
        return files
    
    subfolders = ["dev","test","train"]
    subfolder_index= {subfolder:sorted(read_subfolder(os.path.join(data_root, 'aidatatang', 'corpus', subfolder))) for subfolder in subfolders}
    for subfolder in subfolders:
        indexing = set(subfolder_index[subfolder])
        # sanity
        assert len(indexing) == len(subfolder_index[subfolder])
        subfolder_index[subfolder] = indexing 

    df = pd.DataFrame.from_records(data = lines, columns=["participantID", "sentenceID", "transcript", "toneclass"])
    
    def get_category(x):
        pid = x["participantID"]
        fname = f"{pid}.tar.gz"
        for subfolder in subfolders:
            if fname in subfolder_index[subfolder]:
                return subfolder
        raise ValueError(f"Could not find {fname} in any subfolder")

    df["folder"] = df.apply(get_category, axis=1)
    return df

def read_aidatatang_data(participantID, sentenceID):
    """
    Reads the AIDATATANG dataset for a specific participant and sentence.

    Args:
        participantID (str): The ID of the participant.
        sentenceID (str): The ID of the sentence.

    Returns:
        dict: A dictionary containing the extracted data from the dataset.
            The dictionary has the following keys:
            - 'AudioData': The audio data as a numpy array.
            - 'MetaData': A tuple containing the extracted pinyin.
            - 'Transcript': The transcript of the sentence.

    Raises:
        IndexError: If the dataset file for the specified participant and sentence does not exist.
    """

    fullFileName = f"T0055{participantID}{sentenceID}"
    data_root=os.path.join(os.getcwd(),"data_full")
    makePath = lambda x,y: os.path.join(data_root, 'aidatatang', 'corpus', x, f"{y}.tar.gz")
    subfolders = ["dev","test","train"]
    possiblePaths = [makePath(subfolder,participantID) for subfolder in subfolders]

    zipped_path = [path for path in possiblePaths if os.path.exists(path)][0]

    
    suffixed = {
        'AudioData': (".wav",lambda x: librosa.load(x, sr = 16000, mono=True)[0]) ,
        "MetaData": (".txt", lambda x: tuple(extract_pinyin(x.read().decode("utf-8")))), 
        "Transcript": (".trn", lambda x: x.read().decode("utf-8"))
        }

    with tarfile.open(zipped_path, 'r',) as tar_ref:

        data = {}
        for suf in suffixed.items():
            file_to_extract = f"./{participantID}/{fullFileName}{suf[1][0]}"
            tar_ref.extractfile(file_to_extract)
            data[suf[0]] = suf[1][1](tar_ref.extractfile(file_to_extract))
        
    return data

class PhomemeLibrary():
    def __init__(self, audio_source = ("michigan", "MV1"), keep_loaded = True) -> None:

        # handling source audio
        self.audio_source = audio_source
        self.index = None
        if audio_source[0] == "michigan":
            ds_idx = read_michigan_dataset_index()
            self.index = ds_idx[ds_idx["participantID"] == audio_source[1]]
        else:
            raise Exception(f"{audio_source} as data source is not implemented")

        self.cache = None
        if keep_loaded:
            self.cache = {}
            for idx in range(len(self.index)):
                index_entry = self.index.iloc[idx]
                word_id = index_entry["word"]
                filename = index_entry["filename"]
                tone_class = word_id[-1]
                # sanity
                assert word_id not in self.cache
                self.cache[word_id] = read_michigan_dataset_audio(filename)
        else:
            raise Exception(f"keep_loaded = {keep_loaded} is not implemented")
        
    def getSentence(self, words, convert_fn=lambda x: x.replace("5", "4")):
        """
        Retrieves the audio samples for the given words.

        Args:
            words (list): A list of words.
            convert_fn (function, optional): A function to convert the words. Defaults to lambda x: x.replace("5", "4").

        Returns:
            tuple: A tuple containing the audio samples and the original words.
        """
        audiosamples = []
        for word in words:
            word = convert_fn(word)
            audiosamples.append(self.cache[word])

        return audiosamples, words
        
    def mix_audio(self, audiosamples, overlap=0, add_silence=1, signal_length_seconds=None, min_samples_each_word=0):
        """
        Mixes multiple audio samples together with optional overlap and silence.

        Args:
            audiosamples (list): List of audio samples to be mixed.
            overlap (int or str): Amount of overlap between consecutive audio samples. If "auto" is specified, the overlap is calculated automatically based on the desired signal length.
            add_silence (float): Duration of silence to be added at the beginning and end of the mixed audio, in seconds.
            signal_length_seconds (float): Desired length of the mixed audio signal, in seconds. Only used when overlap is set to "auto".
            min_samples_each_word (int): Minimum number of samples allowed for each word in the mixed audio.

        Returns:
            tuple: A tuple containing the mixed audio signal as a numpy array and a list of delimiter indices indicating the boundaries of each audio sample in the mixed signal.
        """
        frames_to_add = librosa.time_to_samples(add_silence, sr=16000)
        lens = [len(x) for x in audiosamples]
        total_len = 0

        if overlap == "auto" and not (signal_length_seconds is None):
            raise Exception("Auto Overlap does not work well, do use")
            signal_samples = librosa.time_to_samples(signal_length_seconds, sr=16000)
            actual_total_len = np.sum(lens)
            overlap = (actual_total_len - signal_samples) / (len(lens) - 1)
            assert overlap > 0
            for i in lens:
                if overlap > i:
                    raise ValueError(f"Overlap {overlap} is larger than audio sample {i}")

        for idx, l in enumerate(lens):
            if idx == 0:
                total_len += l
            else:
                total_len += l - overlap

        final = np.zeros(total_len + frames_to_add + frames_to_add)

        base_frame_index = frames_to_add
        current_id = base_frame_index
        delims = []
        delims.append(current_id)
        for idx, a in enumerate(audiosamples):
            audLen = len(a)
            if idx == 0:
                final[current_id : current_id + audLen] = a
                current_id = current_id + audLen
            else:
                current_id -= overlap
                if current_id - delims[-1] < min_samples_each_word:
                    current_id = delims[-1] + min_samples_each_word
                delims.append(current_id)
                final[current_id : current_id + audLen] = a
                current_id = current_id + audLen
        delims.append(current_id)
        return final, delims

    def get_sample(self, sentence, overlap=0):
        """
        Retrieves a mixed audio sample and its delimiters for a given sentence.

        Args:
            sentence (str): The input sentence.
            overlap (int, optional): The amount of overlap between consecutive audio segments. Defaults to 0.

        Returns:
            mixed_audio (np.ndarray): The mixed audio sample.
            delimiters (List[Tuple[int, int]]): The delimiters indicating the start and end indices of each audio segment.
        """
        sentence_audio, clsses = self.getSentence(sentence)
        mixed_audio, delimiters = self.mix_audio(sentence_audio, overlap=overlap)
        return mixed_audio, delimiters

class FusedSentenceMichigan(Dataset):
   
    def __init__(self, 
                        dataset_root = os.path.join(os.getcwd(),"data_full"),
                        dataset_index = None,
                        sampling_rate = 16000,
                        preload_audio = True,
                        pad_audio = True,
                        sample_length = 15,
                        feature_options = {"mel_spectrogram"}, #for future iterations
                        audio_source = ("michigan", "MV1"),
                        overlaps = [4000],
                        random_seed = 1234,
                        dataset_size = 500,
                        use_real_data = False
                        ):
            """
            Initializes the dataset object.

            Args:
                dataset_root (str): Root directory of the dataset.
                dataset_index (str): Path to the dataset index file.
                sampling_rate (int): Sampling rate of the audio.
                preload_audio (bool): Whether to preload audio files into memory.
                pad_audio (bool): Whether to pad audio files to a fixed length.
                sample_length (int): Length of audio samples in seconds.
                feature_options (set): Set of feature options to extract from audio.
                audio_source (tuple): Tuple specifying the audio source.
                overlaps (list): List of overlap values for audio samples.
                random_seed (int): Random seed for dataset sampling.
                dataset_size (int): Size of the dataset.
                use_real_data (bool): Whether to use real data or not.
            """
            
            self.properties = {
                "dataset_root": dataset_root,
                "dataset_index": dataset_index,
                "sampling_rate": sampling_rate,
                "preload_audio": preload_audio,
                "pad_audio": pad_audio,
                "sample_length": sample_length,
                "feature_options": feature_options,
                "audio_source": audio_source,
                "overlaps": overlaps,
                "random_seed": random_seed,
                "dataset_size": dataset_size,
                "use_real_data": use_real_data
            }

            
            self.dataset_root = dataset_root
            self.sampling_rate = sampling_rate
            self.pad_samples = librosa.time_to_samples(sample_length, sr=sampling_rate)
            self.preload_audio = preload_audio
            self.pad_audio = pad_audio
            self.features = feature_options

            self.PhomemeLibrary = PhomemeLibrary(audio_source=audio_source,keep_loaded=True)
            self.overlaps = overlaps
            self.use_real_data = use_real_data


            sentence_index_overall = read_aidatatang_index()
            sentence_index_overall[sentence_index_overall["folder"] == 'dev'].drop_duplicates(subset = "transcript", keep= "first")
            sentence_index_overall_wordlens = sentence_index_overall.apply(lambda x: len(x["transcript"]), axis=1)
            mask = sentence_index_overall_wordlens<=10
            self.sentenceIndex = sentence_index_overall[mask]

            blackListed_words = {"Q"}
            filter_sentences_mask = self.sentenceIndex["transcript"].apply(lambda x: len([1 for y in x if len(y)==1]) ==0 )
            self.sentenceIndex =self.sentenceIndex[filter_sentences_mask]

            chosen_ids = np.random.default_rng(seed=random_seed).choice(np.arange(0, len(self.sentenceIndex)), size=dataset_size, replace=False)
            
            self.sentenceIndex = self.sentenceIndex.iloc[chosen_ids]
            print(f"created dataset with {len(self.sentenceIndex)} samples")

    def __len__(self):
        return len(self.sentenceIndex)

    def __getitem__(self, idx, for_plot=False, features_override=None):
        """
        Retrieve the item at the given index from the dataset.

        Args:
            idx (int): The index of the item to retrieve.
            for_plot (bool, optional): Whether the item is being retrieved for plotting. Defaults to False.
            features_override (list, optional): List of features to override the default features. Defaults to None.

        Returns:
            tuple: A tuple containing the following elements:
                - mel_spectrogram_normalised_log_scale (numpy.ndarray): Normalized log-scale mel spectrogram.
                - toneClasses (numpy.ndarray): Array of tone classes.
                - padded_audio_data (numpy.ndarray): Padded audio data.
                - sentence (str): The sentence.
                - delimiter_time (numpy.ndarray): Array of delimiter times.
                - timelimit_start (float): Start time limit.
                - timelimit_end (float): End time limit.
                - originallen (int): The original length of toneClasses.
        """
        features = self.features if features_override is None else features_override

        sentence = self.sentenceIndex.iloc[idx]["transcript"]
        toneClasses = self.sentenceIndex.iloc[idx]["toneclass"]

        if self.use_real_data:
            participantID = self.sentenceIndex.iloc[idx]["participantID"]
            sentenceID	= self.sentenceIndex.iloc[idx]["sentenceID"]
            audio_data  = read_aidatatang_data(participantID, sentenceID)["AudioData"][0:self.pad_samples]
            delimiters = [int(len(audio_data)/(len(toneClasses)+1)*x) for x in range(len(toneClasses)+1)]
            originallen = len(toneClasses)

        else:
            audio_data_list, words = self.PhomemeLibrary.getSentence(sentence)
            combinedAudio, delimiters = self.PhomemeLibrary.mix_audio(audio_data_list, overlap=self.overlaps[0])
            audio_data = combinedAudio
            originallen = len(toneClasses)

        if self.pad_audio:
            padded_audio_data = np.pad(audio_data, (0, max(self.pad_samples - len(audio_data),0)), 'constant')
            toneClasses = np.pad(toneClasses,(0, max(10-originallen,0)), 'constant')
            sentence = "_".join(sentence) 
            delimiter_time = librosa.samples_to_time(delimiters,sr=16000)
            timelimit_start = delimiter_time[0]
            timelimit_end = delimiter_time[-1]
            delimiter_time =  np.pad(delimiter_time,(0, max(11-len(delimiter_time),-1)), 'constant')
        else:
            padded_audio_data = audio_data

        if "mel_spectrogram" in features:
            mel_spectrogram_hop_length = 321 #@sijin any special reason for this btw
            mel_spectrogram_window_length = 1024
            mel_spectrogram_n_fft = 1024
            mel_spectrogram_n_mels = 128
            mel_spectrogram = librosa.feature.melspectrogram(
                y=padded_audio_data, sr=self.sampling_rate, 
                hop_length=mel_spectrogram_hop_length, 
                n_fft=mel_spectrogram_n_fft,
                win_length=mel_spectrogram_window_length,
                n_mels=mel_spectrogram_n_mels,
            ) 
            mel_spectrogram_normalised_log_scale = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)
        else:
            mel_spectrogram_normalised_log_scale = 1

        return mel_spectrogram_normalised_log_scale,toneClasses,padded_audio_data,sentence,delimiter_time,timelimit_start, timelimit_end,originallen

    def get_index_item(self, idx):
        return self.sentenceIndex.iloc[idx]

    def get_properties(self):
        return self.properties

def load_segmentation_model():
    """
    Loads the segmentation model for tone evaluation.

    Returns:
        processor (Wav2Vec2Processor): The processor for the segmentation model.
        model (Wav2Vec2ForCTC): The segmentation model.
    """
    LANG_ID = "zh-CN"
    MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"

    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    return processor, model

def get_segmentations(processor, model, batch_input_audio, DEVICE = "cuda"):
    """
    Get segmentations and transcriptions for a batch of input audio.

    Args:
        processor (Processor): The audio processor.
        model (Model): The model for segmentation and transcription.
        batch_input_audio (list): List of input audio samples.
        DEVICE (str, optional): The device to use for processing. Defaults to "cuda".

    Returns:
        tuple: A tuple containing the predicted segmentation IDs and the transcriptions.
    """
    proccessed = processor(batch_input_audio, return_tensors="pt",sampling_rate=16000).input_values
    proccessed = torch.squeeze(proccessed, dim=(0,1)) 
    proccessed = proccessed.to(DEVICE)

    with torch.no_grad():
        logits = model(proccessed).logits

    predicted_ids = torch.argmax(logits, dim=-1).cpu()
    transcription = processor.batch_decode(predicted_ids)
    return predicted_ids, transcription

def get_timings(prediction_raw_argmax, _wav2vechoplen = 20/1000):
    """
    Calculates the timings for each prediction in the given array.

    Parameters:
    prediction_raw_argmax (numpy.ndarray): Array of prediction values.
    _wav2vechoplen (float): Conversion factor from frame length to seconds

    Returns:
    list: List of timings for each prediction.
    """
    results = []
    for i in range(len(prediction_raw_argmax)):
        mask =  prediction_raw_argmax[i]!=0
        ids = np.arange(len(prediction_raw_argmax[i]))
        partial = ids[mask]*_wav2vechoplen - 10/1000
        results.append(partial)
    return results

def filter_data_batch_mutate(predictions, sig_end):
    """
    Filter the predictions batch based on the sig_end values. Make sure no prediciton exist past end of signal

    Args:
        predictions (list): List of prediction arrays.
        sig_end (list): List of sig_end values.

    Returns:
        list: Filtered predictions batch.
    """
    for i in range(len(predictions)):
        predictions[i] = predictions[i][predictions[i] < sig_end[i].item()]
    return predictions

def doCluster_batch(data, clusters):
    """
    Perform batch clustering on the given data.

    Args:
        data (list): A list of samples to be clustered.
        clusters (list): A list of integers representing the number of clusters for each sample.

    Returns:
        list: A list of lists, where each inner list contains the final elements of each cluster.

    """
    output = []
    for i in range(len(data)):
        sample = data[i]
        n_clusts = clusters[i].item()
        # Create an instance of the KMeans class
        n_clusts = min(n_clusts, len(sample))
        kmeans = KMeans(n_clusters=n_clusts)
        data2 = sample.reshape(-1, 1)
        # Fit the model to the data
        kmeans.fit(data2)
        # Get the cluster labels for each data point
        labels = kmeans.labels_
        # Get the cluster centers
        centers = kmeans.cluster_centers_

        clusts = np.arange(n_clusts)

        d = {x: [] for x in clusts}
        _ = [d[labels[j]].append(sample[j]) for j in range(len(labels))]

        for v in d:
            d[v] = sorted(d[v], reverse=True)

        final = []
        for i in clusts:
            final.append(d[i][-1])

        output.append(final)
    return output

def process_segmentation_results(frame_ids, cluster_sizes, sig_end):
    """
    Process the segmentation results by performing the following steps:
    1. Get the timings for each frame ID.
    2. Filter the timings based on the significant end value.
    3. Cluster the filtered timings using the specified cluster sizes.

    Args:
        frame_ids (list): List of frame IDs.
        cluster_sizes (list): List of cluster sizes.
        sig_end (float): Significant end value.

    Returns:
        list: Clustered timings.

    """
    frames_to_timing = get_timings(frame_ids)
    frames_to_timing_filtered = filter_data_batch_mutate(frames_to_timing, sig_end)
    clustered = doCluster_batch(frames_to_timing_filtered, cluster_sizes)
    return clustered


# Mostly unused
def plot_melspectrogram(mel,toneClasses,padded_audio_data,sentence,delimiter_time,timelimit_start, timelimit_end,originallen, optionalTimings = []):
    fig, ax1= plt.subplots(1, 1, figsize=(5, 3))
    
    # Plot the first graph - Mel Spectrogram
    librosa.display.specshow(mel, sr=16000, hop_length=321, x_axis='time', y_axis='mel', ax=ax1)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Mel Frequency')
    ax1.set_title('Mel Spectrogram')
    
    print(delimiter_time)
    for i in range(len(delimiter_time)):
        ax1.vlines([delimiter_time[i]],ymin = 0, ymax =4096, color = "red")

    for i in range(len(optionalTimings)):
        ax1.vlines([optionalTimings[i]],ymin = 4096, ymax =8192,color = "blue")

    ax1.set_xlim(0, timelimit_end+1)

    plt.tight_layout()
    plt.show()

def split_audio_by_segmentation(audio, delimiters_time, signal_end):
    final_set = delimiters_time + [signal_end]
    return [audio[delimiters[i]:delimiters[i+1]] for i in range(len(delimiters)-1)]

def single_segmentation(audio,originallen = 10, DEVICE = "cuda"):
    processor, model = load_segmentation_model()
    model = model.to(DEVICE)
    audio = torch.unsqueeze(audio, dim=0)
    segmentations = get_segmentations(processor, model, audio, DEVICE = DEVICE)
    predictions, transcription = segmentations
    predictions = predictions.cpu()
    timelimit_end = librosa.samples_to_time(len(audio), sr=16000)
    segementation_times = process_segmentation_results(predictions, originallen, timelimit_end)

    return  predictions, transcription

def single_segmentation_no_load(audio, processor, model,originallen = 10, DEVICE = "cuda"):
    audio = torch.unsqueeze(audio, dim=0)
    audio = torch.unsqueeze(audio, dim=0)

    print(audio.shape)
    segmentations = get_segmentations(processor, model, audio, DEVICE = DEVICE)
    predictions, transcription = segmentations
    predictions = predictions.cpu()
    timelimit_end = torch.tensor([librosa.samples_to_time(audio.shape[-1], sr=16000)])
    originallen = torch.tensor([originallen])
    print(transcription)
    print(originallen.shape)
    print(originallen[0])
    print(timelimit_end,audio.shape[-1])
    segementation_times = process_segmentation_results(predictions, originallen, timelimit_end)

    return  predictions, transcription,segementation_times


np.random.seed(1234)

def run_segmentation(dataset = None, DEVICE = "cuda"):
    """
    Runs the segmentation process on the given dataset.

    Args:
        dataset (Dataset): The dataset to perform segmentation on.
        DEVICE (str): The device to run the segmentation on (default is "cuda").

    Returns:
        tuple: A tuple containing the results of the segmentation process and the properties of the dataset.
            The tuple contains the following elements:
            - sentence_results (list): A list of sentences from the dataset.
            - transcription_results (list): A list of transcriptions generated during the segmentation process.
            - segementation_times_results (list): A list of segmentation times for each sentence.
            - delimiter_time_results (list): A list of delimiter times for each sentence.
            - audioData (list): A list of combined audio data for each sentence.
        dataset_properties (dict): The properties of the dataset.
    """
    processor, model = load_segmentation_model()
    test_data_loader = DataLoader(dataset, batch_size=16, pin_memory=True)
    dataset_properties = dataset.get_properties()

    model = model.to(DEVICE)

    sentence_results = []
    transcription_results = []
    segementation_times_results = []
    delimiter_time_results = []
    audioData = []

    for batch_id, batch in enumerate(tqdm(test_data_loader)):
        mel_spectrogram_normalised_log_scale, toneClasses, combinedAudio, sentence, delimiter_time, timelimit_start, timelimit_end, originallen = batch

        segmentations = get_segmentations(processor, model, combinedAudio, DEVICE = DEVICE)
        predictions, transcription = segmentations
        predictions = predictions.cpu()
        segementation_times = process_segmentation_results(predictions, originallen, timelimit_end)

        sentence_results.append(sentence)
        transcription_results.append(transcription)
        segementation_times_results.append(segementation_times)
        delimiter_time_results.append(delimiter_time)
        audioData.append(combinedAudio)

    results = sentence_results, transcription_results, segementation_times_results, delimiter_time_results,audioData
    dataset_properties = dataset.get_properties()
    return results, dataset_properties

def shallow_concat(x):
    new_x = []
    _ = [new_x.extend(y) for y in x]
    return new_x

def get_alignment(x, real=False):
    """
    Calculate the alignment between segmentation times and delimiter times.

    Args:
        x (dict): A dictionary containing the following keys:
            - "word_count" (int): The number of words for the sentence.
            - "segementation_times_results" (list): A list of segmentation times.
            - "delimiter_time_results" (list): A list of delimiter times.
        real (bool, optional): If True, return a mapping array with no alignment. 
            Defaults to False.

    Returns:
        pandas.Series: A pandas Series containing the following keys:
            - "diffs" (float): The mean difference between segmentation times and delimiter times.
            - "mappingArray" (list): A list representing the mapping array.

    """
    wc = x["word_count"]
    seg = sorted(x["segementation_times_results"], reverse=False)
    gt = x["delimiter_time_results"]
    
    if real:
        d ={
            "diffs":0,
            "mappingArray": [x for x in range(wc)],
        }
        return pd.Series(d)

    if len(seg) == wc:
        d ={
            "diffs":np.mean([abs(gt[i] - seg[i]) for i in range(wc)]),
            "mappingArray": [x for x in range(wc)],
        }
        return pd.Series(d)
    # else we try to align
    mapping_array = []
    for i in range(wc):
        current_gt = gt[i]
        diffs = [abs(current_gt - x) for x in seg]
        closest = np.argmin(diffs)
        mapping_array.append(closest)
    # print(seg)
    # print(mapping_array)
    
    diffs = [abs(gt[x] - seg[mapping_array[x]]) for x in range(len(mapping_array))]
    mapping_array2 = []

    for i in range(wc):
        curr_gt = gt[i]
        curr_mapping = mapping_array[i]
        filtered = [(x,diffs[x]) for x,y in enumerate(mapping_array) if y==curr_mapping]
        filtered = sorted(filtered, key=lambda x: x[1])
        # mapping_array2.append(filtered[0])
        if filtered[0][0] == i:
            mapping_array2.append(curr_mapping)
        else:
            mapping_array2.append(-1)


    # for i in range(len(seg)):
    #     filtered = [x for x,y in enumerate(mapping_array) if y==i]
    #     print(i,filtered)
    #     diffs.append(abs(gt[i] - seg[mapping_array[i]]))

    d ={
        "diffs":np.mean(diffs),
        "mappingArray": mapping_array2,
    }
    return pd.Series(d)

def transcription_to_pinyin(transcription_str):
    return pinyin.get(transcription_str, format="numerical", delimiter="_").split("_")

def processes_segementation_results_global(segmentation_Results, real=False):
    """
    Processes the segmentation results and returns a results DataFrame and an audio DataFrame.

    Args:
        segmentation_Results (tuple): A tuple containing the segmentation results.
            The tuple should have the following elements:
            - sentence_results (list): List of sentence results.
            - transcription_results (list): List of transcription results.
            - segementation_times_results (list): List of segmentation times results.
            - delimiter_time_results (list): List of delimiter time results.
            - audioData (list): List of audio data.

        real (bool, optional): Flag indicating whether the segmentation results are real or not.
            Defaults to False.

    Returns:
        tuple: A tuple containing the processed DataFrame and audio DataFrame.
            The tuple has the following elements:
            - df (pandas.DataFrame): Processed DataFrame with the following columns:
                - sentence_results: Sentence results.
                - transcription_results: Transcription results.
                - word_count: Word count of each sentence.
                - segementation_times_results: Segmentation times results.
                - delimiter_time_results: Delimiter time results.
                - diff: Difference between word count and segmentation times count.
                - err: Error value.
                - mappings: Mappings.
            - aud_df (pandas.DataFrame): Audio DataFrame with the following column:
                - audio: Audio data.
    """
    sentence_results, transcription_results, segementation_times_results, delimiter_time_results,audioData= segmentation_Results
    sentence_results = shallow_concat(sentence_results, )
    transcription_results = shallow_concat(transcription_results)
    segementation_times_results = shallow_concat(segementation_times_results )
    audioData = shallow_concat(audioData)
    audioData = [x.numpy() for x in audioData]
    delimiter_time_results = torch.concat(delimiter_time_results, axis  =0).numpy().tolist()
    
    ds_len = len(sentence_results)


    d = {
        "sentence_results": sentence_results,
        "transcription_results": transcription_results,
        # "segementation_times_results": segementation_times_results,
        # "delimiter_time_results": delimiter_time_results
    }
    df = pd.DataFrame.from_dict(d)
    
    df["transcription_results"] = df["transcription_results"].apply(lambda x: transcription_to_pinyin(x))


    word_count = df["sentence_results"].apply(lambda x: len(x.split("_")))
    df["word_count"] = word_count

    df["segementation_times_results"] = [[]]*ds_len
    df["segementation_times_results"] =segementation_times_results

    df["delimiter_time_results"] = [[]]*ds_len
    df["delimiter_time_results"] = delimiter_time_results

    # if real:
        # df["delimiter_time_results"] = segementation_times_results

    df["diff"] = df.apply(lambda x: x["word_count"] - len(x["segementation_times_results"]), axis=1)

    ga = lambda x:get_alignment(x, real=real)
    df[["err","mappings"]] = df.apply(ga, axis=1)


    aud_df = pd.DataFrame()
    aud_df["audio"] = [[]]*ds_len
    aud_df["audio"] = audioData
    return df, aud_df

def convert_segmentations_to_index(segmentations, convert_fn = lambda x: x.replace("5", "4"), real = False):
    """
    Converts segmentations to index format.

    Args:
        segmentations (DataFrame): The input segmentations DataFrame.
        convert_fn (function, optional): The function used to convert the segmentations. Defaults to lambda x: x.replace("5", "4").
        real (bool, optional): Flag indicating whether to include only real or synthetic data. Defaults to False.

    Returns:
        DataFrame: The converted segmentations in index format.
    """
    finalResults = []
    for i in range(len(segmentations)):
        results_entry = segmentations.iloc[i]
        word_count = results_entry['word_count']
        sentence = results_entry['sentence_results']
        mappings = results_entry['mappings']

        temp = "_".join(results_entry['transcription_results'])
        # replace all <_u_n_k_> with ?
        temp = temp.replace("<_u_n_k_>", "?")
        split = temp.split("_")
        split = [x for x in split if x != ""]

        if real and results_entry['diff'] != 0:
            continue

        split_sentence = sentence.split("_")
        split_sentence = [convert_fn(x) for x in split_sentence]
        for j in range(len(split_sentence)):
            word_id = split_sentence[j] 
            toneclass = int(word_id[-1])
            main_Idx = i
            word_Idx = j
            mapped_onset = mappings[j]
            if mapped_onset == -1:
                transcripted_tone_class = np.random.choice([1,2,3,4])
            else:
                try:
                    string_tone_class = split[mapped_onset][-1]
                    string_tone_class = convert_fn(string_tone_class)
                    if string_tone_class == "?":
                        transcripted_tone_class = np.random.choice([1,2,3,4])
                    else:
                        transcripted_tone_class = int(string_tone_class)
                except:
                    transcripted_tone_class = np.random.choice([1,2,3,4])
            finalResults.append((word_id, toneclass, main_Idx, word_Idx,mapped_onset, transcripted_tone_class))
        
    return pd.DataFrame(finalResults, columns=['word_id', 'toneclass', 'main_Idx', 'word_Idx', 'mapped_onset','transcripted_tone_class'])


def get_audio_sample_at_idx(idx,pSEG_index, pSEG, pSEGAUDIO, sr= 16000, max_time=1.5):
    """
    Retrieves an audio sample from the dataset based on the given index.

    Args:
        idx (int): The index of the audio sample to retrieve.
        pSEG_index (pandas.DataFrame): DataFrame containing index information.
        pSEG (pandas.DataFrame): DataFrame containing sentence information.
        pSEGAUDIO (pandas.DataFrame): DataFrame containing audio information.
        sr (int, optional): The sample rate of the audio. Defaults to 16000.
        max_time (float, optional): The maximum duration of the audio sample in seconds. Defaults to 1.5.

    Returns:
        tuple: A tuple containing the sample information and the audio sample.
    """

    sample_info = pSEG_index.iloc[idx]
    pSEG_idx = sample_info['main_Idx']

    sentence_info = pSEG.iloc[pSEG_idx]
    pSEG_audio = pSEGAUDIO.iloc[pSEG_idx]["audio"]


    segementation_start_time_gt = sentence_info['delimiter_time_results']
    if len(segementation_start_time_gt) == sentence_info['word_count']:
        segementation_start_time_gt = segementation_start_time_gt + [segementation_start_time_gt[-1] + max_time]


    segementation_start_time = sorted(sentence_info['segementation_times_results'] + [segementation_start_time_gt[sentence_info['word_count']]])
    segmentation_onset_mapping = sentence_info['mappings']
    mapped_onset = sample_info['mapped_onset']

    current_mapping = segmentation_onset_mapping[sample_info['word_Idx']]

    # print(sample_info)
    # print(sentence_info)
    # print(segementation_start_time_gt)
    if current_mapping == -1:
        # unmapped, use previous result
        prev_mapping = 0
        for i in range(sample_info['word_Idx'], -1, -1):
            if segmentation_onset_mapping[i] != -1:
                prev_mapping = segmentation_onset_mapping[i]
                break
        # prev_mapping = segmentation_onset_mapping[sample_info['word_Idx'] - 1]
        start_time = segementation_start_time[prev_mapping]
        end_time = segementation_start_time[prev_mapping+1] 
        # print(idx,prev_mapping,prev_mapping+1)
    else:
        start_time = segementation_start_time[current_mapping] 
        end_time = segementation_start_time[current_mapping+1]

    start_samples = librosa.time_to_samples(start_time, sr=sr)
    if end_time-start_time > max_time:
        end_time = start_time + max_time
    end_samples = librosa.time_to_samples(end_time, sr=sr)

    return sample_info,pSEG_audio[start_samples:end_samples]