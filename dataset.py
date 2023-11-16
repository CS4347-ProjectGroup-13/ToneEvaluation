import math
import json
import torch
import librosa
import torchaudio
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import time #delete
import scipy.signal as signal
import pandas as pd
import os
import matplotlib.pyplot as plt


from utils import read_json, save_json, ls, jpath
import sklearn.model_selection

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


def read_michigan_dataset_index(data_root=os.path.join(os.getcwd(),"data_full")):
    """
    Reads the Michigan dataset index and returns a pandas DataFrame containing the participant ID, word, tone class, and filename for each audio file.

    Args:
        data_root (str): The root directory of the Michigan dataset. Defaults to the 'data_full' directory in the current working directory.

    Returns:
        pandas.DataFrame: A DataFrame containing the participant ID, word, tone class, and filename for each audio file.
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
        return (particpantID, word, word_tone_class, filename)
    
    audioData = [parseAudioIndex(filename) for filename in audioIndex]
    return pd.DataFrame.from_records(data=audioData, columns=["participantID", "word", "toneclass", "filename"])

import os
import librosa

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
                    feature_options = {} #for future iterations

                    ):
        
        self.dataset_root = dataset_root
        self.sampling_rate = sampling_rate
        self.pad_samples = librosa.time_to_samples(sample_length, sr=sampling_rate)
        self.preload_audio = preload_audio
        self.pad_audio = pad_audio
        self.features = feature_options

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


        if self.pad_audio:
            padded_audio_data = np.pad(audio_data, (0, max(self.pad_samples - len(audio_data),0)), 'constant')
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
        yinEffectivePostSR = self.sampling_rate/yinPyinHopLength

        pYinDecimationFactor = 4
        pYinSampleRate = self.sampling_rate/pYinDecimationFactor
        pYinMaxFreq = min(300,pYinSampleRate/2)
        pYinFrameLength = int(200/4)
        pYinHopLength = int(50/4)
        pYinEffectivePostSR = pYinSampleRate/pYinHopLength 


        # Thanks @Sijin for getting us started
        results = {}

        #mel_spectrogram
        if "mel_spectrogram" in features:
            mel_spectrogram = librosa.feature.melspectrogram(
                y=padded_audio_data, sr=self.sampling_rate, 
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
            yin = librosa.yin(y=padded_audio_data, fmin=yinPyinMinFreq, fmax=yinPyinMaxFreq, sr=self.sampling_rate, frame_length=yinPyinFrameLength , hop_length=yinPyinHopLength)
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

        # labels 
        word = entry["word"]
        toneclass = entry["toneclass"]

        if for_plot:
            return padded_audio_data,mel_spectrogram_normalised_log_scale, yin_normalised, pyin_fundamental_normalised, word, toneclass
            # out = [x[1] for x in results.items()] + [word, toneclass]
            # return out
        else:
            mel_spectrogram_normalised_log_scale_torch = torch.from_numpy(mel_spectrogram_normalised_log_scale)
            yin_normalised_torch = torch.from_numpy(yin_normalised)
            pyin_normalised_torch = torch.from_numpy(pyin_fundamental_normalised)

            return mel_spectrogram_normalised_log_scale_torch, yin_normalised_torch, pyin_normalised_torch, word, toneclass
            # out = [torch.from_numpy(x[1]) for x in results.items()] + [word, toneclass]
            # return out
        
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


def get_data_loader_michigan(args, test_size = 0.2, split_individual = False):
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

import os
def read_aidatatang_index(data_root):
    transcript_path = os.path.join(data_root, 'aidatatang_200zh', 'transcript', 'aidatatang_200_zh_transcript.txt')
    with open(transcript_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(lines[0])