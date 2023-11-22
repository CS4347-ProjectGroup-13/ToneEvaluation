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


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa.display
import pinyin

# For Segementation Pipeline
def extract_pinyin(sentence_word_list):
    pinyin_word_list = tuple([pinyin.get(x, format="numerical", delimiter=" ") for x in sentence_word_list])
    pinyin_word_list_tone_class = tuple([int(x[-1]) if len(x)>1 else 0 for x in pinyin_word_list])
    return pinyin_word_list,pinyin_word_list_tone_class

def read_aidatatang_index(data_root=os.path.join(os.getcwd(),"data_full")):

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
        
    def getSentence(self, words, convert_fn = lambda x: x.replace("5", "4")):

        audiosamples = []
        for word in words:
            word = convert_fn(word)
            audiosamples.append(self.cache[word])

        return audiosamples, words
        
    def mix_audio(self, 
                  audiosamples, overlap = 0, add_silence = 1, signal_length_seconds = None , min_samples_each_word = 0):
        
        frames_to_add = librosa.time_to_samples(add_silence, sr=16000)
        lens = [len(x) for x in audiosamples]
        total_len = 0


        if overlap == "auto" and not (signal_length_seconds is None):
            raise Exception("Auto Overlap does not work well, do use")
            signal_samples = librosa.time_to_samples(signal_length_seconds, sr=16000)
            actual_total_len = np.sum(lens)
            overlap = (actual_total_len - signal_samples)/(len(lens)-1)
            assert overlap > 0
            for i in lens:
                if overlap > i:
                    raise ValueError(f"Overlap {overlap} is larger than audio sample {i}")
            
        for idx,l in enumerate(lens):
            if idx == 0:
                total_len += l
            else:
                total_len += l - overlap
                
        final = np.zeros(total_len+frames_to_add+frames_to_add)

        base_frame_index = frames_to_add
        current_id = base_frame_index
        delims = []
        delims.append(current_id)
        for idx,a in enumerate(audiosamples):
            audLen = len(a)
            if idx == 0:
                final[current_id:current_id+audLen] = a
                current_id = current_id+audLen
            else:
                current_id -= overlap
                if current_id - delims[-1] < min_samples_each_word:
                    current_id = delims[-1] + min_samples_each_word
                delims.append(current_id)
                final[current_id:current_id+audLen] = a
                current_id = current_id+audLen
        delims.append(current_id)
        return final, delims

    def get_sample(self, sentence, overlap=0):
        sentence_audio,clsses = self.getSentence(sentence)
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
                    dataset_size = 500
                    ):
        
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
            "dataset_size": dataset_size
        }

        
        self.dataset_root = dataset_root
        self.sampling_rate = sampling_rate
        self.pad_samples = librosa.time_to_samples(sample_length, sr=sampling_rate)
        self.preload_audio = preload_audio
        self.pad_audio = pad_audio
        self.features = feature_options

        self.PhomemeLibrary = PhomemeLibrary(audio_source=audio_source,keep_loaded=True)
        self.overlaps = overlaps


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

    def preproccessData(self):
        pass

    def __getitem__(self, idx, for_plot=False, features_override=None):
        features = self.features if features_override is None else features_override

        sentence = self.sentenceIndex.iloc[idx]["transcript"]
        toneClasses = self.sentenceIndex.iloc[idx]["toneclass"]
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
    LANG_ID = "zh-CN"
    MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"

    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    return processor, model

def get_segmentations(processor, model, batch_input_audio):
    # proccessed = processor(batch_input_audio, sampling_rate=16000, return_tensors="pt", padding=True).input_values
    proccessed = processor(batch_input_audio, return_tensors="pt",sampling_rate=16000).input_values
    proccessed = torch.squeeze(proccessed, dim=(0,1)) 
    proccessed = proccessed.to("cuda")


    with torch.no_grad():
        logits = model(proccessed).logits

    # predicted_ids_softmax = torch.softmax(logits, dim=-1)
    predicted_ids = torch.argmax(logits, dim=-1).cpu()
    transcription = processor.batch_decode(predicted_ids)
    return predicted_ids, transcription

def get_timings(prediction_raw_argmax, _wav2vechoplen = 20/1000):
    results = []
    for i in range(len(prediction_raw_argmax)):
        mask =  prediction_raw_argmax[i]!=0
        ids = np.arange(len(prediction_raw_argmax[i]))
        partial = ids[mask]*_wav2vechoplen - 10/1000
        results.append(partial)
    return results

def filter_data_batch_mutate(predictions, sig_end):
    for i in range(len(predictions)):
        predictions[i] = predictions[i][predictions[i]<sig_end[i].item()]
    return predictions

def doCluster_batch(data,clusters):
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
        
        d = {x:[] for x in clusts}
        _ = [d[labels[j]].append(sample[j]) for j in range(len(labels))]
        
        for v in d:
            d[v] = sorted(d[v], reverse=True)

        final = []
        for i in clusts:
            final.append(d[i][-1])

        output.append(final)
    return output

def process_segmentation_results(frame_ids, cluster_sizes,sig_end):
    frames_to_timing = get_timings(frame_ids)
    frames_to_timing_filtered = filter_data_batch_mutate(frames_to_timing, sig_end)
    clustered = doCluster_batch(frames_to_timing_filtered,cluster_sizes)
    return clustered

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

def run_segmentation(dataset = None):
    processor, model = load_segmentation_model()
    test_data_loader = DataLoader(dataset, batch_size=32, pin_memory=True)
    dataset_properties = dataset.get_properties()

    model = model.to("cuda")


    sentence_results = []
    transcription_results = []
    segementation_times_results = []
    delimiter_time_results = []
    audioData = []


    for batch_id, batch in enumerate(tqdm(test_data_loader)):
        mel_spectrogram_normalised_log_scale, toneClasses, combinedAudio, sentence, delimiter_time, timelimit_start, timelimit_end, originallen = batch

        segmentations = get_segmentations(processor, model, combinedAudio)
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

def get_alignment(x):
    wc = x["word_count"]
    seg = sorted(x["segementation_times_results"], reverse=False)
    gt = x["delimiter_time_results"]
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
    
def processes_segementation_results_global(segmentation_Results):
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

    word_count = df["sentence_results"].apply(lambda x: len(x.split("_")))
    df["word_count"] = word_count

    df["segementation_times_results"] = [[]]*ds_len
    df["segementation_times_results"] =segementation_times_results

    df["delimiter_time_results"] = [[]]*ds_len
    df["delimiter_time_results"] = delimiter_time_results

    df["diff"] = df.apply(lambda x: x["word_count"] - len(x["segementation_times_results"]), axis=1)

    df[["err","mappings"]] = df.apply(get_alignment, axis=1)


    aud_df = pd.DataFrame()
    aud_df["audio"] = [[]]*ds_len
    aud_df["audio"] = audioData
    return df, aud_df