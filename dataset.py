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

from utils import read_json, save_json, ls, jpath

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

