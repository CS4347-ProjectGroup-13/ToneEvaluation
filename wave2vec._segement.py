import os, librosa
import pandas as pd
import numpy as np
import pinyin
import torch
from torch.utils.data import Dataset, DataLoader


from dataset import  read_michigan_dataset_index
read_michigan_dataset_index()

def read_michigan_dataset_audio(filename, 
                                data_root=os.path.join(os.getcwd(),"data_full"),
                                sr = 16000,
                                mono=True
                                ):
    filepath = os.path.join(data_root, 'michigan', 'tone_perfect_all_mp3', 'tone_perfect', filename)
    return librosa.load(filepath, sr=sr, mono=mono)[0]


def extract_pinyin(sentence_word_list):
    pinyin_word_list = tuple([pinyin.get(x, format="numerical", delimiter=" ") for x in sentence_word_list])
    pinyin_word_list_tone_class = tuple([int(x[-1]) if len(x)>1 else 0 for x in pinyin_word_list])
    return pinyin_word_list,pinyin_word_list_tone_class

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
                    ):
        
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
            toneClasses = np.pad(toneClasses,(0, max(10-len(toneClasses),0)), 'constant')
            sentence = "_".join(sentence) 
            delimiter_time = librosa.samples_to_time(delimiters,sr=16000)
            delimiter_time =  np.pad(toneClasses,(0, max(11-len(toneClasses),-1)), 'constant')
        else:
            padded_audio_data = audio_data

        return 1,1,1,toneClasses,padded_audio_data,sentence,delimiter_time, originallen

        # Thanks @Sijin for getting us started
        results = {}
        mel_spectrogram_hop_length = 321 #@sijin any special reason for this btw
        mel_spectrogram_window_length = 1024
        mel_spectrogram_n_fft = 1024
        mel_spectrogram_n_mels = 128

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
        
        # labels 
        mel_spectrogram_normalised_log_scale_torch = torch.from_numpy(mel_spectrogram_normalised_log_scale)
        
        # targets
        timelabels_frames = librosa.samples_to_frames(delimiters,hop_length=mel_spectrogram_hop_length)
        word_target = np.zeros(mel_spectrogram_normalised_log_scale.shape[1])
        delim_counter= 0
        for delim_idx in range(len(timelabels_frames)-1):
            word_target[timelabels_frames[delim_idx]:timelabels_frames[delim_idx+1]] = toneClasses[delim_counter]
            delim_counter+=1
         
        
        return mel_spectrogram_normalised_log_scale,word_target,timelabels_frames,toneClasses,combinedAudio,sentence
        return mel_spectrogram_normalised_log_scale_torch, word, toneclass
        
a = FusedSentenceMichigan()

dl = DataLoader(a, batch_size=10, pin_memory=True)
for batchid, b in enumerate(dl):
    data = b
    print(data)
    assert False
