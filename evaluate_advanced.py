from model import *
from hparams import *
from dataset import  get_data_loader, get_data_loader_michigan
import torch

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from dataset import get_data_loader, move_data_to_device

import json
import model
import dataset
import librosa
from dataset import FusedSentenceMichigan,DatasetMichigan
from dataset import processes_segementation_results_global,run_segmentation,convert_segmentations_to_index,get_audio_sample_at_idx

import pickle

USE_REAL_DATA = False

class Metrics:
    def __init__(self):
        self.buffer = {}
        self.results_buffer = []

    def update(self, out, tgt, loss):
        with torch.no_grad():
            out = out.argmax(dim=1)
            out = torch.flatten(out)
            tgt = torch.flatten(tgt)

            acc = accuracy_score(tgt.cpu(), out.cpu())
            f1 = f1_score(tgt.cpu(), out.cpu(), average='macro')

            batch_metric = {
                'loss': loss.item(),
                'accuracy': acc,
                'f1': f1,
            }



            for k in batch_metric:
                if k in self.buffer:
                    self.buffer[k].append(batch_metric[k])
                else:
                    self.buffer[k] = [batch_metric[k]]

            self.results_buffer.append((out.cpu(), tgt.cpu()))

    def get_value(self):
        for k in self.buffer:
            self.buffer[k] = sum(self.buffer[k]) / len(self.buffer[k])
        ret = self.buffer
        ret2 = self.results_buffer
        self.buffer = {}
        self.results_buffer = []

        gt= torch.cat([x[1] for x in ret2]).tolist()
        pred= torch.cat([x[0] for x in ret2]).tolist()
        report= classification_report(gt, pred, output_dict=False, digits=4)

        return ret, report,gt,pred


def validateModelonPregeneratedSegments(model, device, data):
    PSEG_INDEX,PROCESSED_SEGEMENTATION_DATASET,PROCESSED_SEGMENTAION_AUDIO_DATASET = data
    loss_func = nn.CrossEntropyLoss()
    metric = Metrics()
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pbar = tqdm(range(len(PSEG_INDEX)), desc='Validation', unit='sample')
        final_predictions = []

        for sample_idx, _ in enumerate(pbar):
            info, sample = get_audio_sample_at_idx(sample_idx, PSEG_INDEX, PROCESSED_SEGEMENTATION_DATASET, PROCESSED_SEGMENTAION_AUDIO_DATASET)

            mel = DatasetMichigan.preproccessData(sample, features={"mel_spectrogram"},pad_audio=True, pad_samples = librosa.time_to_samples(1.5, sr=16000), sampling_rate = 16000)[1]
            mel_spectrogram_normalised_log_scale_torch = torch.from_numpy(mel).float()

            tone_class = info['toneclass']
            tone_class -= 1 # 0-index
            tone_class = torch.tensor(tone_class)
            tone_class = tone_class.unsqueeze(0)

            x = mel_spectrogram_normalised_log_scale_torch.to(device)
            x = x[None, None, :, :]
            
            tgt = tone_class.to(device)
            out = model(x)
            loss = loss_func(out, tgt)
            metric.update(out, tgt, loss)
            pbar.set_postfix({'Loss': loss.item()})
    metric_test, report,gt,pred = metric.get_value()

    # ConfusionMatrixDisplay.from_predictions(gt, pred, display_labels=["1","2","3","4"]).plot()
    return metric_test, report,gt,pred



def do_speaker(save_model_dir_override ,test_speaker, overlap=1000):
    DATASET =FusedSentenceMichigan(feature_options={},dataset_size=500, overlaps=[overlap],use_real_data=USE_REAL_DATA, audio_source=("michigan",test_speaker))
    SEGMENTATION_RESULTS, DATASET_PROPERTIES = run_segmentation(dataset=DATASET, DEVICE="cuda")
    PROCESSED_SEGEMENTATION_DATASET, PROCESSED_SEGMENTAION_AUDIO_DATASET = processes_segementation_results_global(SEGMENTATION_RESULTS,real=USE_REAL_DATA)
    PSEG_INDEX = convert_segmentations_to_index(PROCESSED_SEGEMENTATION_DATASET, real=USE_REAL_DATA)

    SEGMENTED_STUFF = PSEG_INDEX,PROCESSED_SEGEMENTATION_DATASET,PROCESSED_SEGMENTAION_AUDIO_DATASET

    save_model_dir = f"{Hparams_michigan.args['save_model_dir']}{save_model_dir_override}"
    model = ToneEval_Base(input_shape=(1, 128, 75))
    model.load_state_dict(torch.load(save_model_dir+"/best_model.pth"))
    model.to("cuda")
    model.eval()

    segmentation_results_dir = "./results_segementation"
    results_folder = f"{segmentation_results_dir}/segmementation_{test_speaker}_{overlap}"
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    for_saving = {
        "SEGMENTATION_RESULTS":SEGMENTATION_RESULTS,
        "PROCESSED_SEGEMENTATION_DATASET":PROCESSED_SEGEMENTATION_DATASET,
        "PSEG_INDEX":PSEG_INDEX,
    }
    with open(results_folder + '/segmentation_properties.json', 'w') as f:
        json.dump(DATASET_PROPERTIES, f)

    with open(results_folder + '/segmentation_results.pkl', 'wb') as f:
        # pickle the for_saving
        pickle.dump(for_saving, f)

    metric_test, report,gt,pred = validateModelonPregeneratedSegments(model, "cuda", SEGMENTED_STUFF)
    with open(results_folder + '/segment_testing_log.txt', 'w') as f:
        print(f"====TEST RESULTS  ====  ", file=f)
        print('Split Test Loss, Accuracy: Loss {:.4f} | Accuracy {:.4f}'.format(
            metric_test['loss'],
            metric_test['accuracy']
        ),file=f)
        print(report, file=f)

    with open(results_folder + '/segment_testing_results.txt', 'w') as f:
        json.dump({"gt":gt,"prd":pred},f)

    print(f"Finished {test_speaker} with overlap {overlap}")


def do_eval(config):
    #For the Michigan dataset
    model_dir_override = config["name"] 
    test_speakers = config["test_speakers"]
    Hparams_michigan.args["test_speakers"] = test_speakers

    overlap = config["overlap"]
    print(f"Starting Evaluation using on {config}")
    do_speaker(model_dir_override, test_speakers[0], overlap=overlap)


CONFIGS= [
    {"name":"michigan_MV1", "test_speakers":  ['MV1'], "overlap":0},
    {"name":"michigan_MV1", "test_speakers":  ['MV1'], "overlap":1000},
    {"name":"michigan_MV1", "test_speakers":  ['MV1'], "overlap":2000},
    {"name":"michigan_MV1", "test_speakers":  ['MV1'], "overlap":3000},
    {"name":"michigan_MV1", "test_speakers":  ['MV1'], "overlap":4000},
]

if __name__ == "__main__":
    for i in range(len(CONFIGS)):
        do_eval(CONFIGS[i])