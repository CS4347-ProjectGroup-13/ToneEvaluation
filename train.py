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




import warnings
warnings.filterwarnings("ignore")

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

        return ret, report ,gt, pred


def fit(model, args,learning_params, save_model_dir_override=None, dataloaders = None):
    # Set paths
    dev = args['device']
    print(f"Starting Training on {dev}")
    train_ds, test_ds, data_loader_train, data_loader_test  = dataloaders 
    featureset = data_loader_train.dataset.features
    print(f"Featureset: {featureset}")        
    save_model_dir = f"{args['save_model_dir']}{model.feat_dim}_lr-{learning_params['lr']}"
    if save_model_dir_override is not None:
        save_model_dir = f"{args['save_model_dir']}{save_model_dir_override}"

    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)

    with open(save_model_dir + '/trainingParams.json', 'w') as f:
        print(json.dumps(Hparams_michigan.args), file=f)

    model.to(args['device'])
    optimizer = optim.AdamW(model.parameters(), lr=learning_params['lr'])
    loss_func = nn.CrossEntropyLoss()
    metric = Metrics()
    
    # Start training
    print('Start training...')
    start_time = time.time()
    best_model_id = -1
    min_valid_loss = 10000
    prev_loss = 10000
    threshold = 1e-6

    for epoch in range(1, learning_params['epoch'] + 1):
        model.train()
        
        # Train
        pbar = tqdm(data_loader_train)
        for batch_idx, batch in enumerate(pbar):
            mel_spectrogram_normalised_log_scale_torch, yin_normalised_torch, pyin_normalised_torch, word, tone_class = batch
            tone_class -= 1 # 0-index
        
            x = mel_spectrogram_normalised_log_scale_torch.to(args['device'])
            x = x[:, None, :, :]
            tgt = tone_class.to(args['device'])
            out = model(x)
            loss = loss_func(out, tgt)
            metric.update(out, tgt, loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description('Epoch {}, Loss: {:.4f}'.format(epoch, loss.item()))
        metric_train,_,_,_ = metric.get_value()

        # Validation
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader_test):
                mel_spectrogram_normalised_log_scale_torch, yin_normalised_torch, pyin_normalised_torch, word, tone_class = batch
                tone_class -= 1 # 0-index

                x = mel_spectrogram_normalised_log_scale_torch.to(args['device'])
                x = x[:, None, :, :]
                tgt = tone_class.to(args['device'])
                out = model(x)
                loss = loss_func(out, tgt)
                metric.update(out, tgt, loss)
        metric_test,report,_,_ = metric.get_value()

        # Logging
        print('[Epoch {:02d}], Train Loss: {:.5f}, Valid Loss {:.5f}, Time {:.2f}s'.format(
            epoch, metric_train['loss'], metric_test['loss'], time.time() - start_time,
        ))
        print('Split Train Loss, Accuracy: Loss {:.4f} | Accuracy {:.4f}'.format(
            metric_train['loss'],
            metric_train['accuracy']
        ))
        print('Split Test Loss, Accuracy: Loss {:.4f} | Accuracy {:.4f}'.format(
            metric_test['loss'],
            metric_test['accuracy']
        ))

        print('Classification Report:')
        print(report)

        
        # Save the best model
        saved = False
        if metric_test['loss'] < min_valid_loss:
            min_valid_loss = metric_test['loss']
            best_model_id = epoch

            save_dict = model.state_dict()
            target_model_path = save_model_dir + '/best_model.pth'
            torch.save(save_dict, target_model_path)
            saved = True
        
        with open(save_model_dir + '/training_log.txt', 'a') as f:
            print(f"====EPOCH {epoch} ====  saved:{saved}", file=f)
            print('[Epoch {:02d}], Train Loss: {:.5f}, Valid Loss {:.5f}, Time {:.2f}s'.format(
                epoch, metric_train['loss'], metric_test['loss'], time.time() - start_time,
            ),file=f)
            print('Split Train Loss, Accuracy: Loss {:.4f} | Accuracy {:.4f}'.format(
                metric_train['loss'],
                metric_train['accuracy']
            ),file=f)
            print('Split Test Loss, Accuracy: Loss {:.4f} | Accuracy {:.4f}'.format(
                metric_test['loss'],
                metric_test['accuracy']
            ),file=f)
            print(report, file=f)

        if abs(metric_test['loss'] - prev_loss) < threshold:
            break

        prev_loss = metric_test['loss']



    print('Training done in {:.1f} minutes.'.format((time.time() - start_time) / 60))
    return best_model_id

def test(args, save_model_dir_override=None, dataloaders = None):
    # Set paths
    dev = args['device']
    print(f"Starting Training on {dev}")
    train_ds, test_ds, data_loader_train, data_loader_test  = dataloaders 
    featureset = data_loader_train.dataset.features
    print(f"Featureset: {featureset}")        

    if save_model_dir_override is not None:
        save_model_dir = f"{args['save_model_dir']}{save_model_dir_override}"
    else:
        raise Exception("No model dir specified. Override is actually required")

    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)

    with open(save_model_dir + '/trainingParams.json', 'w') as f:
        print(json.dumps(Hparams_michigan.args), file=f)

    model = ToneEval_Base(input_shape=(1, 128, 75))
    target_model_path = save_model_dir + '/best_model.pth'
    model.load_state_dict(torch.load(target_model_path))
    print(f"Loaded model from {target_model_path}")

    model.to(args['device'])
    loss_func = nn.CrossEntropyLoss()
    metric = Metrics()

    model.eval()
    with torch.no_grad():
        pbar = tqdm(data_loader_test)
        for batch_idx, batch in enumerate(pbar):
            mel_spectrogram_normalised_log_scale_torch, yin_normalised_torch, pyin_normalised_torch, word, tone_class = batch
            tone_class -= 1 # 0-index

            x = mel_spectrogram_normalised_log_scale_torch.to(args['device'])
            x = x[:, None, :, :]
            tgt = tone_class.to(args['device'])
            out = model(x)
            loss = loss_func(out, tgt)
            metric.update(out, tgt, loss)
    metric_test,report,gt,prd = metric.get_value()

    with open(save_model_dir + '/testing_log.txt', 'w') as f:
        print(f"====TEST RESULTS  ====  ", file=f)
        print('Split Test Loss, Accuracy: Loss {:.4f} | Accuracy {:.4f}'.format(
            metric_test['loss'],
            metric_test['accuracy']
        ),file=f)
        print(report, file=f)

    with open(save_model_dir + '/testing_results.txt', 'w') as f:
        json.dump({"gt":gt,"prd":prd},f)

def dotrain_test(config):
    #For the Michigan dataset
    save_dir_override = config["name"] 
    test_speakers = config["test_speakers"]
    Hparams_michigan.args["test_speakers"] = test_speakers

    print(f"Starting Training on {config}")

    train_ds, test_ds, data_loader_train, data_loader_test = get_data_loader_michigan(args=Hparams_michigan.args, test_size=0.4,split_individual=False,test_speakers=Hparams_michigan.args["test_speakers"])
    

    # You may change the learning parameters here.
    learning_params = {
    'epoch': 10,
    'lr': 1e-3,
    }

    model = ToneEval_Base(input_shape=(1, 128, 75))
    best_model_id = fit(model, args=Hparams_michigan.args, learning_params=learning_params,save_model_dir_override=save_dir_override, dataloaders=(train_ds, test_ds, data_loader_train, data_loader_test))

    test(args=Hparams_michigan.args, save_model_dir_override=save_dir_override, dataloaders=(train_ds, test_ds, data_loader_train, data_loader_test))

CONFIGS= [
    {"name":"michigan_MV1", "test_speakers":  ['MV1']},
    {"name":"michigan_MV2", "test_speakers":  ['MV2']},
    {"name":"michigan_MV3", "test_speakers":  ['MV3']},
    {"name":"michigan_FV1", "test_speakers":  ['FV1']},
    {"name":"michigan_FV2", "test_speakers":  ['FV2']},
    {"name":"michigan_FV3", "test_speakers":  ['FV3']},
]
if __name__ == "__main__":
    for i in range(len(CONFIGS)):
        dotrain_test(CONFIGS[i])


    