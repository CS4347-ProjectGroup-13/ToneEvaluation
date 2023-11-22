import math
import torch
import torch.nn as nn

from sklearn.metrics import f1_score

class ToneEval_Base(nn.Module):
    '''
    This the base model for Tone Evaluation.
    '''

    def __init__(self, input_shape, feat_dim=1024):
        '''
        Define Layers
        '''
        super().__init__()
        self.feat_dim = feat_dim

        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=5, stride=3, padding=2)
        self.batch1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.batch3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.batch4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        kernel_sizes = [5, 3, 3, 2, 3, 2, 3, 2]
        strides = [3, 3, 1, 2, 1, 2, 1, 2]
        paddings = [2, 1, 1, 1, 1, 1, 1, 1]
        width = input_shape[1]
        height = input_shape[2]

        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            stride = strides[i]
            padding = paddings[i]
            dilation = 1

            width = math.floor(((width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)
            height = math.floor(((height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)

        self.linear5 = nn.Linear(512 * width * height, feat_dim)
        self.batch5 = nn.BatchNorm1d(feat_dim)

        self.linear6 = nn.Linear(feat_dim, feat_dim)
        self.batch6 = nn.BatchNorm1d(feat_dim)

        self.linear7 = nn.Linear(feat_dim, 4)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.pool1(self.relu(self.batch1(self.conv1(x))))
        x = self.pool2(self.relu(self.batch2(self.conv2(x))))
        x = self.pool3(self.relu(self.batch3(self.conv3(x))))
        x = self.pool4(self.relu(self.batch4(self.conv4(x))))
        
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.batch5(self.linear5(x)))
        x = self.relu(self.batch6(self.linear6(x)))
        x = self.softmax(self.linear7(x))
        
        return x

class ToneEval_Transformer(nn.Module):
    '''
    This the attention-based transformer model for Tone Evaluation.
    '''

    def __init__(self):
        '''
        Define Layers
        '''
        super().__init__()

    def forward(self, x):

        return None

class SegmentLossFunc:
    def __init__(self):
        # self.device = device
        # self.onset_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0], device=device))
        # self.offset_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0], device=device))
        self.onset_criterion = nn.BCELoss()
        self.offset_criterion = nn.BCELoss()

    def get_loss(self, out, tgt):
        out_on, out_off = out
        tgt_on, tgt_off = tgt

        out_on = torch.flatten(out_on)
        out_off = torch.flatten(out_off)
        tgt_on = torch.flatten(tgt_on).type(torch.float32)
        tgt_off = torch.flatten(tgt_off).type(torch.float32)
        
        out_on = torch.sigmoid(out_on)
        out_off = torch.sigmoid(out_off)

        onset_loss = self.onset_criterion(out_on, tgt_on)
        offset_loss = self.offset_criterion(out_off, tgt_off)

        total_loss = onset_loss + offset_loss
        return total_loss, onset_loss, offset_loss


class SegmentMetrics:
    def __init__(self, loss_func):
        self.buffer = {}
        self.loss_func = loss_func

    def update(self, out, tgt, losses=None):
        '''
        Compute metrics for one batch of output and target.
        F1 score for onset and offset,
        Append the results to a list, and link the list to self.buffer[metric_name].
        '''
        with torch.no_grad():
            out_on, out_off = out
            tgt_on, tgt_off = tgt

            if losses == None:
                losses = self.loss_func.get_loss(out, tgt)

            out_on = torch.flatten(out_on)
            out_off = torch.flatten(out_off)
            tgt_on = torch.flatten(tgt_on)
            tgt_off = torch.flatten(tgt_off)

            # generate prediction labels for f1 score computation
            out_on = torch.sigmoid(out_on)
            out_on[out_on >= 0.5] = 1
            out_on[out_on < 0.5] = 0
            out_on = out_on.long()

            out_off = torch.sigmoid(out_off)
            out_off[out_off >= 0.5] = 1
            out_off[out_off < 0.5] = 0
            out_off = out_off.long()

            # metric computation
            onset_f1 = f1_score(tgt_on.cpu(), out_on.cpu())
            offset_f1 = f1_score(tgt_off.cpu(), out_off.cpu())
        
            batch_metric = {
                'loss': losses[0].item(),
                'onset_loss': losses[1].item(),
                'offset_loss': losses[2].item(),
                'onset_f1': onset_f1,
                'offset_f1': offset_f1,
            }

            for k in batch_metric:
                if k in self.buffer:
                    self.buffer[k].append(batch_metric[k])
                else:
                    self.buffer[k] = [batch_metric[k]]

    def get_value(self):
        for k in self.buffer:
            self.buffer[k] = sum(self.buffer[k]) / len(self.buffer[k])
        ret = self.buffer
        self.buffer = {}
        return ret

class ToneSegment_Base(nn.Module):
    '''
    This the base model for Segmentation of multiple utterances.
    '''

    def __init__(self, input_shape):
        '''
        Define Layers
        '''
        super().__init__()
        self.feat_dim = input_shape[1]
        
        channel1 = 16
        channel2 = 32
        channel3 = 64

        self.conv1 = nn.Conv2d(input_shape[0], channel1, kernel_size=(3, 3), padding=(1, 1))
        self.norm1 = nn.BatchNorm2d(channel1)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        
        self.conv2 = nn.Conv2d(channel1, channel2, kernel_size=(3, 3), padding=(1, 1))
        self.norm2 = nn.BatchNorm2d(channel2)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        
        self.conv3 = nn.Conv2d(channel2, channel3, kernel_size=(3, 3), padding=(1, 1))
        self.norm3 = nn.BatchNorm2d(channel3)
        
        self.act = nn.ReLU()
        self.linear = nn.Linear(channel3 * (self.feat_dim // 4), self.feat_dim)

        self.onset_pred = nn.Linear(self.feat_dim, 1)
        self.offset_pred = nn.Linear(self.feat_dim, 1)

    def forward(self, x):
        # [Batch, Channel, Feature, Time]
        x = x.permute(0, 1, 3, 2)

        # [Batch, Channel, Time, Feature]
        x = self.act(self.norm1(self.conv1(x)))
        x = self.pool1(x)
        x = self.act(self.norm2(self.conv2(x)))
        x = self.pool2(x)
        x = self.act(self.norm3(self.conv3(x)))

        x = x.permute(0, 2, 1, 3)
        x = torch.flatten(x, start_dim=2, end_dim=3)
        x = self.linear(x)

        onset_logits = torch.flatten(self.onset_pred(x), start_dim=-2)
        offset_logits = torch.flatten(self.offset_pred(x), start_dim=-2)

        return onset_logits, offset_logits

class ToneSegment_Enhanced(nn.Module):
    '''
    This the base model for Segmentation of multiple utterances.
    '''

    def __init__(self, input_shape):
        '''
        Define Layers
        '''
        super().__init__()
        self.feat_dim = input_shape[1]
        
        channel1 = 16
        channel2 = 32
        channel3 = 64

        self.conv1 = nn.Conv2d(input_shape[0], channel1, kernel_size=(3, 3), padding=(1, 1))
        self.norm1 = nn.BatchNorm2d(channel1)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        
        self.conv2 = nn.Conv2d(channel1, channel2, kernel_size=(3, 3), padding=(1, 1))
        self.norm2 = nn.BatchNorm2d(channel2)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        
        self.conv3 = nn.Conv2d(channel2, channel3, kernel_size=(3, 3), padding=(1, 1))
        self.norm3 = nn.BatchNorm2d(channel3)

        self.linear1 = nn.Linear(channel3 * (self.feat_dim // 4), self.feat_dim)
        self.linear2 = nn.Linear(self.feat_dim, self.feat_dim)

        self.onset_pred = nn.Linear(self.feat_dim, 1)
        self.offset_pred = nn.Linear(self.feat_dim, 1)
        
        self.act = nn.ReLU()

    def forward(self, x):
        # [Batch, Channel, Feature, Time]
        x = x.permute(0, 1, 3, 2)

        # [Batch, Channel, Time, Feature]
        x = self.act(self.norm1(self.conv1(x)))
        x = self.pool1(x)
        x = self.act(self.norm2(self.conv2(x)))
        x = self.pool2(x)
        x = self.act(self.norm3(self.conv3(x)))

        x = x.permute(0, 2, 1, 3)
        x = torch.flatten(x, start_dim=2, end_dim=3)
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        
        onset_logits = torch.flatten(self.onset_pred(x), start_dim=-2)
        offset_logits = torch.flatten(self.offset_pred(x), start_dim=-2)

        return onset_logits, offset_logits