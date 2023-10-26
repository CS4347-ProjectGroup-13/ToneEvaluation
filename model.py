import math
import torch
import torch.nn as nn
# import torch.nn.functional as F

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