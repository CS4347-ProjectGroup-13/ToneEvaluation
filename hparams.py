'''
This file is only used in the ipynb file.
'''

import os
import sys
import torch

class Hparams:
    args = {
        'save_model_dir': './results/',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dataset_root': './data_mini/',
        'sampling_rate': 16000,
        'sample_length': 0.1,  # in second
        'num_workers': 0,  # Number of additional thread for data loading. A large number may freeze your laptop.
        'annotation_path': './data_mini/annotations.json',

        'frame_size': 0.02,
        'batch_size': 8,
    }

class Hparams_michigan:
    args = {
        'save_model_dir': './results/',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dataset_root': os.path.join(os.getcwd(), 'data_full'),
        'sampling_rate': 16000,
        'sample_length': 1.5,  # 1.5 second samples
        'num_workers': 4,  # Number of additional thread for data loading. A large number may freeze your laptop.
        'preload_audio': True,
        'pad_audio': True,

        'batch_size': 32,
    }

class Hparams_synthesized_michigan:
    args = {
        'save_model_dir': './results_segmentation/',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dataset_root': os.path.join(os.getcwd(), 'data_synthesized'),
        'sampling_rate': 16000,
        'sample_length': 10,  # 10 second samples
        'num_workers': 4,  # Number of additional thread for data loading. A large number may freeze your laptop.
        'preload_audio': True,
        'pad_audio': True,

        'batch_size': 32,
    }