import base64
from flask import Flask, render_template, request
from flask import request

import librosa,io
import sys
import os
import pinyin
import json
sys.path.append(os.path.abspath('../'))

from model import ToneEval_Base
from dataset import load_segmentation_model,single_segmentation_no_load


import torch

def loadclassificationsystem():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_classify = ToneEval_Base(input_shape=(1, 128, 75))
    processor_segement, model_segment = load_segmentation_model()

    model_classify.load_state_dict(torch.load('../results/1024_lr-0.001/best_model.pth'))

    model_classify.to(device)
    model_segment.to(device)
    return model_classify,model_segment,processor_segement,device

MODEL_CLASSIFY,MODEL_SEGMENT,PROCESSOR_SEGMENT,DEVICE = loadclassificationsystem()

def run_segmentation(audio,MODEL_SEGMENT,PROCESSOR_SEGMENT, original_len = 2):
     audio = torch.from_numpy(audio).float()
     predictions, transcription, times = single_segmentation_no_load(audio,PROCESSOR_SEGMENT, MODEL_SEGMENT,originallen= original_len, DEVICE=DEVICE)
     print(predictions,transcription,times)
     return predictions, transcription,times

# # model.load_state_dict(torch.load('results/1024_lr-0.001/best_model.pth'))
# aud,sr = librosa.load("audio.wav", sr = 16000, mono=True)

# print(aud.shape)
# run_segmentation(aud,MODEL_SEGMENT,PROCESSOR_SEGMENT)
# exit()

app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():

    # audio = request.get_json()['audio']
    # base64_audio = base64.b64decode(audio)
    # buffer = io.BytesIO(base64_audio)
    # write buffer to file
    # with open('audio.wav', 'wb') as f:
    #     f.write(base64_audio)
    #     f.close()
    # aud,sr = librosa.load(buffer, sr = 16000)
    aud,sr = librosa.load("audio.wav", sr = 16000, mono=True)
    
    predictions, transcription,times = run_segmentation(aud,MODEL_SEGMENT,PROCESSOR_SEGMENT)
    pyn = pinyin.get(transcription[0], format="numerical", delimiter=" ")
    d = {
        "transcription": transcription[0],
        "times": times[0],
        "pinyin": pyn
    }

    return json.dumps(d)

if __name__ == '__main__':
    app.run( port=12345)
