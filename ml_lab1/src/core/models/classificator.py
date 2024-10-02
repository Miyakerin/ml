import joblib
import re
import numpy as np
import pandas as pd

import torch
import whisper
from nltk import SnowballStemmer, WordNetLemmatizer, word_tokenize

from src.core.models.waw_to_mfcc import wav_to_mfcc


class Vocab:
    def __init__(self, data:list, spec_symb=True): # data - list of tokens
        self.spec_list = ['<PAD>', '<UNK>']
        self.create_tokens_set(data)


        if spec_symb:
            self.idx_to_token = {i+2: v for i,v in enumerate(self.tokens)}
            self.token_to_idx = {v: i+2 for i,v in enumerate(self.tokens)}
            self.idx_to_token[0] = '<PAD>'
            self.token_to_idx['<PAD>'] = 0
            self.idx_to_token[1] = '<UNK>'
            self.token_to_idx['<UNK>'] = 1
        else:
            self.idx_to_token = {i: v for i,v in enumerate(self.tokens)}
            self.token_to_idx = {v: i for i,v in enumerate(self.tokens)}


    def create_tokens_set(self, data):
        self.tokens = list(set([item for item in data if item not in self.spec_list]))

    def __len__(self):
        return len(self.token_to_idx)


class Transpose(torch.nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = torch.transpose(x, self.dim0, self.dim1)
        return x


class GetFeaturesFromRNN(torch.nn.Module):
    def forward(self, x):
        x, _ = x
        if len(x.shape) == 3:
            return x[:, -1, :]
        elif len(x.shape) == 2:
            return x[:, -1]


class Model(torch.nn.Module):
    def __init__(self, output_dim,
               vocab_size, embedding_dim,
               in_channels):
        super(Model, self).__init__()
        self.text_features = torch.nn.Sequential(
            torch.nn.Embedding(vocab_size, embedding_dim=embedding_dim, padding_idx=0),
            torch.nn.Dropout(0.25),
            Transpose(dim0=1, dim1=2),
            torch.nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=3, stride=1, padding=0, dilation=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.MaxPool1d(2),
            Transpose(dim0=1, dim1=2),
            torch.nn.LSTM(128, 128, num_layers=2, batch_first=True, dropout=0.5),
            GetFeaturesFromRNN(),
            torch.nn.Flatten()
        )
        self.mfcc_features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=1, padding=0, dilation=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, dilation=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.MaxPool1d(8),
            torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, dilation=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, dilation=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, dilation=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, dilation=1),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )
        self.classificator = torch.nn.Sequential(
            torch.nn.Linear(128+1152, (128+1152)*2),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear((128+1152)*2, (128+1152)//4),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear((128+1152)//4, (128+1152)//8),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear((128+1152)//8, output_dim)
        )


    def forward(self, x0, x1):
        x0 = self.text_features.forward(x0)
        x1 = self.mfcc_features.forward(x1)
        x = torch.cat((x0, x1), 1)
        x = self.classificator.forward(x)
        return x


def clean_text(text):
    stemmer = SnowballStemmer('russian')
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    text = word_tokenize(text)
    text = [lemmatizer.lemmatize(word) for word in text]
    text = ' '.join(text)
    text = word_tokenize(text)
    text = [stemmer.stem(word) for word in text]
    return text

def pad_text(target, max_len):
    diff = max_len - len(target)
    for i in range(diff):
        target.append('<PAD>')
    return target

def label_emo(class_dic, text):
    return class_dic[text]

def text2num(vocab, text):  # text is list of tokens
    return [vocab.token_to_idx.get(token, 1) for token in text]

def num2text(vocab, text):
    return [vocab.idx_to_token.get(token, vocab.spec_list[1]) for token in text]



class Classificator:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if (cls._instance is None):
            cls._instance = super(Classificator, cls).__new__(cls)
            print('loading models')
            state_dict = torch.load('static/ml_models/text-mfcc-custom.model')
            state_dict = state_dict['model_state_dict']
            cls.model = Model(4, 20802, 200, 13)
            cls.model.load_state_dict(state_dict)
            cls.model.eval()

            cls.cls_dict = joblib.load('static/ml_models/cls_dic.pkl')
            cls.vocab = joblib.load('static/ml_models/vocab.pkl')
            cls.whisper_model = whisper.load_model('turbo')
        return cls._instance


    def __init__(self):
        pass

    def classificate(self, uuid):
        text = self.whisper_model.transcribe("static/" + uuid, language='ru')['text']
        text = clean_text(text)
        text = pad_text(text, max_len=21)
        text = text2num(self.vocab, text)
        mfcc = wav_to_mfcc('static/' + uuid)

        with torch.no_grad():
            y_pred = list(self.cls_dict.keys())[torch.argmax(self.model.forward(torch.unsqueeze(torch.tensor(text), 0), torch.unsqueeze(torch.tensor(mfcc), 0)))]
        return y_pred
