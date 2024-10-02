import librosa
import noisereduce as nr
import numpy as np


def wav_to_mfcc(wav_file_path):
    data, sr = librosa.load(wav_file_path, sr=15000)
    data = nr.reduce_noise(data, sr=sr)
    total_length = sr*5
    xt, index = librosa.effects.trim(data, top_db=33)
    xt = np.pad(xt[:total_length], (0, total_length - len(xt)), 'constant')
    mfcc = librosa.feature.mfcc(y=xt, sr=sr, n_mfcc=13, hop_length=512)
    return mfcc