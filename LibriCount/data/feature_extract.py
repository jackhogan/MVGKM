import librosa
import numpy as np
import os
import random

all_audio = [f for f in os.listdir('LibriCount10-0dB/test/') if '.wav' in f]
train_idx = random.sample(range(len(all_audio)), 1000)
test_idx = np.setdiff1d(range(len(all_audio)), train_idx)

n = len(all_audio)

# initialise feature matrices
mfccs = np.zeros((n,10*216))
rms = np.zeros((n,216))
chroma = np.zeros((n, 12*216))

# extract features
for i, f in enumerate(all_audio):
    y, sr = librosa.load(os.path.join('LibriCount10-0dB/test', f))
    mfccs[i,:] = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=10).reshape(-1)
    S, phase = librosa.magphase(librosa.stft(y))
    rms[i,:] = librosa.feature.rms(S=S).reshape(-1)
    chroma[i,:] = librosa.feature.chroma_stft(y=y).reshape(-1)

# extract labels from file names
labels = np.zeros((n,))
for i, f in enumerate(all_audio):
    labels[i] = int(f.split('_')[0])

all_data = {0: mfccs,
           1: rms,
           2: chroma}

# save features and labels to pickle
with open('audio_feats.pkl', 'wb') as file:
    pickle.dump(all_data, file)
with open('audio_labs.pkl', 'wb') as f:
    pickle.dump(labels, f)
