#!/usr/bin/env python3

import numpy as np
import pandas as pd
import librosa
import os
import sys
import time

from datetime import datetime
from pathlib import Path

from src.python.audio_transforms import *
from src.python.model_predict import *
from src.python.graphics import plot_graph

# Hardcoding a few variables
max_chroma_sample = 6145
max_spectrogram_sample = 6145
model_classes = [(0, 'artifact'), (1, 'extra'), (2, 'murmur'), (3, 'normal')]

# Directories
DIR_ROOT = Path().resolve()
DIR_PARENT = Path().resolve().parent

def import_wav(filepath):
  '''
  Takes a filepath and returns the 
  sample rate (sr) and amplitude (x)
  '''
  try:
    x, sr = librosa.load(filepath)
    x, _ = librosa.effects.trim(x)

  except FileNotFoundError:
    raise FileNotFoundError(f'could not file a file at {filepath}')
  
  return x, sr


# ----------------------------------
# MAIN FUNCTION --------------------
# ----------------------------------

def main(wav_path, 
         max_chroma_sample, 
         max_spect_sample,
         dt_string):

    audio_results = {}
    base_path = Path(DIR_ROOT, 'demo_files', 'results')

    # 0. SAVE RECORD SOMEWHERE
    ## Placeholder for now

    # 1. Open wav file with Librosa
    x, sr = import_wav(wav_path)

    # 2. Spectogram
    audio_results['spectogram'] = amp_to_db(
        freq_array = stft_transform(amp_array = x),
        sr = sr,
        ref = np.max
    )
    
    # 3. MFCC
    audio_results['mfcc'] = mfcc_spectogram(
        amp_array = x,
        sr = sr
    )

    # 4. Chromagram
    audio_results['chromagram'] = chromagram(
        amp_array = x,
        sr = sr
    )

    # 5. Create Images (User)
    for key, value in audio_results.items():
        plot_graph(
            audio_array = value,
            viz_type = key,
            out_file = Path(base_path, 'user_images', "_".join([dt_string, key]) + '.png'),
            user = True,
            dpi = 150
        )
        
    # 6. Pad Images
    for key, value in audio_results.items():
        audio_results[key] = pad_along_axis(value, max_spectrogram_sample)
    
    # 6. Create Images (Model)
    img_path = {}
    
    for key, value in audio_results.items():
        file_path = Path(base_path, 'model_images', "_".join([key, dt_string]) + '.png')
        
        plot_graph(
            audio_array = value,
            viz_type = key,
            out_file = file_path,
            user = False,
            dpi = 200
        )
        img_path[key] = str(file_path)

    # Return all 3 images to be pushed to model for predictions
    return img_path


if __name__ == '__main__':
    wav_path = sys.argv[1]

    if not Path(wav_path).is_file():
        raise FileNotFoundError()

    dt_string = str(round(datetime.now().timestamp()))
    
    hb_images = main(
        wav_path,
        max_chroma_sample,
        max_spectrogram_sample,
        dt_string
    )

    results = []
    for key, value in hb_images.items():
        output, predict = predict_heartbeat(key, value, DIR_ROOT)

        results.append(output.detach().numpy()[0])

    results = np.array(results)
    index = results.mean(axis=0).argmax()

    hb_predict = model_classes[index][1].title()

    if hb_predict.lower() == 'artifact':
        m = "Too much backgound noise. Try again!"
    else:
        m = f"Your heartbeat is....... {hb_predict}"

    print(m)
    