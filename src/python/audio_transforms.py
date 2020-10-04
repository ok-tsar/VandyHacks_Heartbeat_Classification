import librosa
import numpy as np


def amp_to_db(freq_array, sr, ref = np.max):
    """
    Convert amplitudes to decibels
    """
    return librosa.amplitude_to_db(freq_array, ref=ref)


## CHROMAGRAM
def chromagram(amp_array, sr, n_fft = 512, hop_length = 100):
    return librosa.feature.chroma_stft(amp_array, sr, n_fft = n_fft, hop_length=hop_length)


## SPECTOGRAMS
def stft_transform(amp_array, n_fft = 2048, hop_length = 100):
    '''
    Short-time Fourier transform of .wav file
    '''
    # STFT Transform
    x_freq = np.abs(librosa.stft(amp_array, 
                                n_fft = n_fft,  
                                hop_length = hop_length))

    return x_freq

def mel_spectogram(amp_array, sr, n_fft = 512, n_mels = 128):
    """
    Create a mel spectorgram
    """
    mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

    mss = librosa.feature.melspectrogram(amp_array, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mss_scaled = librosa.power_to_db(mss, ref=np.max) # log scales power

    return mss_scaled


def mfcc_spectogram(amp_array, sr, n_mfcc=20):
    return librosa.feature.mfcc(y=amp_array, sr=sr, n_mfcc=n_mfcc)


## Pad Images for Modeling
def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0):
    '''
    Pad numpy array with some value (default = 0)

    link: https://stackoverflow.com/questions/19349410/how-to-pad-with-zeros-a-tensor-along-some-axis-python
    '''
    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)