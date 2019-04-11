from numpy import clip, empty
# from scipy.io import wavfile
from matplotlib.colors import Normalize
from matplotlib import cm

# multiplier = 20
# Fs, sound_data = wavfile.read('./2-Linearni Sistemi in Konvolucija/SotW.wav')

def distortion(multiplier, sound_data):
    yn = sound_data[:, 0]
    yn = multiplier * yn
    yn = clip(yn, -1, 1)
    new_sig = empty([yn.size, 2])   # Have to specify that new_sig is a 2D array.
    new_sig[:, 0] = yn
    if len(sound_data[0, :]) > 1:            # If y is 2D
        yn = multiplier[:, 1]
        yn = sound_data * yn
        yn = clip(yn, -1, 1)
        new_sig[:, 1] = yn
    return new_sig


# distortion(A, z)