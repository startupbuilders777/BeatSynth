from __future__ import print_function
import librosa
import librosa.display
import IPython.display
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.style as ms

# Load the example track
y, sr = librosa.load(librosa.util.example_audio_file())

y_orig, sr_orig = librosa.load(librosa.util.example_audio_file(),
                     sr=None)
print(len(y_orig), sr_orig)

# Play it back!
IPython.display.Audio(data=y, rate=sr)

# How about separating harmonic and percussive components?
y_h, y_p = librosa.effects.hpss(y)


plt.figure()
librosa.display.waveplot(y=y, sr=sr)
