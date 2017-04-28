from __future__ import print_function
import librosa
import librosa.display
import IPython.display
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.style as ms

# Load the example track
y, sr = librosa.load("Beats/beat1.mp3")

print(len(y), sr)

#y is data
#sr is rate


# Play it back!
IPython.display.Audio(data=y, rate=sr)

# How about separating harmonic and percussive components?
y_h, y_p = librosa.effects.hpss(y)


plt.figure()
librosa.display.waveplot(y=y, sr=sr)


'''
Short-time Fourier transform underlies most analysis.
librosa.stft returns a complex matrix D.
D[f, t] is the FFT value at frequency f, time (frame) t.
'''

D = librosa.stft(y)
print(D.shape, D.dtype)

'''
Magnitude
'''

S, phase = librosa.magphase(D)
print(S.dtype, phase.dtype, np.allclose(D, S * phase))

'''
Constant-Q transforms
The CQT gives a logarithmically spaced frequency basis.
This representation is more natural for many analysis tasks.
'''

C = librosa.cqt(y, sr=sr)
print(C.shape, C.dtype)


'''
Compute Short Term Fourier Transform with a diff hop length

D = librosa.stft(y2, hop_length=   )
'''


'''
librosa.feature
Standard features:
librosa.feature.melspectrogram
librosa.feature.mfcc
librosa.feature.chroma
Lots more...
Feature manipulation:
librosa.feature.stack_memory
librosa.feature.delta
Most features work either with audio or STFT input
'''

melspec = librosa.feature.melspectrogram(y=y, sr=sr)

# Melspec assumes power, not energy as input
melspec_stft = librosa.feature.melspectrogram(S=S**2, sr=sr)

print(np.allclose(melspec, melspec_stft))

#A basic spectrogram display

plt.figure()
librosa.display.specshow(melspec, y_axis='mel', x_axis='time')
plt.title("melspec")
plt.colorbar()

'''
Pick a feature extractor from the librosa.feature submodule and plot the output with librosa.display.specshow
Bonus: Customize the plot using either specshow arguments or pyplot functions
In [ ]:
# Exercise 1 solution

X = librosa.feature.XX()

plt.figure()

librosa.display.specshow(    )
'''

'''
librosa.beat
Beat tracking and tempo estimation
The beat tracker returns the estimated tempo and beat positions (measured in frames)
'''

tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
print("Tempy", tempo)
print("Beats", beats)

#Sonify it:

clicks = librosa.clicks(frames=beats, sr=sr, length=len(y))

#Audio(data=y + clicks, rate=sr)

#Beats can be used to downsample features


chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
chroma_sync = librosa.util.sync(chroma, beats)

plt.figure(figsize=(6, 3))
plt.subplot(2, 1, 1)
librosa.display.specshow(chroma, y_axis='chroma')
plt.ylabel('Full resolution')
plt.subplot(2, 1, 2)
librosa.display.specshow(chroma_sync, y_axis='chroma')
plt.ylabel('Beat sync')

'''

librosa.segment
Self-similarity / recurrence
Segmentation
Recurrence matrices encode self-similarity
R[i, j] = similarity between frames (i, j)

Librosa computes recurrence between k-nearest neighbors.

'''

R = librosa.segment.recurrence_matrix(chroma_sync)
plt.figure(figsize=(4, 4))
librosa.display.specshow(R)

#Affinity weights for each link
R2 = librosa.segment.recurrence_matrix(chroma_sync,
                                       mode='affinity',
                                       sym=True)


plt.figure(figsize=(5, 4))
librosa.display.specshow(R2)
plt.colorbar()

'''
Exercise 2
Plot a recurrence matrix using different features
Bonus: Use a custom distance metric

'''


'''
librosa DECOMPOSITION:

hpss: Harmonic-percussive source separation
nn_filter: Nearest-neighbor filtering, non-local means, Repet-SIM
decompose: NMF, PCA and friends

'''

D_harm, D_perc = librosa.decompose.hpss(D)

#seperated harmonic and percussion
y_harm = librosa.istft(D_harm)

y_perc = librosa.istft(D_perc)

#Audio(data=y_harm, rate=sr)
#Audio(data=y_perc, rate=sr)


# Fit the model
W, H = librosa.decompose.decompose(S, n_components=16, sort=True)


plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1), plt.title('W')
librosa.display.specshow(librosa.logamplitude(W**2), y_axis='log')
plt.subplot(1, 2, 2), plt.title('H')
librosa.display.specshow(H, x_axis='time')


# Reconstruct the signal using only the first component
S_rec = W[:, :1].dot(H[:1, :])

y_rec = librosa.istft(S_rec * phase)

#Audio(data=y_rec, rate=sr)

'''
Exercise 3¶
Compute a chromagram using only the harmonic component
Bonus: run the beat tracker using only the percussive component
'''


Resource ----------------------------------------------------------------
https://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/
https://aqibsaeed.github.io/2016-09-24-urban-sound-classification-part-2/



plt.show()

