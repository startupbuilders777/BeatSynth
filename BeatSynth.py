import BeatSynthUtility
import librosa
import librosa.display
import IPython.display
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.style as ms
beat_names = ["Playboi Cari 1", "Playboi Carti 2"]
beat_paths = ["Beats/beat1.mp3", "Beats/beat2.mp3"]

beats = BeatSynthUtility.load_sound_files(beat_paths)

#print(beats[0][0].shape)
#print(beats[1][0].shape)


#BeatSynthUtility.plot_specgram(beat_names, beats)
#BeatSynthUtility.plot_waves(beat_names, beats)
#BeatSynthUtility.plot_log_power_specgram(beat_names, beats)


def displayWaveplot(y_arr, sr_arr, names):
    for y, sr, name in zip(y_arr, sr_arr, names):
        plt.figure()
        librosa.display.waveplot(y=y, sr=sr)
        plt.title(name)
    plt.show()


def displaySpecshow(y_arr, sr_arr, names):
    for y, sr, name in zip(y_arr, sr_arr, names):
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        plt.figure(figsize=(6, 3))
        plt.subplot(2, 1, 1)
        librosa.display.specshow(chroma, y_axis='chroma')
        plt.title(name)
    plt.show()

#Returns an array of tuples, first element is tempo, second is beat, third is name.
def getBeatsAndTempos(y_arr, sr_arr, names):
    beatsAndTempos = []
    for y, sr, name in zip(y_arr, sr_arr, names):
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beatsAndTempos.append((temp, beats, name))
    return beatsAndTempos






y_arr = []
sr_arr = []

for beat in beats:
    y_arr.append(beat[0])
    sr_arr.append(beat[1])

beatsAndTempos = getBeatsAndTempos(y_arr, sr_arr, beat_names)

for beatAndTempo in beatsAndTempos:
    print(f"For beatAndTempo, {name}")


displayWaveplot(y_arr, sr_arr, beat_names)
displaySpecshow(y_arr, sr_arr, beat_names)


#plt.figure()
#librosa.display.waveplot(y=beats[0][0], sr=beats[0][1])

#plt.show()
