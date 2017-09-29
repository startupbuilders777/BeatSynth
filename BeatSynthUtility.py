import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import matplotlib.style as ms


import librosa
import librosa.display

def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        raw_sounds.append((X,sr))
    return raw_sounds

def plot_waves(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        librosa.display.waveplot(np.array(f[0]),sr=f[1])
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 1: Waveplot',x=0.5, y=0.915,fontsize=18)
    plt.show()

def plot_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        specgram(np.array(f[0]), Fs=f[1])
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 2: Spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()

def plot_log_power_specgram(sound_names,raw_sounds):
    i = 1
    fig = plt.figure(figsize=(25,60), dpi = 900)
    for n,f in zip(sound_names,raw_sounds):
        plt.subplot(10,1,i)
        D = librosa.logamplitude(np.abs(librosa.stft(f[0]))**2, ref_power=np.max)
        librosa.display.specshow(D,x_axis='time' ,y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle('Figure 3: Log power spectrogram',x=0.5, y=0.915,fontsize=18)
    plt.show()

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(parent_dir,sub_dirs,file_ext="*.wav"):
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
              mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            except Exception as e:
              print("Error encountered while parsing file: ", fn)
              continue
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, fn.split('/')[2].split('-')[1])
    return np.array(features), np.array(labels, dtype = np.int)

#Convolution
def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)

def extract_features(parent_dir, sub_dirs, file_ext="*.wav", bands=60, frames=41):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    labels = []
    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            sound_clip, s = librosa.load(fn)
            label = fn.split('/')[2].split('-')[1]
            for (start, end) in windows(sound_clip, window_size):
                if (len(sound_clip[start:end]) == window_size):
                    signal = sound_clip[start:end]
                    melspec = librosa.feature.melspectrogram(signal, n_mels=bands)
                    logspec = librosa.logamplitude(melspec)
                    logspec = logspec.T.flatten()[:, np.newaxis].T
                    log_specgrams.append(logspec)
                    labels.append(label)

    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])

    return np.array(features), np.array(labels, dtype=np.int)

############################################################################################
#Sound Mixing Utilities###################################################################


def combine(y_arr, sr, name):
    full_y = []
    for y in y_arr:
        print(y.shape)
        full_y += y
    librosa.output.write("Output/" + name, full_y, sr)

    # print(foook[0]
    # print(foook[100])
    # print(foook[12000])
    # print(foook[150000])




    # print(getAudioData("Playboi Carti 1"))

    beats = BeatSynthUtility.load_sound_files(beat_paths)

    # print(beats[0][0].shape)
    # print(beats[1][0].shape)


    # BeatSynthUtility.plot_specgram(beat_names, beats)
    # BeatSynthUtility.plot_waves(beat_names, beats)
    # BeatSynthUtility.plot_log_power_specgram(beat_names, beats)


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

    # Returns an array of tuples, first element is tempo, second is beat, third is name.
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
    '''
    beatsAndTempos = getBeatsAndTempos(y_arr, sr_arr, beat_names)

    for beatAndTempo in beatsAndTempos:
        print("For beatAndTempo", name)
    '''

    # displayWaveplot(y_arr, sr_arr, beat_names)
    # displaySpecshow(y_arr, sr_arr, beat_names)


    # plt.figure()
    # librosa.display.waveplot(y=beats[0][0], sr=beats[0][1])

    # plt.show()











    #y1, sr1 = librosa.load("Beats/beat1.mp3")
#y2, _ = librosa.load("Beats/beat2.mp3", sr=sr1)

#combine([y1, y2], sr1, "Beats1And2.wav")