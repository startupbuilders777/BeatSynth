


import BeatSynthUtility
import librosa
import librosa.display
import IPython.display
import numpy as np
import array

import matplotlib.pyplot as plt
import matplotlib.style as ms
beat_names = ["Playboi Carti 1", "Playboi Carti 2"]
beat_paths = ["Beats/beat1.mp3", "Beats/beat2.mp3"]


retrieveBeat = {
    "Playboi Carti 1": "Beats/beat1.mp3",
    "Playboi Carti 2": "Beats/beat2.mp3",
}

from pydub import AudioSegment
from pydub.utils import get_array_type, get_encoder_name, get_frame_width, get_min_max_value, get_player_name, get_prober_name


def getAudioData(name):
    '''
    sound._data is a bytestring. I'm not sure what input Mpm expects, but you may need to convert the bytestring to an array like so:
    '''
    sound = AudioSegment.from_mp3(retrieveBeat[name])

    bytes_per_sample = sound.sample_width   #1 means 8 bit, 2 meaans 16 bit
    print("BYTES PER SAMPLE: ")
    print(bytes_per_sample)

    bit_depth = sound.sample_width * 8

    frame_rate = sound.frame_rate
    print("FRAME RATE IS: " + str(frame_rate))

    number_of_frames_in_sound = sound.frame_count()
    number_of_frames_in_sound_200ms = sound.frame_count(ms=200)

    print("NUMBER OF FRAMES IS " + str(number_of_frames_in_sound))
    print("NUMBER OF FRAMES IN SOUND PER 200 MS: " + str(number_of_frames_in_sound_200ms))

    array_type = get_array_type(bit_depth)
    print(array_type)
    numeric_array = array.array(array_type, sound.raw_data)
    channel_count = sound.channels
    print("Number of channels in the audio is: ")
    print(channel_count)


    #audio get array of samples

    samples = sound.get_array_of_samples()
    print("SAMPLES ARE")
    print(len(samples))

    left_sound, right_sound = sound.split_to_mono()     #Split it
    print("FRAMES IN LEFT SOUND " + str(left_sound.frame_count()))
    print("FRAMES IN Right SOUND " + str(right_sound.frame_count()))

    print("LEngth of sample left: " + str(len(left_sound.get_array_of_samples())) )
    print("LEngth of sample right: " + str(len(left_sound.get_array_of_samples())) )

    #number_of_frames_in_sound_for_every_20s = sound.frame_count(ms=20000)
    #print("length of song is: " + str(len(samples)/number_of_frames_in_sound_for_every_20s * 20) + " seconds")

    '''
    COLLECTED DATA:
    BYTES PER SAMPLE: 
    2
    FRAME RATE IS: 48000
    NUMBER OF FRAMES IS 7688495.0
    NUMBER OF FRAMES IN SOUND PER 200 MS: 9600.0
    h
    Number of channels in the audio is: 
    2
    SAMPLES ARE
    15376990
    FRAMES IN LEFT SOUND 7688495.0
    FRAMES IN Right SOUND 7688495.0
    LEngth of sample left: 7688495
    LEngth of sample right: 7688495
    15376990
    '''

    counter = 0
    for i in range(0, len(samples)-1):
        if(samples[i] + counter < 10000):
            samples[i] += counter
        if(counter < 500):
            counter += 2
        else:
            counter = 0

        #print(samples[i])
      #  if(i % 2 == 0):
       #     samples[i] = samples[len(samples) - i] #int(samples[i]/2)
       # else:
        #    samples[i] = samples[len(samples) - i] #int(samples[i] - 0.7*samples[i])
        #samples[i] = 10000    #This mutes the sound
        #samples[i+1] = 500

    new_sound = sound._spawn(samples)
    new_sound.export("aaay", format='mp3')

    '''
    note that when using numpy or scipy you will need to convert back to an array before you spawn:

    import array
    import numpy as np
    from pydub import AudioSegment

    sound = AudioSegment.from_file(“sound1.wav”)
    samples = sound.get_array_of_samples()

    shifted_samples = np.right_shift(samples, 1)

    # now you have to convert back to an array.array
    shifted_samples_array = array.array(sound.array_type, shifted_samples)

    new_sound = sound._spawn(shifted_samples_array)
    '''

    return numeric_array
    #raw_data = sound.raw_data
    #return raw_data


foook = getAudioData("Playboi Carti 1")

print(len(foook))

#MAYBE WE SHUD SPLIT THE BEAT, EVERY SO OFTEN AT PLACES WHERE THE BEAT LOOPS

def trimBeat(sound_data):
    #Remove the white noise from the start and the end.
    return 2+2

def beatSplitter(sound_data):
    #Split the beat after every 20 seconds or more depending on where the beat sounds the same as the previous split
    2+2




def getLenghtOfAudioInSeconds(sound):
    return sound.duration_seconds


sound = AudioSegment.from_mp3(retrieveBeat["Playboi Carti 1"])

data = sound.get_array_of_samples()
frames = sound.frame_count()
frames_per_second = len(data)/frames

print("FRAMES PER SECOND ARE: " + str(frames_per_second))

#IT WAS 2 FRAMES PER SECOND

def getLenghtOfAudioData(data):
    return len(data)/2000

print("length of audio from data is: " + str(getLenghtOfAudioData(data)))
print("actual length of audio from data is: " + str(getLenghtOfAudioInSeconds(sound)))


def splitAudio(sound, numberOfSeconds, DEBUG = False):


    #left, right = sound.split_to_mono()
    #left = left.get_array_of_samples()

    #right = right.get_array_of_samples()
    full = sound.get_array_of_samples()

    '''
    duration_in_seconds = sound.duration_seconds
    frames_per_second = len(left)/ duration_in_seconds
    print("FRAMES PER SECOND IS :" + str(frames_per_second))
    number_of_frames_per_split = int(frames_per_second * 20)
    print("Number of frames per split " + str(number_of_frames_per_split))

    
    for i in range(0, int(len(left)/number_of_frames_per_split) - 1):
        if(len(left[i:]) > number_of_frames_per_split):
            itemLeft = left[i:number_of_frames_per_split]
            itemRight = right[i: number_of_frames_per_split]
            splittedLeft.append(itemLeft)
            splittedRight.append(itemRight)
            i += number_of_frames_per_split
            print("fook")
        else:
            print("fook")
    '''
    splittedLeft = []

    duration_in_seconds = sound.duration_seconds
    frames_per_second = len(full) / duration_in_seconds
    numberOfFramesPerSplit = frames_per_second * numberOfSeconds
    numberOfSplits = len(full)/numberOfFramesPerSplit

    lenghtOfData = len(full)
    #sizeOfEachSplit = lenghtOfData/

    a = 0
    for i in range(0, int(numberOfSplits)):
        splittedLeft.append(full[int(a): int(a + numberOfFramesPerSplit)])
        a += numberOfFramesPerSplit


    if(DEBUG == True):
        sound = AudioSegment.from_mp3(file="Beats/beat1.mp3")
        for i in range(0, len(splittedLeft)):
            #
            print(str(i) + "split")
            new_sound = sound._spawn(splittedLeft[i])
            new_sound.export("aaay" + str(i), format='mp3')




    return splittedLeft, frames_per_second, numberOfFramesPerSplit, numberOfSplits


splitAudio(sound, 20)


def BeatSynth():
    import tensorflow as tf

    #INPUTs
    soundFile = None
    X = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    Y = tf.placeholder("float")

    def AutoEncoder():
        learning_rate = 0.01
        num_steps = 3000
        batch_size = 5

    def GAN():
        def generator(x):
            w = tf.Variable(0.0, name="w1")

        def discriminator():
            2+2






#print(foook[0]
#print(foook[100])
#print(foook[12000])
#print(foook[150000])




#print(getAudioData("Playboi Carti 1"))

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
'''
beatsAndTempos = getBeatsAndTempos(y_arr, sr_arr, beat_names)

for beatAndTempo in beatsAndTempos:
    print("For beatAndTempo", name)
'''

#displayWaveplot(y_arr, sr_arr, beat_names)
#displaySpecshow(y_arr, sr_arr, beat_names)


#plt.figure()
#librosa.display.waveplot(y=beats[0][0], sr=beats[0][1])

#plt.show()

