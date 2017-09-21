from pydub import AudioSegment
from os import listdir
from os.path import isfile, join

def get_silence(audio, threshold, interval):
    "get length of silence in seconds from a wav file"

    # swap out pydub import for other types of audio
    song = AudioSegment.from_wav(audio)

    # break into chunks
    chunks = [song[i:i+interval] for i in range(0, len(song), interval)]

    # find number of chunks with dBFS below threshold
    silent_blocks = 0
    for c in chunks:
        if c.dBFS == float('-inf') or c.dBFS < threshold:
            silent_blocks += 1
        else:
            break

    # convert blocks into seconds
    return round(silent_blocks * (interval/1000), 3)

# get files in a directory
audio_path = 'path/to/directory'
audio_files = [i for i in listdir(audio_path) if isfile(join(audio_path, i))]

threshold = -80 # tweak based on signal-to-noise ratio

interval = 1 # ms, increase to speed up

leading_silences = {a: get_silence(join(audio_path, a),
                                   threshold, interval) for a in audio_files}

# to get tab-separated values:
for name, leading_silence in leading_silences.items():
    print(''.join([name, '\t', str(leading_silence)]))