import BeatSynthUtility

beat_names = ["Playboi Cari 1", "Playboi Carti 2"]
beat_paths = ["Beats/beat1.mp3", "Beats/beat2.mp3"]

beats = BeatSynthUtility.load_sound_files(beat_paths)
print(beats[1][0].shape)
print(beats[0][0].shape)

#BeatSynthUtility.plot_specgram(beat_names, beats)
BeatSynthUtility.plot_waves(beat_names, beats)
#BeatSynthUtility.plot_log_power_specgram(beat_names, beats)

