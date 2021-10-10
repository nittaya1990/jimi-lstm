# jimi-lstm

Generate melodies using an LSTM neural network.

[See accompanying tutorial.](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5)

[See accompanying blog post.](https://www.rileynwong.com/blog/2019/2/25/generating-music-with-an-lstm-neural-network)

### Setup

```
$ pip install music21
$ pip install pytorch
$ pip install tqdm
```

### Usage

1. Set training data: Create a folder containing midi files, or use one of the ones provided. `ff_midi_songs` contains music from Final Fantasy, `herbie_midi_songs` contains music by Herbie Hancock.
2. Delete the `data/notes` file in case you change the input songs.
3. In `utils.py`: `for f in glob.glob('herbie_midi_songs/*.mid'):`, edit the folder to the folder containing your training set of midi files.
4. Run `$ python train.py`
5. Run `$ python generate.py`. Your resulting song will be created in the same folder as `output_song.mid`.

### Credits

- https://github.com/rileynwong/lstm-jazz
- https://github.com/InanisV/Generate-music-from-MIDI
