import pickle
import numpy as np

from music21 import instrument, note, stream, chord

import torch

from dataset import MIDIDataset
from utils import get_notes, load_notes

def generate():
    """ Generate a piano midi file """
    notes = load_notes()
    n_vocab = len(set(notes))
    sequence_length = 100

    # Convert notes into numerical input
    dataset = MIDIDataset(notes, n_vocab, sequence_length)
    network_input = dataset.network_input.tolist()

    # Use the model to generate a midi
    model = torch.load(f'jimi_lstm.pt')
    model.eval()
    prediction_output = generate_notes(model, notes, network_input,
                                    len(set(notes)))
    create_midi(prediction_output)


def generate_notes(model, notes, network_input, n_vocab):
    """ Generate notes from neural net based on input sequence of notes. """
    print('Generating notes...')

    # Pick random sequence from input as starting point
    start = np.random.randint(0, len(network_input) - 1)

    int_to_note = dict((number, note) for number, note in enumerate(notes))

    pattern = network_input[start]
    prediction_output = []

    n = 100
    for note_index in range(n):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        my_input = torch.DoubleTensor(prediction_input.tolist())
        prediction = model(my_input).detach().cpu().numpy()

        # Take most probable prediction, convert to note, append to output
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        # Scoot input over by 1 note
        pattern = np.append(pattern, index)
        pattern = pattern[1:len(pattern)]

    return prediction_output


def create_midi(prediction_output):
    print('Creating midi...')
    """ Convert prediction output to notes. Create midi file!!!! """
    offset = 0
    output_notes = []

    stored_instrument = instrument.Guitar()

    # Create Note and Chord objects
    for pattern in prediction_output:
        # Pattern is a Chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = stored_instrument
                notes.append(new_note)

            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else: # Pattern is a note
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = stored_instrument
            output_notes.append(new_note)

        # Increase offset for note
        # Possible extension: ~ RHYTHM ~
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='output_song.mid')


if __name__ == '__main__':
    generate()
