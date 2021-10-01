import glob
import pickle
import numpy as np

from music21 import converter, instrument, note, chord

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with `categorical_crossentropy`.
    Args:
        y: Array-like with class values to be converted into a matrix
            (integers from 0 to `num_classes - 1`).
        num_classes: Total number of classes. If `None`, this would be inferred
            as `max(y) + 1`.
        dtype: The data type expected by the input. Default: `'float32'`.
    Returns:
        A binary matrix representation of the input. The class axis is placed
        last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)

    return categorical


class Net(nn.Module):
    def __init__(self, network_input, n_vocab):
        super(Net, self).__init__()

        self.lstm1 = nn.LSTM(input_size=100, hidden_size=512,
                             dropout=0.3, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=512, dropout=0.3,
                             bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=1024, hidden_size=512,
                             bidirectional=True, batch_first=True)
        self.dense1 = nn.Linear(1024, 256)
        self.dropout = nn.Dropout(0.3)
        self.dense2 = nn.Linear(256, n_vocab)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # print(type(x))
        # print(list(x.size()))
        x = x.permute([2,0,1])
        x, _ = self.lstm1(x)
        # print(list(x.size()))
        x, _ = self.lstm2(x)
        # print(list(x.size()))
        x, _ = self.lstm3(x)
        # print(list(x.size()))
        x = self.dense1(x)
        # print(list(x.size()))
        x = self.dropout(x)
        x = self.dense2(x)
        # print(list(x.size()))
        x = self.softmax(x)

        return x


class MIDIDataset(Dataset):
    def __init__(self, notes, n_vocab, sequence_length=100):
        # Get pitch names
        pitch_names = sorted(set(n for n in notes))

        # Map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(pitch_names))

        network_input = []
        network_output = []

        # Create input sequences and the corresponding outputs
        for i in range(0, (len(notes) - sequence_length), 1):
            seq_in = notes[i:i + sequence_length]
            seq_out = notes[i + sequence_length]

            seq_in_int = [note_to_int[char] for char in seq_in]
            network_input.append(seq_in_int)

            seq_out_int = note_to_int[seq_out]
            network_output.append(seq_out_int)

        n_patterns = len(network_input)

        # Reshape for LSTM layers
        network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

        # Normalize input
        self.network_input = network_input / float(n_vocab)

        # One-hot encode output
        self.network_output = to_categorical(network_output)


    def __len__(self):
        return len(self.network_input)


    def __getitem__(self, i):
        return self.network_input[i], self.network_output[i]


def get_notes():
    """
    Convert midi songs to notes. Serialize when done.
    """

    notes = []

    for f in glob.glob('herbie_midi_songs/*.mid'):
        print('Parsing song: ', f)
        midi = converter.parse(f)
        notes_to_parse = None

        parts = instrument.partitionByInstrument(midi)

        if parts: # if file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else: # notes are flat stucture
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def load_notes():
    """
    Deserialize notes file.
    """
    notes = []

    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    return notes


def train_network():
    """ Train! that! network! """
    notes = load_notes()

    # Number of pitch names
    n_vocab = len(set(notes))
    #network_input, network_output = prepare_sequences(notes, n_vocab)

    dataset = MIDIDataset(notes, n_vocab)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = Net(dataset.network_input, n_vocab)
    model.double()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(200):
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            print('inputs')
            print(inputs.shape)
            print('labels')
            print(labels.shape)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


if __name__ == '__main__':
    train_network()
