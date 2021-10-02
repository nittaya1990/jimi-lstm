import numpy as np

from torch.utils.data import Dataset

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
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
    if not num_classes:
        num_classes = np.max(y) + 1
    return np.eye(num_classes, dtype=dtype)[y]

class MIDIDataset(Dataset):
    def __init__(self, notes, n_vocab, sequence_length=100):
        # Get pitch names
        notes = sorted(set(n for n in notes))

        # Map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(notes))

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

        # Reshape for LSTM layers + normalize
        network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
        self.network_input = network_input / float(n_vocab)

        # One-hot encode output
        self.network_output = to_categorical(network_output, dtype='double')


    def __len__(self):
        return len(self.network_input)


    def __getitem__(self, i):
        return self.network_input[i], self.network_output[i]