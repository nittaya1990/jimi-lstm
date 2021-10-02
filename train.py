import glob
import pickle
import numpy as np

from music21 import converter, instrument, note, chord

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MIDIDataset
from generate import generate_notes
from utils import get_notes, load_notes


class Net(nn.Module):
    def __init__(self, n_vocab, sequence_length=100):
        super(Net, self).__init__()

        self.lstm1 = nn.LSTM(input_size=sequence_length, hidden_size=512,
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


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    notes = load_notes()
    n_vocab = len(set(notes))
    sequence_length = 100

    dataset = MIDIDataset(notes, n_vocab, sequence_length)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = Net(n_vocab, sequence_length)
    model.double()
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        running_loss = 0.0

        with tqdm(loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            for data in tepoch:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = model(inputs)
                l, n, m = outputs.shape
                outputs = torch.reshape(outputs, (n, m))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        avg_loss = running_loss/len(loader)
        lr = optimizer.param_groups[0]['lr']
        print('EPOCH %3d: loss %.5f' % (epoch+1, avg_loss))

    model_cpu = model.cpu()
    torch.save(model, f'jimi_lstm.pt')

    # Use the model to generate a midi
    model = torch.load(f'jimi_lstm.pt')
    model.eval()

    prediction_output = generate_notes(model, notes, dataset.network_input,
                                        len(set(notes)))
    create_midi(prediction_output)
