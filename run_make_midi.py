#!./bin/python
import mido
import numpy as np

import os
import pickle

from midi.parser import array2track
from conf import Path


def make_midi(track):
  print(f'converting {track} output to MIDI..')

def load_meta(idx):
  with open(Path.METADATA_PATH + idx, 'rb') as f:
    res = pickle.load(f)
  return res

if __name__ == '__main__':
  mats = np.load(Path.TRAIN_DATA_PATH + 'train.npy')
  for i,m in enumerate(mats):
    mid = mido.MidiFile(type=0)
    # load file attrs and metadata
    attrs, metadata = load_meta(i)
    print(attrs, metadata)

    # build track 
    m = m.reshape(m.shape[::-1])
    track = array2track(m)


