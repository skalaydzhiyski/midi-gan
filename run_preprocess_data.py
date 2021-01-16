#!./bin/python
import numpy as np
import mido
import os
from midi.parser import midi2array

INPUT_DATA_PATH = './data/input/'
OUTPUT_DATA_PATH = './data/output/'


def is_note(msg):
  return msg.type in ('note_on', 'note_off')

def is_attr(kv):
  return kv[0] not in ('filename', 'track')

def make_data():
  attrs, metadata, mats = [], [], []
  for d in os.listdir(INPUT_DATA_PATH):
    # read in midi
    mid = mido.MidiFile(INPUT_DATA_PATH + d, clip=False)
    # get params and metadata
    a = dict([x for x in mid.__dict__.items() if is_attr(x)])
    m = dict([(i,x) for i,x in enumerate(mid.tracks[0]) if not is_note(x)])
    attrs.append(a)
    metadata.append(m)
    # parse to matrix 
    mat = midi2array(mid)
    mats.append(mat)
  return attrs, metadata, np.array(mats)


if __name__ == "__main__":
  res = make_data()
  print(res)


