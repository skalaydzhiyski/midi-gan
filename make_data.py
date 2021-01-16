#!./bin/python
import mido
import numpy as np

import os
import shutil
import pickle

from midi.parser import midi2array

from conf import Path


def is_note(msg):
  return msg.type in ('note_on', 'note_off')

def is_attr(kv):
  return kv[0] not in ('filename', 'track')

def clean():
  print('clean previous data..')
  dirs = [Path.METADATA_PATH, Path.TRAIN_DATA_PATH]
  for d in dirs:
    if os.path.exists(d):
      shutil.rmtree(d)
    os.mkdir(d)

def make_data():
  attrs, metadata, mats = [], [], []
  for d in os.listdir(Path.INPUT_DATA_PATH):
    print(f'parsing {d} ..')
    # read in midi
    mid = mido.MidiFile(Path.INPUT_DATA_PATH + d, clip=False)
    # get params and metadata
    a = dict([x for x in mid.__dict__.items() if is_attr(x)])
    m = dict([(i,x) for i,x in enumerate(mid.tracks[0]) if not is_note(x)])
    attrs.append(a)
    metadata.append(m)
    # parse to matrix 
    mat = midi2array(mid)
    mats.append(mat)
  return attrs, metadata, np.array(mats)

def serialize(data):
  print('serializing..')
  attrs, metadata, mats = data
  for i,(attr,md) in enumerate(zip(attrs, metadata)):
    with open(Path.METADATA_PATH + f'meta_{i}.pkl', 'wb') as f:
      pickle.dump((attr, md), f)
  np.save(Path.TRAIN_DATA_PATH + 'train.npy', mats)


if __name__ == "__main__":
  res = make_data()
  clean()
  serialize(res)


