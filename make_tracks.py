#!./bin/python
import mido
import numpy as np

import os

from midi.parser import array2track
from conf import Path
from const import Const
from util import get_standard_midi_file


def make_tracks(n_samples=1, out_fname='res.npy'):
  # load generated tracks
  generated_data = np.load(Path.GEN_OUTPUT_PATH + out_fname)
  idxs = np.random.randint(0, generated_data.shape[0], n_samples)
  samples = generated_data[idxs]
  for i,s in enumerate(samples):
    # get default metadata into file 
    mid = get_standard_midi_file()
    # add notes
    s = s.squeeze()
    s = s.reshape(s.shape[::-1])
    track = array2track(s)
    mid.tracks[0] += track
    out_path = Path.OUTPUT_DATA_PATH + str(i) + '.mid'
    mid.save(out_path)
  print("all tracks saved to './data/output")


if __name__ == '__main__':
  make_tracks()

