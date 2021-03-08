import mido
import numpy as np
import matplotlib.pyplot as plt
from multi_channel_parser import *

import os
import sys
import string

# I know these should be in const :|
N_PIANO_NOTES            = 88
DEFAULT_TEMPO            = 500000
DEFAULT_TPM              = 43
DEFAULT_NOTE_OFFSET      = 21
DEFAULT_RELEASE_VELOCITY = 64


def print_data(midis):
  for mid in midis:
    print(mid.__dict__)
    print(len(mid.tracks))
    for t in mid.tracks:
      for note in t[:3]:
        print(note)
    print('')

def parse(midis):
  for idx, mid in enumerate(midis):
    res = mid2arry(mid)
    other = arry2mid(res)
    print(res.shape)
    print(mid)
    print(other)
    print()
    mid.save(f'./data/output/{idx}.mid')


def run():
  midis = []
  for i, d in enumerate(os.listdir('../ext')):
    path = '../ext/'+d
    try:
      mid = mido.MidiFile(path)
      midis.append(mid)
    except:
      pass
  parse(midis)

if __name__ == '__main__':
  run()

