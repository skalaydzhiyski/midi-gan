#!./bin/python
import mido
import numpy as np

import os
import sys
import subprocess as sp

from get_mp3 import download_tracks, split_tracks
from conf import Path


def make_data_dirs():
  print('creating data directories..')
  if not os.path.exists(Path.DATA_PATH):
    os.mkdir(Path.DATA_PATH)
  for d in Path.DATA_DIRS:
    full_path = Path.DATA_PATH + d
    if not os.path.exists(full_path):
      os.mkdir(full_path)

def execute():
  make_data_dirs()
  download_tracks()
  split_tracks()


if __name__ == "__main__":
  execute()

