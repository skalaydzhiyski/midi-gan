#!./bin/python
import mido
import numpy as np

import os
import sys
import subprocess as sp

from preproc import download_tracks, split_tracks, parse_to_midi, make_dataset
from conf import Path
from util import show_func


@show_func
def make_data_dirs():
  if not os.path.exists(Path.DATA_PATH):
    os.mkdir(Path.DATA_PATH)
  for d in Path.DATA_DIRS:
    full_path = Path.DATA_PATH + d
    if not os.path.exists(full_path):
      os.mkdir(full_path)

def execute():
  make_data_dirs()
  # TODO: REMOVE THE FILTER DURATION AND N_LINKS FROM THE FUNC CALL
  download_tracks(n_links=1, filter_duration=10000)
  split_tracks()
  parse_to_midi()
  make_dataset()


if __name__ == "__main__":
  execute()

