import mido
import click

import os
import shutil

from preproc import download_tracks, split_tracks, parse_to_midi, make_dataset
from conf import Path
from util import show_func


@show_func
def make_data_dirs(from_ext):
  if not os.path.exists(Path.DATA_PATH):
    os.mkdir(Path.DATA_PATH)
  for d in Path.DATA_DIRS:
    full_path = Path.DATA_PATH + d
    if not os.path.exists(full_path):
      os.mkdir(full_path)
  # move ext data
  if os.path.exists(Path.EXT_DATA):
    for d in os.listdir(Path.EXT_DATA):
      fname = Path.EXT_DATA + d
      try:
        # if we can load it, we can use it
        _ = mido.MidiFile(fname)
        shutil.copyfile(fname, Path.INPUT_DATA_PATH + d) 
      except Exception as e:
        print(f'skipping {fname}... {str(e)}')

@click.command()
@click.option('--from-ext', is_flag=True,
              help='Determines whether to use midi from ext or download your own')
@show_func
def execute(from_ext):
  make_data_dirs(from_ext)
  if not from_ext:
    download_tracks(n_links=1, filter_duration=10000)
    split_tracks()
    parse_to_midi()
  make_dataset()


if __name__ == "__main__":
  execute()

