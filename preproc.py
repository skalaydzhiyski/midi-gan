#!./bin/python
import mido
import numpy as np

import os
import shutil
import subprocess as sp
import pickle

from midi.parser import midi2array
from downloader import DataDownloader
from conf import Path

# ----------------------------------- UTILS -----------------------------------------

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

  # Here when trying to create a matix with all the tracks - they all have different lengths, i.e can't be cast
  # to the same matrix size
  # TODO: Fix this: print out othe shapes and come up with a standard shape for a midi file for the model inputs.
  return attrs, metadata, np.array(mats)

def serialize(data):
  print('serializing..')
  attrs, metadata, mats = data
  for i,(attr,md) in enumerate(zip(attrs, metadata)):
    with open(Path.METADATA_PATH + f'meta_{i}.pkl', 'wb') as f:
      pickle.dump((attr, md), f)
  np.save(Path.TRAIN_DATA_PATH + 'train.npy', mats)

# ---------------------------------------------------------------------------------

def download_tracks(artist='pink floyd', n_links=5):
  if len(os.listdir(Path.MP3_DOWNLOAD_PATH)) > 0:
    print('Tracks already downloaded')
    return 
  # get links
  dl = DataDownloader() 
  BASE_YOUTUBE_DL_CMD = f'youtube-dl --extract-audio -o "{Path.MP3_DOWNLOAD_PATH}%(id)s.%(ext)s" --match-filter "duration < 800" --restrict-filenames --ignore-errors -x --audio-format mp3 '
  links = dl.get_links(artist, n_links)
  print(links)
  # download
  for l in links:
    youtube_id = l.split('=')[1]
    print(f"downloading {l} / {youtube_id}..")
    sp.call(BASE_YOUTUBE_DL_CMD + youtube_id, shell=True)
    print(f"done!\n")


def split_tracks():
  if len(os.listdir(Path.SPLEETER_PATH)) > 0:
      print("Tracks alredy spleeeted :) nothing to do.")
      return

  # split chunks kand extract melody only
  for fname in os.listdir(Path.MP3_DOWNLOAD_PATH):
    # split the instrumental of a track (i.e. remove drums and vocals)
    full_path = Path.MP3_DOWNLOAD_PATH + fname
    print(f'splitting {fname}')
    spleeter_cmd = f'spleeter separate -p spleeter:4stems -o {Path.SPLEETER_PATH} {full_path}'
    sp.call(spleeter_cmd.split())

    # remove extra parts 
    path = Path.SPLEETER_PATH+fname.split('.')[0]
    for part in os.listdir(path):
      part_path = path + '/' + part
      if 'other' in part:
        gs.rename(part_path, path + '.mp3')
        break
      os.remove(part_path)
    shutil.rmtree(path)


def parse_to_midi():
  if len(os.listdir(Path.INPUT_DATA_PATH)) > 0:
      print("Tracks already parsed to MIDI. nothing to do.")
      return

  for fname in os.listdir(Path.SPLEETER_PATH):
    full_path = Path.SPLEETER_PATH + fname
    out_path = Path.INPUT_DATA_PATH + fname.split('.')[0]
    waon_cmd = f'waon -i {full_path} -o {out_path}.mid'
    sp.call(waon_cmd.split())


def make_dataset():
  res = make_data()
  clean()
  serialize(res)


if __name__ == '__main__':
  pass

