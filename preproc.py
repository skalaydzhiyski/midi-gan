#!./bin/python
import mido
import numpy as np
from youtube_search import YoutubeSearch

import os
import shutil
import subprocess as sp
import pickle

from midi.parser import midi2array
from conf import Path
from const import Const
from util import show_func

# ----------------------------------- UTILS -----------------------------------------

def note(msg):
  return msg.type in ('note_on', 'note_off')

def is_attr(kv):
  return kv[0] not in ('filename', 'track')

def get_links(search_str, n_results):
  res = []
  results = YoutubeSearch(search_str, max_results=n_results).to_dict()
  for r in results:
    res.append(r['url_suffix'])
  return res

def clean():
  print('clean previous data..')
  dirs = [Path.TRAIN_DATA_PATH]
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
    m = dict([(i,x) for i,x in enumerate(mid.tracks[0]) if not note(x)])
    attrs.append(a)
    metadata.append(m)
    # parse to matrix 
    mat = midi2array(mid)
    segs = split_track(mat)
    mats += [segs]
  return np.array(mats)

def split_track(track):
  print(track.shape)
  # clean start of track with padding for the split to have equal lengths
  padding_start = track.shape[1] % Const.TRACK_PART_SIZE
  track = track[padding_start:]
  segments = np.split(track, track.shape[1]/Const.TRACK_PART_SIZE)
  return segments

def serialize(np_data):
  print('serializing..')
  np.save(Path.TRAIN_DATA_PATH + 'train.npy', np_data)

# ---------------------------------------------------------------------------------

@show_func
def download_tracks(artist='pink floyd', n_links=5):
  if len(os.listdir(Path.MP3_DOWNLOAD_PATH)) > 0:
    print('Tracks already downloaded')
    return 
  # get links
  youtube_dl_cmd = f'youtube-dl --extract-audio -o "{Path.MP3_DOWNLOAD_PATH}%(id)s.%(ext)s" --match-filter "duration < 800" --restrict-filenames --ignore-errors -x --audio-format mp3 '
  links = get_links(artist, n_links)
  print(links)
  # download
  for l in links:
    youtube_id = l.split('=')[1]
    print(f"downloading {l} / {youtube_id}..")
    sp.call(youtube_dl_cmd + youtube_id, shell=True)
    print(f"done!\n")


@show_func
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
        os.rename(part_path, path + '.mp3')
        break
      os.remove(part_path)
    shutil.rmtree(path)


@show_func
def parse_to_midi():
  if len(os.listdir(Path.INPUT_DATA_PATH)) > 0:
      print("Tracks already parsed to MIDI. nothing to do.")
      return

  for fname in os.listdir(Path.SPLEETER_PATH):
    full_path = Path.SPLEETER_PATH + fname
    out_path = Path.INPUT_DATA_PATH + fname.split('.')[0]
    waon_cmd = f'waon -i {full_path} -o {out_path}.mid'
    sp.call(waon_cmd.split())


@show_func
def make_dataset():
  res = make_data()
  clean()
  serialize(res)


if __name__ == '__main__':
  pass

