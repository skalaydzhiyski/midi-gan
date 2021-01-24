#!./bin/python
import mido
import numpy as np
from youtube_search import YoutubeSearch

import os
import sys
import shutil
import subprocess as sp
import pickle

#from midi.parser import midi2array
from midi.multi_channel_parser import midi2array
from conf import Path
from const import Const
from util import show_func


def note(msg):
  return msg.type in ('note_on', 'note_off')

def is_attr(kv):
  return kv[0] not in ('filename', 'track')

def get_links(search_str, n_results):
  res = []
  while len(res) == 0:
    results = YoutubeSearch(search_str, max_results=n_results).to_dict()
    for r in results:
      link = r['url_suffix']
      if not r['id'].startswith('-'):
        res.append(r['url_suffix'])
      else:
        n_results += 1
  return res

def clean():
  print('clean previous data..')
  dirs = [Path.TRAIN_DATA_PATH]
  for d in dirs:
    if os.path.exists(d):
      shutil.rmtree(d)
    os.mkdir(d)

@show_func
def split_track(track):
  padding_start = track.shape[1] % Const.TRACK_PART_SIZE
  new = track[:,padding_start:]
  n_parts = new.shape[1]//Const.TRACK_PART_SIZE
  new = new.reshape(new.shape[::-1])
  segments = np.array(np.split(new, n_parts)).reshape(n_parts, Const.N_PIANO_NOTES, -1)
  return segments

def make_data():
  tracks = os.listdir(Path.INPUT_DATA_PATH)
  mid = mido.MidiFile(Path.INPUT_DATA_PATH + tracks[0], clip=False)
  mats = midi2array(mid)
  mats = mats.reshape(mats.shape[::-1])
  res = split_track(mats)
  for d in tracks[1:]:
    print(f'parsing {d} ..')
    # read in midi
    mid = mido.MidiFile(Path.INPUT_DATA_PATH + d, clip=False)
    # parse to matrix 
    mat = midi2array(mid)
    mat = mat.reshape(mat.shape[::-1])
    segs = split_track(mat)
    res = np.concatenate([res, segs], axis=0)
  return res

def serialize(X):
  print('serializing..')
  # add channels to the serialized data to make it easier for training later on 
  np.save(Path.TRAIN_DATA_PATH + 'train.npy', X)


@show_func
def download_tracks(artist='beethoven moonlight sonata', n_links=5, filter_duration=800):
  if len(os.listdir(Path.MP3_DOWNLOAD_PATH)) > 0:
    print('Tracks already downloaded')
    return 
  # get links
  youtube_dl_cmd = f'youtube-dl --extract-audio -o "{Path.MP3_DOWNLOAD_PATH}%(id)s.%(ext)s" --match-filter "duration < {filter_duration}" --restrict-filenames --ignore-errors -x --audio-format mp3 '
  # download
  while len(os.listdir(Path.MP3_DOWNLOAD_PATH)) < n_links:
    links = get_links(artist, n_links)
    for l in links:
      youtube_id = l.split('v=')[1] 
      print(f"downloading {l} / {youtube_id}..")
      sp.call(youtube_dl_cmd + youtube_id, shell=True)
      print(f"done!\n")


@show_func
def split_tracks(classical=True):
  if len(os.listdir(Path.SPLEETER_PATH)) > 0:
      print("Tracks alredy spleeeted :) nothing to do.")
      return

  n_stems = '2' if classical else '4'
  main_part = 'accompaniment' if classical else 'other'

  for fname in os.listdir(Path.MP3_DOWNLOAD_PATH):
    full_path = Path.MP3_DOWNLOAD_PATH + fname
    spleeter_cmd = f'spleeter separate -p spleeter:{n_stems}stems -o {Path.SPLEETER_PATH} {full_path}'
    print(spleeter_cmd)
    sp.call(spleeter_cmd.split())

    # keep only the melody
    path = Path.SPLEETER_PATH+fname.split('.')[0]
    for part in os.listdir(path):
      part_path = path + '/' + part
      if main_part in part:
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

  # clean spleeter dir when done
  for d in os.listdir(Path.SPLEETER_PATH):
    os.remove(Path.SPLEETER_PATH + d)

def make_dataset():
  res = make_data()
  print(f'final dataset shape: {res.shape}')
  clean()
  serialize(res)


if __name__ == '__main__':
  pass

