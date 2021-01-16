#!./bin/python
import mido
import numpy as np

import os
import sys
import shutil
import subprocess as sp

from downloader import DataDownloader
from conf import Path


def download_tracks(artist='pink floyd', n_links=5):
  if len(os.listdir(Path.MP3_DOWNLOAD_PATH)) > 0:
    print('Tracks already downloader')
    return 

  # get links
  dl = DataDownloader() 
  BASE_YOUTUBE_DL_CMD = f'youtube-dl --extract-audio -o "{Path.MP3_DOWNLOAD_PATH}%(id)s.%(ext)s" --match-filter "duration < 800" --restrict-filenames --ignore-errors -x --audio-format mp3 '
  links = dl.get_links(artist, n_links)
  print(links)

  # download and spleet!
  for l in links:
    youtube_id = l.split('=')[1]
    print(f"downloading {l} / {youtube_id}..")
    sp.call(BASE_YOUTUBE_DL_CMD + youtube_id, shell=True)
    print(f"done!\n")

def split_tracks():
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


if __name__ == "__main__":
  # get input 
  #artist = input("Enter artist: ")
  #n_links = input("Number of songs: ")
  artist = "pink floyd"
  n_links = 10
  download_tracks(artist, n_links)

