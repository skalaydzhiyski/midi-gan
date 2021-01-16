import numpy as np
# Uncomment this when the below issues is fixed (see both of the TODO section)
#import librosa
from urllib import request, parse
from tqdm import tqdm
from youtube_search import YoutubeSearch

import os
import pickle
import subprocess as sp

# TODO: Whole class could be refactored heavily :/

class DataDownloader:
  """
  Purpose: download, parse mp3 to wav and decode audio
  to numpy matrix used for training.
  """
  BASE_URL = 'https://www.youtube.com'
  BASE_YOUTUBE_DL_CMD = f'youtube-dl --extract-audio -o "{os.getcwd()}/data/mp3/%(id)s.%(ext)s" --match-filter "duration < 800" --restrict-filenames --ignore-errors -x --audio-format mp3 '

  def __init__(self, size=10, keywords=[]):
    self.size = size
    self.keywords = keywords

  def gen_ds(self, artist, n_res, ignore_download=False):
    res = DataDownloader.get_all_mp3() \
          if ignore_download else self.get_tracks(artist, n_res)
    for idx, fname in enumerate(res):
      if not self.serialize(self.conv_to_wav(fname), fname):
        del res[idx]
    return res

  def conv_to_wav(self, fname, out='data/wav'):
    res_path = f'{out}/{fname}.wav' 
    cmd = f'ffmpeg -i data/mp3/{fname}.mp3 -acodec pcm_s161e -ac 1 -ar 16000 {res_path}'
    try:
      print(f' running cmd: {cmd}')
      sp.call(cmd.split(' '), stdout=sp.DEVNULL)
    except Exception as e:
      print(f' Exception while converting track: {e}')
      return 0
    return res_path

  def serialize(self, wav, out, base_path='data/proc'):
    # TODO: Fix Librosa issue here:
    #  Librosa errors out here, since there is a dependency to llvmlite (PyPI) package.
    #  They don't seem to have wheels for Python 3.9... the solution is to wait or downgrade.
    print(f' serializing {out} !')
    res_path = f'{base_path}/{out}'
    track, sample_freq = librosa.load(wav)
    try:
      # save track data
      np.save(res_path, track)
      # save sample frequency
      with open(res_path + '.pkl', 'wb') as f:
        pickle.dump(sample_freq, res_path)
      print(' OK')
    except Exception as e:
      print(f' Exception while serializing track: {e}')
      return 0
    return res_path

  def get_tracks(self, search_str, n_results):
    links = self.get_links(search_str, n_results)
    print(f' all links: {links}')
    print(f'total results for search string: {len(links)}')
    for link in tqdm(links):
      ytid = link.split('=')[1]
      print(f'Executing "{self.BASE_YOUTUBE_DL_CMD}{ytid}"')
      sp.call(self.BASE_YOUTUBE_DL_CMD + ytid, shell=True)
    return DataDownloader.get_all_mp3()

  def get_links(self, search_str, n_results):
    res = []
    results = YoutubeSearch(search_str, max_results=n_results).to_dict()
    for r in results:
      res.append(r['url_suffix'])
    return res

  @staticmethod
  def get_all_mp3():
    return [os.path.splitext(f)[0] for f in os.listdir(f'{os.getcwd()}/data/mp3') if not f.startswith('.')]

