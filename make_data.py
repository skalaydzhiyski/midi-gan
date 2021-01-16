#!./bin/python
import mido
import numpy as np

import os
import sys
import subprocess as sp

from get_mp3 import download_tracks
from downloader import DataDownloader
from conf import Path


def execute():
  download_tracks()


if __name__ == "__main__":
  execute()

