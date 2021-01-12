#!../bin/python
import mido
import string
import numpy as np
import sys


# make files 
path = './res.mid'
res = mido.MidiFile(path, clip=True)
out = mido.MidiFile(type=0)

# make track
track = mido.MidiTrack()

# copy all the track messages into out
for key, val in res.__dict__.items():
  if key not in ['tracks', 'file']: setattr(out, key, val)

res_track = res.tracks[0]
for msg in res_track:
  new_msg = msg
  #if hasattr(new_msg, 'velocity'):
  #  new_msg.velocity = 65
  track.append(new_msg)
out.tracks.append(track)
# saving 
out.save('mid_new.mid')

