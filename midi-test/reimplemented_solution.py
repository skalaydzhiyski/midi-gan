#!../bin/python
import mido
import numpy as np
import matplotlib.pyplot as plt

import sys
import string

N_PIANO_NOTES            = 88
DEFAULT_TEMPO            = 500000
DEFAULT_TPM              = 43
DEFAULT_NOTE_OFFSET      = 21
DEFAULT_RELEASE_VELOCITY = 64


def plot_midi(files):
  for idx, f in enumerate(files):
    fig = plt.figure()
    fig.suptitle(str(idx))
    plt.plot(
        range(f.shape[0]), np.multiply(np.where(f>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
  plt.show()


def msg2dict(msg):
  res = {}
  # determine type of message
  on_ = None
  if hasattr(msg, 'type') and msg.type == 'note_on':
    on_ = True
  elif hasattr(msg, 'type') and msg.type == 'note_off':
    on_ = False
  # parse params
  res['time'] = msg.time
  if on_ is not None:
    res['note'] = msg.note
    res['velocity'] = msg.velocity
  return (res, on_)


if __name__ == '__main__':
  # encode a track only from the dictionary and see if it's the same. (very thorough I know, but let's do it once and forget all about it).
  res = mido.MidiFile('./res.mid')
  res_track = res.tracks[0]
  out = mido.MidiFile(type=0)

  # copy properties
  for key,value in res.__dict__.items():
    if key not in ['tracks', 'file']:
      setattr(out, key, value)
  print(res)
  print(out)
  print(res.__dict__)
  print(out.__dict__)
  print('-'*50)

  # copy track metadata
  out_track = mido.MidiTrack()
  d = []
  for msg in res_track[:-1]:
    if msg.type not in ['note_on', 'note_off', 'end_of_track']:
      out_track.append(msg)
    else:
      d.append(msg2dict(msg))


  # copy notes from the dict (dict has ALL the info we need)
  for msg, type_ in d:
    t = 'note_on' if type_ else 'note_off'
    msg = mido.Message(type=t, note=msg['note'], velocity=msg['velocity'], time=msg['time'])
    out_track.append(msg)
  out.tracks.append(out_track)
  out.save('from_dict.mid')

  # Now we have to try and parse the dict into a numpy array



