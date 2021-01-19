#!../bin/python
import mido
import numpy as np
import matplotlib.pyplot as plt

import sys
import string

# I know these should be in const :|
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
  on_ = None
  if hasattr(msg, 'type') and msg.type == 'note_on':
    on_ = True
  elif hasattr(msg, 'type') and msg.type == 'note_off':
    on_ = False
  res['time'] = msg.time
  if on_ is not None:
    res['note'] = msg.note
    res['velocity'] = msg.velocity
  return (res, on_)


def switch_note(last_state, note, velocity, on_=True):
  # piano has 88 notes, corresponding to note id 21 to 108, any note out of this range will be ignored
  res = last_state.copy()
  if 21 <= note <= 108:
      res[note-21] = velocity if on_ else 0
  return res


def get_new_state(msg, last_state):
  msg, on_ = msg2dict(msg)
  state = switch_note(last_state, msg['note'], msg['velocity'], on_) 
  return state, msg['time']


def track2matrix(track):
  res = []

  # find the stard of music and ignore the meta for now
  first_note_idx = 0
  for i in range(len(track)):
    if not track[i].type not in ['note_on', 'note_off']:
      first_note_idx = i
      break
  
  last_state, last_time = get_new_state(track[first_note_idx], [0]*N_PIANO_NOTES)
  for i in range(first_note_idx, len(track)):
    msg = track[i]
    if msg.type in ['note_on', 'note_off']:
      new_state, new_time = get_new_state(track[i], last_state)
      if new_time > 0:
        res += [last_state] * new_time
      last_state, last_time = new_state, new_time
  return res


def midi2array(mid):
  # we only have one channel for now
  res = np.array(track2matrix(mid.tracks[0]))
  return res.reshape(res.shape[::-1])


def array2track(arr, tempo=50000, metadata=[]):
  # init and add all metadata
  track = mido.MidiTrack()

  # build transitions (delta)
  arr = np.concatenate([[[0]*arr.shape[1]], arr], axis=0)
  current = arr[1:]
  prev = arr[:-1]
  delta = current - prev

  # parse delta into notes.
  last_time = 0
  for d in delta:
    if d.any():
      notes_on = np.where(d>0)[0]
      notes_on_vel = d[notes_on]
      notes_off = np.where(d<0)[0]

      # used to indicate which note should get the new time
      # (all the rest of the notes in the row will have 0 since they're played AT THE SAME TIME)
      first = True
      for n,v in zip(notes_on, notes_on_vel):
        new_time = last_time if first else 0
        msg = mido.Message('note_on', note=n+DEFAULT_NOTE_OFFSET, velocity=v, time=new_time)
        track.append(msg)
        first = False
      for n in notes_off:
        new_time = last_time if first else 0
        msg = mido.Message('note_off', note=n+DEFAULT_NOTE_OFFSET, velocity=DEFAULT_RELEASE_VELOCITY, time=new_time)
        track.append(msg)
        first = False
      last_time = 1 
    else:
      # if no change -> we UP the time since we're holding the same state
      last_time += 1
  return track


if __name__ == "__main__":
  res_mid = mido.MidiFile('./data/res.mid', clip=False)
  res = midi2array(res_mid)

  # copy attributes 
  other_mid = mido.MidiFile(type=0)
  for key, value in res_mid.__dict__.items():
    if key not in ['filename', 'tracks']: setattr(other_mid, key, value)

  # make metadata 
  metadata = [msg for msg in res_mid.tracks[0] if msg.type not in ['note_on', 'note_off']]

  # generate track from encoding
  track = array2track(res, tempo=DEFAULT_TEMPO, metadata=metadata)
  other_mid.tracks.append(track)
  other_mid.save("./data/other.mid")
  print("done!")
  
