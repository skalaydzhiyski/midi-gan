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
    plt.plot(range(f.shape[0]), np.multiply(np.where(f>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
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
  
  print(f'start building the notes array from {first_note_idx}')
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
  res = track2matrix(mid.tracks[0])
  return np.array(res)


def array2track(arr, tempo=50000, metadata=[]):
  # init and add all metadata
  track = mido.MidiTrack()
  eot = None
  for msg in metadata: 
    if msg.type != 'end_of_track': track.append(msg)
    else: eot = msg

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
      last_time = 0 
    else:
      # if no change -> we UP the time since we're holding the same state
      last_time += 1

  # add the end of track msg 
  track.append(eot)
  return track


if __name__ == '__main__':
  # IMPORTANT -> IF WE WANT TO AUTOMATE THIS WHOLE PROCESS WE NEED TO CHECK FOR SEVERAL THINGS:
  #               (like n_channels, n_tracks, metadata, program, etc..)

  # load midi
  path = 'res.mid'
  mid = mido.MidiFile(path, clip=False)
  print(f'just loaded: {mid}')
  print(f' \nwith dict:')
  for k,v in mid.__dict__.items(): print(k,': ', v)
  print('\nsample:')
  for x in mid.tracks[0][:20]: print(x)

  # now get the numpy array representing the track from our file 
  res = midi2array(mid)
  print('\n Output:')
  print(res)
  print(res.shape)

  # init new MIDI file 
  out = mido.MidiFile(type=0)
  for key, value in mid.__dict__.items():
    if key not in ['file', 'tracks']: setattr(out, key, value)

  # init track
  metadata = [msg for msg in mid.tracks[0] if msg.type not in ['note_on', 'note_off']]
  track = array2track(res, tempo=DEFAULT_TEMPO, metadata=metadata) 
  out.tracks.append(track)
  out.save('other.mid')
  print('done!')

  # Use this to plot the data from both the initial res.mid and the other.mid which is generated based off the encoding of the initial MIDI file.
  #other = mido.MidiFile('other.mid', clip=True)
  #new = midi2array(other)
  #plot_midi([res, new])

