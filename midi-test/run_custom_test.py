#!../bin/python
import mido
import string
import numpy as np

N_PIANO_NOTES            = 88
DEFAULT_TEMPO            = 500000
DEFAULT_TPM              = 43
DEFAULT_NOTE_OFFSET      = 21
DEFAULT_RELEASE_VELOCITY = 64


def plot_midi(res):
  import matplotlib.pyplot as plt
  plt.plot(range(res.shape[0]), np.multiply(np.where(res>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
  plt.title("nocturne_27_2_(c)inoue.mid")
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
    else:
      print(msg.type)
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

  print(delta[:5])

  # parse transitions
  time_ = 0
  for d in delta:
    if d.any():
      notes_on = np.where(d > 0)[0]
      notes_on_vel = d[notes_on]
      notes_off = np.where(d > 0)[0]

      for n,v in zip(notes_on, notes_on_vel):
        msg = mido.Message('note_on', note=n+DEFAULT_NOTE_OFFSET, velocity=v, time=time_)
        track.append(msg)
      for n in off_notes:
        msg = mido.Message('note_off', note=n+DEFAULT_NOTE_OFFSET, velocity=DEFAULT_RELEASE_VELOCITY, time=time_)
        track.append(msg)

      sys.exit(0)
    else: # if no change -> we up the time since we're repeating/holding last state 
      time_ += 1

  # add the end of track msg 
  track.append(eot)
  return track


if __name__ == '__main__':
  # IMPORTANT -> IF WE WANT TO AUTOMATE THIS WHOLE PROCESS WE NEED TO CHECK FOR SEVERAL THINGS:
  #               (like n_channels, n_tracks, metadata, program, etc..)

  # load midi
  path = 'res.mid'
  mid = mido.MidiFile(path, clip=True)
  print(f'just loaded: {mid}')
  print(f' \nwith dict:')
  for k,v in mid.__dict__.items(): print(k,': ', v)
  print(f' \nwith sample:')
  for msg in mid.tracks[0][2:7]: print(msg)

  # now get the numpy array representing the track from our file 
  res = midi2array(mid)
  print('\n Output:')
  print(res)
  print(res.shape)
  #print('\n Example:')
  #print(res[:5])
#  plot_midi(res)

  # init new MIDI file 
  out = mido.MidiFile(type=0)
  for key, value in mid.__dict__.items():
    if key not in ['file', 'tracks']: setattr(out, key, value)

  # init track
  metadata = [msg for msg in mid.tracks[0] if msg.type not in ['note_on', 'note_off']]
  track = array2track(res, tempo=DEFAULT_TEMPO, metadata=metadata) 
  for x in track: print(x)
  out.tracks.append(track)
  out.save('midi_new.res')
  print('done!')

