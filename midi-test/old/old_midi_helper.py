#!../bin/python
import mido
import string
import numpy as np
import sys


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
  result = [0] * 88 if last_state is None else last_state.copy()
  if 21 <= note <= 108:
    result[note-21] = velocity if on_ else 0
  return result


def get_new_state(new_msg, last_state):
  new_msg, on_ = msg2dict(new_msg)
  new_state = switch_note(last_state, note=new_msg['note'], velocity=new_msg['velocity'], on_=on_) if on_ is not None else last_state
  return [new_state, new_msg['time']]


def track2seq(track):
  # piano has 88 notes, corresponding to note id 21 to 108, any note out of the id range will be ignored
  res = []
  last_state, last_time = get_new_state(track[0], [0]*88)
  for i in range(1, len(track)):
    new_state, new_time = get_new_state(track[i], last_state)
    if new_time > 0:
      # this is how we handle time of each step of the track -> we multiply the last state by the number of time 
      # (since the new time is the delta from the last time, i.e. we need to repeat the last state for delta times)
      res += [last_state]*new_time
    last_state, last_time = new_state, new_time
  return res


def mid2arry(mid, min_msg_pct=0.1):
  # for now we assume we only have one track 
  track = mid.tracks[0]
  res = np.array(track2seq(track))
  return np.array(res)


def arry2mid(ary, tempo=500000, ticks_per_beat=480):
  # here we need re-add one row of 0s for the metadata message
  new_ary = np.concatenate([np.array([[0] * 88]), np.array(ary)], axis=0)
  # get the difference (TODO: somehow I thikn we need to fix this part of the processing)
  changes = new_ary[1:] - new_ary[:-1]

  # create a midi file with an empty track
  mid_new = mido.MidiFile(type=0)
  track = mido.MidiTrack()
  track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

  # add difference in the empty track
  last_time = 0
  for ch in changes:
    if set(ch) == {0}:  # no change (if we have no change since last we increase the inteval between notes)
      last_time += 1
    else:
      on_notes = np.where(ch > 0)[0]
      on_notes_vol = ch[on_notes]
      off_notes = np.where(ch < 0)[0]
      first_ = True
      for n, v in zip(on_notes, on_notes_vol):
        new_time = last_time if first_ else 0
        track.append(mido.Message('note_on', note=n + 21, velocity=v, time=new_time))
        first_ = False
      for n in off_notes:
        new_time = last_time if first_ else 0
        track.append(mido.Message('note_off', note=n + 21, velocity=0, time=new_time))
        first_ = False
      last_time = 0

  mid_new.tracks.append(track)
  mid_new.ticks_per_beat = ticks_per_beat
  return mid_new



if __name__ == '__main__':
  # load midi
  path = 'res.mid'
  mid = mido.MidiFile(path, clip=True)
  print(mid)


  # get the tempo and ticks_per_beat
  tpb = mid.ticks_per_beat
  tempo = mid.tracks[0][0].tempo
  print(tpb, tempo)


  res = mid2arry(mid)
  print(res.shape)

  print('saving to other.')
  mid_new = arry2mid(res, tempo*2, tpb)
  print(mid_new.ticks_per_beat)
  print(mid_new)
  mid_new.save('other.mid')


