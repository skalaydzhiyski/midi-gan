#!./bin/python
import mido


def get_standard_midi_file():
  res = mido.MidiFile(type=0)
  res.__dict__ = {'filename': None, 'type': 0, 'ticks_per_beat': 43, 'charset': 'latin1', 'debug': False, 'clip': False, 'tracks': []}
  track = mido.MidiTrack()
  meta_msg = mido.MetaMessage(type='set_tempo', tempo=500000, time=0)
  track.append(meta_msg)
  program_change = mido.Message(type='program_change', program=0, channel=0, time=0)
  track.append(program_change)
  res.tracks.append(track)
  return res


if __name__ == '__main__':
  pass

