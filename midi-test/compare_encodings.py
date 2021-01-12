#!../bin/python
import mido
import numpy as np
import matplotlib.pyplot as plt

from run_custom_test import midi2array

def plot_midi(files):
  for idx, f in enumerate(files):
    fig = plt.figure()
    fig.suptitle(str(idx))
    plt.plot(range(f.shape[0]), np.multiply(np.where(f>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
  plt.show()

def similar(left, right):
  # check if the messages are similar without counting the time property
  return left.type == right.type and left.channel == right.channel and left.note == right.note and left.velocity == right.velocity

# velocities and the notes are sometimes different
def compare_mdi(res, other):
  print('first 10')
  lim = 10
  for x,y in zip(res.tracks[0][:lim], other.tracks[0][:lim]):
    print(f'res: {x}'); print(f'oth: {y}')
    print (x == y, ' -> ', x.time, ' , ', y.time)
    print()
  print('last 10')
  lim = 10
  for x,y in zip(res.tracks[0][-lim:], other.tracks[0][-lim:]):
    print(f'res: {x}'); print(f'oth: {y}')
    print (x == y, ' -> ', x.time, ' , ', y.time)
    print()


def check_similar(res, other):
  similar_notes = 0
  total_notes = 0
  bad = []
  for x,y in zip(res.tracks[0], other.tracks[0]):
    if x.type in ['note_on', 'note_off']:
      sim = similar(x,y)
      similar_notes += int(sim)
      total_notes += 1
      if not sim: bad.append((x,y))
  #print([i for i,x in enumerate(s) if not x])
  print(total_notes)
  print(similar_notes)

  print("bad ones:")
  for x, y in bad[:10]:
    print(x)
    print(y)
    print()


def compare_arrays(res, other):
  lim = 10
  for x,y in zip(res[:lim], other[:lim]):
    print(x); print(y);
    print(88-np.sum(x == y))


# both of the files are different since the encoding is not retaining all the information we have for some reason 
res = mido.MidiFile('res.mid')
other = mido.MidiFile('other.mid')
print('lens of both tracks')
print(len(res.tracks[0]))
print(len(other.tracks[0]))
print()

#compare_mdi(res, other)
check_similar(res, other)


#res = midi2array(res)
#other = midi2array(other)
#compare_arrays(res, other)

#print(f"res: {res.shape}")
#print(f"other: {other.shape}")

