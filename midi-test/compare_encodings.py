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

# velocities and the notes are sometimes different
def compare_mdi(res, other):
  lim = 20
  for x,y in zip(res.tracks[0][:lim], other.tracks[0][:lim]):
    print(f'res: {x}'); print(f'oth: {y}')
    print (x == y, ' -> ', x.time, ' , ', y.time)
    print()

def compare_arrays(res, other):
  lim = 10
  for x,y in zip(res[:lim], other[:lim]):
    print(x); print(y);
    print(88-np.sum(x == y))


# both of the files are different since the encoding is not retaining all the information we have for some reason 
res = mido.MidiFile('res.mid')
other = mido.MidiFile('other.mid')
compare_mdi(res, other)

res = midi2array(res)
other = midi2array(other)
#compare_arrays(res, other)

print(f"res: {res.shape}")
print(f"other: {other.shape}")

