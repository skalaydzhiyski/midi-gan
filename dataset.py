
from const import Const

class Dataset:
  def __init__(self, x, transform):
    self.x = x
    self.transform = transform

  def __getitem__(self, idx):
    if self.transform is not None:
      return self.transform(self.x[idx])
    return self.x[idx]

  def __len__(self):
    return len(self.x)

  def __repr__(self):
    return f'Dataset [shape: {self.x.shape}]'

