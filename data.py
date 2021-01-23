
class Dataset:
  def __init__(self, x, y):
    self.x, self.y = x, y
  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]
  def __len__(self):
    return len(self.x)

