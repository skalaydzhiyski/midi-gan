#!./bin/python
import torch
from torchvision import transforms
import numpy as np

import random

from model import Generator, Discriminator
from const import Const
from conf import Path
from dataset import Dataset


# params
dataroot   = Path.TRAIN_DATA_PATH
workers    = 2
batch_size = 2    # todo: increase
n_epochs   = 1    # todo: increase
lr         = .02
beta1      = .5
ngpu       = 0    # todo: set to the number of gpus in the training machine.
device     = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu') 

# we need reporoducible results
seed = 1948
#seed = random.randint(1, 10000)
random.seed(seed)
torch.manual_seed(seed)


def stats(x):
  return x.mean(), x.std()

def test_loader(train_dl):
  batch = next(iter(train_dl))
  print(batch.shape)

def run_training_loop():
  # build a dataset instance.
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((.5,.5,.5),(.5,.5,.5)) 
  ])
  train_data = np.load('./data/train/train.npy')
  print(stats(torch.tensor(train_data).float()))
  train_ds = Dataset(train_data, transform=transform)
  print(f'Training data: {train_ds}')

  # create a data loader
  train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
  test_loader(train_dl)

  # init models 
  noise = torch.randn(2, 1, 128)
  gen = Generator(noise, ngpu)
  dis = Discriminator(ngpu)
  print(gen, dis)
  
  for i, epoch in enumerate(range(n_epochs)):
    for i, real_batch in enumerate(train_dl):
      # TODO: write the training loop here 

  # Generate samples

  # store samples




if __name__ == '__main__':
  run_training_loop()

