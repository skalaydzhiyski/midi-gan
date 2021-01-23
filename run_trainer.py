#!./bin/python
import torch
import random
 
from model import Generator, Discriminator
from const import Const
from conf import Path
from data import Dataset


# params
dataroot   = Path.TRAIN_DATA_PATH
workers    = 2
batch_size = 2    # todo: increase
n_epochs   = 1    # todo: increase
lr         = 0.2
beta1      = .5
ngpu       = 0    # todo: set to the number of gpus in the training machine.
device     = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu') 

# we need reporoducible results
manual_seed = 1948
manual_seed = random.randint(1, 10000)
print(f'Random seed: {seed}')
random.seed(manual_seed)


def run_training_loop():
  noise = torch.randn(batch_size, 1, 128)
  #gen = Generator(noise, ngpu)
  #print(gen)
  
  # get train.npy into a dataset format 

  # noirmalize the dataset

  # create a data loader

  # write the training loop here:

  # Generate samples

  # store samples




if __name__ == '__main__':
  run_training_loop()

