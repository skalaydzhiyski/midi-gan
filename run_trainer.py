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


def run_training_loop():
  # build a dataset instance.
  transform = transforms.Compose([
      transforms.ToTensor(),
  ])
  train_data = np.load('./data/train/train.npy')
  train_data = train_data[:2]
  print(train_data.shape)
  n_tracks = train_data.shape[0]
  train_data = train_data.reshape(n_tracks, 1, Const.N_PIANO_NOTES, Const.TRACK_PART_SIZE)
  train_ds = Dataset(train_data, transform=transform)
  print(f'Training data: {train_ds}')

  # create a data loader
  train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)

  # init models 
  noise = torch.randn(batch_size, 1, 128)
  gen = Generator(batch_size, ngpu)
  dis = Discriminator(ngpu)
  print(f'{gen}\n{dis}')

  # loss and optim
  criterion = torch.nn.BCELoss()
  real_label = 1.
  fake_label = 0.
  optim_gen = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.9999))
  optim_dis = torch.optim.Adam(dis.parameters(), lr=lr, betas=(beta1, 0.9999))

  # stats and intermediate results
  sample_tracks = []
  losses_dis, losses_gen = [], []
  iters = 0
  print_rate = 1
  save_sample_rate = 250
  sample_batch_size = 2
  fixed_noise = torch.randn(sample_batch_size, 1, Const.NOISE_WIDTH, device=device)
  
  for i, epoch in enumerate(range(n_epochs)):
    for i, real_batch in enumerate(train_dl):
      # Train Discriminator
      dis.zero_grad()

      # run a real batch through discriminator
      label = torch.full((batch_size,), real_label, device=device).reshape(batch_size, 1)
      output = dis(real_batch)

      loss_real = criterion(output, label)
      loss_real.backward()

      D_x = output.mean().item()

      # run a fake batch through discriminator
      noise = torch.randn(batch_size, 1, 128)
      fake_batch = gen(noise)
      label = label.fill_(fake_label)
      output = dis(fake_batch.detach())
      loss_fake = criterion(output, label)
      loss_fake.backward()

      D_G_z1 = output.mean().item()
      total_loss_dis = loss_real + loss_fake
      optim_dis.step()

      # Train Generator
      gen.zero_grad()
      label.fill_(real_label)
      output = dis(fake_batch)
      total_loss_gen = criterion(output, label)
      total_loss_gen.backward()
      D_G_z2 = output.mean().item()
      optim_gen.step()
  
      if i % print_rate == 0:
        print('[%d/%d][%d/%d]\tDis Loss: %.4f\tGen Loss: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, n_epochs, i, len(train_dl), total_loss_dis.item(), total_loss_gen.item(), D_x, D_G_z1, D_G_z2))

      losses_dis.append(total_loss_dis)
      losses_gen.append(total_loss_gen)

      # store sample to measure progress of the generator
      if (iters % save_sample_rate == 0) or ((epoch == n_epochs-1) and (i == len(train_dl)-1)):
        with torch.no_grad():
          fake_track = gen(fixed_noise).detach().cpu()
          fake_track = np.interp(fake_track, (fake_track.min(), fake_track.max()), (0, 127)).astype(int)
        sample_tracks.append(fake_track)
      iters += 1

  # store samples
  print("Training finished. Saving generated tracks..")
  out_path = Path.GEN_OUTPUT_PATH+'res.npy'
  np.save(out_path, np.concatenate(sample_tracks, axis=0))
  print("done.")


if __name__ == '__main__':
  run_training_loop()

