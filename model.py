#!./bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time 


class Generator(nn.Module):
  def __init__(self, noise):
    super().__init__()
    # TODO: try a different model if results are not great where we replace the big kernel size with bigger strides.
    self.noise = noise
    self.bs = noise.shape[0]
    self.tconv1 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=129, stride=1, padding=0, bias=False)
    self.batchnorm1 = nn.BatchNorm1d(256)

    self.tconv2 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=257, stride=1, padding=0, bias=False)
    self.batchnorm2 = nn.BatchNorm1d(512)

    self.tconv3 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=513, stride=1, padding=0, bias=False)
    self.batchnorm3 = nn.BatchNorm1d(1024)

    self.tconv4 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=1025, stride=1, padding=0, bias=False)
    self.batchnorm4 = nn.BatchNorm1d(2048)

    self.tconv5 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=2049, stride=1, padding=0, bias=False)
    self.batchnorm5 = nn.BatchNorm1d(4096)
    self.tconv6 = torch.nn.ConvTranspose2d(4096, 1, (88,4096), 1, 0, bias=False)

  # Faily certian there are better ways to do this. Please refactor. 
  def swap_last_dims(self, x):
    return x.view(self.bs, x.shape[-1], -1)

  def forward(self):
    res = self.tconv1(self.noise)
    res = self.batchnorm1(self.swap_last_dims(res))
    res = F.relu(res)

    res = self.tconv2(self.swap_last_dims(res))
    res = self.batchnorm2(self.swap_last_dims(res))
    res = F.relu(res)

    res = self.tconv3(self.swap_last_dims(res))
    res = self.batchnorm3(self.swap_last_dims(res))
    res = F.relu(res)

    res = self.tconv4(self.swap_last_dims(res))
    res = self.batchnorm4(self.swap_last_dims(res))
    res = F.relu(res)

    res = self.tconv5(self.swap_last_dims(res))
    res = self.batchnorm5(self.swap_last_dims(res))
    res = F.relu(res)

    res = res.reshape(self.bs,4096,1,1)
    res = self.tconv6(res)
    return res


class Descriminator(nn.Module):

  pass


if __name__ == '__main__':
  noise = torch.randn(2,1,128)
  gen = Generator(noise)()
  print(f"Generated track shape: {gen.shape}")
