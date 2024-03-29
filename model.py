import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from const import Const


class Generator(nn.Module):
  def __init__(self, bs, ngpu):
    super(Generator, self).__init__()
    # TODO: try a different model if results are not great where we replace the big kernel size with bigger strides.
    self.ngpu = ngpu
    self.bs = bs

    self.tconv1 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=129, stride=1, padding=0, bias=False)
    self.batchnorm1 = nn.BatchNorm1d(256)

    self.tconv2 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=257, stride=1, padding=0, bias=False)
    self.batchnorm2 = nn.BatchNorm1d(512)

    self.tconv3 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=513, stride=1, padding=0, bias=False)
    self.batchnorm3 = nn.BatchNorm1d(1024)

    self.tconv4 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=1025, stride=1, padding=0, bias=False)
    self.batchnorm4 = nn.BatchNorm1d(2048)

    self.tconv5 = torch.nn.ConvTranspose2d(2048, 1, (Const.N_PIANO_NOTES,2048), 1, 0, bias=False)

  # Fairly certian there are better ways to do this. Please refactor. 
  def swap_last_dims(self, x):
    return x.view(self.bs, x.shape[-1], -1)

  def forward(self, x):
    res = self.tconv1(x)
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

    res = res.reshape(self.bs,2048,1,1)
    res = self.tconv5(res)
    return res


class Discriminator(nn.Module):
  def __init__(self, ngpu):
    super(Discriminator, self).__init__()
    self.ngpu = ngpu
    # 1D Conv over 2D data done through matching the heights of the kernel with the height of input data.
    height = Const.N_PIANO_NOTES
    self.conv1 = nn.Conv2d(1, 8, kernel_size=(height, 8), stride=4, padding=(0,0))
    self.conv2 = nn.Conv1d(8, 16, kernel_size=4,  stride=4, padding=0)
    self.flat = nn.Flatten() 
    self.fn = nn.Linear(2032, 1)
    
  def forward(self,x):
    res = self.conv1(x)
    res = self.conv2(res.squeeze())
    res = self.flat(res)
    res = self.fn(res)
    res = torch.sigmoid(res)
    return res


if __name__ == '__main__':
  #noise = torch.randn(2,1,128)
  #gen = Generator(0)
  #print(gen)
  #res = gen(noise)
  #print(f"Generated track shape: {gen().shape}")

  x = torch.randn(2,1,88,2048)
  dis = Discriminator(0)
  res = dis(x)
  print(f'Shape of discriminator output: {res.shape}')
  print(res)

