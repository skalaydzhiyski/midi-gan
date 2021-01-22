import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
  def __init__(self, noise_dim=128):
    super().__init__()
    self.noise = torch.randn(1,1,noise_dim)
    self.tconv1 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=129, stride=1, padding=0, bias=False)
    self.tconv2 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=257, stride=1, padding=0, bias=False)
    self.tconv3 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=513, stride=1, padding=0, bias=False)
    self.tconv4 = torch.nn.ConvTranspose2d(1024, 1, (88,4096), 1, 0, bias=False)

  def forward(self):
    res = self.tconv1(self.noise)
    res = self.tconv2(res)
    res = self.tconv3(res)
    res = res.reshape(1,1024,1,1)
    res = self.tconv4(res)
    return res

  
