import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
  def __init__(self, noise):
    self.noise = noise
    # Tconv1
    tconv1 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=129, stride=1, padding=0, bias=False)
    # Tconv2
    tconv2 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=257, stride=1, padding=0, bias=False)
    # Tconv3
    tconv3 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=513, stride=1, padding=0, bias=False)
    # Tconv4 to 2D for the piano roll matrix 
    # TODO: Figure this out when the HHKB comes here (and whern there's no more work in Santander)

  
