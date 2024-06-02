import torch.nn as nn
import torch


class NaiveMLPRenderer(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        b = False
        self.net = nn.Sequential(
            nn.Conv2d(in_c, 128, 1, bias=b),
            # nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 1, bias=b),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 1, bias=b),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 1, bias=b),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 1, bias=b),
            #nn.LeakyReLU(),
            #nn.Conv2d(128, 128, 1, bias=b),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 1, bias=b),
            nn.LeakyReLU(),
            nn.Conv2d(64, 3, 1, bias=b),
            nn.LeakyReLU(0.001),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if b: nn.init.zeros_(m.bias)

    def forward(self, x, lv_encoding):
        x = torch.cat([x, lv_encoding], dim=1)
        return self.net(x)



class LogMod(nn.Module):
    def __init__(self):
        super(LogMod, self).__init__()
    def forward(self, x):
        return torch.log(1.0 + x)

class ExpoMod(nn.Module):
    def __init__(self):
        super(ExpoMod, self).__init__()
    def forward(self, x):
        return torch.exp(x) - 1.0

class ClampMod(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.clamp(x, -0.9, 1.0)


class ClampedLogMod(nn.Module):
    def __init__(self):
        super(ClampedLogMod, self).__init__()
    def forward(self, x):
        x = torch.clamp(x, 0.0, 0.999999)
        return torch.log(1.0 + x)

class ClampedExpoMod(nn.Module):
    def __init__(self):
        super(ClampedExpoMod, self).__init__()
    def forward(self, x):
        return torch.clamp(torch.exp(x) - 1.0, 0.0, 0.999999)
