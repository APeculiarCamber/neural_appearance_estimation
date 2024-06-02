import torch
import torch.nn as nn

class HalfZedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, lt : torch.Tensor, vt : torch.Tensor):
        return nn.functional.normalize(lt + vt, dim=1)[:,2:3,:,:]
    def get_feature_count(self):
        return 1

class BaseDirectionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, lt, vt):
        return torch.cat([lt, vt], dim=1)
    def get_feature_count(self):
        return 6

class SineDirectionEncoder(nn.Module):
    def __init__(self, freq_count, start=1, rate=2):
        super().__init__()
        self.freq_count = freq_count
        self.start = start
        self.rate = rate
    
    def forward(self, lt, vt):
        processed = torch.cat([lt, vt, nn.functional.normalize(lt + vt, dim=1)], dim=1)
        coords_pos_enc = [processed]
        for i in range(self.freq_count):
            for j in range(processed.size(1)):
                c = processed[:, j]
                sin = torch.unsqueeze(torch.sin(self.start * (self.rate ** i) * torch.pi * c), 1)
                cos = torch.unsqueeze(torch.cos(self.start * (self.rate ** i) * torch.pi * c), 1)
                coords_pos_enc.extend([sin, cos])
        return torch.cat(coords_pos_enc, dim=1)
    def get_feature_count(self):
        return (2 * self.freq_count * 9) + 9

# TODO: this is such a problem: look at how BIG these are!!!
class SineDirectionEncoder_NN(nn.Module):
    def __init__(self, freq_count, start=1, rate=2, out_dims=9):
        super().__init__()
        self.freq_count = freq_count
        self.start = start
        self.rate = rate
        self.out_dims = out_dims
        input_len = (2 * freq_count * 9) + 9
        self.net = nn.Sequential(
            nn.Conv2d(input_len, input_len*2, 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(2 * input_len, 2 * input_len, 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(2 * input_len, out_dims, 1, bias=False),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        
    def forward(self, lt, vt):
        processed = torch.cat([lt, vt, nn.functional.normalize(lt + vt, dim=1)], dim=1)
        coords_pos_enc = [processed]
        for i in range(self.freq_count):
            for j in range(processed.size(1)):
                c = processed[:, j]
                sin = torch.unsqueeze(torch.sin(self.start * (self.rate ** i) * torch.pi * c), 1)
                cos = torch.unsqueeze(torch.cos(self.start * (self.rate ** i) * torch.pi * c), 1)
                coords_pos_enc.extend([sin, cos])
        return self.net(torch.cat(coords_pos_enc, dim=1))
    def get_feature_count(self):
        return self.out_dims