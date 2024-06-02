import torch
import torch.nn as nn
import brdf_utils
from torch import Tensor as T
import torch.nn.functional as F
from prl_loss_aux import VGGStyleModel, ImageDiscriminator
from prl_config import ConfigData
from torch.optim.lr_scheduler import ExponentialLR
import lpips

class BaseLoss:
    # (out_pred, out_targets, tl, tv) = N, M, 3, H, W;;  (x_rep, in_x, in_l, in_v, in_n) = M, 3, H, W;;
    def forward(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        pass
    # (out_pred, out_targets, tl, tv) = N, M, 3, H, W;; (x_rep, in_x, in_l, in_v, in_n) = M, 3, H, W;;
    def train_update(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        pass
    def epoch_update(self, epoch):
        pass

class LogLoss_sq(nn.Module):
    def __init__(self):
        super(LogLoss_sq, self).__init__()
    
    # (out_pred, out_targets, tl, tv) = N, M, 3, H, W;; (x_rep, in_x, in_l, in_v, in_n) = N, 3, H, W;;
    def forward(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        assert(tl.dim() == 5 and out_preds.dim() == 5)

        l = brdf_utils.normalize(tl,d=-3)
        n = brdf_utils.normalize(in_n,d=-3)

        cos_weights = brdf_utils.tensor_dot(l, n[:, None],d=-3).unsqueeze(-3)
        cos_weights = torch.clamp(cos_weights, min=0.0, max=1.0)

        log_predicted_brdf = torch.log((out_preds*cos_weights) + 1.0)
        log_ground_truth_brdf = torch.log((out_targets*cos_weights) + 1.0)

        return torch.square(log_predicted_brdf - log_ground_truth_brdf).flatten(0, 1).sum(0).mean()
    
    def train_update(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        pass

    def epoch_update(self, epoch):
        pass

class LogLoss_abs(nn.Module):
    def __init__(self):
        super(LogLoss_abs, self).__init__()
    
    def forward(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        
        l = brdf_utils.normalize(tl,d=-3)
        n = brdf_utils.normalize(in_n,d=-3)

        cos_weights = brdf_utils.tensor_dot(l, n[:, None], d=-3).unsqueeze(-3)
        cos_weights = torch.clamp(cos_weights, min=0.0, max=1.0)

        log_predicted_brdf = torch.log((out_preds*cos_weights) + 1.0)
        log_ground_truth_brdf = torch.log((out_targets*cos_weights) + 1.0)

        return torch.abs(log_predicted_brdf - log_ground_truth_brdf).flatten(0, 1).sum(0).mean()
        
    def train_update(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        pass

    def epoch_update(self, epoch):
        pass


class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        return F.mse_loss(out_preds, out_targets, reduction=self.reduction)
    
    def train_update(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        pass

    def epoch_update(self, epoch):
        pass

class MAELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        return F.l1_loss(out_preds, out_targets, reduction=self.reduction)
    
    def train_update(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        pass

    def epoch_update(self, epoch):
        pass

class WrapLoss_KillUpdate(nn.Module):
    def __init__(self, real_loss_module):
        super().__init__()
        self.real_loss_module = real_loss_module
    
    def forward(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        return self.real_loss_module(out_preds, out_targets, tl, tv, x_rep, in_x, in_l, in_v, in_n)
    def train_update(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        pass # KILL TRAINING UPDATES
    def epoch_update(self, epoch):
        pass # KILL EPOCH UPDATES

class MAELoss_SampledByNeuralNorm(nn.Module):
    def __init__(self, percent, squared=True, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.percent = percent
        self.squared = squared
    
    def forward(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):

        num_samples = int(x_rep.size(-1) * x_rep.size(-2) * 0.5)
        inv_x_rep_norm = 1.0 / (torch.norm(x_rep.detach().clone(), p=2, dim=-3) + 1e-7)
        select_per_im = torch.multinomial(inv_x_rep_norm.flatten(-2, -1), num_samples)
        # select_per_im is an N, W matrix where N is the number of images and W is the selected pixels per image
        mask = torch.zeros(x_rep.size(0), x_rep.size(-2) * x_rep.size(-1), 
                           dtype=torch.bool, device=x_rep.device)
        mask[torch.arange(select_per_im.size(0))[:,None],select_per_im] = True
        mask = mask[:,None].unflatten(-1, (x_rep.size(-2), x_rep.size(-1))).repeat(1, out_preds.size(1), 1, 1)
        # mask = (IN, 256 * 256) --> (IN, OUT, 256, 256)
        return F.l1_loss(out_preds.permute(0,1,3,4,2)[mask], out_targets.permute(0,1,3,4,2)[mask], reduction=self.reduction)
    
    def train_update(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        pass

    def epoch_update(self, epoch):
        pass

class MAELoss_SampledByTargetNorm(nn.Module):
    def __init__(self, percent, squared=True, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.percent = percent
        self.squared = squared
    
    def forward(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        num_samples = int(x_rep.size(-1) * x_rep.size(-2) * 0.5)
        target_norm = torch.norm(out_targets, p=2, dim=-3) + 0.5
        select_per_im = torch.multinomial(target_norm.flatten(-2, -1).flatten(0,1), num_samples)
        # select_per_im is an N, W matrix where N is the number of images and W is the selected pixels per image
        mask = torch.zeros(out_targets.size(0) * out_targets.size(1), x_rep.size(-2) * x_rep.size(-1), 
                           dtype=torch.bool, device=x_rep.device)
        mask[torch.arange(select_per_im.size(0))[:,None],select_per_im] = True
        mask.unflatten(0, (out_targets.size(0), out_targets.size(1))).unflatten(-1, (x_rep.size(-2), x_rep.size(-1)))
        # mask = (IN * OUT, 256 * 256) --> (IN, OUT, 256, 256)
        return F.l1_loss(out_preds.permute(0,1,3,4,2)[mask], out_targets.permute(0,1,3,4,2)[mask], reduction=self.reduction)
    
    def train_update(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        pass

    def epoch_update(self, epoch):
        pass



def z_scoring(x_rep : T, thresh=1.5):

    pr_rep = torch.norm(x_rep, dim=1)
    pr_min, pr_max = pr_rep.min(), pr_rep.max()
    pr_rep = (pr_rep - pr_min) / (pr_max - pr_min)
    mean = pr_rep.mean((1, 2))
    
    # Skip outlier detection if the image is all high-value pixels
    if torch.all(mean > 0.9):
        return torch.zeros_like(pr_rep, dtype=torch.bool)
    
    std = torch.std(pr_rep)
    z_scores = (pr_rep - mean[:,None,None]) / std
    
    outliers = z_scores > thresh
    return outliers


class NormFlairLoss(nn.Module):
    def __init__(self, max_norm):
        super().__init__()
        self.max_norm = max_norm
    def forward(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        norm = torch.norm(x_rep, p=2, dim=-3, keepdim=True)
        mask = (norm > self.max_norm).float()
        pen_loss = torch.nn.functional.mse_loss(x_rep * mask, torch.zeros_like(x_rep, device=x_rep.device))
        return pen_loss
    def train_update(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        pass
    def epoch_update(self, epoch):
        pass

class NormFlairLossWithZMask(nn.Module):
    def __init__(self, std_thresh=2.0):
        super().__init__()
        self.std_thresh = std_thresh
    def forward(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        with torch.no_grad():
            z_mask = z_scoring(x_rep, thresh=self.std_thresh).unsqueeze(1)
        pen_loss = torch.nn.functional.mse_loss(x_rep * z_mask, torch.zeros_like(x_rep, device=x_rep.device))
        return pen_loss
    
    def train_update(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        pass
    def epoch_update(self, epoch):
        pass



class MultiSumLoss(nn.Module):
    def __init__(self, losses : list[tuple[nn.Module, float]]):
        super().__init__()
        self.losses = losses
        self.mods = nn.ModuleList([mod for mod, _ in losses])

    def forward(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        final_loss = self.losses[0][0](out_preds, out_targets, tl, tv, x_rep, in_x, in_l, in_v, in_n) * self.losses[0][1]

        for loss_func, loss_mult in self.losses[1:]:
            final_loss = final_loss + (loss_func(out_preds, out_targets, tl, tv, x_rep, in_x, in_l, in_v, in_n) * loss_mult)
        return final_loss
    def train_update(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        for loss_func, loss_mult in self.losses:
            loss_func.train_update(out_preds, out_targets, tl, tv, x_rep, in_x, in_l, in_v, in_n)
        
    def epoch_update(self, epoch):
        for loss_func, loss_mult in self.losses:
            loss_func.epoch_update(epoch)

class AlexPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lpips_net = lpips.LPIPS(net="alex", pretrained=True)

    def alt_forward(self, predictions, targets):
        pred = torch.clamp(predictions, 0.0, 1.0)
        trgt = torch.clamp(targets, 0.0, 1.0)
        per_batch = self.lpips_net.forward(pred, trgt, normalize=True)
        return per_batch.mean()

    def forward(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        return self.alt_forward(out_preds.flatten(0,1), out_targets.flatten(0,1))
    
    def train_update(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        pass
    def epoch_update(self, epoch):
        pass

class LPIPS_With_Gram(nn.Module):
    def __init__(self, m_device):
        super().__init__()
        self.vggnet = VGGStyleModel()
        self.vggnet.initialize(m_device)

    def forward(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        x, y = out_preds.flatten(0,1).clamp(0,1), out_targets.flatten(0,1).clamp(0,1)
        return self.vggnet.VGGLoss((x * 2) - 1, (y * 2) - 1)
    
    def train_update(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        pass
    def epoch_update(self, epoch):
        pass


class DiscriminatorLoss(nn.Module):
    def __init__(self, disc_lr, disc_lr_gamma, config : ConfigData):
        super().__init__()
        self.disc_net = ImageDiscriminator(layers=5).train()
        self.disc_optimizer = torch.optim.Adam(params=self.disc_net.parameters(), lr=disc_lr)
        self.scheduler_disc = ExponentialLR(self.disc_optimizer, gamma=disc_lr_gamma)


    def forward(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        m = out_targets.size(1)
        disc_fake_x, disc_real_input = out_targets.flatten(0,1), in_x.unsqueeze(1).repeat(1, m, 1, 1, 1).flatten(0,1)
        disc_trick_out = self.disc_net(torch.cat((disc_fake_x, disc_real_input.detach().clone()),dim=1))
        disc_loss = F.l1_loss(disc_trick_out,torch.ones_like(disc_trick_out,device=disc_trick_out.device))
        return disc_loss

    def train_update(self, out_preds : T, out_targets : T, tl : T, tv : T, x_rep : T, in_x : T, in_l : T, in_v : T, in_n : T):
        
        m = out_targets.size(1)

        #discriminator training
        self.disc_optimizer.zero_grad()
        
        # List of real renders, then list of real inputs
        assert(in_x.dim() == 4)

        disc_real_x, disc_real_input = out_targets.flatten(0,1), in_x.unsqueeze(1).repeat(1, m, 1, 1, 1).flatten(0,1)
        disc_real_out = self.disc_net(torch.cat((disc_real_x,disc_real_input.detach().clone()),dim=1))
        disc_true_loss = F.mse_loss(disc_real_out, torch.ones_like(disc_real_out,device=disc_real_out.device))
        
        disc_fake_x = out_preds.flatten(0,1)
        disc_fake_out = self.disc_net(torch.cat((disc_fake_x,disc_real_input.detach().clone()),dim=1))
        disc_fake_loss = F.mse_loss(disc_fake_out, torch.zeros_like(disc_fake_out,device=disc_fake_out.device))
        
        disc_wrong_input = torch.roll(in_x.clone().detach(),1,dims=0)
        disc_wrong_out = self.disc_net(torch.cat((disc_wrong_input,in_x.clone().detach()),dim=1))
        disc_wrong_loss = F.mse_loss(disc_wrong_out, torch.zeros_like(disc_wrong_out,device=disc_wrong_out.device)) 

        disc_train_loss = disc_true_loss + disc_fake_loss + disc_wrong_loss

        disc_train_loss.backward()
        self.disc_optimizer.step()


    def epoch_update(self, epoch):
        self.scheduler_disc.step()
        print("Updated discriminator learning rate on epoch", epoch)
