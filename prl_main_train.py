from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch import Tensor as T

import brdf_utils
from torch.optim.lr_scheduler import StepLR

from prl_config import (
    ConfigData, 
    WB_Range, 
    WB_Values, 
)

from prl_nBRDF_dataset import load_fusion_training_data
from prl_loss import ( LogLoss_abs, LogLoss_sq, 
                      MSELoss,
                      MAELoss, MAELoss_SampledByNeuralNorm, MAELoss_SampledByTargetNorm,
                      MultiSumLoss, AlexPerceptualLoss, 
                      LPIPS_With_Gram, DiscriminatorLoss, WrapLoss_KillUpdate)

from models.prl_net import RelitPixelNet, ImSpaceManager
from models.prl_resnet import GenResNetReplicateHA, BasicBlock
from models.prl_pixel_mlp import *
from models.prl_feature_injector import HalfZedEncoder, SineDirectionEncoder_NN

m_device = torch.device("cpu")

def make_comparison_video_no_norm(filename : str, input : T, target : T, pred : T, pred_rep : T):
    assert(input.dim() == 4 and pred_rep.dim() == 4)
    assert(target.dim() == 5 and pred.dim() == 5)
    input = input.unsqueeze(1).repeat(1, target.size(1), 1, 1, 1)
    pred_rep = torch.norm(pred_rep, dim=1).unsqueeze(1).repeat(1, target.size(1), 3, 1, 1)

    g = 1.0 / 2.2
    side_by_side = [t.flatten(0, 1) for t in [input ** g, target ** g, pred_rep, pred ** g]]
    side_by_side = torch.cat(side_by_side, dim=-1) # along width
    # print(f"Rendering to mp4 at {filename}:", side_by_side.shape)
    brdf_utils.writevideo(side_by_side.cpu(), filename, fps=4, desired_resolution=side_by_side.size(-1))


def train_net(epoch : int, model : RelitPixelNet, dataloader : DataLoader, in_optim : Optimizer, render_optim : Optimizer,
              loss_func : MultiSumLoss, render_loss_func : MultiSumLoss):

    space_manager = model.space_manager # MANAGES INPUT AND OUTPUT SPACES TO AND FROM MODEL

    with tqdm(total=len(dataloader), desc=f"Epoch {epoch}", unit="items") as pbar:
        for batch_id, batch in enumerate(dataloader):
            batch : tuple[T, T, T, T, T, T, T, T] = batch
            x, l, v, in_n, tx, tl, tv = [b.to(m_device) for b in batch[:-1]]

            with torch.no_grad():
                tx = space_manager.compress_target_call(tx)

            # RESNET TRAINING
            model.render_net.requires_grad_(False)
            in_optim.zero_grad()
            x_render, x_neural_rep = model.render_multi(x, l, v, tl, tv) # ASSUMES DIM = 5
            f_loss : T = loss_func(x_render, tx, tl, tv, x_neural_rep, x, l, v, in_n)
            f_loss.backward()
            in_optim.step()
            x_render, x_neural_rep = [t.detach().clone() for t in [x_render, x_neural_rep]]
            loss_func.train_update(x_render, tx, tl, tv, x_neural_rep, x, l, v, in_n)
            
            # Render TRAINING
            model.render_net.requires_grad_(True)
            render_optim.zero_grad()
            x_render = model.render_from_neural_rep_multi(x_neural_rep, tl, tv)
            s_loss : T = render_loss_func(x_render, tx, tl, tv, x_neural_rep, x, l, v, in_n)
            s_loss.backward()
            render_optim.step()
            x_render, x_neural_rep = [t.detach().clone() for t in [x_render, x_neural_rep]]
            render_loss_func.train_update(x_render, tx, tl, tv, x_neural_rep, x, l, v, in_n)
            
            del x, tx, x_neural_rep, x_render
            pbar.update(1)


from torch.optim.lr_scheduler import MultiStepLR, ChainedScheduler
def generate_model_training_components(config : ConfigData, num_renders_per_sample, num_samples_per_epoch):
    # MODEL AND MODEL LOADING
    output_depth_factor = 64
    input_encoder, render_encoder = HalfZedEncoder(), SineDirectionEncoder_NN(config.pr_frequencies, out_dims=conf.compressed_freq_encoding_count)
    input_model = GenResNetReplicateHA(BasicBlock, [2,2,2,2], [1,2,2,2], encoder_channels=3 + input_encoder.get_feature_count(), 
                                       output_depth_factor=output_depth_factor, tanh_final_activation=config.use_final_tanh)
    render_model = NaiveMLPRenderer(output_depth_factor + render_encoder.get_feature_count())
    space_manager = ImSpaceManager(LogMod(), ExpoMod(), LogMod(), ExpoMod())
    model = RelitPixelNet(input_encoder, render_encoder, input_model, render_model, space_manager)
    model = model.to(m_device)

    # PARAMETERS & OPTIMIZERS
    in_net_params, in_enc_params = model.resnet.parameters(), model.input_encoder.parameters()
    ren_net_params, ren_enc_params = model.render_net.parameters(), model.render_encoder.parameters()
    in_optimizer = optim.Adam([{'params': in_net_params},
                               {'params': in_enc_params}], lr=config.lr, weight_decay=config.weight_decay)
    render_optimizer = optim.Adam([{'params': ren_net_params},
                                   {'params': ren_enc_params}], lr=config.lr, weight_decay=config.weight_decay)
    input_lr_scheduler = StepLR(in_optimizer, step_size=config.lr_gamma_step, gamma=config.lr_gamma)
    step_render_lr_scheduler = StepLR(render_optimizer, step_size=config.lr_gamma_step, gamma=config.lr_gamma)
    mult_render_lr_scheduler = MultiStepLR(render_optimizer, [20], gamma=config.lr_render_gamma)
    render_lr_scheduler = ChainedScheduler([step_render_lr_scheduler, mult_render_lr_scheduler])

    # DATALOADERS
    tr_dataloader = load_fusion_training_data(batch_size=batch_size, sample_limit=None, crop_size=256, replacement=False,
                                              num_out_samples=num_renders_per_sample, num_samples_per_epoch=num_samples_per_epoch)

    # LOSS FUNCTIONS: MAE Data loss, LPIPS_With_Gram, DiscriminatorLoss
    disc_contrib = 0.03
    disc_loss_mod = DiscriminatorLoss(config.lr, config.lr_gamma, config)
    input_loss_func = MultiSumLoss([(MAELoss(), 1.0), 
                                    (LPIPS_With_Gram(m_device), 0.01), 
                                    (WrapLoss_KillUpdate(disc_loss_mod), disc_contrib),
                                    ]).to(m_device)
    renderer_loss_func = MultiSumLoss([(MAELoss_SampledByNeuralNorm(0.6), 1.0), 
                                    (LPIPS_With_Gram(m_device), 0.01), 
                                    (disc_loss_mod, disc_contrib),
                                    ]).to(m_device)
        
    return model, in_optimizer, render_optimizer, input_lr_scheduler, render_lr_scheduler, \
        tr_dataloader, input_loss_func, renderer_loss_func

def run_plain(config : ConfigData, num_epochs=20, batch_size=2, num_samples_per_epoch=24, num_renders_per_sample=2):

    model, in_optimizer, render_optimizer, input_lr_scheduler, render_lr_scheduler, tr_dataloader, \
     input_loss_func, renderer_loss_func = generate_model_training_components(config, num_renders_per_sample, num_samples_per_epoch)

    for e in range(num_epochs):
        train_net(e, model, tr_dataloader, in_optimizer, render_optimizer, 
                  input_loss_func, renderer_loss_func)
        torch.cuda.empty_cache() 

        input_lr_scheduler.step()
        render_lr_scheduler.step()
        input_loss_func.epoch_update(e)
        renderer_loss_func.epoch_update(e)\
    
    # return the trained model
    return model

if __name__ == "__main__":

    conf = ConfigData()
    conf.lr = 0.00001
    conf.pos_include_half = True
    conf.pixel_feature_depth = 64
    conf.use_final_tanh = False
    conf.lr_gamma = 0.985
    conf.lr_render_gamma = 0.985
    conf.pr_frequencies = 2#16

    conf.crop_size = 32#256 
    conf.compressed_freq_encoding_count = 32

    num_epochs = 4
    batch_size = 2
    num_samples_per_epoch = 10
    num_renders_per_sample = 2

    run_plain(conf, num_epochs, batch_size, num_samples_per_epoch, num_renders_per_sample)