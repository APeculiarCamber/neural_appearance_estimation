import torch
import torch.nn as nn
from torch import Tensor as T
from enum import Enum
from models.prl_feature_injector import BaseDirectionEncoder as Encoder
from brdf_utils import assert_normalized

class NetTrainMode(Enum):
    FULL = 0,
    RESNET_ONLY = 1,
    MLP_ONLY = 2,
    NONE = 3,


from torch import nn


class ImSpaceManager(nn.Module):
    def __init__(self, input_com_func : nn.Module, input_decom_func  : nn.Module, 
                 target_com_func : nn.Module, target_decom_func : nn.Module):
        super().__init__()
        
        self.compress_input = input_com_func
        self.decompress_input = input_decom_func
        self.compress_target = target_com_func
        self.decompress_target = target_decom_func

    def compress_input_call(self, x):
        return self.compress_input(x)

    def decompress_input_call(self, x):
        return self.decompress_input(x)

    def compress_target_call(self, x):
        return self.compress_target(x)

    def decompress_target_call(self, x):
        return self.decompress_target(x)


class SVBRDF_ImportanceSampler:
    def __init__(self, alpha, normals):
        self.alpha = alpha
        self.normals = normals
    def sample(self, x, y):
        return NBRDF_ImportanceSampler(self.alpha[...,y,x].detach().clone(), self.normals[...,y,x].detach().clone())

class NBRDF_ImportanceSampler:
    def __init__(self, alpha, normals):
        self.alpha = alpha
        self.normals = normals
# Need a pipelined system for interacting with TRAINED models...
    # think, do the MAKE ALL --> USE pipeline
class SVNBRDF_Renderer(nn.Module):
    def __init__(self, x_rep : T, importance_sampler : SVBRDF_ImportanceSampler, render_encoder : nn.Module, render_net : nn.Module,
                 space_manager : ImSpaceManager):
        super().__init__()
        '''
        x_rep = (1, N, H, W)
        '''
        self.x_rep = x_rep
        self.render_encoder = render_encoder
        self.render_net = render_net

        self.importance_sampler = importance_sampler
        self.space_manager = space_manager
        self.use_space_fit = True

    def render_all(self, lt : T, vt : T):
        '''
        - lt: N x 3 x H x W tensor of light directions for relighting each N x 3 x H x W pixel in x
        - vt: N x 3 x H x W tensor of view directions for relighting each N x 3 x H x W pixel in x
        '''
        n = lt.size(0)
        assert assert_normalized(lt, d=-3) and assert_normalized(lt, d=-3)
        assert n == lt.size(0) and n == vt.size(0)
        pr_encoded_position = self.render_encoder(lt, vt)
        # x_rep = torch.cat([self.x_rep.repeat(n, 1, 1, 1), pr_encoded_position], dim=1)
        px : torch.Tensor = self.render_net(self.x_rep.repeat(n, 1, 1, 1), pr_encoded_position)
        if self.use_space_fit: px = torch.clamp_min(self.space_manager.decompress_target_call(px), 0.0)
        return px

    def make_nbrdf_from_texel(self, x : int, y : int):
        return NBRDF_Renderer(self.x_rep, self, y, x)

    def render_bilerp_sample(self, l, v, x, y):
        pass

class NBRDF_Renderer(nn.Module):
    def __init__(self, x_rep : T, parent : SVNBRDF_Renderer, y : int, x : int):
        super().__init__()
        '''
        x_rep = (1, N, H, W)
        '''
        self.x_rep = x_rep[:,:,y,x].detach().clone().cuda()
        self.render_encoder = parent.render_encoder
        self.render_net = parent.render_net

        self.importance_sampler = parent.importance_sampler.sample(x, y)
        self.importance_sampler.normals = self.importance_sampler.normals.cuda()
        self.x = x
        self.y = y
        self.cuda()

    def render_texel(self, l, v):
        '''
        - lt: N x 3 tensor of light directions (assumes oriented to pixel already)
        - vt: N x 3 tensor of view directions (assumes oriented to pixel already)
        '''
        assert assert_normalized(l, d=-1) and assert_normalized(l, d=-1)
        assert l.size(0) == v.size(0)

        pr_encoded_position = self.render_encoder(l[:,:,None,None], v[:,:,None,None])
        # x_rep = torch.cat([self.x_rep[:,:,None,None].repeat(l.size(0), 1, 1, 1), pr_encoded_position], dim=1)
        px : torch.Tensor = self.render_net(self.x_rep[:,:,None,None].repeat(l.size(0), 1, 1, 1), pr_encoded_position)
        return px[:,:,0,0] # N, 3

    def device(self):
        return self.x_rep.device
    

class RelitPixelNet(nn.Module):
    """
    MAIN NETWORK

    Takes as input:
        - N list of lit material images
        - N list of light and view for lit material images
        - NxM (or 1xM) list of novel light and view for relit images

    It will handle positional encoding and other preprocessing
    """
    def __init__(self, input_encoder : Encoder, render_encoder : Encoder,
                  input_processor : nn.Module, renderer : nn.Module, space_manager : ImSpaceManager):
        super().__init__()
        self.input_encoder = input_encoder
        self.render_encoder = render_encoder
        self.resnet = input_processor
        self.render_net = renderer
        self.space_manager = space_manager


    def set_train_mode(self, mode : str):
        self.train_mode = mode

        res_grad = mode.upper() in ["INPUT_NET_ONLY", "FULL"]
        self.resnet.requires_grad_(res_grad)
        for p in self.resnet.parameters():
            assert(p.requires_grad == res_grad)
        
        mlp_grad = mode.upper() in ["RENDERER_ONLY", "FULL"]
        self.render_net.requires_grad_(mlp_grad)
        for p in self.render_net.parameters():
            assert(p.requires_grad == mlp_grad)

    def make_neural_feature_rep(self, x : T, lt : T, vt : T):
        '''
        - x : N x PC x H x W tensor of lit material images
        
        - lt: N x 3 x H x W tensor of light directions for each pixel in x
        - vt: N x 3 x H x W tensor of view directions for each pixel in x

        NOTE: Expects images to be linear color space HDR
        '''
        assert assert_normalized(lt, d=-3) and assert_normalized(vt, d=-3)

        assert x.dim() == 4 and lt.dim() >= 3 and vt.dim() >= 3
        if lt.dim() == 3: lt = lt.unsqueeze(0).repeat(len(x), 1, 1, 1)
        if vt.dim() == 3: vt = vt.unsqueeze(0).repeat(len(x), 1, 1, 1)

        x = self.space_manager.compress_input_call(x)

        # CONV POSITIONAL ENCODING
        st_encoded_pos : T = self.input_encoder(lt, vt)
        x = torch.cat([x, st_encoded_pos], dim=1)

        # PROCESS
        rx : T = self.resnet(x, st_encoded_pos)
        return rx


    def render_from_neural_rep(self, x_rep : T, rl_lt : T, rl_vt : T):
        '''
        - x : N x C x H x W tensor of lit material images
        
        - rl_lt: N x 3 x H x W tensor of light directions for relighting each N x 3 x H x W pixel in x
        - rl_vt: N x 3 x H x W tensor of view directions for relighting each N x 3 x H x W pixel in x
        
        NOTE: Expects images to be linear color space HDR
        '''
        assert assert_normalized(rl_lt, d=-3) and assert_normalized(rl_lt, d=-3)

        n = x_rep.size(0)
        assert n == rl_lt.size(0) and n == rl_vt.size(0)

        # MLP POSITIONAL ENCODING
        pr_encoded_position = self.render_encoder(rl_lt, rl_vt)

        # PROCESS
        px : torch.Tensor = self.render_net(x_rep, pr_encoded_position)
        # No post processing here

        return px

    def make_renderer(self, x_rep : T, roughness : T, normals : T):
        assert x_rep.dim() == 4 and x_rep.size(0) == 1
        renderer = SVNBRDF_Renderer(x_rep, SVBRDF_ImportanceSampler(roughness*roughness, normals),
                                     self.render_encoder, self.render_net, self.space_manager)
        return renderer


    def render_from_neural_rep_multi(self, x_rep : T, rl_lt : T, rl_vt : T):
        '''
        - x : N x C x H x W tensor of lit material images
        
        - rl_lt: (N) x M x 3 x H x W tensor of light directions for relighting each N x 3 x H x W pixel in x
        - rl_vt: (N) x M x 3 x H x W tensor of view directions for relighting each N x 3 x H x W pixel in x
        
        NOTE: Expects images to be linear color space HDR
        '''
        assert assert_normalized(rl_lt, d=-3) and assert_normalized(rl_lt, d=-3)

        n = x_rep.size(0)
        rl_dim = rl_lt.dim()
        assert rl_dim == rl_vt.dim()
        m = rl_vt.size(0) if rl_dim == 4 else rl_vt.size(1)

        # MLP POSITIONAL ENCODING
        if rl_dim == 4:
            pr_encoded_position : T = self.render_encoder(rl_lt, rl_vt)
            pr_encoded_position = pr_encoded_position.unsqueeze(0).repeat(n, 1, 1, 1, 1).flatten(0, 1)
        elif rl_dim == 5:
            rl_lt, rl_vt = rl_lt.flatten(0, 1), rl_vt.flatten(0, 1)
            pr_encoded_position : T = self.render_encoder(rl_lt, rl_vt)

        x_rep = x_rep.unsqueeze(1).repeat(1, m, 1, 1, 1).flatten(0, 1)

        # Inject
        # x_rep = torch.cat([x_rep, pr_encoded_position], dim=1)
        px : torch.Tensor = self.render_net(x_rep, pr_encoded_position)

        return px.unflatten(0, (n, m))


    def render_multi(self, x : T, lt : T, vt : T, rl_lt : T, rl_vt : T):
        '''
        LIKE forward but multiple relit light and view conditions can be applied to the same input image for relighting

        - x : N x 3 x H x W tensor of lit material images
        - lt: N x 3 x H x W tensor of light directions for each pixel in x
        - vt: N x 3 x H x W tensor of view directions for each pixel in x
        
        - rl_lt: (N) x M x 3 x H x W tensor of light directions for relighting each N x 3 x H x W pixel in x
        - rl_vt: (N) x M x 3 x H x W tensor of view directions for relighting each N x 3 x H x W pixel in x

        NOTE: Expects images to be linear color space HDR
        '''

        rx = self.make_neural_feature_rep(x, lt, vt)
        px = self.render_from_neural_rep_multi(rx, rl_lt, rl_vt)

        return px, rx
    


    def forward(self, x : T, lt : T, vt : T, rl_lt : T, rl_vt : T):
        '''
        - x : N x 3 x H x W tensor of lit material images
        - lt: N x 3 x H x W tensor of light directions for each pixel in x
        - vt: N x 3 x H x W tensor of view directions for each pixel in x
        
        - rl_lt: (N) x M x 3 x H x W tensor of light directions for relighting each N x 3 x H x W pixel in x
        - rl_vt: (N) x M x 3 x H x W tensor of view directions for relighting each N x 3 x H x W pixel in x

        NOTE: Expects images to be linear color space HDR
        '''

        assert rl_lt.size(0) == x.size(0)
        assert rl_vt.size(0) == x.size(0)
        assert rl_lt.size(1) == rl_vt.size(1)

        rx = self.make_neural_feature_rep(x, lt, vt)
        px = self.render_from_neural_rep(rx, rl_lt, rl_vt)

        return px, rx