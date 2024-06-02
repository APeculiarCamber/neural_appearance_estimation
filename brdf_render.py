import torch
import math as m
import brdf_utils


def generateDirectionMaps(light_pos_world : torch.Tensor, camera_pos_world : torch.Tensor, scale_size : int,
                          sy=0, ey=None, sx=0, ex=None):
    '''
    Convert tensor of light and view positions into directions on a material surfaces with bounds [-1,-1] to [1, 1].

    Returns tensor array of: light_tensor, view_tensor
    '''
    assert light_pos_world.dim() == 2 and camera_pos_world.dim() == 2
    
    ey = scale_size if ey is None else ey
    ex = scale_size if ex is None else ex
    d = light_pos_world.device

    surface = torch.stack(torch.meshgrid(torch.arange(scale_size, device=d), torch.arange(scale_size, device=d), indexing='ij')) + 0.5
    surface = torch.stack([
        2 * (surface[1, :, :] / 256.0) - 1,
        -2 * (surface[0, :, :] / 256.0) + 1,
        torch.zeros(surface[0, :, :].shape, device=d),
    ], axis=0)
    pos = surface
    # Shape(3, scale_size, scale_size)

    # pos shape: (1, 3, rows, cols)
    # cam shape: (pos_count, 3, 1, 1)
    light_tensor = brdf_utils.normalize(light_pos_world[:,:,None,None] - pos[None,:,:,:], d=1)
    view_tensor = brdf_utils.normalize(camera_pos_world[:,:,None,None] - pos[None,:,:,:], d=1)
        
    return light_tensor, view_tensor


def render_image_ggx(l : torch.Tensor, v : torch.Tensor, brdf_params : torch.Tensor) -> torch.Tensor:
    '''
    Perform GGX rendering

    - l : Shape((1), (LV_COUNT), 3, H, W)
    - v : Shape((1), (LV_COUNT), 3, H, W)

    - brdf : : Shape((BRDF_COUNT), (1), BRDF-CHANNELS, H, W)
    '''
    assert brdf_utils.assert_normalized(l, d=-3) and brdf_utils.assert_normalized(v, d=-3)
    
    assert l.dim() == v.dim()
    while l.dim() < 5:
        l = l.unsqueeze(0)
        v = v.unsqueeze(0)

    assert brdf_params.dim() >= 3
    if brdf_params.dim() < 4:
        brdf_params = brdf_params.unsqueeze(0)
    if brdf_params.dim() < 5:
        brdf_params = brdf_params.unsqueeze(1)


    diffuse, specular, roughness, normal = brdf_utils.extract_ggx_parameters(brdf_params)

    INV_PI = 1.0 / m.pi
    EPS = 1e-12

    def GGX(NoH, roughness):
        alpha = roughness  * roughness
        tmp = alpha / torch.clamp((NoH * NoH * (alpha * alpha - 1.0) + 1.0 ), min=1e-8) # this replaces tf.maximum
        return tmp * tmp * INV_PI


    def SmithG(NoV, NoL, roughness):

        def _G1(NoM, k):
            return NoM / (NoM * (1.0 - k ) + k)

        # k = roughness * roughness * 0.5 # TODO TODO TODO 
        k = torch.clamp(roughness * roughness * 0.5, min=1e-8)
        return _G1(NoL,k) * _G1(NoV, k)

    def Fresnel(F0, VoH):
        coeff = VoH * (-5.55473 * VoH - 6.98316)
        return F0 + (1.0 - F0) * torch.pow(2.0, coeff)

    h = brdf_utils.normalize((l+v) * 0.5, -3)

    n = brdf_utils.normalize(normal, -3)

    s = specular
    d = diffuse
    r = torch.cat([roughness,roughness,roughness],dim=-3)

    NoH = torch.clamp(brdf_utils.tensor_dot(n,h,d=-3), min=1e-8).unsqueeze(-3)#.expand(n.size())
    NoV = torch.clamp(brdf_utils.tensor_dot(n,v,d=-3), min=1e-8).unsqueeze(-3)#.expand(n.size())
    NoL = torch.clamp(brdf_utils.tensor_dot(n,l,d=-3), min=1e-8).unsqueeze(-3)#.expand(n.size())
    VoH = torch.clamp(brdf_utils.tensor_dot(v,h,d=-3), min=1e-8).unsqueeze(-3)#.expand(n.size())

    f_d = d * INV_PI
    D = GGX(NoH,r)
    G = SmithG(NoV, NoL, r)
    F = Fresnel(s, VoH)
    f_s = D * G * F / (4.0 * NoL * NoV + EPS)

    res =  (f_d + f_s) * NoL * m.pi

    return res