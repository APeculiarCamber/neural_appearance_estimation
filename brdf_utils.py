import torch
import numpy as np
import imageio.v2 as imageio
from PIL import Image

def normalize(t : torch.Tensor,d=-1) -> torch.Tensor:
    '''
    Normalize (with p=2 norm) a single dimension of the tensor
    '''
    length = torch.norm(t,p=2,dim=d)
    return t/length.unsqueeze(d)

def assert_normalized(t : torch.Tensor, d=-1) -> bool:
    length = torch.norm(t, p=2, dim=d)
    in_range = torch.abs(length - 1.0) < 1e-4
    return torch.all(in_range)

def tensor_dot(a,b,d=-1):
    '''
    Dot product of dimension
    '''
    return torch.sum(a*b,dim=d)

def convert_polar_to_euclid(theta, phi, r):
    '''
    Returns tensor of euclid positions
    '''
    pos = torch.empty((*phi.shape, 3),device=theta.device)
    
    pos[:, 0, ...] = torch.sin(theta) * torch.cos(phi) * r
    pos[:, 1, ...] = torch.sin(theta) * torch.sin(phi) * r
    pos[:, 2, ...] = torch.cos(theta) * r

    return pos

def convert_euclid_to_polar(pos):
    '''
    Returns theta, phi, rho
    '''
    rho = torch.sqrt(pos[:,0,...]**2 + pos[:,1,...]**2 + pos[:,2,...]**2)
    phi = torch.atan2(pos[:,1,...], pos[:,0,...])
    theta = torch.acos(pos[:,2,...] / rho)

    return theta, phi, rho




def saveimage(T,name):
    T=torch.clamp(T,min=0.0,max=1.0).permute(1, 2, 0)
    T=T.detach().squeeze()
    T=(T.numpy().clip(min=0)*255).astype(np.uint8)
    imageio.imsave(name+'.png',T)

def writeimage_to_path(T,path):
    T=torch.clamp(T,min=0.0,max=1.0)
    T=T.detach().squeeze()
    T=(T.numpy().clip(min=0)*255).astype(np.uint8)
    imageio.imsave(f"{path}.png",T)

def makepilimage(T):
    T=torch.clamp(T,min=0.0,max=1.0)
    T=T.detach().squeeze()
    T=(T.numpy().clip(min=0)*255).astype(np.uint8)
    return Image.fromarray(T)


def writevideo(frames,name,fps=30,desired_resolution=512):
    '''
    frames : shape(FRAMES, 3, H, W)
    '''
    frames = torch.nan_to_num(frames, nan=0.0)
    final_frames = [None for _ in range(len(frames))]
    for i in range(len(frames)):
        T = frames[i]
        T=torch.clamp(T,min=0.0,max=1.0)
        T=T.detach().unsqueeze(0)
        size_mod = desired_resolution / T.shape[-1]
        # T = T.permute(2, 0, 1).unsqueeze(1)
        T=torch.nn.functional.interpolate(T, (round(T.shape[-2]*size_mod), desired_resolution), mode='nearest').squeeze().permute(1, 2, 0)
        T=(T.numpy().clip(min=0,max=1)*255).astype(np.uint8)
        final_frames[i] = T
    imageio.mimsave(name+".mp4", final_frames, fps=fps)



def loadImage(path,is_hdr=False):
    a = np.zeros([256,256,3],order='F')
    a = (imageio.imread(path))
    if not is_hdr: a = a / 255.0
    a=np.swapaxes(a,0,2)
    a=np.swapaxes(a,1,2)
    x = torch.from_numpy(a.astype(np.float32))
    return x

def load_dataset_image(path):
    a = np.zeros([256,256,3],order='F')
    a = (imageio.imread(path).astype(float)/255.0)
    a = np.swapaxes(a,0,2)
    a = np.swapaxes(a,1,2)
    a = a[:,:272,:272][:,16:,16:]
    x = torch.from_numpy(a.astype(np.float32))

    return x


def extract_ggx_parameters(mat : torch.Tensor):
    '''
    From mat = Shape(..., 3, H, W), extract the named parameters of the brdf
    return diffuse, specular, roughness, normals
    '''
    diffuse = mat[...,0:3,:,:]
    specular = mat[...,3:6,:,:]
    roughs = mat[...,6:7,:,:]
    normals = mat[...,7:10,:,:]
    return diffuse, specular, roughs, normals


def normalize_xy_normals(brdfs):
    brdfs[...,7:10,:,:] = (brdfs[...,7:10,:,:] * 2.0) - 1.0 # NORMALS, adjust for rendering [0,1]->[-1,1]
    brdfs[...,7:10,:,:] = normalize(brdfs[...,7:10,:,:], d=-3)
    return brdfs





# EXR
import OpenEXR
def save_exr(im : torch.Tensor, fn : str):
    '''
    WARNING SINGLE IMAGES
    '''
    assert im.dim() == 3 and im.size(0) == 3
    im = im.numpy()
    header = OpenEXR.Header(im.shape[1], im.shape[2])
    exr = OpenEXR.OutputFile(fn, header)
    exr.writePixels({'R': im[0].ravel(), 'G': im[1].ravel(), 'B': im[2].ravel()})
    exr.close()

def load_exr(fn : str):
    exr_file = OpenEXR.InputFile(fn)
    dw = exr_file.header()['dataWindow']
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    float_buffer = exr_file.channels(['R', 'G', 'B'])

    r = np.frombuffer(float_buffer[0], dtype=np.float32).reshape(size[::-1])[None]
    g = np.frombuffer(float_buffer[1], dtype=np.float32).reshape(size[::-1])[None]
    b = np.frombuffer(float_buffer[2], dtype=np.float32).reshape(size[::-1])[None]

    return torch.from_numpy(np.concatenate([r,g,b], axis=0)).float()
