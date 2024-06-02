import torch

# TAKEN FROM JAMIE'S SINMR FILES FOR RELEASE

import torch
from torch.utils.data import Dataset
import numpy
import imageio.v2 as imageio
import random
from brdf_render import generateDirectionMaps, render_image_ggx
from brdf_utils import normalize_xy_normals, convert_euclid_to_polar, convert_polar_to_euclid, normalize

# TODO: ensure this is refactored
MIXED_MAT_ROOT = "/big/data/jidema/matfusion-datasets/"

z_height = 4.0

def render_log_output_circle_on_center(brdf : torch.Tensor, num_per_sample : int):
    # circular render for logging
    view = torch.tensor([[0.0, 0.0, z_height]] * num_per_sample)
    l = 2.0 * torch.pi * (torch.arange(0, num_per_sample, dtype=torch.float) / num_per_sample)
    light = torch.stack([torch.cos(l), torch.sin(l), 2.5*torch.ones(l.shape,dtype=torch.float)], dim=-1)
    light = normalize(light, d=-1) * z_height
    lt, vt = generateDirectionMaps(light, view, brdf.size(-1))

    renders = render_image_ggx(lt[None], vt[None], brdf[None,None]).squeeze(0)
    return renders, lt, vt

import random
def render_input_ggx_brdfs_single(brdf : torch.Tensor): 
    height = 4.0

    lv = torch.tensor([[0.0, 0.0, height]])
    lt, vt = generateDirectionMaps(lv, lv, brdf.size(-1))
    renders = render_image_ggx(lt[:], vt[:], brdf[None,:]).squeeze()
    return renders, lt.squeeze(), vt.squeeze()

import math as m
def sample_top_hemisphere(n_samples, steradian, device='cpu'):
    
    # Azimuthal angle
    phi = torch.rand(n_samples, device=device) * (torch.pi * 2)
    
    # Zenith angle
    l = m.cos(steradian)
    u = (torch.rand(n_samples, device=device) * (1.0 - l)) + l
    theta = torch.arccos(u)
    
    # Cartesian
    samples = torch.empty(n_samples, 3, device=device)
    samples[:,0] = torch.sin(theta) * torch.cos(phi)
    samples[:,1] = torch.sin(theta) * torch.sin(phi)
    samples[:,2] = torch.cos(theta)
    
    return samples


def sample_sides_top_hemisphere(n_samples, theta_min, device='cpu'):
    phi = torch.rand(n_samples, device=device) * (torch.pi * 2)
    cos_theta_min = m.cos(theta_min)
    u = torch.rand(n_samples, device=device) * (cos_theta_min)
    theta = torch.arccos(u)
    samples = torch.empty(n_samples, 3, device=device)
    samples[:,0] = torch.sin(theta) * torch.cos(phi)
    samples[:,1] = torch.sin(theta) * torch.sin(phi)
    samples[:,2] = torch.cos(theta)
    return samples

def get_perturbed_mirror_directions(light_dirs, steradian):
    # I - 2.0 * dot(N, I) * N
    light_dirs = light_dirs.clone()
    light_dirs[:,0:2] *= -1.0
    dtp = ((torch.rand(len(light_dirs), 2) * 2.0) - 1.0) * steradian
    theta,phi,rho = convert_euclid_to_polar(light_dirs)
    theta = dtp[:, 0] + theta
    phi = dtp[:, 0] + phi
    light_dirs = convert_polar_to_euclid(theta, phi, rho)
    return light_dirs


def get_jamie_surface_highlight_samples(n_samples):
    sp = (2.0*torch.rand(n_samples, 2))-1.0
    surface_pos = sp + torch.normal(0.0, 2.0, size=(n_samples, 2))

    # REFLECT FROM SURFACE POV
    view_dirs = sample_top_hemisphere(n_samples, torch.pi / 2, device='cpu') * z_height

    light_dists = torch.abs(torch.normal(0.0,2.0,size=(n_samples,))) + 0.5

    reflected_view_dirs = surface_pos - view_dirs[:, 0:2]
    reflected_view_dirs = torch.cat([reflected_view_dirs, view_dirs[:, 2:]], dim=1)
    # CHANGE LIGHT DISTANCE
    light_dirs = light_dists[:, None] * reflected_view_dirs
    # Back to global frame
    light_dirs[:, 0:2] = light_dirs[:, 0:2] + surface_pos

    return view_dirs, light_dirs 
    

def render_output_ggx_brdfs(brdf : torch.Tensor, num_per_sample : int):
    '''
    brdfs: ggx svbrdfs of form { N, 10, 256, 256 }
    renders_per_brdf: number of l-v renders to make for each of the N svbrdfs
    NOTE: STATIC VIEW, 0,0,4

    Returns: 
        - tensor of renders shape:     {N, 3, 256, 256}
        - tensor of lights directions: {N, 3, 256, 256}
        - tensor of view   directions: {N, 3, 256, 256}
    '''
    j_view_dirs, j_light_dirs = get_jamie_surface_highlight_samples(num_per_sample)
    light_dirs, view_dirs = j_light_dirs, j_view_dirs

    lt, vt = generateDirectionMaps(light_dirs, view_dirs, brdf.size(-1))
    # lt, vt = Shape(BRDF_COUNT, RENDERS, 3, H, W), We use implicit broadcasting aggressively here...

    renders = render_image_ggx(lt[None], vt[None], brdf[None,None]).squeeze(0)
    
    return renders, lt, vt


def render_output_ggx_brdfs_jamie(brdf : torch.Tensor, num_per_sample : int):
    '''
    brdfs: ggx svbrdfs of form { N, 10, 256, 256 }
    renders_per_brdf: number of l-v renders to make for each of the N svbrdfs
    NOTE: STATIC VIEW, 0,0,4

    Returns: 
        - tensor of renders shape:     {N, 3, 256, 256}
        - tensor of lights directions: {N, 3, 256, 256}
        - tensor of view   directions: {N, 3, 256, 256}
    '''

    j_view_dirs, j_light_dirs = get_jamie_surface_highlight_samples(num_per_sample)
    light_dirs, view_dirs = j_light_dirs, j_view_dirs

    lt, vt = generateDirectionMaps(light_dirs, view_dirs, brdf.size(-1))
    # lt, vt = Shape(BRDF_COUNT, RENDERS, 3, H, W), We use implicit broadcasting aggressively here...

    renders = render_image_ggx(lt[None], vt[None], brdf[None,None]).squeeze(0)
    
    return renders, lt, vt, light_dirs, view_dirs

def render_output_ggx_brdfs_hemisphere(brdf : torch.Tensor, num_per_sample : int, height):
    '''
    brdfs: ggx svbrdfs of form { N, 10, 256, 256 }
    renders_per_brdf: number of l-v renders to make for each of the N svbrdfs
    NOTE: STATIC VIEW, 0,0,4

    Returns: 
        - tensor of renders shape:     {N, 3, 256, 256}
        - tensor of lights directions: {N, 3, 256, 256}
        - tensor of view   directions: {N, 3, 256, 256}
    '''

    j_view_dirs = sample_top_hemisphere(num_per_sample, torch.pi / 2, brdf.device) * height
    j_light_dirs = sample_top_hemisphere(num_per_sample, torch.pi / 2, brdf.device) * height
    light_dirs, view_dirs = j_light_dirs, j_view_dirs

    lt, vt = generateDirectionMaps(light_dirs, view_dirs, brdf.size(-1))
    # lt, vt = Shape(BRDF_COUNT, RENDERS, 3, H, W), We use implicit broadcasting aggressively here...

    renders = render_image_ggx(lt[None], vt[None], brdf[None,None]).squeeze(0)
    
    return renders, lt, vt, light_dirs, view_dirs


class NaiveCPU_Dataset(Dataset):
    def __init__(self, brdf_dataset : Dataset, num_out_samples=2, inpaint=None):
        self.brdf_dataset = brdf_dataset
        self.num_out_samples = num_out_samples
        
        self.is_inpaint = inpaint != None
        if self.is_inpaint:
            self.inpaint_quantile = inpaint[0]

    def __len__(self):
        return len(self.brdf_dataset)
    
    def __getitem__(self,index):
        while True:
            try:
                brdf = self.brdf_dataset[index]
                brdf = normalize_xy_normals(brdf)

                r, l, v = render_input_ggx_brdfs_single(brdf)
                rough, n = brdf[6], brdf[7:10]

                tr, tl, tv = render_output_ggx_brdfs(brdf, self.num_out_samples)
                return (r, l, v, n, tr, tl, tv, rough)
            except Exception as e:
                print("ERROR:", str(e), type(e))
                index = (index + 1) % len(self)


class CircleRender_CPU_Dataset(Dataset):
    def __init__(self, brdf_dataset : Dataset, num_out_samples=2, inpaint=None):
        self.brdf_dataset = brdf_dataset
        self.num_out_samples = num_out_samples
        
        self.is_inpaint = inpaint != None
        if self.is_inpaint:
            self.inpaint_quantile = inpaint[0]

    def __len__(self):
        return len(self.brdf_dataset)
    
    def __getitem__(self,index):
        while True:
            try:
                brdf = self.brdf_dataset[index]
                brdf = normalize_xy_normals(brdf)

                r, l, v = render_input_ggx_brdfs_single(brdf)
                rough, n = brdf[6], brdf[7:10]

                tr, tl, tv = render_log_output_circle_on_center(brdf, self.num_out_samples)
                return (r, l, v, n, tr, tl, tv, rough)
            except Exception as e:
                print("ERROR:", str(e), type(e))
                index = (index + 1) % len(self)




PATH_TO_INRIA = '/big/data/INRIA/'

class INRIAdataset(Dataset):
    def __init__(self,patchsize=256):
        super(INRIAdataset,self)
        self.patchsize = patchsize
        f = open(PATH_TO_INRIA+'/DeepMaterialsData/trainBlended/filelist.txt','r')
        self.filenames = f.readlines()
        for i in range(len(self.filenames)):
            self.filenames[i] = self.filenames[i].replace('\;',';').replace('\n','')
    def __len__(self):
        return 199068
    def __getitem__(self,index):
        rx = random.randint(0,288-self.patchsize)
        ry = random.randint(0,288-self.patchsize)
        a = numpy.zeros([288,288,10],order='F')
        x = ((imageio.imread('/big/data/INRIA/DeepMaterialsData/trainBlended/'+self.filenames[index])/255.0))
        # order for SINMR is diffuse, specular, roughness, normals
        a[:,:,0:3] = x[:,(4*288):(4*288+288),:]
        a[:,:,3:6] = x[:,(2*288):(2*288+288),:]
        a[:,:,6] = x[:,(3*288):(3*288+288),0]
        a[:,:,7:10] = x[:,(1*288):(1*288+288),:]

        # swap to pytorch format (channels first, not last)
        a=numpy.swapaxes(a,0,2)
        a=numpy.swapaxes(a,1,2)
        # crop based on random value
        a=a[:,rx:rx+self.patchsize,ry:ry+self.patchsize]
        y = torch.from_numpy(a.astype(numpy.float32))
        return y

import os
class Mixed_SVBRDF_Dataset(Dataset):
    def check_and_count(self, dir):
        subdirs = ["diffuse", "specular", "normals", "roughness"]

        if not os.path.exists(dir): raise RuntimeError()
        
        sd_paths = [os.path.join(dir, sd) for sd in subdirs]

        # Check if all subdirectories have the same number of files
        file_counts = set()
        for sd in sd_paths:
            if not os.path.isdir(sd): raise RuntimeError()
            file_counts.add(len(os.listdir(sd)))

        if len(file_counts) != 1: raise RuntimeError()

        brdf_count = file_counts.pop()

        return (brdf_count, sd_paths)


    def __init__(self,dirs,sample_limit=None,patchsize=256):
        super(Mixed_SVBRDF_Dataset,self)
        self.dirs = dirs
        count_and_dirs = zip(*[self.check_and_count(d) for d in self.dirs])
        self.dir_brdf_counts, self.dir_paths = count_and_dirs
        self.num_brdfs = sum(self.dir_brdf_counts) if sample_limit is None else sample_limit
        self.patchsize = patchsize

    def __len__(self):
        return self.num_brdfs
    
    def __getitem__(self,index):
        # Find the correct directory
        k = 0
        while index >= self.dir_brdf_counts[k]: 
            k, index = k + 1, index - self.dir_brdf_counts[k]

        diffuse_dir, specular_dir, normals_dir, roughness_dir = self.dir_paths[k]
        index = index + 1

        diffuse_fn = f"{diffuse_dir}/{str(index).zfill(7)}_diffuse.png"
        specular_fn = f"{specular_dir}/{str(index).zfill(7)}_specular.png"
        normals_fn = f"{normals_dir}/{str(index).zfill(7)}_normals.png"
        roughness_fn = f"{roughness_dir}/{str(index).zfill(7)}_roughness.png"

        d = (imageio.imread(diffuse_fn)/255.0).swapaxes(0, 2).swapaxes(1, 2)
        n = (imageio.imread(normals_fn)/255.0).swapaxes(0, 2).swapaxes(1, 2)
        s = (imageio.imread(specular_fn)/255.0).swapaxes(0, 2).swapaxes(1, 2)
        r = (imageio.imread(roughness_fn)/255.0)
        if r.ndim == 2: r = r[None]
        else: r = r.swapaxes(0, 2).swapaxes(1, 2)[0:1]

        ps = self.patchsize
        rx = random.randint(0,d.shape[1]-self.patchsize)
        ry = random.randint(0,d.shape[2]-self.patchsize)
        d, s = d[:,rx:rx+ps,ry:ry+ps], s[:,rx:rx+ps,ry:ry+ps]
        r, n = r[:,rx:rx+ps,ry:ry+ps], n[:,rx:rx+ps,ry:ry+ps]

        return torch.cat([torch.from_numpy(d), torch.from_numpy(s), torch.from_numpy(r), torch.from_numpy(n)], dim=0).to(torch.float)






from torch.utils.data import RandomSampler, DataLoader

def load_fusion_training_data(batch_size=4, sample_limit=None, crop_size=256, replacement=True, num_out_samples=2, num_samples_per_epoch=40000):
    train_dataset = NaiveCPU_Dataset(Mixed_SVBRDF_Dataset([f'{MIXED_MAT_ROOT}/mixed_svbrdfs', f'{MIXED_MAT_ROOT}/cc0_svbrdfs', f'{MIXED_MAT_ROOT}/inria_svbrdfs'], 
                                         sample_limit=sample_limit, 
                                         patchsize=crop_size
                                         ), num_out_samples=num_out_samples)
    sampler = RandomSampler(train_dataset, replacement=replacement,num_samples=num_samples_per_epoch)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, persistent_workers=True)
    return train_dataloader


class Sam_Test_SVBRDF_Dataset(Dataset):

    def __init__(self,bdir,sample_limit=None,patchsize=256):
        super().__init__()
        
        self.brdf_dirs = [f"{bdir}/{f}" for f in os.listdir(bdir) if os.path.isdir(f'{bdir}/{f}')]
        self.num_brdfs = len(self.brdf_dirs)

        self.patchsize = patchsize

        print("MADE SAM-TEST DATASET WITH", len(self), "BRDFS")

    def __len__(self):
        return self.num_brdfs
    
    def __getitem__(self,index):
        brdf_dir = self.brdf_dirs[index]
        diffuse_fn = f"{brdf_dir}/diffuse.png"
        specular_fn = f"{brdf_dir}/specular.png"
        normals_fn = f"{brdf_dir}/normals.png"
        roughness_fn = f"{brdf_dir}/roughness.png"

        d = (imageio.imread(diffuse_fn)/255.0).swapaxes(0, 2).swapaxes(1, 2)
        n = (imageio.imread(normals_fn)/255.0).swapaxes(0, 2).swapaxes(1, 2)
        s = (imageio.imread(specular_fn)/255.0).swapaxes(0, 2).swapaxes(1, 2)
        r = (imageio.imread(roughness_fn)/255.0)
        if r.ndim == 2: r = r[None]
        else: r = r.swapaxes(0, 2).swapaxes(1, 2)[0:1]

        ps = self.patchsize
        rx = random.randint(0,d.shape[1]-ps)
        ry = random.randint(0,d.shape[2]-ps)
        d, s = d[:,rx:rx+ps,ry:ry+ps]**(2.2), s[:,rx:rx+ps,ry:ry+ps]**(2.2) # NOTE: correction...
        r, n = r[:,rx:rx+ps,ry:ry+ps], n[:,rx:rx+ps,ry:ry+ps]

        return torch.cat([torch.from_numpy(d), torch.from_numpy(s), 
                          torch.from_numpy(r), torch.from_numpy(n)], dim=0).to(torch.float)


def load_sam_testing_data(batch_size=4, crop_size=256, num_samples=256, replacement=True, num_out_samples=2):
    test_dataset = NaiveCPU_Dataset(Sam_Test_SVBRDF_Dataset(f"{MIXED_MAT_ROOT}/samtest-dataset/", patchsize=crop_size),
                                     num_out_samples=num_out_samples)
    sampler = RandomSampler(test_dataset, replacement=replacement, num_samples=num_samples)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, persistent_workers=True)
    return test_dataloader

def load_sam_testing_data_circular(batch_size=4, crop_size=256, num_samples=256, replacement=True, num_out_samples=2):
    test_dataset = CircleRender_CPU_Dataset(Sam_Test_SVBRDF_Dataset(f"/{MIXED_MAT_ROOT}/samtest-dataset/", patchsize=crop_size),
                                     num_out_samples=num_out_samples)
    sampler = RandomSampler(test_dataset, replacement=replacement, num_samples=num_samples)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, persistent_workers=True)
    return test_dataloader
