import torch
from enum import Enum
import math
from copy import copy

m_device = torch.device('cpu')
if torch.cuda.is_available():
    m_device = torch.device("cuda")
else:
    print("ERROR: No CUDA")

def convert_from_enum(e : Enum):
    if isinstance(e, Enum):
        return e.name
    else: return e
def convert_to_enum(s : str, dtype : Enum.__class__):
   return dtype[s]

class WB_Range:
    def __init__(self, start, end, name):
        self.min = start
        self.max = end
        self.name = name
        self.dtype = type(start)

    def enter_into_dict(self):
        return (self.name, {'min': self.min, 'max': self.max})

    def __hash__(self):
        return hash(self.name)

    def convert(self, val):
        return val
    
class WB_Values:
    def __init__(self, val_lst, name):
        self.lst = val_lst
        self.name = name
        self.dtype = type(val_lst)

    def enter_into_dict(self):
        return (self.name, {'values': [convert_from_enum(l) for l in self.lst]})

    def __hash__(self):
        return hash(self.name)

    def convert(self, val):
        if issubclass(self.dtype, Enum): return convert_to_enum(val)
        else: return val

class ConfigData:
    def __init__(self,
                 lr=0.001, lr_gamma=0.99, lr_gamma_step=1000, weight_decay=0.0,
                 epochs=1000, batch_size=4, max_grad_norm=math.inf,
                 crop_size = 128,
                 pen_touch = 0.0,
                 lr_render_gamma=0.2,

                 pos_include_half = False,

                 pr_frequencies=1,

                 pr_sampling_threshold = 0.5,
                 pr_training_epoch_cutoff=0.95,
                 
                 pixel_feature_depth=64, 
                 zero_init_residual=False, depth_factor=64,
                 use_final_tanh=True, tanh_bottleneck=False, tanh_passthroughs=False,
                 renset_final_tanh=True,
                 ):
        self.run_name = "Standard_TODO"

        self.lr = lr
        self.lr_gamma = lr_gamma
        self.lr_render_gamma = lr_render_gamma
        self.lr_gamma_step = lr_gamma_step
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.pen_touch = pen_touch
        self.pr_sampling_threshold = pr_sampling_threshold
        self.pr_training_epoch_cutoff = pr_training_epoch_cutoff

        self.crop_size = crop_size
        self.batch_size = batch_size
        self.epochs = epochs

        self.pos_include_half = pos_include_half

        self.pr_frequencies = pr_frequencies
        self.pr_no_skip = False
        
        # RESNET
        self.pixel_feature_depth = pixel_feature_depth
        self.zero_init_residual = zero_init_residual
        self.depth_factor = depth_factor
        self.use_final_tanh = use_final_tanh
        self.tanh_bottleneck = tanh_bottleneck
        self.tanh_passthroughs = tanh_passthroughs
        self.renset_final_tanh = renset_final_tanh

        self.compressed_freq_encoding_count = 32

    def make_wandb_configuration(self, sweep_name, search_term, metric_name):
        sweep_params = self.get_wandb_parameters()
        param_dict = dict([p.enter_into_dict() for p in sweep_params])
        sweep_config = {
            'method': search_term,
            'name': sweep_name,
            'metric': {'goal': 'minimize', 'name': metric_name},
            'parameters': param_dict}
        return sweep_config


    def get_wandb_parameters(self):
        '''
        Returns the wandb config parameter objects in this config object, along with a list of their wandb names.
        '''
        params = set()
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, (WB_Range, WB_Values)):
                params.add(attr_value)
        params = list(params)
        return params

    def get_wandb_names_to_attr_names(self):
        '''
        Returns (wandb name) --> [Attribute names]
        '''
        place_dict = dict()
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, (WB_Range, WB_Values)):
                wandb_name = attr_value.name
                if wandb_name not in place_dict: place_dict[wandb_name] = []
                place_dict[wandb_name].append(attr_name)
        return place_dict

    def place_wandb_parameters(self, wandb_config):
        '''
        Places wandb selected hyperparameters into the config
        '''
        # DICT: WB_Type.name --> list of attrib names
        wandb_to_attrs = self.get_wandb_names_to_attr_names()
        for wn in wandb_config.keys():
            val = wandb_config.get(wn)
            for attr_name in wandb_to_attrs[wn]:
                vars(self)[attr_name] = vars(self)[attr_name].convert(val)


    def clear_wandb_parameters(self):
        dflt_conf = ConfigData()
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, (WB_Range, WB_Values)):
                vars(self)[attr_name] = vars(dflt_conf)[attr_name]

    def copy(self):
        return copy(self)
