from models.prl_pixel_mlp import *
from models.prl_net import RelitPixelNet

import torch
from torch import Tensor as T
from torch.utils.data import DataLoader
from prl_nBRDF_dataset import load_sam_testing_data
from prl_loss import AlexPerceptualLoss

def eval_net(model : RelitPixelNet, dataloader : DataLoader, m_device):
    model = model.to(m_device)
    loss_func = AlexPerceptualLoss()
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch : tuple[T, T, T, T, T, T, T, T] = batch
            x, l, v, in_n, tx, tl, tv, _ = [b.to(m_device) for b in batch]
            
            x_render, x_neural_rep = model.render_multi(x, l, v, tl, tv) # ASSUMES DIM = 5
            x_render = model.space_manager.decompress_target_call(x_render)

            total_loss =+ loss_func(x_render, tx, tl, tv, x_neural_rep, x, l, v, in_n).item()
    model.train()
    return total_loss

if __name__ == "__main__":
    model = torch.load("model_data/final.pth") # TODO
    eval_dataloader = load_sam_testing_data(batch_size=1, num_samples=50, replacement=False, num_out_samples=256)
    device = torch.device("cuda")

    loss = eval_net(model, eval_dataloader, device)
    print(f"Model Loss: {loss}")
