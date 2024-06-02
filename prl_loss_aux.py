import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w)
        return G


class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return (out)


# vgg definition that conveniently let's you grab the outputs from any layer
class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]




class BaseModel():
    def __init__(self):
        pass
        
    def name(self):
        return 'BaseModel'

    def initialize(self, use_gpu=True, gpu_ids=[0]):
        self.use_gpu = use_gpu
        self.gpu_ids = gpu_ids

    def forward(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, path, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(path, save_filename)
        torch.save(network.state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        print('Loading network from %s'%save_path)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate():
        pass

    def get_image_paths(self):
        return self.image_paths

    def save_done(self, flag=False):
        np.save(os.path.join(self.save_dir, 'done_flag'),flag)
        np.savetxt(os.path.join(self.save_dir, 'done_flag'),[flag,],fmt='%i')








class VGGStyleModel(BaseModel):
    def name(self):
        return 'VGGStyleModel'

    def initialize(self, m_device):
        self.style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
        # self.content_layers = ['r42']
        self.loss_layers = self.style_layers
        self.loss_fns = [GramMSELoss()] * len(self.style_layers)
        if torch.cuda.is_available():
            print('VGG GPU')
            self.loss_fns = [loss_fn.to(m_device) for loss_fn in self.loss_fns]
        self.vgg = VGG()
        # TODO TODO TODO self.vgg.load_state_dict(torch.load('model_data/vgg_conv.pth'))
        for param in self.vgg.parameters():
            param.requires_grad = False
        if torch.cuda.is_available():
            self.vgg.to(m_device)

        print(self.vgg.state_dict().keys())

        self.style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
        # self.content_weights = [1e0]
        self.weights = self.style_weights

    def VGGLoss(self, X, Y):
        style_targets = [GramMatrix()(A).detach() for A in self.vgg(Y, self.style_layers)]
        # content_targets = [A.detach() for A in self.vgg(self.real_B, self.content_layers)]
        targets = style_targets
        out = self.vgg(X, self.loss_layers)
        layer_losses = [self.weights[a] * self.loss_fns[a](A, targets[a]) for a, A in enumerate(out)]
        # print(layer_losses)
        loss = sum(layer_losses)
        self.style_loss = loss
        return self.style_loss


'''
*****************************************************************************************************8
*****************************************************************************************************8
*****************************************************************************************************8
*****************************************************************************************************8
'''


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        # in_channels is computed as the sum of channels per map + the channesl for the rendering (3)
        in_channels = 3 + 3# + sum([textures_mapping[x] for x in texture_maps])

        self.main = NLayerDiscriminator(in_channels=in_channels, n_layers=2)

    def forward(self, x):
        out = self.main(x)

        return out

class DumbPatchDiscriminator(nn.Module):
    def __init__(self):
        super(DumbPatchDiscriminator, self).__init__()
        # in_channels is computed as the sum of channels per map + the channesl for the rendering (3)
        in_channels = 3# + sum([textures_mapping[x] for x in texture_maps])

        self.main = NLayerDiscriminator(in_channels=in_channels, n_layers=2)

    def forward(self, x):
        out = self.main(x)

        return out

class ImageDiscriminator(nn.Module):
    def __init__(self, layers=4):
        super(ImageDiscriminator, self).__init__()
        # in_channels is computed as the sum of channels per map + the channesl for the rendering (3)
        in_channels = 3 + 3#sum([textures_mapping[x] for x in texture_maps])

        n_layers = layers

        self.main = NLayerDiscriminator(in_channels=in_channels, n_layers=n_layers, final_classifier=False)

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(2),
            nn.Conv2d(512, 1, kernel_size=2)
        )


    def forward(self, x):
        out = self.main(x)
        out = self.classifier(out)
        return out

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_features=64, n_layers=3, norm_layer=nn.BatchNorm2d, final_classifier=True, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        kernel_size = 4
        padding = 1

        sequence = [
            nn.Conv2d(in_channels, base_features, kernel_size=kernel_size,
                      stride=2, padding=padding),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(base_features * nf_mult_prev, base_features * nf_mult,
                          kernel_size=kernel_size, stride=2, padding=padding, bias=use_bias),
                norm_layer(base_features * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(base_features * nf_mult_prev, base_features * nf_mult,
                      kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias),
            norm_layer(base_features * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        if final_classifier:
            sequence += [nn.Conv2d(base_features * nf_mult, 1,
                               kernel_size=kernel_size, stride=1, padding=padding)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
