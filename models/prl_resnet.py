import torch
import torch.nn as nn

#what file is this from, what are the big changes

def conv7x7(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=dilation, padding_mode='replicate', groups=groups, bias=False, dilation=dilation)

def conv5x5(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=dilation, padding_mode='replicate', groups=groups, bias=False, dilation=dilation)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, padding_mode='replicate', groups=groups, bias=False, dilation=dilation)

def tranconv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    #if stride==2:
    #    return nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), conv3x3(in_planes, out_planes))
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=stride,
                     groups=groups, bias=False, padding=1, dilation=1, output_padding=0)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def tranconv2x2(in_planes, out_planes, stride=1):
    #if stride==2:
    #    return nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), conv3x3(in_planes, out_planes))

    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride, bias=False)



class InceptionBlock(nn.Module):
    def  __init__(self, inplanes, planes, stride=1):
        super(InceptionBlock,self).__init__()
        self.conv1a = conv3x3(inplanes, planes, stride=1)
        self.conv2a = conv3x3(planes, int(planes/2), stride)
        self.conv1b = conv3x3(inplanes, int(planes/2), stride)
    def forward(self,x):
        x2 = x.clone()
        x = self.conv2a(self.conv1a(x))
        x2=self.conv1b(x2)
        x=torch.cat((x,x2),dim=1)
        return x

class HABlock(nn.Module):
    def  __init__(self, inplanes, planes, stride=1):
        super(HABlock,self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.inorm = nn.InstanceNorm2d(inplanes)
        self.inception = InceptionBlock(inplanes,planes,stride=stride)
        self.relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x2 = x.clone()
        x3 = x.clone()
        x = self.sigmoid(self.conv1(x))
        x2 = self.inorm(x2)
        x2 = self.relu(self.conv2(x2))
        x3 = self.inception(x3)
        return torch.add(torch.mul(x,x2),x3)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if norm_layer is nn.Identity:
            self.use_norm=False
        else:
            self.use_norm=True
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = HABlock(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = HABlock(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if(self.use_norm):
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.use_norm:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class ReverseBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(ReverseBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if norm_layer is nn.Identity:
            self.use_norm=False
        else:
            self.use_norm=True
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if downsample is None:
            self.conv1 = conv3x3(inplanes, planes, stride)
        else:
            self.conv1 = tranconv3x3(inplanes, planes, stride)
        
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if(self.use_norm):
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if(self.use_norm):
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



BOTTLENECK_FUNCS = {
    'identity': None,
    'tanh': lambda x: torch.tanh(x),
    'arctan': lambda x: torch.arctan(x),
    'jamie_tanh': lambda x: (0.75+torch.tanh(x))/1.5,
}
PASSTHROUGH_FUNCS = BOTTLENECK_FUNCS

class GenResNetReplicateHA(nn.Module):

    def __init__(self, block, layers, strides, inplanes=64, use_tanh=False, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, encoder_channels=7, encoder_depth_factor=1, output_depth_factor=3, 
                 extra_decoder_channels=0, tanh_bottleneck=False, tanh_passthroughs=False, tanh_final_activation=True,
                 use_last_skip=True):
        super(GenResNetReplicateHA, self).__init__()

        feature_depth = inplanes

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.use_tanh = use_tanh
        self.tanh_bottleneck = tanh_bottleneck
        self.tanh_passthroughs = tanh_passthroughs
        self.func_bottleneck = BOTTLENECK_FUNCS.get(tanh_bottleneck, None)
        self.func_passthrough = PASSTHROUGH_FUNCS.get(tanh_passthroughs, None)
        self.use_last_skip = use_last_skip

        self.inplanes = int(feature_depth*encoder_depth_factor)
        self.dilation = 1

        # Stride dilation
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False] * (len(layers) - 1)
        if len(replace_stride_with_dilation) != (len(layers) - 1):
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        replace_stride_with_dilation = [False] + replace_stride_with_dilation



        self.groups = groups
        self.base_width = width_per_group
        self.encoder_channels=encoder_channels
        self.extra_decoder_channels=extra_decoder_channels
        self.conv1 = nn.Conv2d(self.encoder_channels, self.inplanes, kernel_size=7, stride=1, padding=3,padding_mode='replicate',bias=False)

        deconv_kernel = 7 # TODO
        self.deconv1 = nn.Conv2d(int(feature_depth*encoder_depth_factor), output_depth_factor, 
                                 kernel_size=deconv_kernel, stride=1, padding=(deconv_kernel // 2),padding_mode='replicate',bias=False) # nn.Conv2d(int(feature_depth*encoder_depth_factor), output_depth_factor, kernel_size=7, stride=1, padding=3,padding_mode='replicate',bias=False)
        self.deconv1_acitvation = nn.Tanh() if tanh_final_activation else nn.LeakyReLU(negative_slope=0.001)
        self.bn1 = norm_layer(self.inplanes)

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        f_planes_depth_list = [int(pow(2, i)*feature_depth*encoder_depth_factor) for i in range(len(layers))]
        b_planes_depth_list = [int(pow(2, i)*feature_depth*encoder_depth_factor) for i in range(len(layers))]
        print("Make forwards with features", f_planes_depth_list)
        print("Make backward with features", f_planes_depth_list)

        # FORWARD
        self.layers = []
        for l in range(len(layers)):
            layer_id = l + 1
            layer = self._make_layer(block, f_planes_depth_list[l], layers[l], stride=strides[l],
                                           dilate=replace_stride_with_dilation[l])
            self.layers.append(layer)
            print(f"Setting forw layer {layer_id} with", f_planes_depth_list[l], layers[l], strides[l])
            setattr(self, f"layer{layer_id}", self.layers[-1])
        
        self.layer_norm = torch.nn.LayerNorm((self.inplanes,))
        torch.nn.init.constant_(self.layer_norm.weight, 1.0)
        torch.nn.init.constant_(self.layer_norm.bias, 0.0)
        
        # BACKWARD
        self.reverse = []
        self.inplanes=self.inplanes+extra_decoder_channels
        for l in range(len(layers)-1, -1, -1):
            layer_id = l + 1
            used_p = self.inplanes
            layer = self._make_reverse_layer(block, b_planes_depth_list[l], layers[l], stride=strides[l],
                                dilate=replace_stride_with_dilation[l])
            if use_last_skip or l != 1:
                self.inplanes=self.inplanes+int(self.inplanes/2)+extra_decoder_channels
            else: #(if don't use_last_skip and l == 1)
                self.inplanes += extra_decoder_channels
            self.reverse.append(layer)
            print(f"Setting back layer {layer_id} with", b_planes_depth_list[l], layers[l], strides[l], ". inplanes =", used_p)
            setattr(self, f"reverse{layer_id}", self.reverse[-1])

        print("tanh final" if tanh_final_activation else "leakyrelu final")
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_reverse_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if self.inplanes != planes * block.expansion:
            if stride!=1:
                downsample = nn.Sequential(
                    tranconv2x2(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion)
                )
        else:
            if stride != 1:
                downsample = nn.Sequential(
                tranconv2x2(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
                )


        layers = []
        if stride != 1:
            layers.append(ReverseBasicBlock(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                        self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        stride=1
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def _forward_impl(self, x):

        x = self.conv1(x)
        # FORWARD
        zs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            zs.append(x) # LAST IS UNUSED
        
        # BOTTLENECKS
        if self.func_passthrough != None:
            zs = [self.func_passthrough(z) for z in zs]

        # zs = [torch.arctan(z) for z in zs]
        x = torch.arctan(x)
        # x = self.layer_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.func_bottleneck != None:
            x=self.func_bottleneck(x)

        # REVERSE
        for i, reverse in enumerate(self.reverse):
            rid = (len(self.reverse) - i) # n...1
            b,_,h,w=x.shape
            if i != 0 and (not hasattr(self, 'use_last_skip') or self.use_last_skip or rid != 1): # don't use last skip if use_last_skip == False
                z = zs[rid - 1]
            else: 
                z = torch.empty(b, 0, h, w, device=x.device)
            x = torch.cat([x, z], dim=1)
            x = reverse(x)

        x = self.deconv1(x)
        # TODO
        # x = torch.tanh(x)
        return x

    def forward(self, x, _):
        return self._forward_impl(x)
