from torchvision import models
import torch.nn.utils.spectral_norm as spectral_norm
import torchvision
from torch import nn
import torch
import torch.nn.functional as F

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class VGGExtractor(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGExtractor, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True).eval()
        blocks = []
        blocks.append(vgg16.features[:4])
        blocks.append(vgg16.features[4:9])
        blocks.append(vgg16.features[9:16])
        blocks.append(vgg16.features[16:23])
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
        output = []
        for block in self.blocks:
            input = block(input)
            output.append(input)
        return output

class G(nn.Module):
    def __init__(self, input_dim=4):
        super().__init__()
        # self.final = unet.UNet(6, 3)
        self.encoder = ContentEncoder(input_dim=input_dim)
        self.decoder = Decoder(dim=self.encoder.output_dim)

    def forward(self, input, residual):
        ip = torch.cat([input, residual], dim=1)
        output = self.decoder(self.encoder(ip))
        # img_out = self.final(torch.cat([output, input], dim=1))
        return (output + input).clamp(-1,1)

class ContentEncoder(nn.Module):
    def __init__(self, n_downsample=2, n_res=4, input_dim=3, dim=64, norm='in', activ='relu', pad_type='reflect'):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, n_upsample=2, n_res=4, dim=64, output_dim=3, res_norm='in', activ='relu', pad_type='reflect'):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class Warper(nn.Module):
    def __init__(self, warp_num):
        super(Warper, self).__init__()
        self.warp_num = warp_num
        self.model = list(models.resnet18(pretrained=True).children())[1:-1]
        self.model.insert(0, nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
        self.model = nn.Sequential(*self.model)
        self.fc = nn.Linear(512, 6*warp_num)
        self.fc.weight.data.zero_()
        values = []
        for i in [1, 0, 0, 0, 1, 0]:
            for _ in range(warp_num):
                values.append(i)
        self.fc.bias.data.copy_(torch.tensor(values, dtype=torch.float))

    def forward(self, input_mask, product):
        input = torch.cat([input_mask, product], dim=1)
        output = self.fc(self.model(input).view(input.size(0), -1))
        affine_transform = output.view(-1, 2, 3, self.warp_num)

        transforms = []
        transformed_products = []

        for i in range(self.warp_num):
            transforms.append(affine_transform[:,:,:,i])
            grid = F.affine_grid(affine_transform[:,:,:,i], product.size())
            transformed_products.append(F.grid_sample(product, grid, padding_mode="border"))

        return transforms, transformed_products

class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.ndf = ndf
        self.local_conv, self.local_cam, self.local_last = self.build_disc(True)
        self.global_conv, self.global_cam, self.global_last = self.build_disc(False)

    def build_disc(self, is_local):
        n_layers = 5 if is_local else 8

        model = [nn.ReflectionPad2d(1),
                 spectral_norm(nn.Conv2d(3, self.ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      spectral_norm(nn.Conv2d(self.ndf * mult, self.ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        model = nn.Sequential(*model)
        cam = CAM(self.ndf*mult, is_disc=True)
        last_conv = spectral_norm(nn.Conv2d(self.ndf*mult, 1, 4, padding=0, stride=1, bias=False))

        return model, cam, last_conv

    def get_feat(self, input, conv, cam, conv_last):
        feats = conv(input)
        cam_output, cam_logit = cam(feats)
        output = conv_last(cam_output)
        return output, cam_logit

    def forward(self, input):
        local, local_cam_logit = self.get_feat(input, self.local_conv, self.local_cam, self.local_last)
        glob, glob_cam_logit = self.get_feat(input, self.global_conv, self.global_cam, self.global_last)
        return local, local_cam_logit, glob, glob_cam_logit

class CAM(nn.Module):
    def __init__(self, in_channel, is_disc=True):
        super().__init__()

        # Class Activation Map
        self.gap_fc = nn.Linear(in_channel, 1, bias=False)
        self.gmp_fc = nn.Linear(in_channel, 1, bias=False)
        self.conv1x1 = nn.Conv2d(in_channel * 2, in_channel, kernel_size=1, stride=1, bias=True)
        if is_disc:
            self.gap_fc = spectral_norm(self.gap_fc)
            self.gmp_fc = spectral_norm(self.gmp_fc)
            self.conv1x1 = spectral_norm(self.conv1x1)
            self.activation = nn.LeakyReLU(0.2, True)
        else:
            self.activation = nn.ReLU(True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        gap = self.avgpool(x)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = self.maxpool(x)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.activation(self.conv1x1(x))

        return x, cam_logit
