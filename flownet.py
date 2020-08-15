import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks, input_channel_size):
        super(FPN, self).__init__()
        self.in_planes = input_channel_size

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  input_channel_size, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(block, 256, num_blocks[4], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(input_channel_size*4, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d( 1024, 256, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        # c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c1 = self.layer1(x) # torch.Size([batch, 256, 64, 48])
        c2 = self.layer2(c1) # torch.Size([batch, 256, 32, 24])
        c3 = self.layer3(c2) # torch.Size([batch, 256, 16, 12])
        c4 = self.layer4(c3) # torch.Size([batch, 256, 8, 6])
        c5 = self.layer5(c4) # torch.Size([batch, 256, 8, 6])


        # Top-down
        p5 = self.toplayer(c5) # torch.Size([batch, 256, 8, 6])
        p4 = self._upsample_add(p5, self.latlayer4(c4)) # torch.Size([batch, 256, 16, 12])
        p3 = self._upsample_add(p4, self.latlayer3(c3)) # torch.Size([batch, 256, 32, 24])
        p2 = self._upsample_add(p3, self.latlayer2(c2)) # torch.Size([batch, 256, 64, 48])
        p1 = self._upsample_add(p2, self.latlayer1(c1)) # torch.Size([batch, 256, 64, 48])

        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        p1 = self.smooth4(p1)

        return p1, p2, p3, p4, p5


class CLothFlowWarper(nn.Module):

    def __init__(self, opt):
        super(CLothFlowWarper, self).__init__()
        self.fpn_source = FPN(Bottleneck, [2,2,2,2,2], 4)
        self.fpn_target = FPN(Bottleneck, [2,2,2,2,2], 1)

        self.encoder_5 = nn.Sequential(nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1))
        self.encoder_4 = nn.Sequential(nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1))
        self.encoder_3 = nn.Sequential(nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1))
        self.encoder_2 = nn.Sequential(nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1))
        self.encoder_1 = nn.Sequential(nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1))

        for encoder in [self.encoder_1, self.encoder_2, self.encoder_3, self.encoder_4, self.encoder_5]:
            encoder[0].bias.data.zero_()

    def tv_loss(self, image):
        # shift one pixel and get difference (for both x and y direction)
        loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
               torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
        return loss

    def forward(self, target, source):
        # source is product
        # target is garment on model masked
        p1_S, p2_S, p3_S, p4_S, p5_S = self.fpn_source(source)
        p1_T, p2_T, p3_T, p4_T, p5_T = self.fpn_target(target)

        f5 = self.encoder_5(torch.cat([p5_S, p5_T], dim=1))
        u5 = F.upsample(f5, size=(f5.shape[2] * 2, f5.shape[3] * 2), mode='nearest')

        warp_4_S = F.grid_sample(p4_S, u5.permute(0,2,3,1))
        f4 = u5 + self.encoder_4(torch.cat([warp_4_S, p4_T], dim=1))
        u4 = F.upsample(f4, size=(f4.shape[2] * 2, f4.shape[3] * 2), mode='nearest')

        warp_3_S = F.grid_sample(p3_S, u4.permute(0,2,3,1))
        f3 = u4 + self.encoder_3(torch.cat([warp_3_S, p3_T], dim=1))
        u3 = F.upsample(f3, size=(f3.shape[2] * 2, f3.shape[3] * 2), mode='nearest')

        warp_2_S = F.grid_sample(p2_S, u3.permute(0,2,3,1))
        f2 = u3 + self.encoder_2(torch.cat([warp_2_S, p2_T], dim=1))
        u2 = F.upsample(f2, size=(f2.shape[2] * 2, f2.shape[3] * 2), mode='nearest')

        warp_1_S = F.grid_sample(p1_S, u2.permute(0,2,3,1))
        f1 = self.encoder_1(torch.cat([warp_1_S, p1_T], dim=1))

        u1 = F.upsample(f1, size=(f1.shape[2] * 2, f1.shape[3] * 2), mode='nearest')
        # grid = F.affine_grid(torch.tensor([[[1,0,0],[0,1,0]] for _ in range(12)], dtype=float), u2.size())
        # print(f1.shape)
        grid = u1.permute(0,2,3,1)
        # warped_cloth = F.grid_sample(inputB, grid, padding_mode='border')
        tv_loss = self.tv_loss(f5) + self.tv_loss(f4) + self.tv_loss(f3) + self.tv_loss(f2) + self.tv_loss(f1)
        return grid, tv_loss