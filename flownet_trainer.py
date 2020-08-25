# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import utils
import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, GMM_k_warps, UnetGenerator, load_checkpoint, save_checkpoint, UNet, \
    GMM_k_warps_Affine
from flownet import CLothFlowWarper
from resnet import Embedder
from torch.utils.tensorboard import SummaryWriter
from visualization import board_add_images
from tqdm import tqdm
from distributed import (
    synchronize,
)

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

def single_gpu_flag(args):
    return not args.distributed or (args.distributed and args.local_rank % torch.cuda.device_count() == 0)


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="flow_warp_clamp")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=20)
    parser.add_argument('-b', '--batch-size', type=int, default=12)

    parser.add_argument('--local_rank', type=int, default=0, help="gpu to use, used for distributed training")

    parser.add_argument("--test", action='store_true')
    parser.add_argument("--warper_type", default="TPS")

    parser.add_argument("--use_gan", action='store_true')
    parser.add_argument("--no_consist", action='store_true')
    parser.add_argument("--stage", default = "CLOTHFLOW")

    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--k_warps", type=int, default=2)

    parser.add_argument('--lr', type=float, default=0.00001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default=100)
    parser.add_argument("--save_count", type=int, default=10000)
    parser.add_argument("--keep_step", type=int, default=200000)
    parser.add_argument("--decay_step", type=int, default=200000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt


def train_cloth_flow(opt, train_loader, generator, generator_module, gmm_model, gmm_model_module, board):
    generator.train()
    gmm_model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    vgg_extractor = VGGExtractor().cuda().eval()

    # optimizer
    optimizer = torch.optim.Adam(list(gmm_model.parameters()), lr=opt.lr, betas=(0.5, 0.999))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']
        im_c = inputs['parse_cloth'].cuda()

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        # cm = (torch.min(c, dim=1)[0].unsqueeze(1) != 1).float().cuda()

        def normalize(x):
            return x * 2 - 1

        def unnormalize(x):
            return (x + 1) / 2

        warp_mask = (torch.min(im_c, dim=1)[0].unsqueeze(1) != 1).float().cuda()
        # torch.cat([agnostic, warp_mask], dim=1)
        grid, flows = gmm_model(warp_mask, torch.cat([c, cm], dim=1))

        tv_loss = 0
        for f in flows:
            tv_loss += compute_tv_loss(f, cm)

        warped_mask = F.grid_sample(normalize(cm), grid, padding_mode='zeros')
        warped_cloth = F.grid_sample(c, grid, padding_mode='zeros')

        # outputs = model(torch.cat([agnostic] + [warped_cloth], 1))
        # p_tryon = F.tanh(outputs)

        visuals = [[im_h, shape, im_pose],
                   [c, normalize(cm), normalize(warp_mask)],
                   [warped_cloth, warped_mask, im_c], ]

        # [p_tryon, im],
        #
        # loss_l1 = criterionL1(p_tryon, im)
        loss_structure = criterionL1(warped_mask, normalize(warp_mask))

        real_feats = vgg_extractor(unnormalize(im_c))
        fake_feats = vgg_extractor(unnormalize(warped_cloth))
        loss_vgg = 0
        for a, b in zip(real_feats, fake_feats):
            interpolated_mask = F.interpolate(warp_mask, size=(a.shape[2], a.shape[3]),
                                              mode='bilinear').repeat(1, a.shape[1], 1, 1)
            loss_vgg += torch.mean(torch.abs(a - b) * interpolated_mask)

        loss = loss_structure * 20 + tv_loss * 2 + loss_vgg * 1  # loss_l1 + loss_vgg +
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % opt.display_count == 0 and single_gpu_flag(opt):
            board_add_images(board, 'combine' + str(step + 1), visuals, step + 1)

            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, tv: %.4f'
                  % (step + 1, t, loss.item(), loss_structure.item(),
                     loss_vgg.item(), tv_loss.item()), flush=True)

        if single_gpu_flag(opt):
            board.add_scalar('LOSS/metric', loss.item(), step + 1)
            board.add_scalar('LOSS/struc', loss_structure.item(), step + 1)
            board.add_scalar('LOSS/VGG', loss_vgg.item(), step + 1)
            board.add_scalar('LOSS/tv', tv_loss.item(), step + 1)

        if (step + 1) % opt.save_count == 0 and single_gpu_flag(opt):
            save_checkpoint(generator_module, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step + 1)))
            save_checkpoint(gmm_model_module,
                            os.path.join(opt.checkpoint_dir, opt.name, 'step_warp_%06d.pth' % (step + 1)))


def compute_tv_loss(image, mask):
    # shift one pixel and get difference (for both x and y direction)
    # interpolated_mask_2 = F.interpolate(mask, size=(image.shape[2]-1, image.shape[3]),
    #                                   mode='bilinear').repeat(1, image.shape[1], 1, 1)
    # interpolated_mask_3 = F.interpolate(mask, size=(image.shape[2], image.shape[3]-1),
    #                                   mode='bilinear').repeat(1, image.shape[1], 1, 1)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
           torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss

def main():
    opt = get_opt()
    print(opt)

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.distributed = n_gpu > 1
    local_rank = opt.local_rank

    if opt.distributed:
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    # create dataset
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)

    board = None
    if single_gpu_flag(opt):
        board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))

    # create model & train & save the final checkpoint
    gmm_model = CLothFlowWarper(opt)
    gmm_model.cuda()

    model = UNet(n_channels=22 + 3, n_classes=3)
    model.cuda()

    if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
        load_checkpoint(model, opt.checkpoint)

    model_module = model
    gmm_model_module = gmm_model
    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)
        model_module = model.module
        gmm_model = torch.nn.parallel.DistributedDataParallel(gmm_model,
                                                              device_ids=[local_rank],
                                                              output_device=local_rank,
                                                              find_unused_parameters=True)
        gmm_model_module = gmm_model.module

    train_cloth_flow(opt, train_loader, model, model_module, gmm_model, gmm_model_module, board)


if __name__ == "__main__":
    main()
