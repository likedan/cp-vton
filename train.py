#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, GMM_k_warps, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint, UNet, CLothFlowWarper, GMM_k_warps_Affine
from resnet import Embedder
from unet import UNet, VGGExtractor, Discriminator, AccDiscriminator
from torch.utils.tensorboard import SummaryWriter
from visualization import board_add_images
from tqdm import tqdm
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
import eval

def single_gpu_flag(args):
    return not args.distributed or (args.distributed and args.local_rank % torch.cuda.device_count() == 0)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "boosting_GMM_l2_confidence")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=16)
    parser.add_argument('-b', '--batch-size', type=int, default=4)

    parser.add_argument('--local_rank', type=int, default=0, help="gpu to use, used for distributed training")

    parser.add_argument("--test",  action='store_true')
    parser.add_argument("--warper_type", default = "TPS")

    parser.add_argument("--use_gan",  action='store_true')
    parser.add_argument("--no_consist",  action='store_true')

    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "CLOTHFLOW")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument("--k_warps", type=int, default = 2)

    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 100)
    parser.add_argument("--save_count", type=int, default = 20000)
    parser.add_argument("--keep_step", type=int, default = 200000)
    parser.add_argument("--decay_step", type=int, default = 200000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

def train_gmm(opt, train_loader, model, board):
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    
    for step in tqdm(range(opt.keep_step + opt.decay_step)):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        im_c =  inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
            
        warped_cloth, grid, theta, confidence_map = model(agnostic, c)
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

        l1_per_pixel = torch.abs(warped_cloth - im_c)

        visuals = [[im_h, shape, im_pose],
                   [c, warped_cloth, im_c], 
                   [warped_grid, (warped_cloth+im)*0.5, im],
                   [torch.cat([confidence_map, confidence_map, confidence_map], dim=1) * 2 - 1, l1_per_pixel, l1_per_pixel]]

        confidence_loss = nn.MSELoss()(l1_per_pixel, 2 - confidence_map * 2)
        l1 = torch.mean(l1_per_pixel)
        loss = l1 + confidence_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine' + str(step+1), visuals, step+1)
        board.add_scalar('Loss/total', loss.item(), step+1)
        board.add_scalar('Loss/l1', l1.item(), step+1)
        board.add_scalar('Loss/confidence_loss', confidence_loss.item(), step+1)
        # t = time.time() - iter_start_time
        # print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))


def train_tom(opt, train_loader, model, model_module, gmm_model, board):

    model.train()
    gmm_model.eval()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    
    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
            
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()

        with torch.no_grad():
            grid, theta = gmm_model(agnostic, c)
            c = F.grid_sample(c, grid, padding_mode='border')
            cm = F.grid_sample(cm, grid, padding_mode='zeros')

        outputs = model(torch.cat([agnostic, c],1))
        p_rendered, m_composite = torch.split(outputs, 3,1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite+ p_rendered * (1 - m_composite)

        visuals = [ [im_h, shape, im_pose], 
                   [c, cm*2-1, m_composite*2-1], 
                   [p_rendered, p_tryon, im]]
            
        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        loss_mask = criterionMask(m_composite, cm)
        loss = loss_l1 + loss_vgg + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if single_gpu_flag(opt):
            if (step+1) % opt.display_count == 0:
                board_add_images(board, str(step + 1), visuals, step + 1)
            board.add_scalar('metric', loss.item(), step+1)
            board.add_scalar('L1', loss_l1.item(), step+1)
            board.add_scalar('VGG', loss_vgg.item(), step+1)
            board.add_scalar('MaskL1', loss_mask.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f'
                    % (step+1, t, loss.item(), loss_l1.item(),
                    loss_vgg.item(), loss_mask.item()), flush=True)

        if (step+1) % opt.save_count == 0 and single_gpu_flag(opt):
            save_checkpoint(model_module, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))


def train_tom_gmm(opt, train_loader, model, model_module, gmm_model, gmm_model_module, board):
    model.train()
    gmm_model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + list(gmm_model.parameters()), lr=opt.lr, betas=(0.5, 0.999))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']
        im_c =  inputs['parse_cloth'].cuda()

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()

        warped_cloth, grid, theta, _ = gmm_model(agnostic, c)
        c = warped_cloth
        cm = F.grid_sample(cm, grid, padding_mode='zeros')

        # grid, theta = model(agnostic, c)
        # warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        # warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        # warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

        outputs = model(torch.cat([agnostic, c], 1))
        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        visuals = [[im_h, shape, im_pose],
                   [c, cm * 2 - 1, m_composite * 2 - 1],
                   [p_rendered, p_tryon, im]]

        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        loss_mask = criterionMask(m_composite, cm)
        loss_warp = criterionL1(c, im_c) * 2

        loss = loss_l1 + loss_vgg + loss_mask + loss_warp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % opt.display_count == 0 and single_gpu_flag(opt):
            board_add_images(board, 'combine' + str(step + 1), visuals, step + 1)
            board.add_scalar('metric', loss.item(), step + 1)
            board.add_scalar('L1', loss_l1.item(), step + 1)
            board.add_scalar('VGG', loss_vgg.item(), step + 1)
            board.add_scalar('MaskL1', loss_mask.item(), step + 1)
            board.add_scalar('Warp', loss_warp.item(), step + 1)

            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f, warp: %.4f'
                  % (step + 1, t, loss.item(), loss_l1.item(),
                     loss_vgg.item(), loss_mask.item(), loss_warp.item()), flush=True)

        if (step + 1) % opt.save_count == 0 and single_gpu_flag(opt):
            save_checkpoint(model_module, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step + 1)))
            save_checkpoint(gmm_model_module, os.path.join(opt.checkpoint_dir, opt.name, 'step_warp_%06d.pth' % (step + 1)))


def train_tom_gmm_multi_warps(opt, train_loader, model, model_module, gmm_model, gmm_model_module, board):
    model.train()
    gmm_model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + list(gmm_model.parameters()), lr=opt.lr, betas=(0.5, 0.999))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']
        im_c =  inputs['parse_cloth'].cuda()

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()

        warped_cloths, grids, thetas = gmm_model(agnostic, c)

        # c = warped_cloth
        # cm = F.grid_sample(cm, grid, padding_mode='zeros')
        #
        # # grid, theta = model(agnostic, c)
        # # warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        # # warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        # # warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

        outputs = model(torch.cat([agnostic] + warped_cloths, 1))
        p_tryon = F.tanh(outputs)

        visuals = [[im_h, shape, im_pose],
                   [c, cm * 2 - 1],
                   [p_tryon, im],
                   warped_cloths]

        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)

        transform_loss = 0
        for k in range(1, opt.k_warps + 1):
            warp_losses = []
            for i in range(k):
                per_pixel_loss = torch.abs(warped_cloths[i] - im_c)
                # merge spatial dimension to one
                per_pixel_loss = per_pixel_loss.view(im_c.size(0), -1).unsqueeze(2)
                warp_losses.append(per_pixel_loss)

            warp_losses = torch.cat(warp_losses, dim=2)
            per_pixel_min_loss = torch.min(warp_losses, dim=2)[0]

            transform_loss += torch.mean(per_pixel_min_loss)
        mse_criterion = nn.MSELoss()
        transform_l2_loss = 0
        if opt.k_warps > 1:
            for i in range(opt.k_warps - 1):
                transform_l2_loss += mse_criterion(thetas[0], thetas[i + 1])

        loss_warp = transform_loss + transform_l2_loss * .5
        loss_warp /= opt.k_warps

        loss = loss_l1 + loss_vgg + loss_warp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % opt.display_count == 0 and single_gpu_flag(opt):
            board_add_images(board, 'combine' + str(step + 1), visuals, step + 1)

            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, warp: %.4f'
                  % (step + 1, t, loss.item(), loss_l1.item(),
                     loss_vgg.item(), loss_warp.item()), flush=True)

        board.add_scalar('metric', loss.item(), step + 1)
        board.add_scalar('L1', loss_l1.item(), step + 1)
        board.add_scalar('VGG', loss_vgg.item(), step + 1)
        board.add_scalar('Warp', loss_warp.item(), step + 1)

        if (step + 1) % opt.save_count == 0 and single_gpu_flag(opt):
            save_checkpoint(model_module, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step + 1)))
            save_checkpoint(gmm_model_module, os.path.join(opt.checkpoint_dir, opt.name, 'step_warp_%06d.pth' % (step + 1)))


def train_cloth_flow(opt, train_loader, model, model_module, gmm_model, gmm_model_module, board):
    model.train()
    gmm_model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss(cuda=True)
    criterionMask = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + list(gmm_model.parameters()), lr=opt.lr, betas=(0.5, 0.999))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']
        im_c =  inputs['parse_cloth'].cuda()

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()

        warp_mask = (torch.min(im_c, dim=1)[0].unsqueeze(1) != 1).float().cuda()
        # torch.cat([agnostic, warp_mask], dim=1)
        grid, tv_loss = gmm_model(warp_mask, torch.cat([c, cm], dim=1))
        warped_mask = F.grid_sample(cm, grid, padding_mode='border')
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')

        # outputs = model(torch.cat([agnostic] + [warped_cloth], 1))
        # p_tryon = F.tanh(outputs)


        def normalize(x):
            return x * 2 - 1

        def unnormalize(x):
            return (x + 1) / 2

        visuals = [[im_h, shape, im_pose],
                   [c, normalize(cm), normalize(warp_mask.repeat(1,3,1,1))],
                   [warped_cloth, normalize(warped_mask), im_c],]
        # [p_tryon, im],
        #
        # loss_l1 = criterionL1(p_tryon, im)
        # print(torch.max(warped_mask), torch.min(warped_mask), torch.max(warp_mask), torch.min(warp_mask))
        loss_warp = criterionL1(warped_mask, warp_mask)
        loss_vgg = criterionVGG(warped_cloth, im_c, mask=warp_mask)
        # loss_vgg = criterionVGG(normalize(unnormalize(warped_cloth) * warp_mask), normalize(unnormalize(im_c) * warp_mask))

        loss_l1 = loss_warp

        loss = loss_warp * 10 + tv_loss * 50 + loss_vgg # loss_l1 + loss_vgg +
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % opt.display_count == 0 and single_gpu_flag(opt):
            board_add_images(board, 'combine' + str(step + 1), visuals, step + 1)

            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, warp: %.4f, tv: %.4f'
                  % (step + 1, t, loss.item(), loss_l1.item(),
                     loss_vgg.item(), loss_warp.item(), tv_loss.item()), flush=True)

        board.add_scalar('LOSS/metric', loss.item(), step + 1)
        board.add_scalar('LOSS/L1', loss_l1.item(), step + 1)
        board.add_scalar('LOSS/VGG', loss_vgg.item(), step + 1)
        board.add_scalar('LOSS/Warp', loss_warp.item(), step + 1)
        board.add_scalar('LOSS/tv', tv_loss.item(), step + 1)

        if (step + 1) % opt.save_count == 0 and single_gpu_flag(opt):
            save_checkpoint(model_module, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step + 1)))
            save_checkpoint(gmm_model_module, os.path.join(opt.checkpoint_dir, opt.name, 'step_warp_%06d.pth' % (step + 1)))


def main():
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))

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
        board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))

    # create model & train & save the final checkpoint
    if opt.stage == 'GMM':
        model = GMM(opt)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_gmm(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'gmm_final.pth'))
    elif opt.stage == 'TOM':

        gmm_model = GMM(opt)
        load_checkpoint(gmm_model, "checkpoints/gmm_train_new/step_020000.pth")
        gmm_model.cuda()

        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        model.cuda()

        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)

        model_module = model
        if opt.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                                       device_ids=[local_rank],
                                                                       output_device=local_rank,
                                                                       find_unused_parameters=True)
            model_module = model.module


        train_tom(opt, train_loader, model, model_module, gmm_model, board)
        if single_gpu_flag(opt):
            save_checkpoint(model_module, os.path.join(opt.checkpoint_dir, opt.name, 'tom_final.pth'))
    elif opt.stage == 'TOM+WARP':

        gmm_model = GMM(opt)
        gmm_model.cuda()

        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        model.cuda()
        # if opt.distributed:
        #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
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


        train_tom_gmm(opt, train_loader, model, model_module, gmm_model, gmm_model_module, board)

    elif opt.stage == 'TOM+WARP+kWarps':

        if opt.warper_type == "TPS":
            gmm_model = GMM_k_warps(opt)
        else:
            gmm_model = GMM_k_warps_Affine(opt)

        gmm_model.cuda()

        model = UNet(n_channels=22 + 3 * opt.k_warps, n_classes=3)
        model.cuda()

        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
            load_checkpoint(gmm_model, opt.checkpoint.replace("step_", "step_warp_"))

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

        train_tom_gmm_multi_warps(opt, train_loader, model, model_module, gmm_model, gmm_model_module, board)

    elif opt.stage == 'CLOTHFLOW':

        gmm_model = CLothFlowWarper(opt)
        gmm_model.cuda()

        model = UNet(n_channels=22 + 3, n_classes=3)
        model.cuda()

        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
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
        if single_gpu_flag(opt):
            save_checkpoint(model_module, os.path.join(opt.checkpoint_dir, opt.name, 'tom_final.pth'))

    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)


    print('Finished training %s, nameed: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
