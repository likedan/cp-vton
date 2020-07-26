#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, GMM_k_warps, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint, UNet, CLothFlowWarper
from resnet import Embedder
from unet import UNet, VGGExtractor, Discriminator, AccDiscriminator
from torch.utils.tensorboard import SummaryWriter
from visualization import board_add_images
from tqdm import tqdm
from torchvision.utils import save_image
import calculate_inception_score

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "boosting_GMM_l2_confidence")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=16)
    parser.add_argument('-b', '--batch-size', type=int, default=4)

    parser.add_argument('--local_rank', type=int, default=0, help="gpu to use, used for distributed training")
    parser.add_argument("--use_gan",  action='store_true')
    parser.add_argument("--no_consist",  action='store_true')

    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "test")
    parser.add_argument("--stage", default = "TOM+WARP+kWarps")
    parser.add_argument("--data_list", default = "test_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument("--k_warps", type=int, default = 4)

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 100)
    parser.add_argument("--save_count", type=int, default = 10000)
    parser.add_argument("--keep_step", type=int, default = 100000)
    parser.add_argument("--decay_step", type=int, default = 100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

def normalize(x):
    x = ((x+1)/2).clamp(0,1)
    return x


def test_tom_gmm_multi_warps(opt, loader, model, gmm_model):

    model.eval()
    gmm_model.eval()

    test_files_dir = "test_files_dir/" + opt.name
    os.makedirs(test_files_dir, exist_ok=True)
    os.makedirs(os.path.join(test_files_dir, "gt"), exist_ok=True)
    os.makedirs(os.path.join(test_files_dir, "output"), exist_ok=True)

    for index, (inputs, inputs_2) in tqdm(enumerate(loader.data_loader)):
        im = inputs['image'].cuda()
        agnostic = inputs['agnostic'].cuda()

        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()

        c_2 = inputs_2['cloth'].cuda()
        cm_2 = inputs_2['cloth_mask'].cuda()

        with torch.no_grad():
            warped_cloths, grids, thetas = gmm_model(agnostic, c_2)
            outputs = model(torch.cat([agnostic] + warped_cloths, 1))
        p_tryon = F.tanh(outputs)

        calculate_inception_score.inception_score(p_tryon)

        for b_i in range(im.shape[0]):
            save_image(normalize(im[b_i].cpu()),
                       os.path.join(test_files_dir, "gt", str(index * opt.batch_size + b_i) + ".jpg"))
            save_image(normalize(p_tryon[b_i].cpu()),
                       os.path.join(test_files_dir, "output", str(index * opt.batch_size + b_i) + ".jpg"))

def main():
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.distributed = n_gpu > 1
    local_rank = opt.local_rank

    # create dataset
    dataset = CPDataset(opt)

    # create dataloader
    loader = CPDataLoader(opt, dataset)

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

        gmm_model = GMM_k_warps(opt)
        gmm_model.cuda()

        model = UNet(n_channels=22 + 3 * opt.k_warps, n_classes=3)
        model.cuda()

        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
            load_checkpoint(gmm_model, opt.checkpoint.replace("step_", "step_warp_"))


        test_tom_gmm_multi_warps(opt, loader, model, gmm_model)


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
