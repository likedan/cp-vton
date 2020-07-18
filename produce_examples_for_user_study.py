# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from tryon_net import G

import utils
import argparse
import os
from torchvision.utils import save_image
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint
from resnet import Embedder
from unet import UNet, VGGExtractor, Discriminator
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from visualization import tensor_list_for_board
def normalize(x):
    x = ((x+1)/2).clamp(0,1)
    return x


def single_gpu_flag(args):
    return not args.distributed or (args.distributed and args.local_rank % torch.cuda.device_count() == 0)


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="test_vton")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=16)
    parser.add_argument('-b', '--batch-size', type=int, default=32)

    parser.add_argument('--local_rank', type=int, default=1, help="gpu to use, used for distributed training")

    parser.add_argument("--use_gan", action='store_true')

    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="test")
    parser.add_argument("--stage", default="residual")
    parser.add_argument("--data_list", default="test_files/one_person_different_cloth.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default=20)
    parser.add_argument("--save_count", type=int, default=5000)
    parser.add_argument("--keep_step", type=int, default=100000)
    parser.add_argument("--decay_step", type=int, default=100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

def test_residual(opt, loader, model, gmm_model, generator):

    model.eval()
    gmm_model.eval()
    generator.eval()

    test_files_dir = "result_files_dir/" + opt.name
    os.makedirs(test_files_dir, exist_ok=True)

    for i, (inputs, inputs_2) in tqdm(enumerate(loader), total=len(loader)):

        im = inputs['image'].cuda()
        agnostic = inputs['agnostic'].cuda()

        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()

        c_2 = inputs_2['cloth'].cuda()
        cm_2 = inputs_2['cloth_mask'].cuda()

        with torch.no_grad():
            grid, theta = gmm_model(agnostic, c)
            c = F.grid_sample(c, grid, padding_mode='border')
            cm = F.grid_sample(cm, grid, padding_mode='zeros')

            outputs = generator(torch.cat([agnostic, c], 1))
            p_rendered, m_composite = torch.split(outputs, 3, 1)
            p_rendered = F.tanh(p_rendered)
            m_composite = F.sigmoid(m_composite)
            transfer_1 = c * m_composite + p_rendered * (1 - m_composite)

            grid_2, theta_2 = gmm_model(agnostic, c_2)
            c_2 = F.grid_sample(c_2, grid_2, padding_mode='border')
            cm_2 = F.grid_sample(cm_2, grid_2, padding_mode='zeros')

            outputs_2 = generator(torch.cat([agnostic, c_2], 1))
            p_rendered_2, m_composite_2 = torch.split(outputs_2, 3, 1)
            p_rendered_2 = F.tanh(p_rendered_2)
            m_composite_2 = F.sigmoid(m_composite_2)
            transfer_2 = c_2 * m_composite_2 + p_rendered_2 * (1 - m_composite_2)

            gt_residual = (torch.mean(im, dim=1) - torch.mean(transfer_2, dim=1)).unsqueeze(1)

            output_1 = model(transfer_1.detach(), gt_residual.detach())

            output_residual = torch.cat([gt_residual, gt_residual, gt_residual], dim=1).cpu()

            visuals = [[output_residual],[transfer_1], [output_1], [output_1-transfer_1]]
            tensors = tensor_list_for_board(visuals)
            for b_i in range(transfer_1.shape[0]):
                save_image(tensors[b_i], os.path.join(test_files_dir, str(i * opt.batch_size + b_i) + ".jpg"))



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
    dataset = CPDataset(opt)

    # create dataloader
    loader = CPDataLoader(opt, dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.workers, pin_memory=True, sampler=None)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)


    gmm_model = GMM(opt)
    load_checkpoint(gmm_model, "checkpoints/gmm_train_new/step_020000.pth")
    gmm_model.cuda()

    generator_model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
    load_checkpoint(generator_model, "checkpoints/tom_train_new_2/step_040000.pth")
    generator_model.cuda()

    embedder_model = Embedder()
    load_checkpoint(embedder_model, "checkpoints/identity_train_64_dim/step_020000.pth")
    embedder_model = embedder_model.embedder_b.cuda()

    model = G()
    model.cuda()

    if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
        load_checkpoint(model, opt.checkpoint)

    test_residual(opt, data_loader, model, gmm_model, generator_model)

    print('Finished training %s, nameed: %s!' % (opt.stage, opt.name))


if __name__ == "__main__":
    main()