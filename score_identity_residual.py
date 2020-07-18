import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint
from resnet import Embedder
from unet import UNet, VGGExtractor, Discriminator, AccDiscriminator
from torch.utils.tensorboard import SummaryWriter
from tryon_net import G
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

mse_criterion = nn.MSELoss()

def single_gpu_flag(args):
    return not args.distributed or (args.distributed and args.local_rank % torch.cuda.device_count() == 0)


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="residual_resnet")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=16)
    parser.add_argument('-b', '--batch-size', type=int, default=4)

    parser.add_argument('--local_rank', type=int, default=0, help="gpu to use, used for distributed training")

    parser.add_argument("--use_gan", action='store_true', default=True)
    parser.add_argument("--no_consist", action='store_true')

    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="test")
    parser.add_argument("--stage", default="residual_old")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default=100)
    parser.add_argument("--save_count", type=int, default=5000)
    parser.add_argument("--keep_step", type=int, default=100000)
    parser.add_argument("--decay_step", type=int, default=100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

opt = get_opt()
print(opt)
print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))

n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
opt.distributed = n_gpu > 1
local_rank = opt.local_rank

# create dataset
train_dataset = CPDataset(opt)

# create dataloader
train_loader = CPDataLoader(opt, train_dataset)
data_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=opt.batch_size, shuffle=False,
                num_workers=opt.workers, pin_memory=True)

gmm_model = GMM(opt)
load_checkpoint(gmm_model, "checkpoints/gmm_train_new/step_020000.pth")
gmm_model.cuda()

generator = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
load_checkpoint(generator, "checkpoints/tom_train_new_2/step_070000.pth")
generator.cuda()

embedder_model = Embedder()
load_checkpoint(embedder_model, "checkpoints/identity_embedding_for_test/step_045000.pth")
image_embedder = embedder_model.embedder_b.cuda()
prod_embedder = embedder_model.embedder_a.cuda()

model = G()
if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
    load_checkpoint(model, opt.checkpoint)
model.cuda()

model.eval()
gmm_model.eval()
image_embedder.eval()
generator.eval()
prod_embedder.eval()

pbar = tqdm(enumerate(data_loader), total=len(data_loader))

product_embeddings = []
outfit_embeddings = []
transfer_embeddings = []

product_embeddings_gt = []
outfit_embeddings_gt = []

for i, (inputs, inputs_2) in pbar:

    im = inputs['image'].cuda()
    im_pose = inputs['pose_image']
    im_h = inputs['head']
    shape = inputs['shape']

    agnostic = inputs['agnostic'].cuda()

    c = inputs['cloth'].cuda()
    cm = inputs['cloth_mask'].cuda()
    c_2 = inputs_2['cloth'].cuda()
    cm_2 = inputs_2['cloth_mask'].cuda()

    with torch.no_grad():
        grid, theta = gmm_model(agnostic, c)
        c = F.grid_sample(c, grid, padding_mode='border')
        cm = F.grid_sample(cm, grid, padding_mode='zeros')

        grid_2, theta_2 = gmm_model(agnostic, c_2)
        c_2 = F.grid_sample(c_2, grid_2, padding_mode='border')
        cm_2 = F.grid_sample(cm_2, grid_2, padding_mode='zeros')

        outputs = generator(torch.cat([agnostic, c], 1))
        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        transfer_1 = c * m_composite + p_rendered * (1 - m_composite)

        outputs_2 = generator(torch.cat([agnostic, c_2], 1))
        p_rendered_2, m_composite_2 = torch.split(outputs_2, 3, 1)
        p_rendered_2 = F.tanh(p_rendered_2)
        m_composite_2 = F.sigmoid(m_composite_2)
        transfer_2 = c_2 * m_composite_2 + p_rendered_2 * (1 - m_composite_2)

        gt_residual = (torch.mean(im, dim=1) - torch.mean(transfer_1, dim=1)).unsqueeze(1)

        output_1 = model(transfer_1.detach(), gt_residual.detach())

        embedding_o = image_embedder(output_1).cpu()
        embedding_t = image_embedder(transfer_1).cpu()
        embedding_p = prod_embedder(inputs['cloth'].cuda()).cpu()

        product_embeddings.append(embedding_p)
        outfit_embeddings.append(embedding_o)
        transfer_embeddings.append(embedding_t)

        embedding_o = image_embedder(inputs['image'].cuda()).cpu()
        embedding_p = prod_embedder(inputs_2['cloth'].cuda()).cpu()
        product_embeddings_gt.append(embedding_p)
        outfit_embeddings_gt.append(embedding_o)

product_embeddings = torch.cat(product_embeddings, dim=0).numpy()
outfit_embeddings = torch.cat(outfit_embeddings, dim=0).numpy()
transfer_embeddings = torch.cat(transfer_embeddings, dim=0).numpy()

product_embeddings_gt = torch.cat(product_embeddings_gt, dim=0).numpy()
outfit_embeddings_gt = torch.cat(outfit_embeddings_gt, dim=0).numpy()

def get_correct_count(outfit_embeddings, product_embeddings, top_k=5):
    correct_count = 0

    for i in tqdm(range(outfit_embeddings.shape[0])):

        scores = []
        for j in range(product_embeddings.shape[0]):
            scores.append(distance.euclidean(outfit_embeddings[i], product_embeddings[j]))
        sort_index = np.argsort(scores).tolist()
        if i in sort_index[0:top_k]:
            correct_count += 1
    return correct_count

def get_correct_match_count(outfit_embeddings, product_embeddings, transfer_embeddings, top_k=1):
    correct_count = 0

    for i in tqdm(range(outfit_embeddings.shape[0])):

        scores_o = []
        for j in range(product_embeddings.shape[0]):
            scores_o.append(distance.euclidean(outfit_embeddings[i], product_embeddings[j]))
        sort_index_o = np.argsort(scores_o).tolist()

        scores_t = []
        for j in range(product_embeddings.shape[0]):
            scores_t.append(distance.euclidean(transfer_embeddings[i], product_embeddings[j]))
        sort_index_t = np.argsort(scores_t).tolist()
        correct_count += len(set(sort_index_o[0: top_k]).intersection(set(sort_index_t[0: top_k]))) / top_k
    return correct_count

# def build_tree(vectors, dim=64):
#     a = AnnoyIndex(dim, 'euclidean')
#     for i, v in enumerate(vectors):
#         a.add_item(i, v)
#     a.build(-1)
#     return a
#
# product_tree = build_tree(product_embeddings)

dataset_size = len(train_dataset)

correct_count = get_correct_count(transfer_embeddings, product_embeddings, top_k=1)
print("acc_transfer", correct_count , correct_count / dataset_size)

correct_count = get_correct_match_count(outfit_embeddings, product_embeddings, transfer_embeddings, top_k=1)
print("acc 1", correct_count, correct_count / dataset_size)
correct_count = get_correct_match_count(outfit_embeddings, product_embeddings, transfer_embeddings, top_k=5)
print("acc 5", correct_count, correct_count / dataset_size)



