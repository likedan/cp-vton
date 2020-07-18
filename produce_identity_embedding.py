# coding=utf-8
import torch
from annoy import AnnoyIndex

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint
from resnet import Embedder
from torch.utils.tensorboard import SummaryWriter
import json
from tqdm import tqdm
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)


def single_gpu_flag(args):
    return not args.distributed or (args.distributed and args.local_rank % torch.cuda.device_count() == 0)


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="identity")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('-b', '--batch-size', type=int, default=128)

    parser.add_argument('--local_rank', type=int, default=0, help="gpu to use, used for distributed training")

    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="test")
    parser.add_argument("--stage", default="identity")
    parser.add_argument("--data_list", default="test_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/identity_train_64_dim/step_020000.pth', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default=20)
    parser.add_argument("--save_count", type=int, default=100)
    parser.add_argument("--keep_step", type=int, default=100000)
    parser.add_argument("--decay_step", type=int, default=100000)
    parser.add_argument("--shuffle", default='False')

    opt = parser.parse_args()
    return opt

def test_identity_embedding(opt, train_loader, model, board):
    model.cuda()
    model.eval()

    pbar = tqdm(train_loader, total=len(train_loader))

    names = []
    embeddings_from_product = []
    embeddings_from_model = []
    for inputs_1, inputs_2 in pbar:

        img_1 = inputs_1['cloth'].cuda()
        img_ou_1 = inputs_1['image'].cuda()
        with torch.no_grad():
            pred_prod_embedding_1, pred_outfit_embedding_1 = model(img_1, img_ou_1)

        embeddings_from_product.append(pred_prod_embedding_1.cpu().detach())
        embeddings_from_model.append(pred_outfit_embedding_1.cpu().detach())

        names += inputs_1["im_name"]

    embeddings_from_product = torch.cat(embeddings_from_product).numpy()
    embeddings_from_model = torch.cat(embeddings_from_model).numpy()

    with open(os.path.join('data/identity_embedding_test.json'), 'w') as outfile:
        json.dump({"embeddings_from_product": embeddings_from_product.tolist(), "embeddings_from_model": embeddings_from_model.tolist(), "names": names}, outfile)




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

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.workers, pin_memory=True)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)

    board = None
    if single_gpu_flag(opt):
        board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))

    model = Embedder()
    if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
        load_checkpoint(model, opt.checkpoint)
    test_identity_embedding(opt, train_loader, model, board)
    print('Finished training %s, nameed: %s!' % (opt.stage, opt.name))


if __name__ == "__main__":
    main()
