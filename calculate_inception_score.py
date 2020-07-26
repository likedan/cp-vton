
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import os
from torchvision.models.inception import inception_v3
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from PIL import Image
from tqdm import tqdm
import numpy as np
from scipy.stats import entropy
import torchvision.transforms as transforms


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in tqdm(enumerate(dataloader, 0)):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder_dir):
        self.folder_dir = folder_dir
        self.imgs = os.listdir(folder_dir)
        self.transform = transforms.Compose([
                                 transforms.ToTensor(),
                                 # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])

    def __getitem__(self, index):
        im = Image.open(os.path.join(self.folder_dir, self.imgs[index]))
        im = self.transform(im)
        return im

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', type=str,
                        help=('Path to the generated images or '
                              'to .npz statistic files'))
    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise RuntimeError('Invalid path: %s' % args.path)

    output_dataset = ImageDataset(args.path)

    print("Calculating Inception Score...")
    print(inception_score(output_dataset, cuda=True, batch_size=8, resize=False, splits=10))