from __future__ import print_function, division
import os
import argparse
import torch.nn as nn
from skimage import io
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from datasets import __datasets__
from models import __models__
from utils import *
from utils.KittiColormap import *

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='MobileStereoNet')
parser.add_argument('--model', default='MSNet2D', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--dataset', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--testlist', required=True, help='testing list')
parser.add_argument('--loadckpt', required=True, help='load the weights from a specific checkpoint')
parser.add_argument('--colored', default=1, help='save colored or save for benchmark submission')

# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()

# load parameters
print("Loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])


def test(args):
    print("Generating the disparity maps...")

    os.makedirs('./predictions', exist_ok=True)

    for batch_idx, sample in enumerate(TestImgLoader):

        disp_est_tn = test_sample(sample)
        disp_est_np = tensor2numpy(disp_est_tn)
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]

        for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):

            assert len(disp_est.shape) == 2

            disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
            name = fn.split('/')
            fn = os.path.join("predictions", '_'.join(name[2:]))

            if float(args.colored) == 1:
                disp_est = kitti_colormap(disp_est)
                cv2.imwrite(fn, disp_est)
            else:
                disp_est = np.round(disp_est * 256).astype(np.uint16)
                io.imsave(fn, disp_est)

    print("Done!")


@make_nograd_func
def test_sample(sample):
    model.eval()
    disp_ests = model(sample['left'].cuda(), sample['right'].cuda())
    return disp_ests[-1]


if __name__ == '__main__':
    test(args)
