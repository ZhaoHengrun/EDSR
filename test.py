import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math
import scipy.io as sio
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="PyTorch EDSR Test")
parser.add_argument("--cuda", default=True, action="store_true", help="use cuda?")
parser.add_argument("--model", default="checkpoints/model_edsr.pth", type=str, help="model path")
parser.add_argument("--image", default="butterfly_GT", type=str, help="image name")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


opt = parser.parse_args()
cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model = torch.load(opt.model)["model"]

im_gt = sio.loadmat("Set5/" + opt.image + ".mat")['im_gt']
im_b = sio.loadmat("Set5/" + opt.image + ".mat")['im_b']
im_l = sio.loadmat("Set5/" + opt.image + ".mat")['im_l']

im_gt = im_gt.astype(float).astype(np.uint8)
im_b = im_b.astype(float).astype(np.uint8)
im_l = im_l.astype(float).astype(np.uint8)

im_input = im_l.astype(np.float32).transpose(2, 0, 1)
im_input = im_input.reshape(1, im_input.shape[0], im_input.shape[1], im_input.shape[2])
im_input = Variable(torch.from_numpy(im_input / 255.).float())

if cuda:
    model = model.cuda()
    im_input = im_input.cuda()
else:
    model = model.cpu()

start_time = time.time()
out = model(im_input)
elapsed_time = time.time() - start_time

out = out.cpu()

im_h = out.data[0].numpy().astype(np.float32)

im_h = im_h * 255.
im_h[im_h < 0] = 0
im_h[im_h > 255.] = 255.
im_h = im_h.transpose(1, 2, 0)

print("Scale=", opt.scale)
print("It takes {}s for processing".format(elapsed_time))

fig = plt.figure()
plt.imshow(im_h.astype(np.uint8))
plt.title('EDSR')
plt.show()
plt.imshow(im_gt)
plt.title('gt')
plt.show()
plt.imshow(im_b)
plt.title('bicubic')
plt.show()
