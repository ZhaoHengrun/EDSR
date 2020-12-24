from __future__ import print_function

import argparse
import torch
import numpy as np
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from dataset import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_image', type=str, default='datasets/test/test.png', help='input image to use')
parser.add_argument('--input_LR_path', type=str, default='datasets/test/DIV2K/LR/', help='input path to use')
parser.add_argument('--input_HR_path', type=str, default='datasets/test/DIV2K/HR/', help='input path to use')
parser.add_argument('--model', type=str, default='checkpoints/m.pth', help='model file to use')
parser.add_argument('--output_path', default='results/', type=str, help='where to save the output image')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
opt = parser.parse_args()

print(opt)


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


# def calc_psnr(img1, img2):
#     criterion = nn.MSELoss()
#     mse = criterion(img1, img2)
#     return 10 * log10(1 / mse.item())


loader = transforms.Compose([
    transforms.ToTensor()])

path = opt.input_LR_path
path_HR = opt.input_HR_path

crop_size = 256
scale = 4
image_nums = len([lists for lists in listdir(path) if is_image_file('{}/{}'.format(path, lists))])
print(image_nums)
psnr_avg = 0
psnr_avg_bicubic = 0
for i in listdir(path):
    if is_image_file(i):
        with torch.no_grad():
            img_name = i.split('.')
            img_num = img_name[0]

            img_original = Image.open('{}{}'.format(path_HR, i))
            img_original_ybr = img_original.convert('YCbCr')
            img_original_y, _, _ = img_original_ybr.split()

            img_LR = Image.open('{}{}'.format(path, i))

            if len(np.array(img_LR).shape) != 3:
                img_LR = img_LR.convert('RGB')

            img_to_tensor = ToTensor()
            input = img_to_tensor(img_LR)
            input = Variable(torch.unsqueeze(input, dim=0).float(), requires_grad=False)

            model = torch.load(opt.model, map_location='cuda:0')
            # model = torch.load(opt.model)['model']
            # model = torch.load(opt.model, map_location='cpu')['model']
            if opt.cuda:
                model = model.cuda()
                input = input.cuda()

            out = model(input)

            out = out.cpu()

            im_h = out.data[0].numpy().astype(np.float32)
            im_h = im_h * 255.
            im_h = np.clip(im_h, 0., 255.)
            im_h = im_h.transpose(1, 2, 0)
            im_h_pil = Image.fromarray(im_h.astype(np.uint8))
            im_h_pil_ybr = im_h_pil.convert('YCbCr')
            im_h_pil_y, _, _ = im_h_pil_ybr.split()

            # fig = plt.figure()
            # plt.imshow(im_h.astype(np.uint8))
            # plt.title('EDSR')
            # plt.show()

            psnr_val = calc_psnr(loader(im_h_pil_y), loader(img_original_y))
            psnr_avg += psnr_val
            print(psnr_val)

            im_h_pil.save('{}output/{}.png'.format(opt.output_path, img_num))
            img_original.save('{}gt/{}.png'.format(opt.output_path, img_num))
            # print('output image saved to ', opt.output_path)
psnr_avg = psnr_avg / image_nums
print('psnr_avg', psnr_avg)
