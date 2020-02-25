import argparse
import numpy as np
import torchvision
import torch
import time

import torch.nn.functional as F
from imageio import imread, imwrite
from PIL import Image
from glob import glob

import st_helper
import utils

def run_style_transfer(content_path, style_path, content_weight, max_scale, coords, use_guidance, regions, output_path='./output.png', print_freq=100, use_sinkhorn=False, sinkhorn_reg=0.1, sinkhorn_maxiter=30):

    smallest_size = 64
    start = time.time()

    content_image = torchvision.transforms.functional.to_tensor(Image.open(content_path).convert('RGB')) - 0.5
    _, content_H, content_W = content_image.size()
    print('content image size {}x{}'.format(content_H, content_W))

    style_image = torchvision.transforms.functional.to_tensor(Image.open(style_path).convert('RGB')) - 0.5
    _, style_H, style_W = style_image.size()
    print('style image size {}x{}'.format(style_H, style_W))


    big_image_size = (int(content_H * 512 / content_W), 512) if content_H < content_W else (512 , int(content_W * 512 / content_H))
    content_image_big = F.interpolate(content_image.unsqueeze(0), size=big_image_size, mode='bilinear')

    if torch.cuda.is_available():
        content_image_big = content_image_big.cuda()

    for scale in range(1, max_scale+1):
        t0 = time.time()

        scaled_size = smallest_size*(2**(scale-1))

        print('Processing scale {}/{}, size {}...'.format(scale, max_scale, scaled_size))

        content_scaled_size = (int(content_H * scaled_size / content_W), scaled_size) if content_H < content_W else (scaled_size , int(content_W * scaled_size / content_H))
        # style_scaled_size = (int(style_H * scaled_size / style_W), scaled_size) if style_H < style_W else (scaled_size , int(style_W * scaled_size / style_H))

        lr = 2e-3

        ### Load Style and Content Image ###
        content_image_scaled = F.interpolate(content_image.unsqueeze(0), size=content_scaled_size, mode='bilinear')

        if torch.cuda.is_available():
            content_image_scaled = content_image_scaled.cuda()


        style_image_mean = style_image.unsqueeze(0).mean(dim=(2, 3), keepdim=True)
        if torch.cuda.is_available():
            style_image_mean = style_image_mean.cuda()

        ### Compute bottom level of laplaccian pyramid for content image at current scale ###
        scaled_H, scaled_W = content_image_scaled.size(2), content_image_scaled.size(3)
        content_image_downsampled = F.interpolate(content_image_scaled, (scaled_H//2, scaled_W//2), mode='bilinear')
        bottom_laplacian = content_image_scaled - F.interpolate(content_image_downsampled, (scaled_H, scaled_W), mode='bilinear')

        canvas = F.interpolate(bottom_laplacian.clamp(-0.5, 0.5), (scaled_H, scaled_W),mode='bilinear')[0].cpu().numpy().transpose(1,2,0)

        if scale == 1:
            canvas = F.interpolate(content_image_scaled, (scaled_H//2, scaled_W//2),mode='bilinear')[0].cpu().numpy().transpose(1,2,0)

        # Initialize by zeroing out all but highest and lowest levels of Laplaccian Pyramid
        # Otherwise bilinearly upsample previous scales output and add back bottom level of Laplaccian pyramid for current scale of content image
        if scale == 1:
            stylized_im = style_image_mean + bottom_laplacian
        elif scale > 1 and scale < max_scale:
            stylized_im = F.interpolate(stylized_im.clone(), (scaled_H, scaled_W), mode='bilinear') + bottom_laplacian
        elif scale == max_scale:
            stylized_im = F.interpolate(stylized_im.clone(), (scaled_H, scaled_W), mode='bilinear')
            lr = 1e-3

        ### Style Transfer at this scale ###
        stylized_im, final_loss = st_helper.style_transfer(stylized_im, content_image_scaled, style_path, output_path, scale, scaled_size, 0., use_guidance=use_guidance, coords=coords, content_weight=content_weight, lr=lr, regions=regions, print_freq=print_freq, use_sinkhorn=use_sinkhorn, sinkhorn_reg=sinkhorn_reg, sinkhorn_maxiter=sinkhorn_maxiter)

        canvas = F.interpolate(stylized_im.clamp(-0.5, 0.5), (scaled_H, scaled_W),mode='bilinear')[0].detach().cpu().numpy().transpose(1,2,0)

        ### Decrease Content Weight for next scale ###
        content_weight /= 2.0
        print('...done in {:.1f} sec, final loss {:.4f}'.format(time.time()-t0, final_loss.item()))

    print('Finished in {:.1f} secs' .format(time.time()-start))

    canvas = torch.clamp(stylized_im[0],-0.5,0.5).data.cpu().numpy().transpose(1,2,0)
    imwrite(output_path,canvas)
    return final_loss , stylized_im

if __name__=='__main__':
    parser = argparse.ArgumentParser('style transfer by relaxed optimal transport')
    parser.add_argument('--content_path', help="path of content img", required=True)
    parser.add_argument('--style_path', help="path of style img", required=True)
    parser.add_argument('--output_path', help="path of output img", default='output.png')
    parser.add_argument('--content_weight', type=float, help='no padding used', default=0.5)
    parser.add_argument('--max_scale', type=int, help='max scale for the style transfer', default=4)
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument('--content_guidance_path', default='', help="path of content guidance region image")
    parser.add_argument('--style_guidance_path', default='', help="path of style guidance regions image")
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency for the loss')
    parser.add_argument('--use_sinkhorn', action='store_true', help='use sinkhorn algo. for the earth mover distance')
    parser.add_argument('--sinkhorn_reg', type=float, help='reg param for sinkhorn', default=0.1)
    parser.add_argument('--sinkhorn_maxiter', type=int, default=30, help='number of interations for sinkohrn algo')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    content_path = args.content_path
    style_path = args.style_path
    content_weight = 16 * args.content_weight
    max_scale = args.max_scale
    use_guidance_region = args.content_guidance_path and args.style_guidance_path
    use_guidance_points = False


    paths = glob(style_path+'*')
    losses = []
    ims = []


    ### Preprocess User Guidance if Required ###
    coords=0.
    if use_guidance_region:
        regions = utils.extract_regions(args.content_guidance_path, args.style_guidance_path)
    else:
        try:
            regions = [[imread(content_path)[:,:,0]*0.+1.], [imread(style_path)[:,:,0]*0.+1.]]
        except:
            regions = [[imread(content_path)[:,:]*0.+1.], [imread(style_path)[:,:]*0.+1.]]

    ### Style Transfer and save output ###
    loss, canvas = run_style_transfer(content_path,style_path,content_weight,max_scale,coords,use_guidance_points,regions, args.output_path, print_freq=args.print_freq, use_sinkhorn=args.use_sinkhorn, sinkhorn_reg=args.sinkhorn_reg, sinkhorn_maxiter=args.sinkhorn_maxiter)
