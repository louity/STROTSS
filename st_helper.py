from glob import glob
import shutil

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from imageio import imwrite
import numpy as np

import utils
import vgg_pt
import pyr_lap
from stylize_objectives import objective_class

def style_transfer(stylized_im, content_im, style_path, output_path,
                   scale, long_side, mask, content_weight=0., use_guidance=False,
                   regions=0, coords=0, lr=2e-3, save_intermediate=False,
                   print_freq=100, max_iter=250, resample_freq=1, use_pyr=True):


    ### Keep track of current output image for GUI ###
    canvas = utils.aug_canvas(stylized_im, scale, 0)
    imwrite(output_path, canvas)

    #### Define feature extractor ###
    cnn = vgg_pt.Vgg16_pt()
    if torch.cuda.is_available():
        cnn = cnn.cuda()

    phi = lambda x: cnn.forward(x)
    phi2 = lambda x, y, z: cnn.forward_cat(x,z,samps=y,forward_func=cnn.forward)


    #### Optimize over laplaccian pyramid instead of pixels directly ####


    ### Define Optimizer ###
    if use_pyr:
        s_pyr = pyr_lap.dec_lap_pyr(stylized_im, 5)
        s_pyr = [Variable(li.data,requires_grad=True) for li in s_pyr]
    else:
        s_pyr = [Variable(stylized_im.data,requires_grad=True)]

    optimizer =  optim.RMSprop(s_pyr,lr=lr)

    ### Pre-Extract Content Features ###
    z_c = cnn(content_im)

    ### Pre-Extract Style Features from a Folder###
    paths = glob(style_path+'*')[::3]

    ### Create Objective Object ###
    objective_wrapper = 0
    objective_wrapper = objective_class(objective='remd_dp_g')

    z_s_all = []
    for ri in range(len(regions[1])):
        z_s, style_ims = utils.load_style_folder(phi2, paths, regions,ri, n_samps=-1, subsamps=1000, scale=long_side, inner=5)
        z_s_all.append(z_s)

    ### Extract guidance features if required ###
    gs = np.array([0.])
    if use_guidance:
        gs = utils.load_style_guidance(phi, style_path, coords[:,2:], scale=long_side)


    ### Randomly choose spatial locations to extract features from ###
    if use_pyr:
        stylized_im = pyr_lap.syn_lap_pyr(s_pyr)
    else:
        stylized_im = s_pyr[0]

    for ri in range(len(regions[0])):
        r_temp = regions[0][ri]
        r_temp = torch.from_numpy(r_temp).unsqueeze(0).unsqueeze(0).contiguous()
        r = F.interpolate(r_temp, (stylized_im.size(3),stylized_im.size(2)), mode='bilinear')[0,0,:,:].numpy()

        if r.max()<0.1:
            r = np.greater(r+1.,0.5)
        else:
            r = np.greater(r,0.5)

        objective_wrapper.init_inds(z_c, z_s_all,r,ri)

    if use_guidance:
        objective_wrapper.init_g_inds(coords, stylized_im)



    for i in range(max_iter):

        ### zero out gradients and compute output image from pyramid ##
        optimizer.zero_grad()
        if use_pyr:
            stylized_im = pyr_lap.syn_lap_pyr(s_pyr)
        else:
            stylized_im = s_pyr[0]

        ## Dramatically Resample Large Set of Spatial Locations ##
        if i==0 or i % (resample_freq*10) == 0:
            for ri in range(len(regions[0])):

                r_temp = regions[0][ri]
                r_temp = torch.from_numpy(r_temp).unsqueeze(0).unsqueeze(0).contiguous()
                r = F.interpolate(r_temp, (stylized_im.size(3),stylized_im.size(2)), mode='bilinear')[0,0,:,:].numpy()

                if r.max()<0.1:
                    r = np.greater(r+1.,0.5)
                else:
                    r = np.greater(r,0.5)

                objective_wrapper.init_inds(z_c, z_s_all,r,ri)

        ## Subsample spatial locations to compute loss over ##
        if i==0 or i%resample_freq == 0:
            objective_wrapper.shuffle_feature_inds()

        ## Extract Features from Current Output
        z_x = cnn(stylized_im)

        ## Compute Objective and take gradient step ##
        loss = objective_wrapper.eval(z_x, z_c, z_s_all, gs, 0., content_weight=content_weight,moment_weight=1.0)

        loss.backward()
        optimizer.step()

        ## Periodically save output image for GUI ###
        if save_intermediate and (i+1) % 20 == 0:
            canvas = utils.aug_canvas(stylized_im, scale, i)
            imwrite(output_path, canvas)

        ### Periodically Report Loss and Save Current Image ###
        if i % print_freq == 0:
            print('step {}/{}, loss {:.4f}'.format(i, max_iter, loss.item()))

    return stylized_im, loss
