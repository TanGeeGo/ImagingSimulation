import os, time, glob, cv2
import tifffile
import numpy as np
import torch
import torchvision.transforms as transforms
import scipy.io as sio
from utils import *
from option.option_20220330 import args
from deformable_unet import DFUNet

def compute_fld_info(img):
    [h, w, c] = img.shape
    h_range = np.arange(0, h, 1)
    w_range = np.arange(0, w, 1)
    img_fld_w, img_fld_h = np.meshgrid(w_range, h_range)
    img_fld_h = ((img_fld_h - (h-1)/2) / ((h-1)/2)).astype(np.float32)
    img_fld_w = ((img_fld_w - (w-1)/2) / ((w-1)/2)).astype(np.float32)
    img_fld_h = np.expand_dims(img_fld_h, -1)
    img_fld_w = np.expand_dims(img_fld_w, -1)
    img_wz_fld = np.concatenate([img, img_fld_h, img_fld_w], 2)
    return img_wz_fld

def crop_small_patch(in_img_wz_fld, patch_half_size=100):
    [H_img, W_img, C] = in_img_wz_fld.shape
    patch_1 = in_img_wz_fld[int(H_img/2)-patch_half_size : int(H_img/2)+patch_half_size,
                            int(W_img/2)-patch_half_size : int(W_img/2)+patch_half_size, :]
    patch_2 = in_img_wz_fld[1413-patch_half_size : 1413+patch_half_size,
                            2119-patch_half_size : 2119+patch_half_size, :]
    patch_3 = in_img_wz_fld[1017-patch_half_size : 1017+patch_half_size,
                            1526-patch_half_size : 1526+patch_half_size, :]
    patch_4 = in_img_wz_fld[277-patch_half_size : 277+patch_half_size,
                            415-patch_half_size : 415+patch_half_size, :]
    patch_list = [patch_1, patch_2, patch_3, patch_4]
    return patch_list

def crop_patch(padded_in_img_wz_fld, patch_size=500, pad_size=100):
    patch_list = []
    [H_img, W_img, C] = padded_in_img_wz_fld.shape
    H_num = int((H_img-pad_size) / patch_size)
    W_num = int((W_img-pad_size) / patch_size)
    for h_index in range(H_num):
        for w_index in range(W_num):
            patch = padded_in_img_wz_fld[patch_size*h_index : patch_size*(h_index+1)+pad_size,
                                         patch_size*w_index : patch_size*(w_index+1)+pad_size, :]
            patch_list.append(patch)

    return patch_list

def sew_up_img(out_patch_list, patch_size=500, pad_size=100, img_size=[3000, 4000]):
    rgb = np.zeros((img_size[0], img_size[1], 3))
    for patch_index in range(len(out_patch_list)):
        # w seq first, h seq second
        h_index = patch_index // 8
        w_index = patch_index - h_index*8
        patch_data = out_patch_list[patch_index].copy()

        patch_data = patch_data[int(pad_size/2) : int(patch_size+pad_size/2), int(pad_size/2) : int(patch_size+pad_size/2), :]
        rgb[h_index*patch_size : (h_index+1)*patch_size, w_index*patch_size : (w_index+1)*patch_size] = patch_data

    return rgb

def postprocessing(img, ccm):
    # copy image
    # img_out = np.zeros_like(img)
    # apply gamma
    img_out = np.power((img+1e-8), 0.454)
    img_out = np.clip(img_out, 0.0, 1.0)
    # apply ccm
    img_out = apply_cmatrix(img_out, ccm)
    img_out = np.clip(img_out, 0.0, 1.0)
    return img_out

def apply_cmatrix(img, ccm):
    if not np.size(img, 2) == 3:
        raise ValueError('Incorrect channel dimension!')
    
    img_out = np.zeros_like(img)
    img_out[:, :, 0] = ccm[0, 0] * img[:, :, 0] + ccm[0, 1] * img[:, :, 1] + ccm[0, 2] * img[:, :, 2]
    img_out[:, :, 1] = ccm[1, 0] * img[:, :, 0] + ccm[1, 1] * img[:, :, 1] + ccm[1, 2] * img[:, :, 2]
    img_out[:, :, 2] = ccm[2, 0] * img[:, :, 0] + ccm[2, 1] * img[:, :, 1] + ccm[2, 2] * img[:, :, 2]
    return img_out

def evaluate_net():
    create_dir(args.real_dst_tiff_path)
    print('Testing path is %s' % args.real_blur_src_path)
    blurred_src_file_list = sorted(glob.glob(args.real_blur_src_path + '/*.tiff' ))
    blurred_head_file_list = sorted(glob.glob(args.real_blur_src_path + '/*.mat' ))

    # Build model
    input_channel, output_channel = 5, 3

    model = DFUNet(input_channel, output_channel, args.n_channel, args.offset_channel)

    if torch.cuda.is_available():
        model_dict = torch.load(args.ckpt_dir_test_real + '/model_%04d_dict.pth' % args.epoch_test_real)
        model.load_state_dict(model_dict)
        model = model.cuda()
        print('Finish loading the model of the %dth epoch' % args.epoch_test_real)
    else:
        print('There are not available cuda devices !')

    model.eval()

    #=================#
    for index in range(len(blurred_src_file_list)):
        out_patch_list = []
        img_name = os.path.split(blurred_src_file_list[index])[-1].split('.')[0]
        in_img = tifffile.imread(blurred_src_file_list[index])
        head = sio.loadmat(blurred_head_file_list[index])
        wb, ccm = head['wb'], head['ccm']

        in_img = np.asarray(in_img / 65535, np.float32)

        # normalize to 0~1
        in_img = (in_img - np.min(in_img)) / (np.max(in_img) - np.min(in_img)) 

        # compute field
        in_img_wz_fld = compute_fld_info(in_img)
        [h, w, c] = in_img_wz_fld.shape
        padded_in_img_wz_fld = np.pad(in_img_wz_fld, ((50, 50), (50, 50), (0, 0)), 'edge')
        # crop_patch
        patch_list = crop_patch(padded_in_img_wz_fld, patch_size=500, pad_size=100)
        # concat in and gt, gt->in
        print('process img: %s' % blurred_src_file_list[index])
        for i in range(len(patch_list)):
            in_patch = patch_list[i].copy()
            in_patch = transforms.functional.to_tensor(in_patch)
            in_patch = in_patch.unsqueeze_(0).float()
            if torch.cuda.is_available():
                in_patch = in_patch.cuda()

            torch.cuda.synchronize()
            with torch.no_grad():
                out_patch = model(in_patch)
            torch.cuda.synchronize()

            rgb_patch = out_patch.cpu().detach().numpy().transpose((0, 2, 3, 1))
            rgb_patch = np.clip(rgb_patch[0], 0, 1)
            out_patch_list.append(rgb_patch)

        rgb = sew_up_img(out_patch_list, patch_size=500, pad_size=100, img_size=[3000, 4000])
        # postprocessing
        in_img = postprocessing(in_img, ccm)
        rgb = postprocessing(rgb, ccm)
        # save image
        cv2.imwrite(args.real_dst_tiff_path + '/' + img_name + '_ipt.png', (in_img[..., ::-1]*255).astype(np.uint8))
        cv2.imwrite(args.real_dst_tiff_path + '/' + img_name + '_out.png', (rgb[..., ::-1]*255).astype(np.uint8))
#-------------------------------------------------------------------------------
#        for patch_index in range(len(out_patch_list)):
#            # save image
#            img_patch = patch_list[patch_index]
#            create_dir(args.real_dst_png_path + '/' + img_name)
#            imwrite(args.real_dst_png_path + '/' + img_name + '/real_test_deblur_%02d.png' %patch_index, np.uint8(out_patch_list[patch_index]*255))
#            imwrite(args.real_dst_png_path + '/' + img_name + '/real_test_src_%02d.png' % patch_index, np.uint8(img_patch[:, :, 0:3]*255))
#-------------------------------------------------------------------------------
        print('real test img of %s saved!' %blurred_src_file_list[index])

    return 0


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print('Use {} GPU, which order is {:s}th'.format(torch.cuda.device_count(), args.gpu))

    evaluate_net()
