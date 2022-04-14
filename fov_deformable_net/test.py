import os, time, glob, cv2
import numpy as np
import matplotlib.image as mpimg
from skimage.measure import compare_psnr, compare_ssim
import torch
import torchvision.transforms as transforms

from utils import *
from option.option_20201230 import args
from model.__init__ import make_model

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

def test_info_generator(gt_src_file_list, psnr, ssim, test_time, test_time_avr, result_txt_path):
    '''
    record the psnr, ssim and the time consuming of each test
        gt_src_file_list: file list, list[]
        psnr: testing psnr recorder, list[]
        ssim: testing ssim recorder, list[]
        test_time: testing time recorder, list[]
        test_time_avr: average testing time, float
        result_txt_path: path to save the test info, str
    '''
    f_info_head = open(result_txt_path, 'w')
    for i in range(len(gt_src_file_list)):
        f_info_head.write('src_file: %s: psnr: %f, ssim: %f, time: %f \n' %(os.path.basename(gt_src_file_list[i]), psnr[i], ssim[i], test_time[i]))

    f_info_head.write('average time: %f' %(test_time_avr))
    f_info_head.close()
    return 0



def evaluate_net():
    create_dir(args.result_png_path)
    print('Testing path is %s' % args.blur_src_path)
    blurred_src_file_list = sorted(glob.glob(args.blur_src_path + '/*.png' ))
    gt_src_file_list = sorted(glob.glob(args.gt_src_path + '/*.png'))

    if args.gt_src_path:
        psnr = np.zeros(len(gt_src_file_list))
        ssim = np.zeros(len(gt_src_file_list))
        test_time = np.zeros(len(gt_src_file_list) * 96)

    # Build model
    input_channel, output_channel = 5, 3

    model = make_model(input_channel, output_channel, args)

    if torch.cuda.is_available():
        model_dict = torch.load(args.ckpt_dir_test + '/model_%04d_dict.pth' % args.epoch_test)
        model.load_state_dict(model_dict)
        model = model.cuda()
        print('Finish loading the model of the %dth epoch' % args.epoch_test)
    else:
        print('There are not available cuda devices !')

    model.eval()

    #=================#
    for index in range(len(gt_src_file_list)):
        out_patch_list = []
        img_name = os.path.split(gt_src_file_list[index])[-1].split('.')[0]
        
        # read the image
        gt_img = cv2.imread(gt_src_file_list[index])
        gt_img = gt_img[..., ::-1]
        gt_img = np.asarray(gt_img / 255, np.float64)
        in_img = cv2.imread(blurred_src_file_list[index])
        in_img = in_img[..., ::-1]
        in_img = np.asarray(in_img / 255, np.float64)

        # add noise
        if args.sigma:
            noise = np.random.normal(loc=0, scale=args.sigma/255.0, size=in_img.shape)
            in_img = in_img + noise
            in_img = np.clip(in_img, 0.0, 1.0)

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
            start_time = time.time()
            with torch.no_grad():
                out_patch = model(in_patch)
            torch.cuda.synchronize()
            test_time[index * 96 + i] = time.time() - start_time

            rgb_patch = out_patch.cpu().detach().numpy().transpose((0, 2, 3, 1))
            rgb_patch = np.clip(rgb_patch[0], 0, 1)
            out_patch_list.append(rgb_patch)

        rgb = sew_up_img(out_patch_list, patch_size=500, pad_size=100, img_size=[3000, 4000])
        
        # compare psnr and ssim
        psnr[index] = compare_psnr(gt_img, rgb)
        ssim[index] = compare_ssim(gt_img, rgb, multichannel=True)
        # save image
        rgb = rgb[..., ::-1]
        cv2.imwrite(args.result_png_path + '/' + img_name + ".png", np.uint8(rgb*255))
        print('test image: %s saved!' %img_name)

    test_time_avr = 0
    #===========
    #print psnr,ssim
    for i in range(len(gt_src_file_list)):
        print('src_file: %s: ' %(os.path.split(gt_src_file_list[i])[-1].split('.')[0]))
        if args.gt_src_path:
            print('psnr: %f, ssim: %f, average time: %f' % (psnr[i], ssim[i], test_time[i]))

        if i > 0:
            test_time_avr += test_time[i]

    test_time_avr = test_time_avr / (len(gt_src_file_list)-1)
    print('average time: %f' % (test_time_avr))
    # save the psnr, ssim information
    result_txt_path = args.result_png_path + '/' + "test_result.txt"
    test_info_generator(gt_src_file_list, psnr, ssim, test_time, test_time_avr, result_txt_path)
    return 0

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print('Use {} GPU, which order is {:s}th'.format(torch.cuda.device_count(), args.gpu))

    evaluate_net()
