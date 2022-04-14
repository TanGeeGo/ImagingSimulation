from random import uniform
import h5py
import cv2
import os
import glob
import time
import tifffile
import numpy as np

from utils import create_dir

def crop_patch(img, half_patch_size, stride, random_crop):
    """
    crop image into patches
    input args:
        img: input image array, np.array
        half_patch_size: half of patch size, int
        stride: stride of neighbor patch, int
        random_crop: if random crop the input image, bool
    """
    patch_list = []
    [h, w, c] = img.shape
    ######################################################################################
    # calculate the fov information
    h_range = np.arange(0, h, 1)
    w_range = np.arange(0, w, 1)
    img_fld_w, img_fld_h = np.meshgrid(w_range, h_range)
    img_fld_h = ((img_fld_h - (h-1)/2) / ((h-1)/2)).astype(np.float32)
    img_fld_w = ((img_fld_w - (w-1)/2) / ((w-1)/2)).astype(np.float32)
    img_fld_h = np.expand_dims(img_fld_h, -1)
    img_fld_w = np.expand_dims(img_fld_w, -1)
    img_wz_fld = np.concatenate([img, img_fld_h, img_fld_w], 2)
    ######################################################################################
    if random_crop:
        crop_num = 100
        pos = [(np.random.randint(half_patch_size, h - half_patch_size), \
            np.random.randint(half_patch_size, w - half_patch_size)) \
            for i in range(crop_num)]
    else:
        pos = [(ht, wt) for ht in range(half_patch_size, h, stride) \
            for wt in range(half_patch_size, w, stride)]

    for (ht, wt) in pos:
        cropped_img = img_wz_fld[ht - half_patch_size:ht + half_patch_size, wt - half_patch_size:wt + half_patch_size, :]
        patch_list.append(cropped_img)

    return patch_list

def mask_img(img_wz_fld, fov, fov_interval):
    """
    mask the img out of the range of fov
    """
    mask = np.where((fov_interval[0] <= fov) and (fov <= fov_interval[1]), 1, 0)
    # mask the out range pixel of img
    img_wz_fld[..., 0:6][mask == 0] = 0
    return img_wz_fld

def crop_patch_wzfov(img, half_patch_size, stride, random_crop, splited_fov, if_mask):
    """
    crop image into patches
    input args:
        img: input image array, np.array
        half_patch_size: half of patch size, int
        stride: stride of neighbor patch, int
        random_crop: if random crop the input image, bool
    """
    patch_list = []
    [h, w, c] = img.shape
    ######################################################################################
    # calculate the fov information
    h_range = np.arange(0, h, 1)
    w_range = np.arange(0, w, 1)
    img_fld_w, img_fld_h = np.meshgrid(w_range, h_range)
    img_fld_h = ((img_fld_h - (h-1)/2) / ((h-1)/2)).astype(np.float32)
    img_fld_w = ((img_fld_w - (w-1)/2) / ((w-1)/2)).astype(np.float32)
    img_fld_h = np.expand_dims(img_fld_h, -1)
    img_fld_w = np.expand_dims(img_fld_w, -1)
    img_wz_fld = np.concatenate([img, img_fld_h, img_fld_w], 2)
    ######################################################################################
    if random_crop:
        crop_num = 100
        pos = [(np.random.randint(half_patch_size, h - half_patch_size), \
            np.random.randint(half_patch_size, w - half_patch_size)) \
            for i in range(crop_num)]
    else:
        pos = [(ht, wt) for ht in range(half_patch_size, h, stride) \
            for wt in range(half_patch_size, w, stride)]

    for (ht, wt) in pos:
        cropped_img = img_wz_fld[ht - half_patch_size:ht + half_patch_size, wt - half_patch_size:wt + half_patch_size, :]
        # judge whether this cropped image is in the interval of fov
        cropped_fov = cropped_img[:, :, -2:] # fov information
        normalized_fov = np.sqrt(np.sum(np.power(cropped_fov, 2), 2)) / np.sqrt(1.0 + 1.0)
        
        if (splited_fov[0] <= np.max(normalized_fov)) and (np.min(normalized_fov) <= splited_fov[1]):
            if if_mask:
                cropped_img = mask_img(cropped_img, normalized_fov, splited_fov)
            
            patch_list.append(cropped_img) # include the image in the fov interval

    return patch_list

def gen_dataset(src_input_files, src_label_files, dst_path, date_index, splited_fov, if_mask):
    """
    generating datasets:
    input args: 
        src_input_files: input image files list, list[]
        src_label_files: label image files list, list[]
        dst_path: path for saving h5py file, str
    """
    # h5py file pathname, record the fov information
    h5py_path = dst_path + "/dataset_" + date_index + "_fov_" + \
                str(int(splited_fov[0] * 10)) + "_" + str(int(splited_fov[1] * 10)) + ".h5"
    h5f = h5py.File(h5py_path, 'w')

    for img_idx in range(len(src_input_files)):
        print("Now processing img pairs of %s", os.path.basename(src_input_files[img_idx]))
        img_input = tifffile.imread(src_input_files[img_idx])
        img_label = tifffile.imread(src_label_files[img_idx])

        # normalize the input and the label
        img_input = np.asarray(img_input / 65535, np.float32)
        img_label = np.asarray(img_label / 65535, np.float32)

        # concate input and label together
        img_pair = np.concatenate([img_input, img_label], 2)

        # crop the patch 
        if splited_fov == [0.0, 1.0]:
            patch_list = crop_patch(img_pair, 100, 100, False)
        else:
            patch_list = crop_patch_wzfov(img_pair, 100, 100, False, splited_fov, if_mask)

        # save the patches into h5py file
        for patch_idx in range(len(patch_list)):
            data = patch_list[patch_idx].copy()
            h5f.create_dataset(str(img_idx)+'_'+str(patch_idx), shape=(200,200,8), data=data)

    h5f.close()


if __name__ == "__main__":
    # generating train/valid/test datasets
    date_ind = "2022xxxx" # date information for h5py file
    dataset_type = "valid" # type of dataset "train" or "valid"
    camera_idx = "camera0x" # index of camera "camera01" to "camera05" 
    base_path = "./synthetic_datasets" # system path 
    input_dir = "input_rgb_2022xxxx" # input data dir
    label_dir = "label_rgb" # label data dir
    if_mask = False # whether add mask

    src_input_path = os.path.join(base_path, camera_idx, dataset_type + "_datasets", input_dir)
    src_label_path = os.path.join(base_path, camera_idx, dataset_type + "_datasets", label_dir)
    dst_path = os.path.join(base_path, camera_idx, dataset_type + "_datasets", "h5py_file")
    create_dir(dst_path)

    src_input_files = sorted(glob.glob(src_input_path + "/*.tiff"))
    src_label_files = sorted(glob.glob(src_label_path + "/*.tiff"))

    splited_fov = [0.0, 1.0]
    print("start dataset generation!")
    # generate one dataset in one step, in one image, split fov by the interval
    for interval_idx in range(len(splited_fov)-1):
        gen_dataset(src_input_files, src_label_files, dst_path, date_ind, \
                    [splited_fov[interval_idx], splited_fov[interval_idx+1]], if_mask)