import argparse
import numpy as np

parser = argparse.ArgumentParser(description='SADNET')
# File paths
parser.add_argument('--src_path', type=str, 
                    default="./dataset_train.h5",
                    help='training dataset path')
parser.add_argument('--val_path', type=str, 
                    default="./dataset_valid.h5",
                    help='validating dataset path, if not, set None')
parser.add_argument('--ckpt_dir', type=str, 
                    default="./ckpt_dir",
                    help='model directory')
parser.add_argument('--log_dir', type=str, 
                    default="./log_dir",
                    help='log directory')
# Hardware specifications
parser.add_argument('--gpu', type=str, default="2",
                    help='GPUs')

# Training parameters
parser.add_argument('--batch_size', type=int, default=32,
                    help='training batch size')
parser.add_argument('--val_batch_size', type=int, default=4,
                    help='validing batch size')
parser.add_argument('--patch_size', type=int, default=128,
                    help='training patch size')
parser.add_argument('--sigma', type=int, default=2,
                    help='noise sigma')
parser.add_argument('--n_epoch', type=int, default=30,
                    help='the number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--milestone', type=int, default=5,
                    help='the epochs for weight decay')
parser.add_argument('--val_epoch', type=int, default=1,
                    help='do validation per every N epochs')
parser.add_argument('--save_val_img', type=bool, default=True,
                    help='save the last validated image for comparison')
parser.add_argument('--val_patch_size', type=int, default=512,
                    help='patch size in validation dataset')
parser.add_argument('--save_epoch', type=int, default=1,
                    help='save model per every N epochs')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for every milestone')
parser.add_argument('--finetune', type=bool, default=False,
                    help='if finetune model, set True')
parser.add_argument('--init_epoch', type=int, default=0,
                    help='if finetune model, set the initial epoch')
parser.add_argument('--wz_process', type=bool, default=False,
                    help='calculate loss in RGB domain')
parser.add_argument('--ccm', type=np.array, 
                    default=np.array([[ 1.93994141, -0.73925781, -0.20068359],
                                      [-0.28857422,  1.59741211, -0.30883789],
                                      [-0.0078125 , -0.45654297,  1.46435547]]),
                    help='ccm')

# loss
parser.add_argument('--t_loss', type=str, default='L2',
                    help='training loss: L2, L1, L2_wz_TV, L2_wz_Perceptual, L2_wz_SSIM')
parser.add_argument('--tv_weight', type=float, default=4e-8,
                    help='tvloss weight')
parser.add_argument('--style_weight', type=float, default=1,
                    help='style weight of perceptual loss')
parser.add_argument('--content_weight', type=float, default=1,
                    help='content weight of perceptual loss')
parser.add_argument('--ssim_weight', type=float, default=2e-1,
                    help='ssim weight of ssim loss')                   


# model
parser.add_argument('--NetName', default='DFUNet',
                    help='model name')
parser.add_argument('--n_channel', type=int, default=32,
                    help='number of convolutional channels')
parser.add_argument('--offset_channel', type=int, default=32,
                    help='number of offset channels')
parser.add_argument('--fov_att', type=bool, default=False,
                    help='whether using FOV attention block')
parser.add_argument('--kernel_size', type=list, default=[5],
                    help='kernel size to be estimated')
parser.add_argument('--color', type=bool, default=True,
                    help='color image or gray image')

# test
# File paths
parser.add_argument('--gt_src_path', type=str, default="./test_datasets/label",
                    help='testing clear image path, if not, set None')
parser.add_argument('--blur_src_path', type=str, default="./test_datasets/input_20201228",
                    help='testing noisy image path')
parser.add_argument('--result_png_path', type=str, default="./test_result",
                    help='result directory')
parser.add_argument('--ckpt_dir_test', type=str, default="./ckpt_dir",
                    help='model directory')
parser.add_argument('--epoch_test', type=int, default=30,
                    help='the epoch for testing')

# test_real
# File paths
parser.add_argument('--real_blur_src_path', type=str, 
                    default="./camera04",
                    help='testing real abberation image path, if not, set None')
parser.add_argument('--real_dst_tiff_path', type=str, 
                    default="./real_test_result",
                    help='result directory')
parser.add_argument('--ckpt_dir_test_real', type=str, 
                    default="./ckpt_dir",
                    help='model directory')
parser.add_argument('--epoch_test_real', type=int, default=30,
                    help='the epoch for testing real image')

args = parser.parse_args()
