import os
import cv2
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable

from option.option_20220330 import args
from loss import *
from utils import *
from deformable_unet import DFUNet
from dataloader import Dataset_from_h5

def lr_adjust(optimizer, epoch, init_lr=1e-4, step_size=20, gamma=0.5):
    lr = init_lr * gamma ** (epoch // step_size)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train():

    # Load dataset
    dataset = Dataset_from_h5(src_path=args.src_path, recrop_patch_size=args.patch_size, sigma=args.sigma, train=True)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    dataset_val = Dataset_from_h5(src_path=args.val_path, recrop_patch_size=args.val_patch_size, sigma=args.sigma, train=False)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=args.val_batch_size, shuffle=False, num_workers=8, drop_last=True)
    print('Training path of {:s};\nValidation path of {:s};'.format(args.src_path, args.val_path))
    # Build model
    input_channel, output_channel = 5, 3
    model = DFUNet(input_channel, output_channel, args.n_channel, args.offset_channel)
    model.initialize_weights()

    if args.finetune:
        model_dict = torch.load(args.ckpt_dir+'model_%04d_dict.pth' % args.init_epoch)
        model.load_state_dict(model_dict)

    if args.t_loss == 'L2':
        criterion = torch.nn.MSELoss()
        print('Training with L2Loss!')
    elif args.t_loss == 'L1':
        criterion = torch.nn.L1Loss()
        print('Training with L1Loss!')
    elif args.t_loss == 'L2_wz_TV':
        criterion = L2_wz_TV(args)
        print('Training with L2 and TV Loss!')
    elif args.t_loss == 'L2_wz_Perceptual':
        criterion = L2_wz_Perceptual(args)
        print('Training with L2 and Perceptual Loss!')

    if torch.cuda.is_available():
        print('Use {} GPU, which order is {:s}th'.format(torch.cuda.device_count(), args.gpu))
        if torch.cuda.device_count() > 1:
            #model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

        criterion = criterion.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # writer = SummaryWriter(args.log_dir)
    ccm = torch.from_numpy(np.ascontiguousarray(args.ccm)).float().cuda()

    for epoch in range(args.init_epoch, args.n_epoch):
        loss_sum = 0
        step_lr_adjust(optimizer, epoch, init_lr=args.lr, step_size=args.milestone, gamma=args.gamma)
        print('Epoch {}, lr {}'.format(epoch+1, optimizer.param_groups[0]['lr']))
        start_time = time.time()
        for i, data in enumerate(dataloader):
            input, label = data
            if torch.cuda.is_available():
                input, label = input.cuda(), label.cuda()
            input, label = Variable(input), Variable(label)

            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            output = model(input)

            # whether with post processing
            if args.wz_process:
                print('Calculate Loss with post processing')
                label = process(label, ccm)
                output = process(output, ccm)

            # calculate loss
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

            if (i % 100 == 0) and (i != 0) :
                loss_avg = loss_sum / 100
                loss_sum = 0.0
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.8f} Time: {:4.4f}s".format(
                    epoch + 1, args.n_epoch, i + 1, len(dataloader), loss_avg, time.time()-start_time))
                start_time = time.time()
                # Record train loss
                # writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
                # # Record learning rate
                # writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
        # save model
        if epoch % args.save_epoch == 0:
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), os.path.join(args.ckpt_dir, 'model_%04d_dict.pth' % (epoch+1)))
            else:
                torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'model_%04d_dict.pth' % (epoch+1)))

        # validation
        if epoch % args.val_epoch == 0:
            psnr = 0
            loss_val = 0
            model.eval()
            for i, data in enumerate(dataloader_val):
                input, label = data
                if torch.cuda.is_available():
                    input, label = input.cuda(), label.cuda()
                input, label = Variable(input), Variable(label)

                test_out = model(input)
                test_out.detach_()

                # compute loss
                loss_val += criterion(test_out, label).item()
                rgb_out = test_out.cpu().numpy().transpose((0,2,3,1))
                clean = label.cpu().numpy().transpose((0,2,3,1))
                for num in range(rgb_out.shape[0]):
                    deblurred = np.clip(rgb_out[num], 0, 1)
                    psnr += compare_psnr(clean[num], deblurred)
            img_nums = rgb_out.shape[0] * len(dataloader_val)
            psnr = psnr / img_nums
            loss_val = loss_val / len(dataloader_val)
            print('Validating: {:0>3} , loss: {:.8f}, PSNR: {:4.4f}'.format(img_nums, loss_val, psnr))
            # writer.add_scalars('Loss_group', {'valid_loss': loss_val}, epoch)
            # writer.add_scalar('valid_psnr', psnr, epoch)
            if args.save_val_img:
                cv2.imwrite(args.ckpt_dir+"img/%04d_deblurred.png" % epoch, deblurred[..., ::-1])

    # writer.close()

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    create_dir(args.log_dir)
    create_dir(args.ckpt_dir)
    if args.save_val_img:
        create_dir(args.ckpt_dir+'_img/')
    train()
