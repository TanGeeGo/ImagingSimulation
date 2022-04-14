import torch
import torch.nn.functional as F
from math import exp
from torch import nn
from torchvision import models
from torch.autograd import Variable
from utils import normalize_tensor_transform

class L2_wz_TV(nn.Module):
    def __init__(self, args):
        super(L2_wz_TV, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.TV_WEIGHT = args.tv_weight

    def forward(self, out_images, target_images):
        # MSELoss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + self.TV_WEIGHT * tv_loss

class L2_wz_Perceptual(nn.Module):
    def __init__(self, args):
        super(L2_wz_Perceptual, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.per_loss = PerceptualLoss()
        self.STYLE_WEIGHT = args.style_weight
        self.CONTENT_WEIGHT = args.content_weight

    def forward(self, out_images, target_images):
        # MSELoss
        image_loss = self.mse_loss(out_images, target_images)
        # Perceptual Loss
        out_images_norm, target_images_norm = normalize_tensor_transform(out_images, target_images)
        style_loss, content_loss = self.per_loss(out_images_norm, target_images_norm)
        # print(style_loss.data, content_loss.data)
        return image_loss + self.STYLE_WEIGHT * style_loss.data + self.CONTENT_WEIGHT * content_loss.data

class L2_wz_SSIM(nn.Module):
    def __init__(self, args):
        super(L2_wz_SSIM, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss()
        self.SSIM_WEIGHT = args.ssim_weight

    def forward(self, out_images, target_images):
        # MSELoss
        image_loss = self.mse_loss(out_images, target_images)
        # SSIMLoss
        ssim_loss = self.ssim_loss(out_images, target_images)
        return image_loss + self.SSIM_WEIGHT * ssim_loss

#----------------------------------------------------------------------------------

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()
        
        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])    

        self.mse_loss = nn.MSELoss()
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def _gram(self, x):
        (bs, ch, h, w) = x.size()
        f = x.view(bs, ch, w*h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (ch * h * w)
        return G

    def forward(self, pred_img, targ_img):
        h_relu_1_2_pred_img = self.to_relu_1_2(pred_img)
        h_relu_1_2_targ_img = self.to_relu_1_2(targ_img)
        style_loss_1_2 = self.mse_loss(self._gram(h_relu_1_2_pred_img), self._gram(h_relu_1_2_targ_img))
        
        h_relu_2_2_pred_img = self.to_relu_2_2(h_relu_1_2_pred_img)
        h_relu_2_2_targ_img = self.to_relu_2_2(h_relu_1_2_targ_img)
        style_loss_2_2 = self.mse_loss(self._gram(h_relu_2_2_pred_img), self._gram(h_relu_2_2_targ_img))
        
        h_relu_3_3_pred_img = self.to_relu_3_3(h_relu_2_2_pred_img)
        h_relu_3_3_targ_img = self.to_relu_3_3(h_relu_2_2_targ_img)
        style_loss_3_3 = self.mse_loss(self._gram(h_relu_3_3_pred_img), self._gram(h_relu_3_3_targ_img))

        h_relu_4_3_pred_img = self.to_relu_4_3(h_relu_3_3_pred_img)
        h_relu_4_3_targ_img = self.to_relu_4_3(h_relu_3_3_targ_img)
        style_loss_4_3 = self.mse_loss(self._gram(h_relu_4_3_pred_img), self._gram(h_relu_4_3_targ_img))
        
        style_loss_tol = style_loss_1_2 + style_loss_2_2 + style_loss_3_3 + style_loss_4_3
        # content loss (h_relu_2_2)
        content_loss_tol = style_loss_2_2
        return style_loss_tol, content_loss_tol

class SSIMLoss(nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average = True):
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, pred_img, targ_img):
        (_, channel, _, _) = pred_img.size()

        if channel == self.channel and self.window.data.type() == pred_img.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)

            if pred_img.is_cuda():
                window = window.cuda()
            window = window.type_as(pred_img)

            self.window = window
            self.channel = channel
        
        return self._ssim(pred_img, targ_img, window, self.window_size, channel, self.size_average)
