%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% variable declaration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dataset image path. the images in the object image path have the same
% resolution with the camera sensor
object_img_path = './dataset/object_plane_image/';
img_path_dir = dir(object_img_path);
img_file_num = length(img_path_dir) - 2;
image_img_path = './dataset/image_plane_image/';
% PSF cell saving path
PSF_cell_path = '../PSF_generation/PSF_cell/';
% white balance information path
wb_path = './white_balance_info.mat';
% color correction matrix path
ccm_path = './color_correction_info.mat';
% bayer pattern of sensor
bayer_pattern = 'BGGR';
% image size which equals to the sensor resolution
H = 3000; W = 4000;
% patch length between different PSFs
patch_length = 10;
% psf number in column and line
PSF_h_num = H / patch_length;
PSF_w_num = W / patch_length;
% PSF uniform size which is the largest size of PSF 
PSF_uniform_size = 10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% imaging simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for img_index = 1:img_file_num
    img_file_path = strcat(object_img_path, '\', img_path_dir(img_index+2).name);
    img = im2double(imread(img_file_path));
    % inverse gamma compression
    img_inv_gamma = imadjust(img, [0 1], [0 1], 2.2);
    % inverse color correction matrix
    img_inv_ccm = apply_ccm(img_inv_gamma, ccm, 1);
    % randomly choose a color temperature
    color_temperature_index = round(12 * rand);
    wb = wb(color_temperature_index, :);
    % inverse white balance 
    img_inv_wb = apply_wb(img_inv_ccm, wb, 1);
    % optical PSF convolution
    img_aberration = patch_conv(img_inv_wb, PSF_cell_path, PSF_h_num, ...
                                PSF_w_num, patch_length, PSF_uniform_size);
    % mosaicing
    img_mosaiced = mosaicing(img_aberration, bayer_pattern);
    % demosaicing
    img_demosaiced = demosaic(img_mosaiced, bayer_pattern);
    % white balance
    img_wz_wb = apply_wb(img_demosaiced, wb, 0);
    % color correction matrix
    img_wz_ccm = apply_ccm(img_wz_wb, ccm, 0);
    % gamma compression
    img_wz_gamma = imadjust(img, [0 1], [0 1], 1/2.2);
    % convert to uint8
    img_uint8 = im2uint8(img_wz_gamma);
    % saving 
    image_img_file_path = strcat(image_img_path, '\', img_path_dir(img_index+2).name);
    imwrite(img_uint8, image_img_file_path);
end