function [wav_PSF_pad, ori_sample_pos] = pad_PSF(wav_PSF, ...
                                                 PSF_center_delta_pixel, ...
                                                 data_cell_length)
[PSF_h, PSF_w] = size(wav_PSF);
if (PSF_center_delta_pixel(2) < 0)&&(PSF_center_delta_pixel(1) <= 0)
    % pad right bottom
    pad_h = abs(floor(PSF_center_delta_pixel(1))) * data_cell_length;
    pad_w = abs(floor(PSF_center_delta_pixel(2))) * data_cell_length;
    wav_PSF_pad = zeros(PSF_h + pad_h, PSF_w + pad_w);
    wav_PSF_pad(1:PSF_h, 1:PSF_w) = wav_PSF;
    ori_sample_pos = round([-PSF_center_delta_pixel(1), ...
                            -PSF_center_delta_pixel(2)] * data_cell_length);
elseif (PSF_center_delta_pixel(2) >= 0)&&(PSF_center_delta_pixel(1) < 0)
    % pad left bottom
    pad_h = abs(floor(PSF_center_delta_pixel(1))) * data_cell_length;
    pad_w = abs(ceil(PSF_center_delta_pixel(2))) * data_cell_length;
    wav_PSF_pad = zeros(PSF_h + pad_h, PSF_w + pad_w);
    wav_PSF_pad(1:PSF_h, pad_w + 1:pad_w + PSF_w) = wav_PSF;
    ori_sample_pos = round([-PSF_center_delta_pixel(1)*data_cell_length, ...
                            pad_w - PSF_center_delta_pixel(2)*data_cell_length]);
elseif (PSF_center_delta_pixel(2) > 0)&&(PSF_center_delta_pixel(1) >= 0)
    % pad left top
    pad_h = abs(ceil(PSF_center_delta_pixel(1))) * data_cell_length;
    pad_w = abs(ceil(PSF_center_delta_pixel(2))) * data_cell_length;
    wav_PSF_pad = zeros(PSF_h + pad_h, PSF_w + pad_w);
    wav_PSF_pad(pad_h + 1:PSF_h + pad_h, pad_w + 1:pad_w + PSF_w) = wav_PSF;
    ori_sample_pos = round([pad_h - PSF_center_delta_pixel(1)*data_cell_length, ...
                            pad_w - PSF_center_delta_pixel(2)*data_cell_length]);
elseif (PSF_center_delta_pixel(2) <= 0)&&(PSF_center_delta_pixel(1) >= 0)
    % pad right top
    pad_h = abs(ceil(PSF_center_delta_pixel(1))) * data_cell_length;
    pad_w = abs(floor(PSF_center_delta_pixel(2))) * data_cell_length;
    wav_PSF_pad = zeros(PSF_h + pad_h, PSF_w + pad_w);
    wav_PSF_pad(pad_h + 1:PSF_h + pad_h, 1:PSF_w) = wav_PSF;
    ori_sample_pos = round([pad_h - PSF_center_delta_pixel(1)*data_cell_length, ...
                            -PSF_center_delta_pixel(2)*data_cell_length]);
end
end