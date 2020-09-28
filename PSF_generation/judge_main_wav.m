function [main_center_h, PSF_fld_tmp, PSF_wav_tmp] = judge_main_wav(fld_sample_pre, fld_index_int)
% define the wavelength of main light 
% notice!! this data is different between optical designs!!
% please re-calibrate the result for a new optical design!!
if (0.00 <= fld_index_int)&&(fld_index_int < 0.03)
    % wavelength of main light is 730nm
    fld_sample_mat = strcat(fld_sample_pre, '\wav_073.mat');
elseif (0.04 <= fld_index_int)&&(fld_index_int < 1.23)
    % wavelength of main light is 400nm
    fld_sample_mat = strcat(fld_sample_pre, '\wav_040.mat');
elseif (1.24 <= fld_index_int)&&(fld_index_int < 2.51)
    % wavelength of main light is 730nm
    fld_sample_mat = strcat(fld_sample_pre, '\wav_073.mat');
elseif (2.52 <= fld_index_int)&&(fld_index_int < 2.71)
    % wavelength of main light is 400nm
    fld_sample_mat = strcat(fld_sample_pre, '\wav_040.mat');
elseif (2.72 <= fld_index_int)&&(fld_index_int < 3.71)
    % wavelength of main light is 730nm
    fld_sample_mat = strcat(fld_sample_pre, '\wav_073.mat');
elseif (3.71 <= fld_index_int)&&(fld_index_int < 3.73)
    % wavelength of main light is 460nm
    fld_sample_mat = strcat(fld_sample_pre, '\wav_046.mat');
elseif (3.73 <= fld_index_int)&&(fld_index_int < 3.77)
    % wavelength of main light is 450nm
    fld_sample_mat = strcat(fld_sample_pre, '\wav_045.mat');
elseif (3.77 <= fld_index_int)&&(fld_index_int < 3.80)
    % wavelength of main light is 440nm
    fld_sample_mat = strcat(fld_sample_pre, '\wav_044.mat');
elseif (3.80 <= fld_index_int)&&(fld_index_int < 3.82)
    % wavelength of main light is 430nm
    fld_sample_mat = strcat(fld_sample_pre, '\wav_043.mat');
elseif (3.82 <= fld_index_int)&&(fld_index_int <= 4.00)
    % wavelength of main light is 400nm
    fld_sample_mat = strcat(fld_sample_pre, '\wav_040.mat');    
else 
    error('Invalid field value!');
end
% read out the simulation information
wav_txt = load(fld_sample_mat);
wav_txt = wav_txt.wav_txt;
PSF_wav = wav_txt{9};
PSF_center = wav_txt{16};
PSF_wav_tmp = split(PSF_wav, ' ');
PSF_center_tmp = split(PSF_center, ' ');
% the position, field, wavelength of main light
main_center_h = str2double(PSF_center_tmp{8});
PSF_fld_tmp = str2double(PSF_wav_tmp{5});
PSF_wav_tmp = str2double(PSF_wav_tmp{1});
end