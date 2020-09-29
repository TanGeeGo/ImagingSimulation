%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% variable declaration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% path of PSF_info
PSF_info_folder = '.\PSF_info\';
% path to save PSF_cell
PSF_cell_folder = '.\PSF_cell\';
% full field value of sensor
full_field = 4.00;
% wave number to synthetic a three channel PSF, which is defined by  
% (wave distribution range)/(wave_interval)
wave_num = 340/10;
% H, W resolutions of sensor
img_h = 3000; img_w = 4000;
% center of the image
img_h_cent = (img_h + 1) / 2; img_w_cent = (img_w + 1) / 2;
% pixel distance between two PSFs
tile_length = 10;
% pixel size in microns
pixel_length = 1.60;
% sample interval of field in millimeters
fld_sample_interval = 0.02;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sensor response integral path
wave_dist_path = '.\wav_response\wav_dist_cell_avr_itvl_380_10nm_780.mat';
wave_dist_cell = load(wave_dist_path);
wave_dist_cell = wave_dist_cell.wav_dist_cell;
% three experiment results of sensor response, load one
wave_dist_r = wave_dist_cell{1, 1}; wave_dist_g = wave_dist_cell{1, 2}; wave_dist_b = wave_dist_cell{1, 3};
IMA_response = cat(1, wave_dist_r, wave_dist_g, wave_dist_b);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% whether load the illumination information
load_illumination = false;
% relative illumination information
illumination_path = '.\illumination_info\illumination_info.xlsx';
Rlt_illumination = interp_relative_illumination(illumination_path, wave_num);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Point spread function calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for h_index = tile_length/2 : tile_length : img_h/2
    for w_index = tile_length/2 : tile_length : img_w/2
        
        % print first mark information
        formatSpec_first = strcat('field of (h:%d, w:%d); (h:%d, w:%d); ' , ...
                                  '(h:%d, w:%d); (h:%d, w:%d) is processing!\n');
        fprintf(formatSpec_first, h_index, w_index, ...
                                  h_index, img_w - w_index, ...
                                  img_h - h_index, w_index, ...
                                  img_h - h_index, img_w - w_index);
        
        % calculate the distance between sample position to the image center
        % the field_sample_dist is in millimeters 
        fld_sample_delta_h = img_h_cent - h_index;
        fld_sample_delta_w = img_w_cent - w_index;
        fld_sample_delta = [fld_sample_delta_h, fld_sample_delta_w];
        fld_sample_dist = sqrt((fld_sample_delta_h)^2 + ...
                               (fld_sample_delta_w)^2) * ...
                               pixel_length * 0.001;
        % calculate the field position
        [fld_index, fld_index_int] = compute_field_info(fld_sample_dist, fld_sample_interval);
        fld_sample_prepath = strcat(PSF_info_folder, fld_index);
        % judge the main light position in the imaging plane, set the
        % wavelength that closest to the image center to the main light 
        [main_center_h, PSF_fld_tmp, PSF_wav_tmp] = judge_main_wav(fld_sample_pre, fld_index_int);
        
        % print the second mark information
        formatSpec_scd = strcat('main light wav is %3.1f nm;', ...
                                ' field value is %3.3f; ', ...
                                'center of main light is %06.8f\n');
        fprintf(formatSpec_scd, PSF_wav_tmp*1000, PSF_fld_tmp, main_center_h);
        
        % initialize the cell to store Point Spread Function
        % the first column is the top-left, the second column is the
        % top-right, the third column is the left-bottom, the fourth column
        % is the right-bottom
        PSF_cell = cell(wave_num, 4);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for wave_index = 1:wave_num
            % load the PSF information, the wavelength varies from 400nm to 700nm
            wave_mat_path = strcat(fld_sample_prepath, '\wav_', ...
                                   num2str(39 + wave_index, '%03d'), '.mat');
            wave_mat = load(wave_mat_path);
            wave_PSF = wave_mat.wav_PSF;
            wave_txt = wave_mat.wav_txt;
            PSF_area = wave_txt{11};
            PSF_area_tmp = split(PSF_area, ' ');
            PSF_space = wave_txt{10};
            PSF_space_tmp = split(PSF_space, ' ');
            PSF_center = wave_txt{16};
            PSF_center_tmp = split(PSF_center, ' ');
            % PSF data interval, in microns
            PSF_data_space = str2double(PSF_space_tmp{4});
            % n data in PSF cover the one pixel
            data_cell_length = round(pixel_length / PSF_data_space);
            % the distance of PSF to the center of image
            PSF_center_h = str2double(PSF_center_tmp{8});
            % compute the deviation between this wavelength and the main
            % light wavelength
            PSF_center_delta_mm = PSF_center_h - main_center_h;
            % swift the PSF, to generate a central symmetric PSF
            [PSF_H, PSF_W] = size(wave_PSF);
            wave_PSF = cat(1, wave_PSF(PSF_H, :), wave_PSF);
            wave_PSF = cat(2, wave_PSF, wave_PSF(:, 1));
            % rotate the PSF according to the field position
            % calculate four PSFs of different angle, 1 -> top-left,
            % 2 -> top-right, 3 -> left-bottom, 4 -> right-bottom
            for rotat_index = 1:4
                % calculate the rotate angle of PSF
                fld_sample_delta_angle = compute_delta_angle(fld_sample_delta, ...
                                                             rotat_index);
                wave_PSF_rotat = imrotate(wave_PSF, fld_sample_delta_angle, ...
                                         'bilinear','crop');
                % compute the deviation after rotating, right-bot is
                % positive, left-top is negative
                PSF_center_delta_pixel = compute_h_w_delta(PSF_center_delta_mm, ...
                                                           fld_sample_delta_angle, ...
                                                           pixel_length);
                [PSF_H, PSF_W] = size(wave_PSF_rotat);
                % padding int pixel number
                pixel_num = ceil(PSF_H / data_cell_length);
                if mod(pixel_num, 2) == 0
                    pixel_num = pixel_num + 1;
                end
                pad_pixel_num = (pixel_num * data_cell_length - PSF_H) / 2;
                % first pad the array
                wave_PSF_pad_fir = padarray(wave_PSF_rotat, [pad_pixel_num pad_pixel_num], 0, 'both');
                % according to the delta_h and delta_w, judge the position.
                % pay attention to the different pad direction has
                % different start point!!
                [wave_PSF_pad_scd, ori_sample_pos] = pad_PSF(wave_PSF_pad_fir, ...
                                                             PSF_center_delta_pixel, ...
                                                             data_cell_length);
                % initialize the PSF
                pixel_PSF = zeros(pixel_num, pixel_num);
                for h_pixel_index = 1:pixel_num
                    for w_pixel_index = 1:pixel_num
                        % the h_range and w_range in wave_PSF
                        h_wave_range = ori_sample_pos(1) + (h_pixel_index-1)*data_cell_length+1 : ...
                            ori_sample_pos(1) + (h_pixel_index)*data_cell_length;
                        w_wave_range = ori_sample_pos(2) + (w_pixel_index-1)*data_cell_length+1 : ...
                            ori_sample_pos(2) + (w_pixel_index)*data_cell_length;
                        pixel_cell_PSF = wave_PSF_pad_scd(h_wave_range, w_wave_range);
                        % accumulate
                        pixel_PSF(h_pixel_index, w_pixel_index) = sum(pixel_cell_PSF, 'all');
                    end
                end
                PSF_cell{wave_index, rotat_index} = pixel_PSF ./ sum(pixel_PSF, 'all');
            end
        end
        % print the third mark information
        formatSpec_mid_one = strcat('field of (h:%d, w:%d); (h:%d, w:%d); ' , ...
                                    '(h:%d, w:%d); (h:%d, w:%d) is reunioned!\n');
        fprintf(formatSpec_mid_one, h_index, w_index, ...
                                    h_index, img_w - w_index, ...
                                    img_h - h_index, w_index, ...
                                    img_h - h_index, img_w - w_index);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % load the sensor response
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % initialize the sensor response PSF cell
        PSF_rsp_cell = cell(1, 4);
        for rotat_index = 1:4
            PSF_cell_tmp = PSF_cell(:, rotat_index);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % pad the PSF_cell and make every PSF of different wavelength
            % to the same size
            % load illumination response
            if load_illumination
                [PSF_rch, PSF_gch, PSF_bch] = ...
                    load_wave_response_illuminate(PSF_cell_tmp, IMA_response, ...
                                                  Rlt_illumination, fld_index_int);
            else
                [PSF_rch, PSF_gch, PSF_bch] = ...
                    load_wave_response(PSF_cell_tmp, IMA_response);
            end
            PSF_wave_response = cat(3, PSF_rch, PSF_gch, PSF_bch);
            PSF_rsp_cell{rotat_index} = PSF_wave_response;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % save the four direction PSFs
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        PSF_info = PSF_rsp_cell{1};
        save(strcat(PSF_cell_folder, 'PSF_cell_', ...
             num2str((h_index-5)/10+1, '%03d'), '_', ...
             num2str((w_index-5)/10+1, '%03d'), '.mat'), 'PSF_info');
        PSF_info = PSF_rsp_cell{2};
        save(strcat(PSF_cell_folder, 'PSF_cell_', ...
             num2str((h_index-5)/10+1, '%03d'), '_', ...
             num2str((img_w-w_index-5)/10+1, '%03d'), '.mat'), 'PSF_info');
        PSF_info = PSF_rsp_cell{3};
        save(strcat(PSF_cell_folder, 'PSF_cell_', ...
             num2str((img_h-h_index-5)/10+1, '%03d'), '_', ...
             num2str((w_index-5)/10+1, '%03d'), '.mat'), 'PSF_info');
        PSF_info = PSF_rsp_cell{4};
        save(strcat(PSF_cell_folder, 'PSF_cell_', ...
             num2str((img_h-h_index-5)/10+1, '%03d'), '_', ...
             num2str((img_w-w_index-5)/10+1, '%03d'), '.mat'), 'PSF_info');
        % print the final mark information
        formatSpec_final = strcat('response and illumination of (h:%d, w:%d); (h:%d, w:%d); ' , ...
                                    '(h:%d, w:%d); (h:%d, w:%d) is saved!\n');
        fprintf(formatSpec_final, h_index, w_index, ...
                                  h_index, img_w - w_index, ...
                                  img_h - h_index, w_index, ...
                                  img_h - h_index, img_w - w_index);
    end
end
