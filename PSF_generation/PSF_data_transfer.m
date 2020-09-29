%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% variable declaration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PSF data source path
PSF_data_folder = '.\PSF_data\';
% PSF information save path
PSF_info_folder = '.\PSF_info\';
% wave number to synthetic a three channel PSF, which is defined by  
% (wave distribution range)/(wave_interval)
wave_num = 340/10;
% sample interval of field in millimeters
fld_sample_interval = 0.02;
% the max field range and the min field range
fld_max_value = 4.00; fld_min_value = 0.00;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PSF data transfer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for fld_index = fld_min_value:fld_sample_interval:fld_max_value
    tic
    [status, msg, msgID] = mkdir(strcat(PSF_info_folder, fld_index_str));
    fld_index_str = num2str(round(fld_index * 100), '%03d');
    % convert the field information to string
    fld_index_str = strcat('PSF_info_fld_', fld_index_str);
    % field information excel path
    fld_sample_excel = strcat(PSF_data_folder, fld_index_str, '\PSF_info.xlsx');
    % run every wave information
    for wave_index = 1:wave_num
        % load the PSF information of a wavelength
        [wav_PSF, wav_txt, ~] = xlsread(fld_sample_excel, wave_index);
        PSF_wav = wav_txt{9};
        PSF_wav_tmp = split(PSF_wav, ' ');
        % check the wavelength and the field information
        PSF_fld_tmp = str2double(PSF_wav_tmp{5});
        PSF_wav_tmp_1 = str2double(PSF_wav_tmp{1});
        assert(round(PSF_fld_tmp*100) == round(fld_index*100),...
            'sampling field of %f is wrong, the field value in the Excel is %f', ...
            fld_index, PSF_fld_tmp);
        assert(round(PSF_wav_tmp_1*1000) == round((0.74-wave_index*0.01)*1000), ...
            'sampling wavelength is wrong, the field position is %02.2f,出错文件为 %s，采样波长为 %f，计算波长为 %f', ...
            fld_index, fld_index_str, PSF_wav_tmp_1, (0.74-wave_index*0.01));
        PSF_wav_tmp_2 = round(100 * str2double(PSF_wav_tmp{1}));
        % save the wave PSF and the wave txt information
        mat_path = strcat(PSF_info_folder, fld_index_str, '\wav_', num2str(PSF_wav_tmp_2, '%03d'), '.mat');
        save(mat_path, 'wav_PSF', 'wav_txt');
    end
    toc
    % print the mark information
    formatSpec = 'field of %02.2f is finished!\n';
    fprintf(formatSpec, fld_index);
end