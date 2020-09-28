function [fld_index,fld_index_int] = compute_field_info(fld_sample_dist,fld_sample_interval)
div = round(fld_sample_dist / fld_sample_interval);
% compute the nearest index
index = div * fld_sample_interval;
index_int = round(index * 100);
% transfer to string
index_int_str = num2str(index_int, '%03d');
% concate the string
fld_index = strcat('PSF_info_fld_', index_int_str);
fld_index_int = index;
end

