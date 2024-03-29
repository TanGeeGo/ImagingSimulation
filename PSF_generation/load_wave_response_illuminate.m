function [PSF_rch, PSF_gch, PSF_bch] = ...
            load_wave_response_illuminate(PSF_cell, IMA_response, ...
                                          Rlt_illumination, ...
                                          fld_index_int)
% the wave sequence of PSF_cell is 400nm-700nm��the wave sequence of 
% response is 380-780nm
% initialize
PSF_rch = zeros(size(PSF_cell{1}));
PSF_gch = zeros(size(PSF_cell{1}));
PSF_bch = zeros(size(PSF_cell{1}));
IMA_respon_rch = IMA_response(1, :);
IMA_respon_gch = IMA_response(2, :);
IMA_respon_bch = IMA_response(3, :);
wave_number = length(PSF_cell);
for wave_index = 1:wave_number
    PSF_rch = PSF_rch + IMA_respon_rch(wave_index + 2) * ...
              Rlt_illumination{wave_index}(fld_index_int) * ...
              PSF_cell{wave_index};
    PSF_gch = PSF_gch + IMA_respon_gch(wave_index + 2) * ...
              Rlt_illumination{wave_index}(fld_index_int) * ...
              PSF_cell{wave_index};
    PSF_bch = PSF_bch + IMA_respon_bch(wave_index + 2) * ...
              Rlt_illumination{wave_index}(fld_index_int) * ...
              PSF_cell{wave_index};
end
end