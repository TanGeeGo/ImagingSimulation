function PSF_center_delta_pixel = compute_h_w_delta(PSF_center_delta_mm,fld_sample_delta_angle,pixel_length)
if (-180 < fld_sample_delta_angle)&&(fld_sample_delta_angle <= -90) 
    PSF_center_delta_mm_h = PSF_center_delta_mm * ...
                            cosd(fld_sample_delta_angle + 180);
    PSF_center_delta_mm_w = PSF_center_delta_mm * ...
                            sind(fld_sample_delta_angle + 180);
elseif (-90 < fld_sample_delta_angle)&&(fld_sample_delta_angle <= 0)
    PSF_center_delta_mm_h = -PSF_center_delta_mm * ...
                            sind(fld_sample_delta_angle + 90);
    PSF_center_delta_mm_w = PSF_center_delta_mm * ...
                            cosd(fld_sample_delta_angle + 90);
elseif (0 < fld_sample_delta_angle)&&(fld_sample_delta_angle <= 90)
    PSF_center_delta_mm_h = -PSF_center_delta_mm * ...
                            sind(90 - fld_sample_delta_angle);
    PSF_center_delta_mm_w = -PSF_center_delta_mm * ...
                            cosd(90 - fld_sample_delta_angle);
elseif (90 < fld_sample_delta_angle)&&(fld_sample_delta_angle <= 180)
    PSF_center_delta_mm_h = PSF_center_delta_mm * ...
                            cosd(180 - fld_sample_delta_angle);
    PSF_center_delta_mm_w = -PSF_center_delta_mm * ...
                            sind(180 - fld_sample_delta_angle);
end
PSF_center_delta_pixel_h = (PSF_center_delta_mm_h * 1000) / pixel_length;
PSF_center_delta_pixel_w = (PSF_center_delta_mm_w * 1000) / pixel_length;
PSF_center_delta_pixel = [PSF_center_delta_pixel_h, PSF_center_delta_pixel_w];
end

