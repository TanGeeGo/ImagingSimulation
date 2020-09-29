function img_out = apply_ccm(img, ccm, inverse_ccm)
% inverse the CCM of image
if inverse_ccm
    % inverse the color correction matrix
    ccm = inv(ccm);
end
img_r = ccm(1, 1) * img(:, :, 1) + ccm(1, 2) * img(:, :, 2) + ccm(1, 3) * img(:, :, 3);
img_g = ccm(2, 1) * img(:, :, 1) + ccm(2, 2) * img(:, :, 2) + ccm(2, 3) * img(:, :, 3);
img_b = ccm(3, 1) * img(:, :, 1) + ccm(3, 2) * img(:, :, 2) + ccm(3, 3) * img(:, :, 3);
    
img_out = cat(3, img_r, img_g, img_b);
end

