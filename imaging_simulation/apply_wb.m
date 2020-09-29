function img_out = apply_wb(img, wb, inverse_wb)
% inverse or add white balance of image
if inverse_wb
    wb = 1 ./ wb;
end
img_out = cat(3, img(:, :, 1) .* wb(1), img(:, :, 2) .* wb(2), img(:, :, 3) .* wb(3));
end

