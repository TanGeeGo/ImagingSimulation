function img_mosaiced = mosaicing(img_aberration, bayer_pattern)
[H, W, ~] = size(img_aberration);
img_mosaiced = zeros(H, W);
bayer_pattern = upper(bayer_pattern);
if strcmp(bayer_pattern, 'BGGR')
    img_mosaiced(1:2:end, 1:2:end) = img_aberration(1:2:end, 1:2:end, 3);
    img_mosaiced(2:2:end, 1:2:end) = img_aberration(2:2:end, 1:2:end, 2);
    img_mosaiced(1:2:end, 2:2:end) = img_aberration(1:2:end, 2:2:end, 2);
    img_mosaiced(2:2:end, 2:2:end) = img_aberration(2:2:end, 2:2:end, 1);
elseif strcmp(bayer_pattern, 'GBRG')
    img_mosaiced(1:2:end, 1:2:end) = img_aberration(1:2:end, 1:2:end, 2);
    img_mosaiced(2:2:end, 1:2:end) = img_aberration(2:2:end, 1:2:end, 1);
    img_mosaiced(1:2:end, 2:2:end) = img_aberration(1:2:end, 2:2:end, 3);
    img_mosaiced(2:2:end, 2:2:end) = img_aberration(2:2:end, 2:2:end, 2);
elseif strcmp(bayer_pattern, 'GRBG')
    img_mosaiced(1:2:end, 1:2:end) = img_aberration(1:2:end, 1:2:end, 2);
    img_mosaiced(2:2:end, 1:2:end) = img_aberration(2:2:end, 1:2:end, 3);
    img_mosaiced(1:2:end, 2:2:end) = img_aberration(1:2:end, 2:2:end, 1);
    img_mosaiced(2:2:end, 2:2:end) = img_aberration(2:2:end, 2:2:end, 2);
elseif strcmp(bayer_pattern, 'RGGB')
    img_mosaiced(1:2:end, 1:2:end) = img_aberration(1:2:end, 1:2:end, 1);
    img_mosaiced(2:2:end, 1:2:end) = img_aberration(2:2:end, 1:2:end, 2);
    img_mosaiced(1:2:end, 2:2:end) = img_aberration(1:2:end, 2:2:end, 2);
    img_mosaiced(2:2:end, 2:2:end) = img_aberration(2:2:end, 2:2:end, 3);
else
    error('Unknown Bayer Pattern of %s!', bayer_pattern);
end
end

