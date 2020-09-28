function fld_sample_delta_angle = compute_delta_angle(fld_sample_delta,rotat_index)
switch rotat_index
    case 2
        fld_sample_delta(2) = -fld_sample_delta(2);
    case 3
        fld_sample_delta(1) = -fld_sample_delta(1);
    case 4
        fld_sample_delta(1) = -fld_sample_delta(1);
        fld_sample_delta(2) = -fld_sample_delta(2);
end
% compute the rotate angle by delta h
if fld_sample_delta(1) >= 0
    % rotate angle in [-90, 90]
    angle_tangent = fld_sample_delta(2) / fld_sample_delta(1);
    fld_sample_delta_angle = atand(angle_tangent);
elseif fld_sample_delta(1) < 0
    % rotate angle in [-180, -90] and [90, 180]
    angle_tangent = fld_sample_delta(2) / - fld_sample_delta(1);
    if fld_sample_delta(2) >= 0
        % rotate angle in [90, 180]
        fld_sample_delta_angle = 180 - atand(angle_tangent);
    elseif fld_sample_delta(2) < 0
        % rotate angle in [-180, -90]
        fld_sample_delta_angle = -180 - atand(angle_tangent);
    end
end
end

