function relative_illumination_cell = interp_relative_illumination(illumination_path, wave_num)
% initialize the relative illumination cell
relative_illumination_cell = cell(wave_num, 1);
for relative_wave_index = 1:wave_num
    % read out the illumination data
    [relative_illumination, ~, ~] = xlsread(illumination_path, relative_wave_index);
    % interpt the relative illumination data
    relative_illumination_interp = griddedInterpolant(relative_illumination(:, 1), ...
                                                      relative_illumination(:, 2), 'pchip');
    % save the handle of interpt 
    relative_illumination_cell{relative_wave_index} = relative_illumination_interp;
end
end

