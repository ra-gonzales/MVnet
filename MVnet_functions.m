function varargout = MVnet_functions(varargin)
    if (nargout)
        [varargout{1:nargout}] = feval(varargin{:});
    else
        feval(varargin{:});
    end
end

%% Load dicom data
function [IM, ResolutionXY, TimeVector] = load_dicom_data(data_sample_path)
    % List dicom files
    dicom_dir = dir(fullfile(data_sample_path, '*.dcm'));
    % Load information of first dicom file
    info = dicominfo(fullfile(dicom_dir(1).folder, dicom_dir(1).name));
    % Save ResolutionXY
    ResolutionXY = info.PixelSpacing;
    % Initialize IM and TimeVector
    IM = single(zeros(info.Rows, info.Columns, info.CardiacNumberOfImages));
    TimeVector = zeros(1, info.CardiacNumberOfImages);
    % Save trigger times for sorting the cine
    for i=1:info.CardiacNumberOfImages
        TimeVector(1,i) = dicominfo(fullfile(dicom_dir(i).folder, dicom_dir(i).name)).TriggerTime;
    end
    % Sort the trigger times and save TimeVector
    [TimeVector, index] = sort(TimeVector / 1000);
    % Save IM with sorted trigger times
    for i=1:info.CardiacNumberOfImages
        IM(:,:,i) = single(dicomread(fullfile(dicom_dir(index(i)).folder, dicom_dir(index(i)).name)));
    end
end

%% Pipeline function
function [AV_out] = pipeline(IM_in,Rxy_in,net_1st,net_2nd,GPU)
    IM_in = IM_in / max(IM_in(:));
    % 3. Resize input image for 1st stage
    % ----------------------
    nn_row_1 = 160;
    nn_col_1 = 160;
    % ----------------------
    [IM_1,~] = resize_dims(IM_in,[],size(IM_in),[nn_row_1 nn_col_1]);
    
    % 4. Prepare data for network - 1st stage
    x_1 = prepare_data_network(IM_1(:,:,:));
    
    % 5. Predict points in 1st stage
    if GPU
        y_1 = net_1st.predict(x_1,'ExecutionEnvironment','gpu');
    else
        y_1 = net_1st.predict(x_1);
    end
    
    % 6. Resize back predicted AV points to (out-2)
    [~,AV_1] = resize_dims([],y_1,size(IM_1),size(IM_in));
    
    % 7. Center valve from 1st frame, same resolution
    [IM_centered,AV_centered,info_centered] = center_valve(IM_in,AV_1);
    
    % 8. Resize IM_centered to a fixed resolution, including AV_center
    % ----------------------
    Rxy_fixed = [1.5 1.5];
    % ----------------------
    [IM_fixed,AV_fixed,Rxy_tmp] = fix_resolution(IM_centered,AV_centered,size(IM_centered),Rxy_in,Rxy_fixed);
    
    % 9. Rotate IM_fixed, including AV_center
    [IM_rotated,AV_rotated,info_rotated] = rotate_heart(IM_fixed,AV_fixed);
    
    % 10. Flip heart (if needed)
    [IM_flipped,AV_flipped,info_flipped] = flip_heart(IM_rotated,AV_rotated);
    
    % 11. Crop IM_rotated, including AV_center
    % ----------------------
    row_half = 29;%70
    col_half = 40;%40;
    % ----------------------
    [IM_cropped,AV_cropped,info_cropped] = crop_heart(IM_flipped,AV_flipped,row_half,col_half);
    
    % 12. Resize IM_cropped for 2nd stage (it is only the double)
    nn_row_2 = (row_half*2+1)*2;
    nn_col_2 = (col_half*2+1)*2;
    [IM_2,~] = resize_dims(IM_cropped,AV_cropped,size(IM_cropped),[nn_row_2 nn_col_2]);
    
    % 13. Prepare data for network - 2nd stage
    x_2 = prepare_data_network(IM_2);
    
    % 14. Predict points in 2nd stage
    if GPU
        y_2 = net_2nd.predict(x_2,'ExecutionEnvironment','gpu');
    else
        y_2 = net_2nd.predict(x_2);
    end
    
    % 15. Resize back predicted AV points to (out-11)
    [~,AV_back_1] = resize_dims([],y_2,size(IM_2),size(IM_cropped));
    
    % 16. Uncropped dimensions of AV points, to (out-10)
    AV_back_2 = AV_back_1 - repmat(info_cropped,1,size(AV_back_1,2)/2);
    
    % 17. Unflip AV points to original orientation (out-9)
    AV_back_3 = unflip_heart(AV_back_2,info_flipped);
    
    % 17. Unrotate back AV points to (out-8)
    AV_back_4 = unrotate_heart(AV_back_3,info_rotated);
    
    % 18. Unsize AV points to unfixed (out-7)
    [~,AV_back_5] = fix_resolution([],AV_back_4,size(IM_fixed),Rxy_tmp,Rxy_in);
    
    % 19. Uncenter AV points to initial (out-6,2)
    AV_out = AV_back_5 - repmat(info_centered,1,size(AV_back_5,2)/2);
    
    % ----------------------------------------
    % Doing the first stage twice
    % ----------------------------------------
    
    AV_out_tmp = AV_out; clear AV_out;
    
    % 7. Center valve from 1st frame, same resolution
    [IM_centered,AV_centered,info_centered] = center_valve(IM_in,AV_out_tmp);
    
    % 8. Resize IM_centered to a fixed resolution, including AV_center
    % ----------------------
    Rxy_fixed = [1.5 1.5];
    % ----------------------
    [IM_fixed,AV_fixed,Rxy_tmp] = fix_resolution(IM_centered,AV_centered,size(IM_centered),Rxy_in,Rxy_fixed);
    
    % 9. Rotate IM_fixed, including AV_center
    [IM_rotated,AV_rotated,info_rotated] = rotate_heart(IM_fixed,AV_fixed);
    
    % 10. Flip heart (if needed)
    [IM_flipped,AV_flipped,info_flipped] = flip_heart(IM_rotated,AV_rotated);
    
    % 11. Crop IM_rotated, including AV_center
    % ----------------------
    row_half = 29;%70
    col_half = 40;%40;
    % ----------------------
    [IM_cropped,AV_cropped,info_cropped] = crop_heart(IM_flipped,AV_flipped,row_half,col_half);
    
    % 12. Resize IM_cropped for 2nd stage (it is only the double)
    nn_row_2 = (row_half*2+1)*2;
    nn_col_2 = (col_half*2+1)*2;
    [IM_2,~] = resize_dims(IM_cropped,AV_cropped,size(IM_cropped),[nn_row_2 nn_col_2]);
    
    % 13. Prepare data for network - 2nd stage
    x_2 = prepare_data_network(IM_2);
    
    % 14. Predict points in 2nd stage
    if GPU
        y_2 = net_2nd.predict(x_2,'ExecutionEnvironment','gpu');
    else
        y_2 = net_2nd.predict(x_2);
    end
    
    % 15. Resize back predicted AV points to (out-11)
    [~,AV_back_1] = resize_dims([],y_2,size(IM_2),size(IM_cropped));
    
    % 16. Uncropped dimensions of AV points, to (out-10)
    AV_back_2 = AV_back_1 - repmat(info_cropped,1,size(AV_back_1,2)/2);
    
    % 17. Unflip AV points to original orientation (out-9)
    AV_back_3 = unflip_heart(AV_back_2,info_flipped);
    
    % 17. Unrotate back AV points to (out-8)
    AV_back_4 = unrotate_heart(AV_back_3,info_rotated);
    
    % 18. Unsize AV points to unfixed (out-7)
    [~,AV_back_5] = fix_resolution([],AV_back_4,size(IM_fixed),Rxy_tmp,Rxy_in);
    
    % 19. Uncenter AV points to initial (out-6,2)
    AV_out = AV_back_5 - repmat(info_centered,1,size(AV_back_5,2)/2);
    
end

%% Derive clinical metrics
function [MVdist, MVvel, MAPSE, MVs, MVe, MVa] = get_clinical(MV_2ch, MV_4ch, Rxy_2ch, TimeVector_2ch, Rxy_4ch, TimeVector_4ch)
    % Process for 2ch
    % Pixel to mm
    MV_2ch = MV_2ch.*[Rxy_2ch(2) Rxy_2ch(1) Rxy_2ch(2) Rxy_2ch(1)];
    %-- Set a matrix with e' points [X Y] of the frame
    % Get number of frames
    fr_2ch = size(MV_2ch,1);
    % Distribute points
    pe_1 = [MV_2ch(1,1) MV_2ch(1,2)];
    pe_2 = [MV_2ch(1,3) MV_2ch(1,4)];
    %-- Get slope and intercept of initial mitral valve plane
    m_i = (pe_2(2)-pe_1(2))/(pe_2(1)-pe_1(1));
    b_i = pe_1(2)-m_i*pe_1(1);
    %-- Get distance moved for each reamining phase
    dist_lat = zeros(fr_2ch,1); dist_sep = zeros(fr_2ch,1);
    for k=2:fr_2ch
        p_lat = MV_2ch(k,1:2); p_sep = MV_2ch(k,3:4);
        if abs(m_i) == Inf
            dist_lat(k,1) = pe_1(1)-p_lat(1);
            dist_sep(k,1) = pe_2(1)-p_sep(1);
        else
            dist_lat(k,1) = (m_i*p_lat(1)-p_lat(2)+b_i)/sqrt(1+m_i^2);
            dist_sep(k,1) = (m_i*p_sep(1)-p_sep(2)+b_i)/sqrt(1+m_i^2);
        end
    end
    %-- Adjust curve direction
    if mean(dist_lat) > 0
        dist_lat = -dist_lat;
    end
    if mean(dist_sep) > 0
        dist_sep = -dist_sep;
    end
    %-- Get global MVdist
    MVdist_2ch = (dist_lat+dist_sep)/2;
    % Process for 4ch
    % Pixel to mm
    MV_4ch = MV_4ch.*[Rxy_4ch(2) Rxy_4ch(1) Rxy_4ch(2) Rxy_4ch(1)];
    %-- Set a matrix with e' points [X Y] of the frame
    % Get number of frames
    fr_4ch = size(MV_4ch,1);
    % Distribute points
    pe_1 = [MV_4ch(1,1) MV_4ch(1,2)];
    pe_2 = [MV_4ch(1,3) MV_4ch(1,4)];
    %-- Get slope and intercept of initial mitral valve plane
    m_i = (pe_2(2)-pe_1(2))/(pe_2(1)-pe_1(1));
    b_i = pe_1(2)-m_i*pe_1(1);
    %-- Get distance moved for each reamining phase
    dist_lat = zeros(fr_4ch,1); dist_sep = zeros(fr_4ch,1);
    for k=2:fr_4ch
        p_lat = MV_4ch(k,1:2); p_sep = MV_4ch(k,3:4);
        if abs(m_i) == Inf
            dist_lat(k,1) = pe_1(1)-p_lat(1);
            dist_sep(k,1) = pe_2(1)-p_sep(1);
        else
            dist_lat(k,1) = (m_i*p_lat(1)-p_lat(2)+b_i)/sqrt(1+m_i^2);
            dist_sep(k,1) = (m_i*p_sep(1)-p_sep(2)+b_i)/sqrt(1+m_i^2);
        end
    end
    %-- Adjust curve direction
    if mean(dist_lat) > 0
        dist_lat = -dist_lat;
    end
    if mean(dist_sep) > 0
        dist_sep = -dist_sep;
    end
    %-- Get global AVPD
    MVdist_4ch = (dist_lat+dist_sep)/2;
    endT_2ch = TimeVector_2ch(end);
    endT_4ch = TimeVector_4ch(end);
    % Interpolate to have same dimensions
    if fr_2ch < fr_4ch
        fr_max = fr_4ch;
        MVdist_2ch = interp1(linspace(0,endT_2ch,fr_2ch)',MVdist_2ch,linspace(0,endT_2ch,fr_max)','spline');
    elseif fr_4ch <= fr_2ch
        fr_max = fr_2ch;
        MVdist_4ch = interp1(linspace(0,endT_4ch,fr_4ch)',MVdist_4ch,linspace(0,endT_4ch,fr_max)','spline');
    end
    % Average of AVPD
    MVdist = (MVdist_2ch+MVdist_4ch)/2;
    endT = (endT_2ch+endT_4ch)/2;
    %-- Smooth curve
    x = linspace(-endT,endT*2,fr_max*3)';
    ft = fittype('smoothingspline');
    % Smooth process for average
    [fitresult, ~] = fit(x,[MVdist; MVdist; MVdist],ft);
    MVvel = differentiate(fitresult,x)/10;
    MVvel = MVvel(fr_max+1:fr_max*2);
    MVdist = fitresult(x);
    MVdist = MVdist(fr_max+1:fr_max*2);
    % Get PD (value, loc)
    MAPSE = zeros(1,2);
    [MAPSE(1,1),MAPSE(1,2)] = min(MVdist);
    MAPSE(1,1) = abs(MAPSE(1,1));
    % Get velocities
    [MVs, MVe, MVa] = findpeakvelocities(MVdist, MVvel);
end

function [sprime,eprime,aprime] = findpeakvelocities(dist,vel)
    [~,EST] = min(dist);
    s_curve = [vel(1:EST-1); vel(EST:end)*0];
    d_curve = [vel(1:EST-1)*0; vel(EST:end); 0];
    
    [value, loc] = min(s_curve);
    sprime = [value, loc];

    [pks, loc] = findpeaks(d_curve,'SortStr','descend');
    if isempty(loc)
        eprime = [0 0];
        aprime = [0 0];
    elseif length(loc) == 1
        eprime = [pks(1) loc(1)];
        aprime = [0 length(vel)];
    else
        pks = pks(1:2);
        loc = loc(1:2);
        [~,index] = min(loc);
        eprime = [pks(index) loc(index)];
        [~,index] = max(loc);
        aprime = [pks(index) loc(index)];
    end
    sprime = sprime(1); eprime = eprime(1); aprime = aprime(1);

end

%% [3] Resize IM with desired dimensions
function [IM_resized,AV_resized] = resize_dims(IM_in,AV_in,size_in,size_out)
    % Get IM_in dimensions
    row_in = size_in(1); col_in = size_in(2); fr = size_in(3);
    % Get IM_out dimensions
    row_out = size_out(1); col_out = size_out(2);
    % Initialize IM_resized and AV_resized
    IM_resized = zeros(row_out,col_out,fr);
    AV_resized = zeros(size(AV_in,1),size(AV_in,2));
    % Resize every frame
    if ~isempty(IM_in)
        IM_resized = imresize3(IM_in,[row_out col_out fr]);
    end
    if ~isempty(AV_in)
        AV_resized = AV_in .* repmat([row_out/row_in col_out/col_in],1,size(AV_in,2)/2);
    end
end

%% [4] Prepare data for network
function x_out = prepare_data_network(x_in)
    % Get x_in dimensions
    [row_in,col_in,fr] = size(x_in);
    % Initialize output
    x_out = zeros(row_in,col_in,1,fr);
    % Store data in output
    for i=1:fr
        tmp = x_in(:,:,i);
        x_out(:,:,1,i) = (tmp-median(tmp(:)))/iqr(tmp(:));
    end
end

%% [7] Function for centering valve
function [IM_centered,AV_centered,info_centered] = center_valve(IM_in,AV_in)
    % Get size of IM_in
    [row_in,col_in,fr] = size(IM_in);
    % Get center of the valve in integer coordinates
    % [From first AV point to last AV point]
    center_valve_xy = [round((AV_in(1,1)+AV_in(1,end-1))/2) round((AV_in(1,2)+AV_in(1,end))/2)];
    % Get dx and dy
    dx = col_in-2*center_valve_xy(2); a_dx = abs(dx);
    dy = row_in-2*center_valve_xy(1); a_dy = abs(dy);
    % Initiate expanded IM with zeros
    IM_centered = zeros(row_in+a_dy,col_in+a_dx,fr);
    if dx < 0
        dx = 0;
    end
    if dy < 0
        dy = 0;
    end
    IM_centered(1+dy:dy+row_in,1+dx:dx+col_in,:) = IM_in;
    % Output info_centered
    info_centered = [dy dx];
    % Change in x and y points
    AV_centered = AV_in + repmat(info_centered,1,size(AV_in,2)/2);
end

%% [8] Function for resizing IM and AV at fixed resolution
function [IM_fixed,AV_fixed,Rxy_tmp] = fix_resolution(IM_in,AV_in,size_in,Rxy_in,Rxy_out)
    % Get IM_in dimensions
    row_in = size_in(1); col_in = size_in(2); fr = size_in(3);
    % New dimensiones for resizing
    row_out = ceil(Rxy_in(2)*row_in/Rxy_out(2));
    col_out = ceil(Rxy_in(1)*col_in/Rxy_out(1));
    % Initialize IM_fixed and Rxy_tmp
    IM_fixed = zeros(row_out,col_out,fr);
    AV_fixed = zeros(size(AV_in,1),size(AV_in,2));
    Rxy_tmp = zeros(1,2);
    % Real resolutions
    Rxy_tmp(1,2) = Rxy_in(2)*row_in/row_out;
    Rxy_tmp(1,1) = Rxy_in(1)*col_in/col_out;
    % Resize IM and AV in each frame
    if ~isempty(IM_in)
        IM_fixed = imresize3(IM_in,[row_out col_out fr]);
    end
    if ~isempty(AV_in)
        AV_fixed = AV_in .* repmat([row_out/row_in col_out/col_in],1,size(AV_in,2)/2);
    end
end

%% [9] Function for rotating heart
function [IM_rotated,AV_rotated,info_rotated] = rotate_heart(IM_in,AV_in)
    % Set variables
    [row_in,col_in,fr] = size(IM_in);
    ImCenter = [row_in;col_in]/2;
    % Set angle for rotation
    % [From first AV point to last AV point]
    rot = atan((AV_in(1,1)-AV_in(1,end-1))/(AV_in(1,2)-AV_in(1,end)))*180/pi;
    % Rotation matrix
    RotMatrix = [cosd(rot) -sind(rot); sind(rot) cosd(rot)];
    % Initialize IM_rotated and AV_rotated
    IM_rotated = zeros(row_in,col_in,fr);
    AV_rotated = zeros(size(AV_in,1),size(AV_in,2));
    % Rotate image
    for i=1:fr
        IM_rotated(:,:,i) = imrotate(IM_in(:,:,i),rot,'bicubic','crop');
    end
    % Rotate points in every frame
    % Assign variables
    a = RotMatrix(1,1); b = RotMatrix(1,2);
    c = RotMatrix(2,1); d = RotMatrix(2,2);
    e = ImCenter(1,1); f = ImCenter(2,1);
    % Rotate points
    AV_rotated(:,1:2:end) = a*AV_in(:,1:2:end)-a*e+b*AV_in(:,2:2:end)-b*f+e;
    AV_rotated(:,2:2:end) = c*AV_in(:,1:2:end)-c*e+d*AV_in(:,2:2:end)-d*f+f;
    % Output rotation parameter
    info_rotated = [RotMatrix ImCenter];
end

%% 10. Function for flipping heart (if needed)
function [IM_flipped,AV_flipped,info_flipped] = flip_heart(IM_in,AV_in)
    % Initiliaze output
    IM_flipped = IM_in;
    AV_flipped = AV_in;
    info_flipped = zeros(2,2);
    % Check if fliplr is needed
    if AV_flipped(1,2) > AV_flipped(1,end)
        % Middle variable
        x_m = size(IM_in,2)/2;
        % Flip image
        IM_flipped = fliplr(IM_flipped);
        % Flip points
        AV_flipped(:,2:2:end) = 2*x_m - AV_flipped(:,2:2:end) + 1;
        % Save info
        info_flipped(1,:) = [1 2*x_m];
    end
    % Check if flipud is needed
    if mean(AV_flipped(2:end,1)) < AV_flipped(1,1) ...
            && mean(AV_flipped(2:end,end-1)) < AV_flipped(1,end-1)
        % Middle variable
        y_m = size(IM_in,1)/2;
        % Flip image
        IM_flipped = flipud(IM_flipped);
        % Flip points
        AV_flipped(:,1:2:end) = 2*y_m - AV_flipped(:,1:2:end) + 1;
        % Save info
        info_flipped(2,:) = [1 2*y_m];
    end
end

%% [11] Function for cropping rotated and centered heart
function [IM_cropped,AV_cropped,info_cropped] = crop_heart(IM_in,AV_in,row_half,col_half)
    % Get IM_in dimensions
    [row_in,col_in,fr] = size(IM_in);
    % Get final dimensions
    row_out = row_half*2+1; col_out = col_half*2+1;
    % Add zeros if needed
    row_part = 0; col_part = 0;
    if row_in < row_out
        row_diff = row_out-row_in;
        row_part = ceil(row_diff/2);
        row_zeros = zeros(row_part,col_in,fr);
        IM_in = [row_zeros; IM_in; row_zeros];
    end
    if col_in < col_out
        col_diff = col_out-col_in;
        col_part = ceil(col_diff/2);
        col_zeros = zeros(size(IM_in,1),col_part,fr);
        IM_in = [col_zeros IM_in col_zeros];
    end
    % Get center
    center = [round(size(IM_in,1)/2) round(size(IM_in,2)/2)];
    % Save new cropped IM
    IM_cropped = IM_in(center(1)-row_half:center(1)+row_half,center(2)-col_half:center(2)+col_half,:);
    % Reassign eXY
    dX = row_out/2; dY = col_out/2;
    % Final dX and dY
    final_dX = dX-center(1)+row_part; final_dY = dY-center(2)+col_part;
    % Output info_cropped
    info_cropped = [final_dX final_dY];
    % Get AV
    AV_cropped = AV_in + repmat(info_cropped,1,size(AV_in,2)/2);
end

%% [17] Function for unflip AV points
function AV_unflipped = unflip_heart(AV_flipped,info_flipped)
    % Initialize output
    AV_unflipped = AV_flipped;
    % Flip if it was flipped (up to down)
    if info_flipped(2,1) == 1
        AV_unflipped(:,1:2:end) = info_flipped(2,2) - AV_unflipped(:,1:2:end) + 1;
    end
    % Flip if it was flipped (left to right)
    if info_flipped(1,1) == 1
        AV_unflipped(:,2:2:end) = info_flipped(1,2) - AV_unflipped(:,2:2:end) + 1;
    end
end

%% [18] Function for unrotating AV points
function AV_unrotated = unrotate_heart(AV_rotated,info_rotated)
    % Unpack info_rotated
    RotMatrix = info_rotated(:,1:2);
    ImCenter = info_rotated(:,3);
    % Initialize output
    AV_unrotated = AV_rotated;
    % Assign variables
    a = RotMatrix(1,1); b = RotMatrix(1,2);
    c = RotMatrix(2,1); d = RotMatrix(2,2);
    e = ImCenter(1,1); f = ImCenter(2,1);
    % Unrotate in every frame
    AV_unrotated(:,1:2:end) = (d*AV_rotated(:,1:2:end)+a*d*e-d*e-b*AV_rotated(:,2:2:end)-b*c*e+b*f)/(a*d-b*c);
    AV_unrotated(:,2:2:end) = (AV_rotated(:,2:2:end)+c*e+d*f-f-c*AV_unrotated(:,1:2:end))/d;
end
