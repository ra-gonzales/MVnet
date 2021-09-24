% Enable GPU processing
GPU = true;

% Load MVnet networks
load(fullfile(pwd, 'models', 'MVnet_2ch_1st.mat'));
load(fullfile(pwd, 'models', 'MVnet_2ch_2nd.mat'));
load(fullfile(pwd, 'models', 'MVnet_4ch_1st.mat'));
load(fullfile(pwd, 'models', 'MVnet_4ch_2nd.mat'));

% Load anonymized dicom data from a subject
data_sample_path_2ch = fullfile(pwd, 'data_sample', '2ch');
data_sample_path_4ch = fullfile(pwd, 'data_sample', '4ch');
[IM_2ch, Rxy_2ch, TimeVector_2ch] = MVnet_functions('load_dicom_data', data_sample_path_2ch);
[IM_4ch, Rxy_4ch, TimeVector_4ch] = MVnet_functions('load_dicom_data', data_sample_path_4ch);

% Track the mitral valve valve
MV_2ch = MVnet_functions('pipeline',IM_2ch,Rxy_2ch,MVnet_2ch_1st,MVnet_2ch_2nd,GPU);
MV_4ch = MVnet_functions('pipeline',IM_4ch,Rxy_4ch,MVnet_4ch_1st,MVnet_4ch_2nd,GPU);

% Derive clinical metrics
[MVdist, MVvel, MAPSE, MVs, MVe, MVa] = MVnet_functions('get_clinical', ...
    MV_2ch, MV_4ch, Rxy_2ch, TimeVector_2ch, Rxy_4ch, TimeVector_4ch);

% Visualize the tracking
figure,
for i=1:size(MV_4ch,1)
    subplot(1,2,1),
    imagesc(IM_2ch(:,:,i)),colormap(gray), axis image, hold on,
    plot(MV_2ch(i,2),MV_2ch(i,1),'*r'), hold on,
    plot(MV_2ch(i,4),MV_2ch(i,3),'*g'),
    subplot(1,2,2),
    imagesc(IM_4ch(:,:,i)),colormap(gray), axis image, hold on,
    plot(MV_4ch(i,2),MV_4ch(i,1),'*r'), hold on,
    plot(MV_4ch(i,4),MV_4ch(i,3),'*g'),
    pause(0.2);
end
