%% FORCE Implementation
% This script implements the Fully Online and Automated Artifact Removal
% for Brain-Computing Interface to work with the convoluted muscle ('mus')
% or eye ('eye') artifact data.

%% Start clean
clc;
close all;
clearvars;

%% Settings
trial = 'mus';  % Trial to clean 
                % - 'mus' = Muscle artifacts
                % - 'eye' = Eye movement artifacts
ssvep = 1;      % Select block trial to analyze
                % - 1 = 10 Hz
                % - 2 = 12 Hz
                % - 3 = 15 Hz
plot_raw = 0;   % Boolean to plot raw data
plot_clean = 0; % Boolean to plot clean data
plot_psd = 1;   % Boolean to plos PSD of raw and clean data

% - Constants
SSVEP_STR = {'10', '12', '15'}; % String to include in plots

%% Load and select data
% - Determine directory
par_dir = pwd();    % Get current parent directory
data_dir = [par_dir, '\Data\Convolved'];  % Directory of data

% - Load data
if strcmp(trial,'mus')
    load([data_dir,'\conv_mus_data.mat']);
elseif strcmp(trial,'eye')
    load([data_dir,'conv_eye_data.mat']);
else
    disp(['Warning!', newline ,'Wrong type of trial selected'])
    return
end

% - Select
EEG_raw = conv{1,ssvep};    % Raw data [\muV]
srate = double(srate);      % Sampling rate [Hz]
time = 0:1/srate:(size(EEG_raw,1)-1)/srate; % Time vector [sec]

%% Plot raw data
if plot_raw
    figure(1)
    
    for i = 1:size(EEG_raw, 2)
        plot(time, (EEG_raw(:,i) - (i-1)*1e-6))
        hold on
    end
    xlabel('Time [sec]')
    ylabel('RAW EEG [\muV]')
end

%% Create channels structure
% - Channels need to be in a structure variable with the labels property
channels.theta = zeros(length(chans),1);
channels.labels = chans';

%% Implement FORCE
EEG_clean = zeros(size(EEG_raw));   % Initialize array to store cleaned data
win_length = 1*srate;               % Window length [n_samples]
N = win_length*3;     % Number of windows [N]

disp('Start FORCE...')

for win_pos = 1:win_length:N    
    win = win_pos:(win_pos+win_length)-1;       % Window [n]
        
    tic;    % Start timer
    cleanEEG = FORCe(EEG_raw(win,:)', srate, channels', 0);
    disp(['Time taken to clean 1s EEG = ' num2str(toc) 's.']);
        
    EEG_clean(win,:) = cleanEEG';   % Put together the cleaned EEG time series.
end

%% Plot clean data
if plot_clean
    figure(2)
    
    for i = 1:size(EEG_clean,2)
        plot(time, EEG_clean(:,i) - (i-1)*1e-6)
        hold on;
    end
    ylabel('Cleaned EEG [\muV]')
    xlabel('Time [sec]')
end

%% Plot PSD of raw and clean data
[psd_raw, f_raw] = pwelch(EEG_raw, size(EEG_raw,1), [], [] ,srate);
[psd_clean, f_clean] = pwelch(EEG_clean, size(EEG_clean,1), [], [] ,srate);

% Calculate PSD of artifact free data
ref_chans = find(ismember(clean_chans, chans)); % Find reference channels
EEG_na = clean(:,ref_chans, ssvep);     % EEG data with no artifacts [\muV]
[psd_na, f_na] = pwelch(EEG_na, size(EEG_na,1), [], [], srate);

x_axis = [0, 35];
y_max = max([max(psd_raw,[],'all'), max(psd_clean,[],'all'), max(psd_na,[],'all')]);
y_min = min([min(psd_raw,[],'all'), min(psd_clean,[],'all'), min(psd_na,[],'all')]);
%y_axis = [0.9*y_min, 10*y_max];
y_scale = 'linear';

switch y_scale
    case 'linear'
        y_axis = [-inf, inf];
    case 'log'
        y_axis = [0.9*y_min, 10*y_max];
end

if plot_psd
    figure(3)
    sgtitle(['SSVEP - ', SSVEP_STR{ssvep}, ' Hz'])
    
    subplot(1,3,1)
    plot(f_na, psd_na)
    ylabel('PSD [V^2/Hz]')
    title("No artifact"+newline+"(original)")
    set(gca, 'yscale', y_scale)
    axis([x_axis, y_axis])
    grid()
    
    subplot(1,3,2)
    plot(f_raw, psd_raw)
    xlabel('Frequency [Hz]')
    title("RAW" + newline + "(clean + artifact)")
    set(gca, 'yscale', y_scale)
    axis([x_axis, y_axis])
    grid()

    subplot(1,3,3)
    plot(f_clean, psd_clean)
    title("Clean"+newline+"(after FORCE)")
    set(gca, 'yscale', y_scale)
    axis([x_axis, y_axis])
    grid()
end