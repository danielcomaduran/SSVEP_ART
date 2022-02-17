# EDF to NPZ
# - This script takes an .EDF file from the TUH EEG dataset and stores the folowwing values in a .NPZ file
# - Values: 
# -- EEG = samples x channels x frequencies [\muV]
# -- srate = sampling rate [Hz]
# -- ssvep = ssvep stimulus frequencies [Hz]
# -- chans = [x,y,z,channel name]

#%% Import modules
import os
import mne
import numpy as np
import pandas as pd
from scipy import signal

#%% Settings
# - Edit this section according to your needs
# - Note: The script saves all 64 channels of data, just from the selected frequencies
file_name = '00000630_s002_t001'    # Input file name. Note: doesn't need the .edf
save_file = True          # Boolean to save data to file
output_file = 'AS02'        # Output file name - Artifact subject \#

#%% EEG data
dir_name = os.path.dirname(os.path.abspath(__file__))
edf_data = mne.io.read_raw_edf(dir_name+'\\'+file_name+'.edf')

# - EEG data
eeg = edf_data.get_data()  # EEG [V]

# - Transpose EEG data to have channels as columns
[a, b] = np.shape(eeg)
if b > a:
    eeg = eeg.T

# - Extra information
srate = edf_data.info['sfreq']  # Data sampling rate [Hz]
chans = edf_data.info.ch_names  # Channel information [name]
for c in range(len(chans)):     # Eliminate extra characters in channel information
    t_chan = chans[c]
    t_chan = t_chan.replace('EEG ','')
    t_chan = t_chan.replace('-REF','')
    chans[c] = t_chan

# - Resample data to new sample rate
nsrate = 250    # [Hz]

# -- Original data
l_oeeg = len(eeg)   # Length of original EEG [n]
eeg = signal.resample(eeg, int((l_oeeg*nsrate)/srate)) # Reample data
# -- Resampled data
l_neeg = len(eeg)   # Length of new EEG signal [n]
time = np.linspace(0, l_neeg/nsrate, l_neeg)   # New time vector
srate = nsrate # New sampling rate [Hz]

#%% Artifact annotations
full_artifact_notes = pd.read_csv(dir_name+'\\labels_01_tcp_ar.csv', header=4)
sub_artifact_notes = full_artifact_notes[full_artifact_notes['# key'].isin([file_name])]    # Subset for current trial

# - Separate eye and muscle artifacts
eye_artifacts = sub_artifact_notes[sub_artifact_notes[' artifact_label'].isin(['eyem'])]
mus_artifacts = sub_artifact_notes[sub_artifact_notes[' artifact_label'].isin(['musc'])]


#%% Trim data to eye and muscle artifacts
# - Trimmed data is stored in a list with each element being one artifact
# - Each element is a np.array with dimensions = samples x channels

# - Initialize lists
eye_eeg = np.array([None] * eye_artifacts.shape[0], dtype='object')
mus_eeg = np.array([None] * mus_artifacts.shape[0], dtype='object')

# - Eye artifacts
for eye in range(len(eye_artifacts)):
    [start_time, stop_time] = eye_artifacts.iloc[eye][[' start_time', ' stop_time']]
    eye_eeg[eye] = eeg[(time>=start_time)&(time<=stop_time), :]

# - Muscle artifacts
for mus in range(len(mus_artifacts)):
    [start_time, stop_time] = mus_artifacts.iloc[mus][[' start_time', ' stop_time']]
    mus_eeg[mus] = eeg[(time>=start_time)&(time<=stop_time), :]


#%% Save .NPZ data files
if save_file:
    np.savez(output_file+'.npz', eye_eeg=eye_eeg, mus_eeg=mus_eeg, srate=srate, chans=chans)