#%% Import libraries
import os
import numpy as np
from scipy.io import loadmat

def mat_to_npz(file, save_file=True, save_loc='Data\\Imported'):
    """
    Import a .mat file and convert to a .npz file. Meant to be used with the BETA dataset.

    Parameters
    ----------
        file : str
            Absolute path of the file to be imported
        save_file: bool, optional
            Boolean to save the data
        save_loc: str, optional
            Relative path of the saved data. Only necessary if save_file == True 

    Returns
    -------
        npz: list
            List with the same variables as the .npz file    

    Notes
    -----
    The .npz file is organized as follows:
        - EEG: samples x channels x frequencies [uV]
        - srate: sampling rate [Hz]
        - ssvep: SSVEP stimulus frequencies [Hz]
        - chans: EEG channels  with columns [loc_x, loc_y, loc_z, channel_name]
    """

    # Settings
    ssvep_freqs = [10,12,15]    # SSVEP frequencies to extract

    # Load and separate data
    mat_data = loadmat(file, simplify_cells=True)   
    eeg_data = mat_data['data']['EEG'] # EEG data [channels x samples x block x ssvep_freqs] [uV]
    sup_data = mat_data['data']['suppl_info']   # Sumplementary information
    srate = sup_data['srate']       # Data sampling rate [Hz]   
    stim_freqs = sup_data['freqs']  # Stimulation frequencies [Hz]
    chans = sup_data['chan']        # Channel information [x,y,z,name]

    a = 0
    # Organize data by SSVEP frequency
    # for i,ssvep_freq in enumerate(ssvep_freqs):
    #     temp_eeg = np.isin(stim_freqs, ssvep_freqs)
    #     trim_data = eeg_data[:,:,]


#%% Settings
# - Edit this section according to your needs
# - Note: The script saves all 64 channels of data, just from the selected frequencies
# file_name = 'S1'            # Input file name. Note: doesn't need the .mat
# save_file = True            # Boolean to save data
# output_file = 'S01'         # Output file name
# ssvep_freqs = [10, 12, 15]  # SSVEP frequencies to extract

# #%% Load data
# dir_name = os.path.dirname(os.path.abspath(__file__))
# mat_data = loadmat(dir_name+'\\'+file_name, simplify_cells=True)  # Load mat file
# eeg_data = mat_data['data']['EEG']          # EEG data. channels x samples x block x ssvep_freqs



# #%% Select frequencies to keep
# # - Each file has 4 blocks of data (trials) change _block_ to select the one to keep
# block = 1
# sel_freqs = np.isin(stim_freqs, ssvep_freqs)                # Search for index of selected frequencies
# trim_data = eeg_data[:,:,block, sel_freqs].swapaxes(0,1)    # Output data. samples x channels x frequencies

# #%% Save data to NPY file
# if save_file:
#     np.savez(output_file, eeg=trim_data, srate=srate, ssvep=ssvep_freqs, chans=chans)