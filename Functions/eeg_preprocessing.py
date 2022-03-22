"""
    EEG Preprocessing
    -----------------
    These functions are used to preprocess EEG data

"""

#%% Import libraries
import numpy as np
import scipy.signal as signal

#%% Detrend
def detrend(eeg, poly=1):
    """
        Remove the polynomial trend of an EEG signal. The trend can be calculated for the complete data or in epochs.

        Parameters
        ----------
            eeg: numpy matrix or tensor
                Numpy matrix or tensor to detrend.
                If 2D matrix, each column will be detrended.
                If >2D matrix, each column in each extra dimension will be detrended
            poly: int, optional
                Polynomial order to detrend the data. If left blank, poly = 1
            epoch; int, optional
                Number of samples to take for each EEG epoch. 
                Must comply with len(eeg)%epoc = 1 

        Returns
        -------
            eeg_detrend: numpy matrix or tensor
                Numpy matrix or tensor of the detrended EEG data                
    """

    # - Determine if eeg is a 2D matrix or tensor
    shape = np.shape(eeg)
    dim = np.size(shape)

    if dim < 2:
        print('EEG is not a 2D matrix')
        return
    elif dim == 2:
        eeg_detrend= signal.poly