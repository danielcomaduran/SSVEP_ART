"""
    Artifact Removal Tools
    -----------------
    These functions are used to remove artifacts from preprocessed EEG data

"""

## Import libraries
from copy import copy
import os
import time
# from pyrsistent import T
import pyts as ts
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.signal as signal
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from numpy import matlib as matlib, ndarray
from pyts.decomposition import SingularSpectrumAnalysis


#%% Remove Eyeblinks - Multiple channels
def remove_eyeblinks(eeg_raw, srate, window_length = 125, n_clusters = 4, fd_threshold = 1.4, ssa_threshold = 0.01):    
    """
        This function removes eyeblink artifacts from EEG data using an SSA approach implementation was adapted from
        [Maddirala & Veluvolu 2021] 

    Parameters
    ----------
        eeg_raw: array_like 
            Raw EEG data to be cleaned
        srate: int or float
            Sampling rate of the raw EEG signal [Hz]
        window_lenght: float or int, optional
            Length of window used to create the SSA matrix 
            If window_length.type() == float : window_length must be in msec 
            If window_length.type() == int: window_length is the number of sample points
        fd_threshold: float, optional
            Fractal dimension threshold 
        ssa_threshold: floar, optional
            Singular Spectrum Analysis threshold

    Returns
    -------
        eeg_clean: array like
            EEG signal without the artifacts
        eeg_artifact: array like
            Artifacts found in EEG signal

    Notes
    -----
        Return values have the same shape as eeg_raw
    """

    #%% Organize data
    # - Determine if number of dimensions of data
    shape = np.shape(eeg_raw)
    dimensions = np.ndim(eeg_raw)
    data_reshape = False    # Initialize flag to know if the data was reshaped

    if dimensions == 1:     # If data is one dimension, expand to row matrix
        eeg_raw = np.reshape(eeg_raw, (1,-1))
    elif dimensions == 2:   # If data is a matrix, Make sure data is in row vectors (i.e., [channels x samples])
        if shape[0] > shape[1]:
            data_reshape = True # Flag to know that the data was reshaped
            eeg_raw = eeg_raw.T
    else:
        print("Warning, data dimension not accepted")
        return None

    # - Determine embedding matrix sizes
    shape = np.shape(eeg_raw)   # Update shape value
    n_channels = shape[0]       # Number of channels 
    N = shape[1]                # Length of EEG signal
    
    #%% Create embedding matrix
    # - M = Number of rows
    if type(window_length) == int:
        M = window_length
    elif type(window_length) == float:
        M = np.floor(window_length * srate).astype(int) 
    else: 
        print('Data type of window_length is incorrect \n Data type should be "int" or "float"')
        return None

    # - K = Number of columns
    K = N - M + 1

    # - Create embedding matrix with the correspongding indices of the vector data
    idx_col = np.arange(K).reshape(1,-1).repeat(M, axis=0)
    idx_row = np.arange(M).reshape(-1,1).repeat(K, axis=1)
    idx_mat = idx_col + idx_row

    #%% Decomposed 
    # - Preallocate variables
    eeg_component = np.zeros((n_clusters, N))   # Temporary matrix of n_clusters EEG components reconstructed from eeg_decomposed
    eeg_artifact = np.zeros_like(eeg_raw)       # EEG + Artifact (before SSA)
    artifact = np.zeros_like(eeg_raw)           # Artifact signal (after SSA)
    
    for channel in range(n_channels):
        eeg_embedded = eeg_raw[channel, idx_mat]
        
        #%% Calculate features from embedded matrix
        f1 = np.sum(eeg_embedded**2, axis=0)            # Energy [V^2]
        f2 = np.sqrt(np.var(np.diff(eeg_embedded,axis=0),axis=0) / np.var(eeg_embedded,axis=0)) # H_mobility
        f3 = stats.kurtosis(eeg_embedded, axis=0)       # Kurtosis
        f4 = eeg_embedded.max(0) - eeg_embedded.min(0)  # Range
        eeg_features = np.array((f1,f2,f3,f4))

        #%% Perform Kmeans classification
        kmeans = KMeans(n_clusters=n_clusters).fit(eeg_features.T)

        # - Preallocate variables
        eeg_component = np.zeros((n_clusters, N))

        #%% Compute decomposed matrices
        for cluster in range(n_clusters):
            eeg_decomposed = np.zeros((M,K))    # Temporary matrix to store the decomposed EEG for each cluster [M,K] 
            
            # - Determine columns to copy based on the kmeans label
            copy_index = (kmeans.labels_==cluster) 
            eeg_decomposed[:,copy_index] = eeg_embedded[:,copy_index]
                    
            # Reconstruct signal from antidiagonal averages
            eeg_component[cluster, :] = mean_antidiag(eeg_decomposed)
            
        #%% Calculate Fractal Dimension (FD)        
        # - Normalize EEG to unit square
        x_norm = np.repeat(np.reshape(np.linspace(0, 1, N),[-1,1]),n_clusters,axis=1)
        y_num = eeg_component - matlib.repmat(np.amin(eeg_component, axis=1, keepdims=True), 1, N)
        y_den = matlib.repmat(np.amax(eeg_component, axis=1, keepdims=True) - np.amin(eeg_component, axis=1, keepdims=True), 1, N)
        y_norm = np.divide(y_num, y_den).T
        z = np.array([x_norm, y_norm]) # 3D Matrix to store x_norm and y_norm [sample x [x,y] x n_cluster]

        # - Calculate fractal dimension
        # l = np.sum(np.sum(np.abs(np.diff(z, axis=1)), axis=0), axis=0)  # Calculate length of signal (l1-norm for each n_cluster) [A.U.]
        l = np.sum(np.sqrt(np.sum(np.square(np.diff(z, axis=1)), axis=0)), axis=0)  # Calculate length of signal (l2-norm for each n_cluster) [A.U.]
        fd = 1 + (np.log(l) / np.log(2*(N-1)))

        # - Binary artifact creation
        fd_mask = fd < fd_threshold                                                 # Apply threshold to FD to determine artifact components
        eeg_mask = np.sum(eeg_component[fd_mask,:],0)                               # Vector with artifact points != 0
        eeg_artifact[channel,:] = eeg_raw[channel,:] * (eeg_mask != 0).astype(int)  # Multiply mask with original to get eeg_artifact with correct values [V]

        #%% Singular Spectrum Analysis
        # - Singular Value Decomposition
        artifact_embedded = eeg_artifact[channel,idx_mat]   # Create multivariate matrix for each channel
        [u, s, vh] = np.linalg.svd(artifact_embedded)       # Calculate SVD for multivariate matrix

        # - Determine number of groups
        eigen_ratio = (s / np.sum(s)) > ssa_threshold                                       # Keep only eigenvectors > ssa_threshold
        vh_sub = vh[0:np.size(s)]                                                           # Select subset of unitary arrays
        artifact_sub = u[:,eigen_ratio] @ np.diag(s[eigen_ratio]) @ vh_sub[eigen_ratio,:]   # Artifact with subset of eigenvectors
        
        # Reconstruct signals from antidiagonal averages
        artifact[channel,:] = mean_antidiag(artifact_sub)
        
    #%% Subtract artifact signal from original to get clean_eeg
    eeg_clean = eeg_raw - artifact
    
    #%% Return data in original shape
    if data_reshape:
        return eeg_clean.T, artifact.T
    else:
        return eeg_clean, artifact

def mean_antidiag(input_mat):
    """
        This function returns the mean of the antidiagonal components of a matrix

        Parameters
        ----------
            input_mat: array_like
                Matrix with shape [i,k] for which the mean of the antidiagonal components will be calculated.\n
                Must be a 2D matrix.

        Returns
        -------
            average_vect: array_like
                1D vector containing the mean values of the antidiagonal components
    """

    input_shape = np.shape(input_mat)   # Shape of input matrix
    input_flip = np.fliplr(input_mat)   # Flip input matrix from left to right
    average_vect = np.zeros(np.sum(input_shape)-1)  # Preallocate vector with average values

    # Make sure that input matrix is 2D
    if len(input_shape)!=2:
        print('Matrix must be 2D')
        return None

    # Calculate mean across diagonals
    # - Organize values from end to start to get right order
    for i_diag, k_diag in enumerate(range(-input_shape[0]+1,input_shape[1])):
        average_vect[-i_diag] = np.mean(np.diag(input_flip, k=k_diag))
    
    return average_vect