from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import os
import numpy as np

# %% Settings
percentage_cpus = 50    # Percentage of CPUs to be used in multithreading [int]
n_channels = 15         # Number of EEG channels [int]
srate = 250             # Sampling rate [Hz]
eeg_time = 10           # Duration of fake EEG signals [sec]
eeg_range = 10e-6       # Range of EEG signals [V]
svd_threshold = 0.01    # Threshold for eigenvalues in SVD

# Determine number of CPUs
total_cpus = os.cpu_count()
used_cpus = np.floor(total_cpus*(percentage_cpus/100)).astype(int)
values = np.arange(n_channels)

# %% EEG Matrix
# Generate EEG matrix
eeg = np.random.rand(n_channels, srate*eeg_time) * eeg_range
eeg_time_vector = np.linspace(0,eeg_time/srate,eeg_time*srate)

# Generate EEG embedded tensor
N = eeg_time*srate  # Number of samples per channel 
M = 125             # Number of samples per window [int]
K = N - M + 1       # Number of columns

# Create embedding matrix with the corresponding indices of the vector data
idx_col = np.arange(K).reshape(1,-1).repeat(M, axis=0)
idx_row = np.arange(M).reshape(-1,1).repeat(K, axis=1)
idx_mat = idx_col + idx_row

# Preallocate and fill embedded matrix
eeg_embedded = np.zeros((M, K, n_channels)) # Preallocate tensor

for i in range(n_channels):
    eeg_embedded[:,:,i] = eeg[i, idx_mat]

# %% SVD for single channel
def svd_channel(input_chan):
    [u, s, vh] = np.linalg.svd(input_chan)   # Calculate SVD for multivariate matrix

    # Determine number of groups
    eigen_ratio = (s / np.sum(s)) > svd_threshold                                       # Keep only eigenvectors > ssa_threshold
    vh_sub = vh[0:np.size(s)]                                                           # Select subset of unitary arrays
    input_subset = u[:,eigen_ratio] @ np.diag(s[eigen_ratio]) @ vh_sub[eigen_ratio,:]   # Input signal with subset of eigenvectors

    # Calculate antidiagonal matrix
    average_vect = np.zeros(N)              # Preallocate average vector
    input_shape = np.shape(input_chan)      # Determine shape of embedded matrix
    input_flip = np.fliplr(input_subset)    # Flip eigenvector subset matrix

    for i_diag, k_diag in enumerate(range(-input_shape[0]+1,input_shape[1])):
        average_vect[-i_diag] = np.mean(np.diag(input_flip, k=k_diag))
    
    return average_vect

# %% Square function - from example
def square(n):
    return n * n

# %% Run svd channel in for loop
sub_eeg = np.zeros_like(eeg)
for i in range(n_channels):
    sub_eeg[i,:] = svd_channel(eeg_embedded[:,:,i])

# %% Main function    
def main():
    vect_chans = np.arange(n_channels)
    vect_n = (N * np.ones(n_channels)).astype(int)
    eeg_result = np.zeros_like(eeg)
    with ProcessPoolExecutor(max_workers = used_cpus) as executor:
        results = executor.map(svd_channel, eeg_embedded[:,:,vect_chans])   # Enable this to try Daniel's function
        # results = executor.map(square, values)    # Enable this to try the documentation example
    
    for result in results:
        print(result)

    a = 0   # Breakpoint line

if __name__ == '__main__':
    main()