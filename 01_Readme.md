# SSVEP_ART
Steady State Visual Evoked Potentials Artifact Removal Tool based on Python

## Description
--------------
This tool is meant to be used to remove blink, eye movement, and EMG artifacts from EEG data. Initially the dataset used with this tool was taken from the following sites:

- [Temple University database](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#i_rsyn). This dataset includes artifactual EEG data (e.g., eye movements, blinks, EMG artifacts). Detailed instructions on how to download the data can be found inside the `Data//Temple artifact data` folder.
- [BETA dataset](http://bci.med.tsinghua.edu.cn/download.html). This dataset is assumed to be clean SSVEP data. Detailed instructions on how to import the data into Python (Numpy) can be found inside the `Data//BETA dataset` folder.

## Temple University database
-----------------------------
To download the [Temple University database](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#i_rsyn) you have to register to the database to get a password and a user name to download the data. The username and password I got are:
- **User:** nedc
- **Password:** nedc_resources

The instructions on how to download the database are not very clear or updated.
I downloaded the database following these steps:

1) Download the free [MobaXTerm client](https://mobaxterm.mobatek.net/).
2) Create a new SSH session with the following parameters:
    - **Remote host:** www.isip.piconepress.com
    - **User:** nedc
    - **Port:** leave as is, in my case it was 22
3) Type in the password (i.e., nedc_resources). Note that there is no visual feedback of whats being typed, so be careful.
4) Once you are log in, type the following command in the main window to download the data:
    - `rsync -auxvL nedc@www.isip.piconepres.com:data/server_data_location computer_save_location`
    - Where `server_data_location` is the location in the database server of the data you want to download. In the case of the EEG artifact data, the location is `tuh_eeg_artifact/v2.0.0/`.
    - And `computer_save_location` is the address of where you want your data to be stored. I could not get this to work so I just used a period sign (`.`). My data was saved in this location:

        `C:\Users\danie\AppData\Local\Temp\Mxt215\tmp\home_danie`


5) Copy or move the data to your desired location.

## Workflow
-----------
The general workflow of the SSVEP ART is as follows

1. Create dataset
    1. Import the BETA dataset (clean).
    2. Import and partition the Temple University dataset. Partition according to the desired artifact types (i.e., eye movement, motion artifact, muscle activity).
    3. Convolute clean and artifactual datasets for testing purposes
2. Test dataset
    1. Start with supervised ICA removal as described here

## Notes
--------
### **2112**
Before finding the Temple and BETA datasets, I attemped to use an [EEG signal generator script](https://github.com/nikk-nikaznan/SSVEP-Neural-Generative-Models). However, the script did not specify which input was required. It seemed that the data required a tensor of [channels $\times$ data samples $\times$ 2]. The paper, nor the Github repository, explained what the third dimenson of the tensor needed to be.

### **2201**
After getting the BETA and Temple University datasets together, my first attempt was to use the Fully Online and Automated Artifact Removal for Brain-Computer Interfacing ([FORCE](http://www.iandaly.org/force/)) to separate artifacts from the clean signal. FORCE is implemented in Matlab, using some funcitons from [EEGlab](https://sccn.ucsd.edu/eeglab/index.php). This approach is contained in the [Matlab functions](/Matlab%20functions/) folder.