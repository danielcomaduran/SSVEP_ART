# EEG Artifact database
## Temple university
Artifact data was downloaded from the [Temple University database](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#i_rsyn).

You have to register to the database to get a password and a user name to download the data. The username and password I got are:
- **User:** nedc
- **Password:** nedc_resources

## How to download databse?
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
 

