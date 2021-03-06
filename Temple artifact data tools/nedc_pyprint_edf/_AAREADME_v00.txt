file: _AAREADME.txt

This directory contains Python code to open an EDF file, print the
header information, load the signal, and print the signal values. It
is a stripped down version of several of our utilities.

To learn more about the data, please review this document:

 Ferrell, S., Mathew, V., Ahsan, T., & Picone, J. (2019). The Temple
 University Hospital EEG Corpus: Electrode Location and Channel
 abels. Philadelphia, Pennsylvania, USA.
 https://www.isip.piconepress.com/publications/reports/2019/tuh_eeg/electrodes/

Included in this distribution is a simple EDF file, example.edf. This file
contains the following information:

---
nedc_000_[1]: more example.edf 
0       00008836 M 01-JAN-1952 00008836 Age:60                                  
        Startdate 30-APR-2012 00008836_s003 00 X                                
        30.04.1208.44.117936    EDF                                         5   
    1.00000030  EEG FP1-REF     EEG FP2-REF     EEG F3-REF      EEG F4-REF      
EEG C3-REF      EEG C4-REF      EEG P3-REF      EEG P4-REF      EEG O1-REF      
EEG O2-REF      EEG F7-REF      EEG F8-REF      EEG T3-REF      EEG T4-REF      
EEG T5-REF      EEG T6-REF      EEG A1-REF      EEG A2-REF      EEG FZ-REF      
EEG CZ-REF      EEG PZ-REF      EEG ROC-REF     EEG LOC-REF     EEG EKG1-REF    
EEG T1-REF      EEG T2-REF      PHOTIC-REF      IBI             BURSTS          
SUPPR           Unknown                                                         
                Unknown                     
...
---

Notice that the first three channels are "EEG FP1-REF", "EEG FP2-REF",
"EEG F3-REF", and there is an additional channel "EEG T3-REF". We will focus
on these channels in the examples below.

The use of this program is best illustrated with a simple example::

===============================================================================
Differencing Two Channels
===============================================================================

EEG labels can be unpredictable. For example, the channel "EEG FP1-REF" could
just be labeled "FP1-REF" at another institution. Therefore, we support
two matching modes to identify channels: exact and partial.

The parameter file param_00.txt simply selects the first channel that matches
"FP1-REF" and prints its values to stdout. This block enables this:

 channel_selection = (null)
 select_mode = select
 match_mode = exact
 montage =  0, FP1: EEG FP1-REF
 montage =  1, FP2: EEG FP2-REF
 montage =  2, FP1-FP2: EEG FP1-REF -- EEG FP2-REF

This montage spec creates a new signal where the first channel, labeled
FP1 is the channel labeled "EEG FP1-REF" in the original file. The second
channel, labeled FP2, is the channel labeled "EEG FP2-REF" in the original
file. The third channel, labeled "FP1-FP2" is the difference of these
two channels.

Here is the system output:

./nedc_pyprint_edf.py example.edf params_00.txt
edf   file: example.edf
param file: params_00.txt

<---------- start of header ---------->
	Block 1: Version Information
	 version = [0       ]

	Block 2: Local Patient Information
	 lpti_patient_id = [00008836]
	 lpti_gender = [M]
	 lpti_dob = [01-JAN-1952]
	 lpti_full_name = [00008836]
	 lpti_age = [Age:60]

	Block 3: Local Recording Information
	 lrci_start_date_label = [Startdate]
	 lrci_start_date = [30-APR-2012]
	 lrci_eeg_id = [00008836_s003]
	 lrci_tech = [00]
	 lrci_machine = [X]

	Block 4: General Header Information
	 ghdi_start_date = [30.04.12]
	 ghdi_start_time = [08.44.11]
	 ghdi_hsize = [7936]
	 ghdi_file_type = [EDF  ]
	 ghdi_reserved = [                                       ]
	 ghdi_num_recs = [5]
	 ghdi_dur_rec = [1.000000]
	 ghdi_nsig_rec = [30]

	Block 5: Channel-Specific Information
	 chan_labels (30) = [EEG FP1-REF], [EEG FP2-REF], [EEG F3-REF], [EEG F4-REF], [EEG C3-REF], [EEG C4-REF], [EEG P3-REF], [EEG P4-REF], [EEG O1-REF], [EEG O2-REF], [EEG F7-REF], [EEG F8-REF], [EEG T3-REF], [EEG T4-REF], [EEG T5-REF], [EEG T6-REF], [EEG A1-REF], [EEG A2-REF], [EEG FZ-REF], [EEG CZ-REF], [EEG PZ-REF], [EEG ROC-REF], [EEG LOC-REF], [EEG EKG1-REF], [EEG T1-REF], [EEG T2-REF], [PHOTIC-REF], [IBI], [BURSTS], [SUPPR]
	 chan_trans_type (30) = [Unknown], [Unknown], [Unknown], [Unknown], [Unknown], [Unknown], [Unknown], [Unknown], [Unknown], [Unknown], [Unknown], [Unknown], [Unknown], [Unknown], [Unknown], [Unknown], [Unknown], [Unknown], [Unknown], [Unknown], [Unknown], [Unknown], [Unknown], [Unknown], [Unknown], [Unknown], [Unknown], [Unknown], [Unknown], [Unknown]
	 chan_phys_dim (30) = [uV], [uV], [uV], [uV], [uV], [uV], [uV], [uV], [uV], [uV], [uV], [uV], [uV], [uV], [uV], [uV], [uV], [uV], [uV], [uV], [uV], [uV], [uV], [uV], [uV], [uV], [uV], [sec], [brst/min], [%]
	 chan_phys_min (30) = [ -4999.840], [ -4999.840], [ -4999.840], [ -4999.840], [ -4999.840], [ -4999.840], [ -4999.840], [ -4999.840], [ -4999.840], [ -4999.840], [ -4999.840], [ -4999.840], [ -4999.840], [ -4999.840], [ -4999.840], [ -4999.840], [ -4999.840], [ -4999.840], [ -4999.840], [ -4999.840], [ -4999.840], [ -4999.840], [ -4999.840], [ -4999.840], [ -4999.840], [ -4999.840], [-32767.000], [ -3276.700], [ -3276.700], [ -3276.700]
	 chan_phys_max (30) = [  4999.847], [  4999.847], [  4999.847], [  4999.847], [  4999.847], [  4999.847], [  4999.847], [  4999.847], [  4999.847], [  4999.847], [  4999.847], [  4999.847], [  4999.847], [  4999.847], [  4999.847], [  4999.847], [  4999.847], [  4999.847], [  4999.847], [  4999.847], [  4999.847], [  4999.847], [  4999.847], [  4999.847], [  4999.847], [  4999.847], [ 32767.000], [  3276.700], [  3276.700], [  3276.700]
	 chan_dig_min (30) = [    -32767], [    -32767], [    -32767], [    -32767], [    -32767], [    -32767], [    -32767], [    -32767], [    -32767], [    -32767], [    -32767], [    -32767], [    -32767], [    -32767], [    -32767], [    -32767], [    -32767], [    -32767], [    -32767], [    -32767], [    -32767], [    -32767], [    -32767], [    -32767], [    -32767], [    -32767], [    -32767], [    -32767], [    -32767], [    -32767]
	 chan_dig_max (30) = [     32767], [     32767], [     32767], [     32767], [     32767], [     32767], [     32767], [     32767], [     32767], [     32767], [     32767], [     32767], [     32767], [     32767], [     32767], [     32767], [     32767], [     32767], [     32767], [     32767], [     32767], [     32767], [     32767], [     32767], [     32767], [     32767], [     32767], [     32767], [     32767], [     32767]
	 chan_prefilt (30) = [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0], [HP:0.000 Hz LP:0.0 Hz N:0.0]
	 chan_rec_size (30) = [       250], [       250], [       250], [       250], [       250], [       250], [       250], [       250], [       250], [       250], [       250], [       250], [       250], [       250], [       250], [       250], [       250], [       250], [       250], [       250], [       250], [       250], [       250], [       250], [       250], [       250], [       250], [         1], [         1], [         1]
	
	Block 6: Derived Values
	 hdr_sample_frequency =      250.0
	 hdr_num_channels_signal =         30
	 hdr_num_channels_annotation =          0
	 duration of recording (secs) =        5.0
	 per channel sample frequencies:
	  channel[   0]:      250.0 Hz (EEG FP1-REF)
	  channel[   1]:      250.0 Hz (EEG FP2-REF)
	  channel[   2]:      250.0 Hz (EEG F3-REF)
	  channel[   3]:      250.0 Hz (EEG F4-REF)
	  channel[   4]:      250.0 Hz (EEG C3-REF)
	  channel[   5]:      250.0 Hz (EEG C4-REF)
	  channel[   6]:      250.0 Hz (EEG P3-REF)
	  channel[   7]:      250.0 Hz (EEG P4-REF)
	  channel[   8]:      250.0 Hz (EEG O1-REF)
	  channel[   9]:      250.0 Hz (EEG O2-REF)
	  channel[  10]:      250.0 Hz (EEG F7-REF)
	  channel[  11]:      250.0 Hz (EEG F8-REF)
	  channel[  12]:      250.0 Hz (EEG T3-REF)
	  channel[  13]:      250.0 Hz (EEG T4-REF)
	  channel[  14]:      250.0 Hz (EEG T5-REF)
	  channel[  15]:      250.0 Hz (EEG T6-REF)
	  channel[  16]:      250.0 Hz (EEG A1-REF)
	  channel[  17]:      250.0 Hz (EEG A2-REF)
	  channel[  18]:      250.0 Hz (EEG FZ-REF)
	  channel[  19]:      250.0 Hz (EEG CZ-REF)
	  channel[  20]:      250.0 Hz (EEG PZ-REF)
	  channel[  21]:      250.0 Hz (EEG ROC-REF)
	  channel[  22]:      250.0 Hz (EEG LOC-REF)
	  channel[  23]:      250.0 Hz (EEG EKG1-REF)
	  channel[  24]:      250.0 Hz (EEG T1-REF)
	  channel[  25]:      250.0 Hz (EEG T2-REF)
	  channel[  26]:      250.0 Hz (PHOTIC-REF)
	  channel[  27]:        1.0 Hz (IBI)
	  channel[  28]:        1.0 Hz (BURSTS)
	  channel[  29]:        1.0 Hz (SUPPR)
<---------- end   of header ---------->

<---------- start of signal ---------->
       0: 4497.9858  4435.5774    62.4084 
       1: 4490.8142  4430.2368    60.5773 
       2: 4485.0159  4423.9808    61.0351 
       3: 4479.6753  4418.3350    61.3403 
       4: 4472.8088  4412.3841    60.4248 
       5: 4467.6209  4406.4332    61.1877 
       6: 4461.0596  4400.6348    60.4248 
       7: 4455.5664  4394.8365    60.7299 
       8: 4449.4629  4389.0382    60.4248 
       9: 4443.5120  4383.2398    60.2722 
     100: 3939.0568  3883.8200    55.2368 
     101: 3933.8688  3878.7846    55.0842 
     102: 3928.6808  3873.5967    55.0842 
     103: 3923.4929  3868.4087    55.0842 
     104: 3918.3049  3863.2207    55.0842 
     105: 3912.9643  3858.1853    54.7790 
     106: 3907.7763  3852.9973    54.7790 
     107: 3902.5883  3847.9619    54.6264 
     108: 3897.4004  3842.7739    54.6264 
     109: 3892.2124  3837.5859    54.6264 
     200: 3739.3194  3688.9655    50.3540 
     201: 3795.7769  3756.2567    39.5202 
     202: 3671.4179  3674.6222    -3.2043 
     203: 3794.2510  3730.0116    64.2395 
     204: 3679.8102  3684.9982    -5.1880 
     205: 3751.2213  3700.8673    50.3540 
     206: 3706.2079  3691.1017    15.1062 
     207: 3711.0907  3680.5731    30.5176 
     208: 3716.2787  3686.2189    30.0598 
     209: 3691.8646  3670.1972    21.6675 
     300:-3453.5155  -217.1289 -3236.3866 
     301:-3480.0658  4938.8119 -8418.8777 
     302:-3488.7633  4480.1331 -7968.8964 
     303:-3462.8234  4751.5867 -8214.4101 
     304:-3520.0438  4683.0748 -8203.1186 
     305:-3659.2038  4490.8142 -8150.0180 
     306:-3666.3755  4358.2154 -8024.5909 
     307:-3763.5739  4283.4474 -8047.0213 
     308:-3945.4585  3863.0681 -7808.5266 
     309:-3945.4585  3932.9533 -7878.4118 
     400:-2989.9539  4026.4896 -7016.4435 
     401:-3105.1576  3828.1255 -6933.2831 
     402:-3163.1410  3724.2132 -6887.3542 
     403:-3207.0863  3617.8596 -6824.9458 
     404:-3215.6312  3596.4973 -6812.1285 
     405:-3089.2885  3803.7115 -6893.0000 
     406:-3117.2121  3744.0496 -6861.2617 
     407:-3178.3998  3636.6279 -6815.0276 
     408:-3169.5497  3637.0856 -6806.6353 
     409:-3075.7082  3786.3165 -6862.0247 
     500:-2816.3090  2998.9636 -5815.2725 
     501:-2811.1210  2993.3178 -5804.4388 
     502:-2779.5353  3018.4948 -5798.0301 
     503:-2792.8105  2988.8928 -5781.7032 
     504:-2797.6933  2973.1762 -5770.8695 
     505:-2714.2278  3112.4889 -5826.7166 
     506:-2655.3289  3204.4993 -5859.8282 
     507:-2747.1867  3041.0778 -5788.2645 
     508:-2660.0591  3185.4258 -5845.4849 
     509:-2630.6097  3212.8916 -5843.5013 
     600:-2190.2414  3067.0177 -5257.2591 
     601:-2225.6417  3002.7783 -5228.4200 
     602:-2179.4076  3085.3282 -5264.7359 
     603:-2147.5168  3127.4425 -5274.9593 
     604:-2120.5088  3164.5213 -5285.0300 
     605:-2026.9725  3326.1117 -5353.0842 
     606:-1977.8392  3401.0323 -5378.8715 
     607:-1897.7306  3547.9744 -5445.7050 
     608:-1884.6081  3558.3503 -5442.9584 
     609:-1876.3683  3571.0151 -5447.3834 
     700:-1057.2772  2592.3171 -3649.5943 
     701:-1096.6448  2527.6199 -3624.2647 
     702:-1056.3616  2583.7722 -3640.1339 
     703:-1107.7837  2510.0723 -3617.8561 
     704:-1257.1671  2278.9019 -3536.0690 
     705:-1098.4759  2523.5001 -3621.9759 
     706:-1040.3399  2589.4180 -3629.7579 
     707: -989.6808  2651.9790 -3641.6597 
     708: -959.9262  2709.1994 -3669.1255 
     709: -912.3188  2745.8204 -3658.1392 
     800:-1312.5565  1610.4148 -2922.9713 
     801:-1205.2873  1799.7763 -3005.0636 
     802:-1121.9744  1929.3233 -3051.2977 
     803:-1227.7177  1734.3161 -2962.0338 
     804:-1278.0717  1654.0549 -2932.1266 
     805:-1272.8837  1659.3955 -2932.2792 
     806:-1287.5321  1623.2322 -2910.7643 
     807:-1360.9268  1497.4999 -2858.4267 
     808:-1345.2103  1515.8104 -2861.0207 
     809:-1245.7231  1677.8586 -2923.5817 
     900: -966.0297  1739.0463 -2705.0760 
     901:-1002.4982  1650.3928 -2652.8910 
     902: -866.5425  1876.3753 -2742.9178 
     903: -773.1587  2027.2846 -2800.4434 
     904: -751.4913  2055.3608 -2806.8521 
     905: -710.2926  2128.9081 -2839.2007 
     906: -762.9354  2026.5217 -2789.4570 
     907: -729.8238  2071.2299 -2801.0537 
     908: -683.1320  2174.0741 -2857.2060 
     909: -619.6554  2280.1226 -2899.7780 
    1000:  696.5667  3854.8284 -3158.2617 
    1001:  765.6889  3961.6398 -3195.9509 
    1002:  679.4768  3796.6924 -3117.2156 
    1003: 1047.2134  4476.6235 -3429.4102 
    1004: 1131.1366  4654.3883 -3523.2516 
    1005: 1160.5861  4690.2464 -3529.6603 
    1006: 1316.3782  4900.8175 -3584.4393 
    1007: 1385.8056  4487.6099 -3101.8042 
    1008:    0.0035     0.0035     0.0000 
    1009:    0.0035     0.0035     0.0000 
    1100:    0.0035     0.0035     0.0000 
    1101:    0.0035     0.0035     0.0000 
    1102:    0.0035     0.0035     0.0000 
    1103:    0.0035     0.0035     0.0000 
    1104:    0.0035     0.0035     0.0000 
    1105:    0.0035     0.0035     0.0000 
    1106:    0.0035     0.0035     0.0000 
    1107:    0.0035     0.0035     0.0000 
    1108:    0.0035     0.0035     0.0000 
    1109:    0.0035     0.0035     0.0000 
    1200:    0.0035     0.0035     0.0000 
    1201:    0.0035     0.0035     0.0000 
    1202:    0.0035     0.0035     0.0000 
    1203:    0.0035     0.0035     0.0000 
    1204:    0.0035     0.0035     0.0000 
    1205:    0.0035     0.0035     0.0000 
    1206:    0.0035     0.0035     0.0000 
    1207:    0.0035     0.0035     0.0000 
    1208:    0.0035     0.0035     0.0000 
    1209:    0.0035     0.0035     0.0000 
<---------- end   of signal ---------->

===============================================================================
Questions about the use of this software should be directed 
to help@nedcdata.org.

Enjoy,

The Neural Engineering Data Consortium
