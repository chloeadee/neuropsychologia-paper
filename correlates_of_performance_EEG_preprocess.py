"""
Individual alpha frequency and aperiodic exponent estimatation for Poseidon

Author: Zachariah R Cross, 2020

"""

# Here, we are loading the modules needed for this analysis
import mne
import glob
import yasa
import pandas as pd
import os.path as op
import seaborn as sns
import matplotlib.pyplot as plt
from philistine.mne import savgol_iaf

sns.set(style='white', font_scale=1.2)

rest_files = glob.glob('PR*.vhdr')

# Set parameters for EOG and reference combinations
eye_chan_1 = ['HEOR','HEOL','VEOL','VEOU']
eye_chan_2 = ['E1','E2']
eye_chan_3 = ['VEOL','VEOR']
eye_chan_4 = ['EOGH','EOGV']
eye_chan_5 = ['Fp1','Fp2']
eye_chan_6 = ['FP1','FP2']

ref_chan_1 = ['M1','M2']
ref_chan_2 = ['TP9','TP10']

# extract information of the file extensions (e.g., brainvision, neuroscan)
for r in rest_files:
    file = op.split(r)[-1][-4:]

# set which electrodes we want to use for IAF estimation
electrodes = ['P3','P4','O1','O2','P7','P8']

# create .txt file to save IAF values
outfile = open('iaf_output.txt','w')
header = "subj"+"\t"+"measure"+"\t"+"value"+"\n"
outfile.write(header)

montage = 'standard_1020'

## ---------------------------------------------------------------------------
## Basic Pre-Processing
## ---------------------------------------------------------------------------

# Now we loop through each of our resting-state recordings
for r in rest_files:
    print("processing file " + r)
    subj = op.split(r)[1][0:4] # extract first 4 characters (i.e., CP01)

# Now we are reading in our raw EEG file based on the file format.
# We have included multiple 'if' statement here, so that the script will 
# automatically detect which sort of EEG file you are reading in based on
# what we defined on line 31. We are also setting our EOG channels based on
# what is listed for each file based on what we defined on lines 23 - 25.
    if file == "vhdr":
        find_eog = mne.io.read_raw_brainvision(r)
        if 'HEOR' in find_eog.ch_names:
            raw = mne.io.read_raw_brainvision(r, eog = [eye_chan_1], preload=True)
            print("h_used")
        elif 'E1' in find_eog.ch_names:
            raw = mne.io.read_raw_brainvision(r, eog = [eye_chan_2], preload=True)
            print("e_used")
        elif 'VEOL' in find_eog.ch_names:
            raw = mne.io.read_raw_brainvision(r, eog = [eye_chan_3], preload=True)
            print("v_used")  
        elif 'EOGH' in find_eog.ch_names:
            raw = mne.io.read_raw_brainvision(r, eog = [eye_chan_4], preload=True)
            print("combined_used")
        elif 'Fp1' in find_eog.ch_names:
            raw = mne.io.read_raw_brainvision(r, eog = [eye_chan_5], preload=True)
            print("fp_used")
        elif 'FP1' in find_eog.ch_names:
            raw = mne.io.read_raw_brainvision(r, eog = [eye_chan_6], preload=True)
            print("fp_used")
    elif file == ".cnt":
        find_eog = mne.io.read_raw_cnt(r)
        if 'HEOR' in find_eog.ch_names:
            raw = mne.io.read_raw_cnt(r, eog = [eye_chan_1], preload=True)
            print("h_used")
        elif 'E1' in find_eog.ch_names:
            raw = mne.io.read_raw_cnt(r, eog = [eye_chan_2], preload=True)
            print("e_used")
        elif 'VEOL' in find_eog.ch_names:
            raw = mne.io.read_raw_cnt(r, eog = [eye_chan_3], preload=True)
            print("v_used")
        elif 'EOGH' in find_eog.ch_names:
            raw = mne.io.read_raw_cnt(r, eog = [eye_chan_4], preload=True)
            print("combined_used")
        elif 'Fp1' in find_eog.ch_names:
            raw = mne.io.read_raw_cnt(r, eog = [eye_chan_5], preload=True)
            print("fp_used")
    elif file == ".edf":
        find_eog = mne.io.read_raw_edf(r)
        if 'HEOR' in find_eog.ch_names:
            raw = mne.io.read_raw_edf(r, eog = [eye_chan_1], preload=True)
            print("h_used")
        elif 'E1' in find_eog.ch_names:
            raw = mne.io.read_raw_edf(r, eog = [eye_chan_2], preload=True)
            print("e_used")
        elif 'VEOL' in find_eog.ch_names:
            raw = mne.io.read_raw_edf(r, eog = [eye_chan_3], preload=True)
            print("v_used")
        elif 'EOGH' in find_eog.ch_names:
            raw = mne.io.read_raw_edf(r, eog = [eye_chan_4], preload=True)
            print("combined_used")
        elif 'Fp1' in find_eog.ch_names:
            raw = mne.io.read_raw_edf(r, eog = [eye_chan_5], preload=True)
            print("fp_used")
    elif file == ".set":
        find_eog = mne.io.read_raw_eeglab(r)
        if 'HEOR' in find_eog.ch_names:
            raw = mne.io.read_raw_eeglab(r, eog = [eye_chan_1], preload=True)
            print("h_used")
        elif 'E1' in find_eog.ch_names:
            raw = mne.io.read_raw_eeglab(r, eog = [eye_chan_2], preload=True)
            print("e_used")
        elif 'VEOL' in find_eog.ch_names:
            raw = mne.io.read_raw_eeglab(r, eog = [eye_chan_3], preload=True)
            print("v_used")
        elif 'EOGH' in find_eog.ch_names:
            raw = mne.io.read_raw_eeglab(r, eog = [eye_chan_4], preload=True)
            print("combined_used")
        elif 'Fp1' in find_eog.ch_names:
            raw = mne.io.read_raw_eeglab(r, eog = [eye_chan_5], preload=True)
            print("fp_used")
    
    else:
      print("unable to determine file format and/or EOG parameters")
   
    raw.drop_channels(['FP1', 'FP2'])
    
    raw.set_montage(montage)
    
    # re-reference our EEG data to whether the file contains linked mastoids
    # (i.e., M1, M2) or TP9 and TP10.
    if 'M1' in raw.ch_names:
        raw = mne.io.set_eeg_reference(raw,ref_chan_1)[0]
    elif 'TP9' in raw.ch_names:
        raw = mne.io.set_eeg_reference(raw,ref_chan_2)[0]
        
    else:
      print("unable to determine reference")
      
    # downsample to 250 Hz
    raw = raw.resample(250)
    
    # apply a basic preprocessing step to the data i.e., filter from 1-40 Hz.
    raw = raw.filter(1, 40.,
                    l_trans_bandwidth='auto',
                     h_trans_bandwidth='auto',
                     filter_length='auto',
                     method='fir',
                     fir_window='hamming',
                     phase='zero',
                     n_jobs=2)

    picks = list()
    for e in electrodes:
        index = raw.ch_names.index(e)
        picks.append(index)
        
## ---------------------------------------------------------------------------
## IAF Estimation
## ---------------------------------------------------------------------------
    
    # get peak alpha frequency (paf) and centre of gravity (cog) estimates
    paf, cog, ablimits = savgol_iaf(raw, picks=picks, fmin=7, fmax=13)

    # write the paf and cog values into our .txt file that we created on
    # line 38 and save it to our working directory
    outfile.write(subj+"\t"+"paf\t"+str(paf)+"\n")
    outfile.write(subj+"\t"+"cog\t"+str(cog)+"\n")
    
    # finally, let's save the figure showing peak IAF for each subject
    plt.title('IAF at occipital-parietal channels' + ' for ' + subj)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectral Density ($uV^2$/Hz)')
    plt.savefig('iaf_figures/' + subj + '.png');
    plt.close();
    
## ---------------------------------------------------------------------------
## Aperiodic Estimation
## ---------------------------------------------------------------------------
    
    # set parameters for analysis
    data = raw.get_data()
    sf = raw.info['sfreq']
    chan = raw.ch_names
    
    # let's check our channel list, sampling frequency and length of the data
    print('Chan =', chan)
    print('Sampling frequency =', sf, 'Hz')
    print('Data shape =', data.shape)
    print('Duration =', data.shape[1] / sf, 'seconds')
    
    # now let's calculate the original power spectrum (PSD)
    from scipy.signal import welch
    
    win = int(4 * sf)                           # window size is 4 seconds
    freqs, psd = welch(data, sf, nperseg=win)   # single or multi-channel data
    
    print(freqs.shape, psd.shape)               # psd shape (n_chan, n_freq)
    
    # now we can plot our PSD - 19 refers to the channel we are selecting
    # here, 19 is Iz. This is a good choice because we can easily see 
    # the peak alpha power.
    plt.plot(freqs, psd[3, :], 'k', lw=2.5)
    plt.fill_between(freqs, psd[3, :], cmap='Spectral')
    plt.xlim(1, 30)
    plt.yscale('log')
    sns.despine()
    plt.title(chan[3])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD log($uV^2$/Hz)')   
    plt.savefig('irasa_figures/' + subj +'_PSD.png')
    plt.close();
    
    # apply the IRASA technique - this is the function to calculate
    # and separate the aperiodic from the true oscillatory activity
    freqs, psd_aperiodic, psd_osc = yasa.irasa(data, sf, ch_names=chan, 
                                               band=(1, 30), win_sec=4, 
                                               return_fit=False)
    
    # now let's plot the aperiodic component on a linear-log scale
    plt.plot(freqs, psd_aperiodic[3, :], 'k', lw=2.5)
    plt.fill_between(freqs, psd_aperiodic[3, :], cmap='Spectral')
    plt.xlim(1, 30)
    plt.yscale('log')
    sns.despine()
    plt.title('Aperiodic component at ' + chan[3])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD log($uV^2$/Hz)')
    plt.savefig('irasa_figures/' + subj + '_aperiodic.png')
    plt.close();
    
    # and oscillatory component on a linear-linear scale
    plt.plot(freqs, psd_osc[3, :], 'k', lw=2.5)
    plt.fill_between(freqs, psd_osc[3, :], cmap='Spectral')
    plt.xlim(1, 30)
    sns.despine()
    plt.title('Oscillatory component at ' + chan[3])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD log($uV^2$/Hz)')
    plt.savefig('irasa_figures/' + subj + '_oscillatory.png')
    plt.close();
    
    # finally, let's fit the fractal component (1/f)
    # here, we are fitting an exponential function to the aperiodic 
    # power spectrum and return the fit parameters (intercept, slope), the R^2 
    # of the fit, and the standard deviation of the oscillatory component.
    
    freqs, psd_aperiodic, psd_osc, fit_params = yasa.irasa(data, sf, 
                                                           ch_names=chan)
    fit_params
    
    # add an additional column to include subject number
    fit_params['subj'] = subj
    
    # save output file for each subject
    # add to dataframe 
    fit_params.to_csv('irasa_data/' + subj + '_aperiodic.csv', header = True)
    
    df_export = pd.DataFrame(fit_params)
    if op.isfile('aperiodic.csv'):
        df_export.to_csv('aperiodic.csv', sep=',', mode='a', header=False)
    else:
        df_export.to_csv('aperiodic.csv', sep=',', mode='a', header=True)

outfile.close()