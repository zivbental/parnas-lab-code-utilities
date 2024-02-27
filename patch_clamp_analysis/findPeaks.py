

'''
This code goal is to take raw records from abf files and process them into graphs with appropriate statistcs
As an input, you need to provide a directory for the code to work on.
Its output would be excel file with all of the data variables stored in a convinient way, with corresponding graphs.
This code assumes that the directory contains .abf files with the following formula:
{recordID}_{protein-name}_{doseofTreatment+Ligandname}_{holding-potential}_{date}.abf
any deviation from this naming would probably result in an error.
'''

### Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pyabf
from scipy.signal import butter, lfilter

### Functions

# This function goal is to take the name of a given file and break it into meaningful variables
def breakName(fileName):
    splits = fileName.split('_')
    recordID = splits[0]
    proteinName = splits[1]
    dose = splits[2]
    holdPotantial = splits[3]
    recordDate = splits[4].split('.')[0]
    return [recordID, proteinName, dose, holdPotantial, recordDate]

# filters out strings not ending with '.abf'.
def filter_strings_by_extension(base_path):
    string_list = os.listdir(base_path)
    filtered_list = [s for s in string_list if s.endswith('.abf')]
    return filtered_list

def createRecordDF(files_path):
    listOfRecords = filter_strings_by_extension(filesPath) # List all the files in a given DF
    df = pd.DataFrame(columns = ['Record ID', 'Protein Name', 'Dose', 'Holding potential', 'Date', 'Response (pA)'])
    # record = listOfRecords[0]
    # recordGroups = breakName(record)
    # recordGroups.append(peakCurrent(record, 2.5))
    # df.loc[len(df)] = recordGroups
    # print(df)
    # Check
    
    for record in listOfRecords:
        recordGroups = breakName(record)
        recordGroups.append(peakCurrent(record, 2.8))
        df.loc[len(df)] = recordGroups
    print(df)

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Extract peak current from .abf file
def peakCurrent(filePath, timeOfPicosprizer):
    abf = pyabf.ABF(f'patch_clamp_analysis\\test_files\\{filePath}')
    abf.setSweep(0)
    sweep = abf.sweepY
    sweep_filtered = butter_lowpass_filter(sweep, 30, 10000, 5)
    # Takes the value of time = timeOfPicosprizer as a reference BEFORE picosprizer ligand application
    ref_val = sweep_filtered[int(timeOfPicosprizer*1000)]
    # Create numpy vector of the time following the ligand application (5 seconds) and substract all numbers by the reference value
    responseVector = np.asarray(sweep_filtered[int(timeOfPicosprizer*10000):int((timeOfPicosprizer+2)*10000)])
    deltaOfResponseVector = np.subtract(responseVector, ref_val)
    # Now i'll find the maximum value of absolute numbers in that vector. this is the place where I get the highest (or lowest) current in response to the ligand
    locOfPeakCurrent = np.where(np.absolute(deltaOfResponseVector) == np.max(np.absolute(deltaOfResponseVector)))
    peakCurrent = np.subtract(responseVector[locOfPeakCurrent], ref_val)
    # plt.plot(sweep_filtered)
    # plt.plot(responseVector)
    # plt.plot(locOfPeakCurrent, responseVector[locOfPeakCurrent], 'go')
    # plt.show()
    return peakCurrent[0]


### Main code
# Set the path of files to be analysed
filesPath = 'C:\\Users\\zivbe\\Documents\\codes\\parnas_codes\\patch_clamp_analysis\\test_files'
# List the files names of a given directory
createRecordDF(filesPath)




