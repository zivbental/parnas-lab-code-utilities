

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
        recordGroups.append(peakCurrent(record,files_path, 2.8))
        df.loc[len(df)] = recordGroups
    print(df)

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
def peakCurrent(fileName, filePath, timeOfPicosprizer):
    abf = pyabf.ABF(f'{filePath}\\{fileName}')
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
    # If I take the location of the peak current, I might get artifacts that result from noise. Therefore, i'll average an area of 30 sample from each side of this location and this will be my peak current value
    windowSize = 500 # Samples from each side
    startIndex = max(0, locOfPeakCurrent[0][0] - windowSize)
    endIndex = min(len(responseVector), locOfPeakCurrent[0][0] + windowSize + 1)
    averageOfResponse = np.average(responseVector[startIndex:endIndex])
    peakCurrent = np.subtract(averageOfResponse, ref_val)
    locOfPeakCurrent[0][0] += (timeOfPicosprizer)*10000
    plotTrace(sweep, sweep_filtered, locOfPeakCurrent[0][0], averageOfResponse, fileName)
    return peakCurrent

def plotTrace(sweep, sweep_filtered, locOfPeakCurrent, averageOfResponse, fileName):
    # Plot raw trace
    plt.plot(sweep)
    # Plot filtered trace
    plt.plot(sweep_filtered)
    # Plot the location of peak current
    plt.plot(locOfPeakCurrent, averageOfResponse, 'go')
    # Get information about the file
    recordData = breakName(fileName)
    plt.title(f'{recordData}')
    plt.ylabel('I(pA)')
    plt.xlabel('Time(traces)')
    plt.show()


### Main code
# Set the path of files to be analysed
filesPath = 'C:\\Users\\zivbe\\Documents\\codes\\parnas_codes\\patch_clamp_analysis\\test_files'
# Set where to save the results data and graphs
resultsPath = ''
# Perform analysis to the files within this folder
createRecordDF(filesPath)




