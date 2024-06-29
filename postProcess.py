import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
from astropy.io import fits
from scipy.interpolate import interp1d
import pandas as pd

SolarRadius = 6.957e+10 #cm
Parsec = 3.086e+18 #cm

AllFiles = glob.glob('data/*.h5')
StellarParams = pd.read_csv("database/StellarParams.csv")

for FileItem in AllFiles:

    print("\n"*3)
    print ("Processing file: ", FileItem)
    # Open the HDF5 file
    groups = []
    data = h5py.File(FileItem, 'r')
    # Visit all the objects in the file
    #groups = f.visititems(get_all_groups) 

    AllTime =  []
    AllFlux = []
    FluxCube = []

    fig, ax = plt.subplots(figsize=(12,6), nrows=1, ncols=2)
    ax_twin = ax[0].twinx()
    
    for counter, group in enumerate(data.keys()):
        if counter==0:
            RefFilter = str(data[group]['Filter'][()])
            
            if 'G102' in RefFilter  :
                SensitivityCurve= fits.open("sensitivityCurves/WFC3.IR.G102.1st.sens.2.fits")
                WavelengthSelect = [8200, 11350]
            elif 'G141' in RefFilter:
                SensitivityCurve = fits.open("sensitivityCurves/WFC3.IR.G141.1st.sens.2.fits")
                WavelengthSelect = [11000, 16700]
            WLenSCurve = SensitivityCurve[1].data['WAVELENGTH']
            SCurveVal = SensitivityCurve[1].data['SENSITIVITY']/np.max(SensitivityCurve[1].data['SENSITIVITY'])
        else:
            Filter = str(data[group]['Filter'][()])
            assert Filter==RefFilter, "The filters are not the same"
            #for dset in data[group].keys():      
            #    print (dset)
            #input("Wait here...")
        Wavelength = data[group]['Wavelength'][:]
        FluxOpt = data[group]['Flux_Opt'][:]
        FluxBox = data[group]['Flux_Opt'][:]
        Date = data[group]['Time'][()]
        TimeStart = data[group]['ExposureStart'][()]
        TimeEnd = data[group]['ExposureEnd'][()]
        Filter = data[group]['Time'][()]

        AllTime.append((TimeStart+TimeEnd)/2.0)
        AllFlux.append(np.sum(FluxOpt))
        FluxCube.append(FluxOpt)
        
      
        ax[0].plot(Wavelength, FluxOpt)
        ax[0].plot(Wavelength, FluxBox)
    
    ax_twin.plot(WLenSCurve, SCurveVal, "r--")
    ax[0].set_xlabel("Wavelength")
    ax[0].set_ylabel("Flux")
    ax[0].set_title("Flux vs Wavelength")

    #Remove the first orbit

    AllTime = np.array(AllTime)
    AllFlux = np.array(AllFlux)
    FluxCube = np.array(FluxCube)

    ArrangeIndex = np.argsort(AllTime)
    
    AllTime = AllTime[ArrangeIndex]
    AllFlux = AllFlux[ArrangeIndex]
    FluxCube = FluxCube[ArrangeIndex]

    DiffTime = np.diff(AllTime)

    NumOrbits = np.sum(DiffTime>0.02083)+1
    Locations = np.where(DiffTime>0.02083)[0]
    
    LocationStart = np.append([0], Locations+1)
    LocationEnd = np.append(Locations, len(AllTime)-1)+1
    SelectIndex = np.zeros(len(AllTime)).astype(np.bool)
    
    
    MedianValues = []
    for LStart, LEnd in zip(LocationStart, LocationEnd):
        #Take the median of all the flux
        CurrentFluxMedian = np.nanmedian(FluxCube[LStart:LEnd])
        MedianValues.append(CurrentFluxMedian)

    MedianValues = np.array(MedianValues)

    sorted_vals = np.sort(MedianValues)[::-1]
    if len(sorted_vals)<2:
        print("The list of the sorted values is less than 2. Check why this is the case.")
        continue
    second_largest_val = sorted_vals[1]
    FirstCounter = np.argmax(MedianValues)
    SecondCounter = np.where(MedianValues == second_largest_val)[0][0]

    Counter = 0
    for LStart, LEnd in zip(LocationStart, LocationEnd):
        if Counter == FirstCounter or Counter==SecondCounter:
             SelectIndex[LStart:LEnd] = True
        Counter+=1
  
    
    
    ax[1].plot(AllTime[SelectIndex], AllFlux[SelectIndex], "ro")
    ax[1].plot(AllTime[~SelectIndex], AllFlux[~SelectIndex], "ko")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Flux")
    plt.tight_layout()
    plt.savefig("Figures/"+FileItem.split('/')[-1].replace('.h5', '_1.png'))
    plt.close('all')


    #Now add the flux to the plot
    SelectFlux = FluxCube[SelectIndex] 
    MedianFlux = np.median(SelectFlux, axis=0)
    FluxSTD = np.std(SelectFlux, axis=0)/np.sqrt(np.sum(SelectIndex))

    
    TargetName = FileItem.split('/')[-1].split("_")[0].replace('.h5', '').replace("-", "") 
    print(TargetName)
    StellarParamsIndex = StellarParams['Target']==TargetName
    
    assert np.sum(StellarParamsIndex)==1, "The target name is not unique"

    Factor = ((StellarParams['Radius'][StellarParamsIndex].values[0]*SolarRadius)/(StellarParams['Dist'][StellarParamsIndex].values[0]*Parsec))**2
    MedianFlux = MedianFlux*Factor
    FluxSTD = FluxSTD*Factor
   
   

    #Create interpolator
    InterpolatedCurve = np.interp(Wavelength, WLenSCurve, SCurveVal)     
    DataSelectIndex = (Wavelength>WavelengthSelect[0]) & (Wavelength<WavelengthSelect[1])
    
    Title = FileItem.split('/')[-1].replace('.h5', '')
    fig, ax = plt.subplots(figsize=(12,8), nrows=1, ncols=1)
    ax.errorbar(Wavelength[DataSelectIndex], MedianFlux[DataSelectIndex], yerr=FluxSTD[DataSelectIndex], marker='o', linestyle='None', color='k', capsize=2)
    ax.errorbar(Wavelength[~DataSelectIndex], MedianFlux[~DataSelectIndex], yerr=FluxSTD[~DataSelectIndex], marker='o', linestyle='None', color='r', capsize=2)
    ax.errorbar(Wavelength, MedianFlux/InterpolatedCurve, yerr=FluxSTD, marker='o', linestyle='None', color='b', capsize=2)
    ax_twin = ax.twinx()
    ax_twin.plot(WLenSCurve, SCurveVal, "r--")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")
    ax.set_title("Flux vs Wavelength")
    #ax.set_ylim('log')
    ax.set_title(Title)
    plt.tight_layout()
    plt.savefig("Figures/"+FileItem.split('/')[-1].replace('.h5', '_2.png'))
    plt.close()

    #Define the range of the data to be saved.
    SaveName = "ProcessedData/"+FileItem.split('/')[-1].replace('.h5', '.csv')

    RemoveIndex = np.logical_or(MedianFlux<0, np.isnan(MedianFlux))

    Wavelength = Wavelength[~RemoveIndex]
    MedianFlux = MedianFlux[~RemoveIndex]
    FluxSTD = FluxSTD[~RemoveIndex]
    

    #Save the data in the right format
    np.savetxt(SaveName, np.array([Wavelength, MedianFlux, FluxSTD]).T, delimiter=',', header='Wavelength,Flux,FluxErr',comments='')

