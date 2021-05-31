from __future__ import print_function
import numpy as np
import sys
import os
import datetime
import pandas as pd
from pandas import Series, DataFrame
'''
TO run this python code:
    "python XYBuilder.py path forward/back"
for example:
    "python XYBuilder.py /Users/yushanzhang/Desktop/SWAN2 forward"

This code first counts datenumber# = m from getfilename(path_to_count) function, return how many total data sets we have

For xdt.txt, 3 (INPUT) +12*2 (WIND u, v)+357*2 (Currents u, v) = 741 columns;
the files are INPUT-YYYY_MM_DD-##.dat; WIND-YYYY_MM_DD-##.dat; Currents-YYYY_MM_DD-##.dat

For ydt.txt, 3105 columns; all outputs in Hsig and first data point for Tper
the files are Hsig_YYYY_MM_DD-##.comp; Tper_YYYY_MM_DD-##.comp
longitude:  237.5 to 238.211 and latitude: 36.5 to 36.98 with 0.01+; 72*49 = 3528;
excluding -999, 3104 Hsig + 1 Tper, so totally 3105 columns.
'''

def getfilename(path_to_count):
    # Count forecast# = m and ensembles# = n,
    #m is from counting how many Currents files in the folder and n is from counting how many INPUT-first date
    os.chdir(path_to_count)
    namelist = [name for name in os.listdir('.') if name[0:4] == "Hsig"]
    datenm = []
    for i in range(0, len(namelist)):
        datenm.append(namelist[i].replace('.','-').replace('_','-').split('-')[1]+str('_')+namelist[i].replace('.','-').replace('_','-').split('-')[2]+str('_')+namelist[i].replace('.','-').replace('_','-').split('-')[3]+str('-')+namelist[i].replace('.','-').replace('_','-').split('-')[4])
    print(len(datenm))
   # print(datenm)
    return datenm

def readX(thisdirectory, datenm):
    #read the outputs file and return the outputdt with all the Hsig, RTpeak, PkDir
    currents = np.loadtxt(thisdirectory+'Currents_Inputs/Currents-'+datenm[0]+'.dat', delimiter = ',').reshape(1, 850)
    m, n = (np.delete(currents, (np.where(currents[0,:] == -999)), axis = 1)).shape
    xdt = np.zeros((len(datenm),27+n))
    currents = np.zeros((len(datenm), 850))
    currents2 = np.zeros((len(datenm), n))
    for datelen in range(0, len(datenm)):
        with open(thisdirectory+'SWAN_Inputs/INPUT-'+datenm[datelen]+'.dat') as f:
            filelist1 = f.readlines()[18:19]
            for line1 in filelist1:
                this_line1 = line1.split()
                xdt[datelen][0] = this_line1[9]    #first column Hsig
                xdt[datelen][1] = this_line1[10]  #second column RTpeak
                xdt[datelen][2] = this_line1[11]  #thrid colu,n PkDir
        f.close()
        xdt[datelen][3:27] = np.loadtxt(thisdirectory+'Winds_Inputs/WIND-'+datenm[datelen]+'.dat', delimiter = ',').reshape(1,24)
        currents[datelen][:] = np.loadtxt(thisdirectory+'Currents_Inputs/Currents-'+datenm[datelen]+'.dat', delimiter = ',').reshape(1,850)
        currents2 = np.delete(currents, (np.where(currents[datelen, :] == -999)),axis = 1)
    xdt[:, 27:n+27] = currents2
    print(currents2)
    np.savetxt(thisdirectory+'xdt.txt', xdt, fmt = '%5.4f', delimiter = '\t')
    return xdt


def readY(thisdirectory, datenm):
    hsigtemp = np.genfromtxt(thisdirectory+'SWAN_Outputs/Hsig_'+datenm[0]+'.comp')#, usecols = [0,1,2,3,4,5], invalid_raise = False)
    hsigtemp = hsigtemp.reshape(1, len(hsigtemp)*6)
    hsigtemp2 = np.delete(hsigtemp, (np.where(hsigtemp[:] == -9)), axis = 1)
    m, n = hsigtemp2.shape
    print(m, n+1)
    ydt = np.zeros((len(datenm), (1+n)+1))
    for datelen in range(0, len(datenm)):
        hsigtemp = np.genfromtxt(thisdirectory+'SWAN_Outputs/Hsig_'+datenm[datelen]+'.comp')
        hsigtemp = hsigtemp.reshape(1, len(hsigtemp)*6)
        ydt[datelen,1:n+1] = np.delete(hsigtemp, (np.where(hsigtemp[:] == -9)), axis = 1)
        ydt[datelen][0] = hsigtemp[0][0]

    for datelen in range(0, len(datenm)):
        tpertemp = np.genfromtxt(thisdirectory+'SWAN_Outputs/Tper_'+datenm[datelen]+'.comp')
        ydt[datelen][n+1] = tpertemp[0][0]  #only first period
#        tpertemp = tpertemp.reshape(1, len(tpertemp)*6)          #if taken all the period
#        ydt[datelen,n+2:2*(n+1)] = np.delete(tpertemp, (np.where(tpertemp[:] == -9)), axis = 1)
    print(ydt)
    np.savetxt(thisdirectory+'ydt.txt', ydt, fmt = '%5.4f', delimiter = '\t')
    return ydt


def renewXY(xdt, ydt):
    ## I'm not sure why this doesn't work
    ## -999 should denote missing values in currents data
    ## (i.e. values collected over land ).
    ## That isn't the case though so can just return
    row, col = (np.where(xdt[:, :] == -999))
    dataindex = []
    if (len(row) == 0): return dataindex

    for i in range(0, len(row)/714):
        dataindex.append(row[i*714])
    print(dataindex)
    return dataindex

def renewX(dataindex, xdt):
    xdt2 = np.delete(xdt, dataindex, axis = 0)
    np.savetxt(thisdirectory+'xdt1.txt', xdt2, fmt = '%5.4f', delimiter = '\t')

    ydt = np.loadtxt(thisdirectory+'ydt.txt')
    ydt2 = np.delete(ydt, dataindex, axis = 0)
    np.savetxt(thisdirectory+'ydt1.txt', ydt2, fmt = '%5.4f', delimiter = '\t')


def runmain():
    global thisdirectory
    thisdirectory = os.getcwd() + '/'
    print(thisdirectory)
    #foldername = 'Wind_trial'
    datenm = getfilename(thisdirectory + 'SWAN_Outputs/')
    xdt = readX(thisdirectory, datenm)
    ydt = readY(thisdirectory, datenm)
    dataindex = renewXY(xdt, ydt)
    renewX(dataindex, xdt)



runmain()
