# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Toolbox to process EEG data
Software toolbox for processing EEG data from the TUH EEG corpus
Results used in:

David O. Nahmias, Kimberly L. Kontson, David A. Soltysik, and F. Civillico, Eugene.
Consistency of quantitative electroencephalography features in a large clinical data set. Journal of Neural Engineering, 16(066044), 2019.

If you have found this software useful please consider citing our publication.

Public domain license
"""

""" Disclaimer:
This software and documentation (the "Software") were developed at the Food and Drug Administration (FDA) by employees
of the Federal Government in the course of their official duties. Pursuant to Title 17, Section 105 of the United States Code,
this work is not subject to copyright protection and is in the public domain. Permission is hereby granted, free of charge,
to any person obtaining a copy of the Software, to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, or sell copies of the Software or derivatives,
and to permit persons to whom the Software is furnished to do so. FDA assumes no responsibility whatsoever for use by other
parties of the Software, its source code, documentation or compiled executables, and makes no guarantees, expressed or implied,
about its quality, reliability, or any other characteristic. Further, use of this code in no way implies endorsement by the FDA
or confers any advantage in regulatory decisions. Although this software can be redistributed and/or modified freely, we ask that
any derivative works bear some notice that they are derived from it, and any modified versions bear some notice that they have been modified.
"""

__author__ = 'David Nahmias'
__copyright__ = 'No copyright - US Government, 2020 , EEG processing toolbox'
__credits__ = ['David Nahmias']
__license__ = 'Public domain'
__version__ = '0.0.1'
__maintainer__ = 'David Nahmias'
__email__ = 'david.nahmias@fda.hhs.gov'
__status__ = 'alpha'

"""
Created on Wed Nov 23 10:20:33 2016

@author: David.Nahmias
"""
#import matlab.engine
import sys
import numpy as np
from scipy import fftpack
from scipy import signal
from scipy import stats
import pywt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import itertools
import mne
import time
import datetime
from string import punctuation
import glob
from multiprocessing import Pool
import heapq
#import caffe
#import lmdb
import warnings
import pdb
from lempel_ziv_complexity import lempel_ziv_complexity
import resampy
#from multiprocessing.dummy import Pool as ThreadPool 
#ENG = matlab.engine.start_matlab('-nojvm')
#import tensorflow as tf

#DIRECTORY = '/media/david/FA324B89324B4A39/'
#DIRECTORY = '/media/david/WD 2TB EXT/EEGcorpus/'
#DIRECTORY = '/media/david/Data1/EEGCorpus06/'
DIRECTORY = '/media/david/Data1/NEDC/tuh_eeg/v1.0.0/edf/'


class Subject:
    def __init__(self):
        self.name = 0
        self.session = 0
        self.male = 0
        self.female = 0
        self.noGender = 0
        self.goodAge = 0
        self.noAge = 0
        self.age = []
        self.medFound = 0
        self.goodEDF = 0
        self.eegSysFound = 0
        self.normalEEG = 0
        self.abnormalEEG = 0
        self.noImpressionEEG = 0
        self.valueErrorCount = 0
        self.noEEGerr = 0
        self.keyCount = np.zeros([5,1])# [0,0,0,0,0]
        self.keywords = []
        self.specialCount = 0 #Keep count
        self.specialSubj = [] #Keep list
        self.genderNP = np.zeros([3,1])
        self.subjEdfFiles = []
        self.singleTXTdata = []
        self.subjEEG = []
        self.prevSubjName = 0
        self.subjGender = ''
        self.subjAge = 0
        self.subjMed = []
        self.subjNormalState = 2
        self.fileName = ''
        self.edfFiles = []
        self.date = []



#eng = matlab.engine.start_matlab('-nojvm')
def scatterEEGdataMATLAB(data,eng):
    matData = matlab.double(data.transpose().tolist())
    matScatter = eng.scattering(matData)
    scatter = np.array(matScatter._data.tolist())
    scatter = scatter.reshape(matScatter.size).transpose()
    return scatter

def scatterEEGdata(sn,data):
    scatter = sn.transform(data)
    return scatter

def toNumeric(numWord):
    ''' 
    Given the textual form of a number, return the numerical form of the number. 
    @PARAM: numWord - string containing number's textual form using 
        hyphens (i.e. "seventy-two" not "seventy two") 
    '''
    numeric = 0
    numbers = {"one":1, "two":2, "three":3, "four":4, "five":5, "six":6, "seven":7,
               "eight":8, "nine":9, "ten":10, "eleven":11, "twelve":12, "thirteen":13, 
               "fourteen":14, "fifteen":15, "sixteen":16, "seventeen":17, "eighteen":18,
               "nineteen":19, "twenty":20, "thirty":30, "forty":40, "fifty":50, "sixty":60, 
               "seventy":70, "eighty":80, "ninety":90, "hundred":100}
    numParts = numWord.split('-') 
    for part in range(len(numParts)):
        try:
            numeric += numbers.get(numParts[part])
        except TypeError: 
            numeric = 0
            break
    return numeric

def readEDF(filenameEDF):
    valueErrorCount = ''
    #filenameEDF = filenameEDF.split('/')[-1]
    edfFileStr = filenameEDF.split('_t')[1]
    if edfFileStr.split('.')[0].isdigit():
        edfFileNum = edfFileStr.split('.')[0]
    else:
        edfFileNum = '99'
        print('No EDF file number found.')
    try:
        eeg = mne.io.read_raw_edf(filenameEDF,preload=True,verbose='CRITICAL')
    except ValueError:
        eeg = []
        valueErrorCount += 'valueError'
    except RuntimeError:
        eeg = []
        valueErrorCount += 'runtimeError'
    except MemoryError:
        eeg = []
        valueErrorCount += 'memoryError'

    name = filenameEDF.split('/')[8]
    session = filenameEDF.split('/')[9].split('s')[1]
    textFile = filenameEDF.split('_t')[0]+'.txt'

    return eeg,valueErrorCount

def readNotes(filenameNotes):
    f = open(filenameNotes,'rU') #new line on /r and /n
    age = ''
    gender =''
    medication = []
    eegSys = ''
    normalState = 2
    normalStateFound = 0
    ageFound = 0
    keywords =['notFound','notFound','notFound','notFound','notFound'] #number of KW
    lines = f.readlines()
    for i in range(len(lines)):
        #Get Keywords
        keyword1 = 0
        keyword2 = 0
        keyword3 = 0
        keyword4 = 0
        keyword5 = 0
        prevPrevWord = ''
        prevWord = ''
        for word in lines[i].split():
            word = word.rstrip(punctuation)
            if (word[0:6].lower() == 'epilep') and (keyword1 == 0):
                keywords[0] = 'epilepsy'
                keyword1 = 1
            if (word[0:6].lower() == 'seizur') and (keyword2 == 0):
                keywords[1] = 'seizure'
                keyword2 = 1
            if (word[0:7].lower() == 'concuss') and (keyword3 == 0):
                keywords[2] = 'concussion'
                keyword3 = 1
            if (word[0:6].lower() == 'tremor') and (keyword4 == 0):
                keywords[3] = 'tremor'
                keyword4 = 0
            if (word[-3:].lower() == 'tbi') and (keyword5 == 0):
                keywords[4] = 'tbi'
                keyword5 = 1
            if (prevPrevWord[0:4] != 'anox') and (prevPrevWord[0:4] != 'diff') and (prevWord[0:5].lower() == 'brain') and (word[0:5].lower() == 'injur') and (keyword5 == 0):
                keywords[4] = 'tbi'
                keyword5 = 1
            if (prevWord[0:5].lower() == 'traum') and (word[0:5].lower() == 'brain') and (keyword5 == 0):
                keywords[4] = 'tbi'
                keyword5 = 1
            prevPrevWord = prevWord
            prevWord = word

        curLineComp = lines[i].replace(';',':').split(':')[0].strip().upper()
        #Get Age
        if (curLineComp == 'CLINICAL HISTORY') or ('HISTORY' in curLineComp) or (curLineComp.split(' ')[0] == 'CLINICAL'):
            l = 0 #letters in line
            filteredLines = filter(None, lines[i].rstrip().replace(';',':').split(':')) 
            j=i #use current line
            if len(filteredLines) == 1:
                if j < (len(lines)-1):
                    j=i+1 #go to next line if no ; or :
            if ageFound == 0:
                #when age in numerical form
                while not lines[j][l].isdigit():
                    l += 1
                    if l == len(lines[j]):
                        break
                if l != len(lines[j]):  
                    while lines[j][l].isdigit():
                        age += lines[j][l]
                        l += 1
                if len(age) > 0:
                    if (int(age) > 0) and (int(age) < 120):
                        ageFound = 1
            if ageFound == 0:
                #when age in textual form
                words = lines[j].lower().split()
                trigger = 0
                counter = 0
                while counter < len(words):
                    if "year" in words[counter]: 
                        trigger = 1
                        break
                    counter += 1
                if trigger == 1: 
                    currentIndex = 0
                    nextIndex = 1
                    while nextIndex < len(words[counter]):
                       if ((words[counter][currentIndex] == '-') and (words[counter][nextIndex] == 'y')):
                          break
                       currentIndex += 1
                       nextIndex += 1
                    age = str(toNumeric(words[counter][0:currentIndex])) 
		    if len(age) > 0:
                          if (int(age) > 0) and (int(age) < 120):
                              ageFound = 1
            #Get Gender
            for word in lines[j].split():#replaced ' '
                wordComp = word.rstrip(punctuation).lower()
                if (wordComp == 'male') or (wordComp == 'man') or (wordComp == 'm') or (wordComp == 'gentleman') or (wordComp == 'gentlemen') or (wordComp == 'boy'):
                    gender += 'male'
                elif (wordComp == 'female') or (wordComp == 'woman') or (wordComp == 'f') or (wordComp == 'lady') or (wordComp == 'girl'):
                    gender += 'female'
        #Get Medications
        elif (curLineComp == 'MEDICATIONS') or (curLineComp == 'MEDICATION'):
            if len(lines[i].replace(';',':').split(':')) == 1:
                if i+1 < len(lines):
                    j=i+1 #go to next line if no ; or :
                else:
                    j=i
            else:
                j=i #use current line
            for word in lines[j].split():#replaced ' '
                wordComp = word.lower().rstrip(punctuation).strip().rstrip('.')
                #pdb.set_trace()

                if (wordComp[0:10] != 'medication') and (wordComp[0:3] != 'and') and (len(wordComp) > 2):
                    if wordComp == 'acid':
                        medication[-1] += '-acid'
                    else:
                        medication.append(wordComp)
        #Get EEG System
        elif (curLineComp == 'INTRODUCTION'):
            if len(lines[i].replace(';',':').rstrip().split(':')) == 1:
                if i+1 < len(lines):
                    j=i+1 #go to next line if no ; or :
                else:
                    j=i            
            else:
                j=i #use current line
            for word in lines[j].split():#replaced ' '
                wordComp = word.rstrip('.,:').lower()
                if (wordComp == '10-20') or (wordComp == '10/20'):
                    eegSys = '10-20'
        
        #Get Normal/Abnormal, 0-normal, 1-abnormal, 2-not found
        elif (curLineComp == 'IMPRESSION'):
            #pdb.set_trace()
            if len(lines[i].replace(';',':').rstrip().split(':')) == 1:
                if i+1 < len(lines):
                    j=i+1 #go to next line if no ; or :
                else:
                    j=i
            elif len(lines[i].replace(';',':').rstrip().split(':')) == 2:
                if lines[i].replace(';',':').rstrip().split(':')[1] == '':
                    if i+1 < len(lines):
                        j=i+1 #go to next line if nothing after :
                    else:
                        j=i
                else:
                    j=i
            else:
                j=i #use current line

            counterW = 0
            for word in lines[j].split():#replaced ' '
                if normalStateFound == 0:
                    wordComp = word.rstrip().rstrip('.,:').lower()
                    if (wordComp == 'normal') or (wordComp == 'normality'):
                        normalState = 0
                        normalStateFound = 1
                    elif (wordComp == 'abnormal') or (wordComp == 'abnormality'):
                        if lines[j].split(' ')[counterW-1].rstrip().rstrip('.,:').lower() == 'no':
                            normalState = 0
                            normalStateFound = 1
                        else:
                            normalState = 1
                            normalStateFound = 1
                counterW += 1
    if age=='':
        age=0
    return gender,int(age),medication,eegSys,keywords,normalState
    
def preProcessEEG(eegR):    
    channelsUni = eegR.ch_names
    #print channelsUni
    #channelNames = [x.encode('UTF8') for x in channelsUni]
    tenTwenty = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','FZ','CZ','PZ']
    chNamesTT = ['EEG FP1','EEG FP2','EEG F3','EEG F4','EEG C3','EEG C4','EEG P3','EEG P4','EEG O1','EEG O2','EEG F7','EEG F8','EEG T3','EGG T4','EEG T5','EEG T6','EEG FZ','EEG CZ','EEG PZ']
    #Find EEG Channels
    eegChannelNum = []
    eegChannelNames = []
    for eegCh in tenTwenty:
        chNum = 0
        for ch in channelsUni:
            if eegCh in ch:
                eegChannelNum.append(chNum)
                eegChannelNames.append(eegCh)
                break
            chNum += 1

    #Filter    
    eegR.filter(l_freq=0.5, h_freq=40.0,verbose='CRITICAL')  # band-pass filter data
    
    #Automatic? But includes non-EEG signals
    #eegR.set_eeg_reference() 
    #refEEGmean = np.mean(refEEGdata,axis=0)

    #Rereference only EEG data
    data, times = eegR[:]
    srate = 1/(times[1]-times[0]) #time in seconds => Hz
    #print 'Sampled at:',srate
    numSamples = np.size(data,axis=1)
    eegData = np.zeros([len(eegChannelNum),numSamples])    
    refEEGdata = np.zeros([len(eegChannelNum),numSamples])
    #EEGdata = np.zeros([len(eegChannelNum),int(np.ceil(numSamples*(250/srate)))])
    desiredSRate = 80
    ch=0
    for i in eegChannelNum:
        eegData[ch,:] = data[i,:]
        ch+=1

    timeAvg = np.mean(eegData,axis=0)
    for j in range(numSamples):
        refEEGdata[:,j] = eegData[:,j] - timeAvg[j]
    #print 'Before:',np.size(refEEGdata),'srate:',srate
    if srate != desiredSRate:
        eegData = np.zeros([len(eegChannelNum),int(np.ceil(numSamples*(desiredSRate/srate)))])
        for k in range(np.size(refEEGdata,0)):
            eegData[k,:] = signal.resample(refEEGdata[k,:],int(np.ceil(numSamples*(desiredSRate/srate))))
        srate = desiredSRate
        #print 'After:',np.size(eegData),'srate:',srate
        return eegData,eegChannelNames,srate
    else:
        return refEEGdata,eegChannelNames,srate

    #ch=0
    #for k in range(len(eegChannelNum)):
    #    data[k,:] = refEEGdata[ch,:]
    #    ch+=1
        
    #eegR[:] = data
    #eegR[0:np.size(refEEGdata,axis=0)] = refEEGdata
    #eegR.ch_names = chNamesTT

    #Find ICA    
    #ica = mne.preprocessing.ICA(n_components=0.95, method='fastica')
    #picks = mne.pick_types(eegR.info, meg=False, eeg=True, eog=False,
    #                   stim=False, exclude='bads')
    #print picks
    #ica.fit(eegR, picks=picks, decim=3, reject=dict(grad=4000e-13))

    #return eegData,eegChannelNames,srate #eegR,channelsUni,eegChannelNum,

def getCurveLength(data,numPart=1):
    channelNum = np.shape(data)[0]
    samples = np.shape(data)[1]
    N = int(np.floor(samples/numPart))
    D = 0
    CL = np.zeros([channelNum,numPart])
    for ch in range(channelNum):
        for n in range(0,numPart):
            startInd = (n*(N-D))
            endInd = (n+1)*(N-D)-1 + D
            cumsumCL = 0
            for i in range(startInd,endInd-1):
                cumsumCL += np.abs(data[ch,i]-data[ch,i+1])
            CL[ch,n] = cumsumCL
    return CL

def getEnergy(data,numPart=1):
    channelNum = np.shape(data)[0]
    samples = np.shape(data)[1]
    N = int(np.floor(samples/numPart))
    D = 0
    E = np.zeros([channelNum,numPart])
    for ch in range(channelNum):
        for n in range(0,numPart):
            startInd = (n*(N-D))
            endInd = (n+1)*(N-D)-1 + D
            cumsumE = 0
            for i in range(startInd,endInd):
                cumsumE += data[ch,i]**2
            E[ch,n] = 1./N*(cumsumE)
    return E

def getNonlinearEnergy(data,numPart=1):
    channelNum = np.shape(data)[0]
    samples = np.shape(data)[1]
    N = int(np.floor(samples/numPart))
    D = 0
    NE = np.zeros([channelNum,numPart])
    for ch in range(channelNum):
        for n in range(0,numPart):
            startInd = (n*(N-D))
            endInd = (n+1)*(N-D)-1 + D
            cumsumNE = 0
            for i in range(startInd+1,endInd-1):
                cumsumNE += data[ch,i]**2-(data[ch,i-1]*data[ch,i+1])
            NE[ch,n] = 1./N*(cumsumNE)
    return NE  

def getSpectralEntropyFirst(data,numPart=1):
    channelNum = np.shape(data)[0]
    samples = np.shape(data)[1]
    N = int(np.floor(samples/numPart))
    D = 0
    SE = np.zeros([channelNum,numPart])
    for ch in range(channelNum):
        dataDFT = np.fft.fft(data[ch,:])
        for n in range(0,numPart):
            startInd = (n*(N-D))
            endInd = (n+1)*(N-D)-1 + D
            cumsumSE = 0
            for i in range(startInd,endInd-1):
                cumsumSE += dataDFT[i]*np.log2(dataDFT[i])
            #pdb.set_trace()
            SE[ch,n] = abs(-cumsumSE)
    return SE

def getSpectralEntropy(data,numPart=1):
    channelNum = np.shape(data)[0]
    samples = np.shape(data)[1]
    N = int(np.floor(samples/numPart))
    D = 0
    SE = np.zeros([channelNum,numPart])
    for ch in range(channelNum):
        dataDFT = np.fft.fft(data[ch,:])
        for n in range(0,numPart):
            startInd = (n*(N-D))
            endInd = (n+1)*(N-D)-1 + D
            cumsumSE = 0
            for i in range(startInd,endInd-1):
                specPow = (1./N)*np.square(dataDFT[i])
                cumsumSE += specPow*np.log2(specPow)
            #pdb.set_trace()
            SE[ch,n] = -np.linalg.norm(cumsumSE)
    return SE

def getSixPower(data,numPart=1):
    channelNum = np.shape(data)[0]
    samples = np.shape(data)[1]
    N = int(np.floor(samples/numPart))
    D = 0
    SP = np.zeros([channelNum,numPart])
    for ch in range(channelNum):
        for n in range(0,numPart):
            startInd = (n*(N-D))
            endInd = (n+1)*(N-D)-1 + D
            cumsumSP = 0
            for i in range(startInd,endInd):
                cumsumSP += data[ch,i]**6
            SP[ch,n] = 1./N*(cumsumSP)
    return SP

def getSecondThirdFeat(data,numPart=1):
    channelNum = np.shape(data)[0]
    samples = np.shape(data)[1]
    mini = np.zeros([channelNum,numPart])
    maxi = np.zeros([channelNum,numPart])
    medi = np.zeros([channelNum,numPart])
    mean = np.zeros([channelNum,numPart])
    var = np.zeros([channelNum,numPart])
    std = np.zeros([channelNum,numPart])
    skew = np.zeros([channelNum,numPart])
    kurt = np.zeros([channelNum,numPart])
    #slope = np.zeros(channelNum)
    inte = np.zeros([channelNum,numPart])
    #deriv = np.zeros(channelNum)
    sumi = np.zeros([channelNum,numPart])
    N = int(np.floor(samples/numPart))
    D = 0
    for ch in range(channelNum):
        for n in range(0,numPart):
            startInd = (n*(N-D))
            endInd = (n+1)*(N-D)-1 + D
            mini[ch,n] = np.min(data[ch,startInd:endInd])
            maxi[ch,n] = np.max(data[ch,startInd:endInd])
            medi[ch,n] = np.median(data[ch,startInd:endInd])
            mean[ch,n] = np.mean(data[ch,startInd:endInd])
            var[ch,n] = np.var(data[ch,startInd:endInd])
            std[ch,n] = np.std(data[ch,startInd:endInd])
            skew[ch,n] = stats.skew(data[ch,startInd:endInd])
            kurt[ch,n] = stats.kurtosis(data[ch,startInd:endInd])
            inte[ch,n] = np.trapz(data[ch,startInd:endInd])
            sumi[ch,n] = np.sum(data[ch,startInd:endInd])

        #np.array(np.concatenate((mini,maxi,medi,mean,var,std,skew,kurt,inte,sumi),axis=0))
    return mini,maxi,medi,mean,var,std,skew,kurt,inte,sumi

def getWaveletTransform(data,numPart=1,family='db4',lev=5):
    channelNum = np.shape(data)[0]
    samples = np.shape(data)[1]
    waveA = np.zeros([lev,channelNum,numPart])
    waveD = np.zeros([lev,channelNum,numPart])
    N = int(np.floor(samples/numPart))
    D = 0
    for ch in range(channelNum):
        for n in range(0,numPart):
            startInd = (n*(N-D))
            endInd = (n+1)*(N-D)-1 + D
            #cA,cD = pywt.dwt(data[ch,startInd:endInd], family)
            #wave = pywt.wavedec(data[ch,startInd:endInd],family,mode='symmetric',level=lev)
            wave = pywt.WaveletPacket(data=data[ch,startInd:endInd], wavelet=family, mode='symmetric')
            #waveA[:,ch,n] = heapq.nlargest(10,np.abs(cA))
            #waveD[:,ch,n] = heapq.nlargest(10,np.abs(cD))
            #waveA[:,ch,n] = cA[np.argpartition(np.abs(cA),-lev)[-lev:]]
            #waveD[:,ch,n] = cD[np.argpartition(np.abs(cD),-lev)[-lev:]]
            #pdb.set_trace()


    return waveA,waveD

def getLZC(data,numPart=1):
    channelNum = np.shape(data)[0]
    samples = np.shape(data)[1]
    lzc = np.zeros([channelNum,numPart])
    N = int(np.floor(samples/numPart))
    D = 0
    for ch in range(channelNum):
        for n in range(0,numPart):
            startInd = (n*(N-D))
            endInd = (n+1)*(N-D)-1 + D
            binData = convertBinarization(data[ch,startInd:endInd])
            lzc[ch,n] = lempel_ziv_complexity(binData)

    return lzc

def convertBinarization(data):
    if data.ndim == 1:
        thresh = np.mean(data)
        binarizationData = data >= thresh
    
    elif data.ndim == 2:
        channelNum = np.shape(data)[0]
        binarizationData = np.zeros(np.shape(data))
        for ch in range(channelNum):
            thresh = np.mean(data[ch,:])
            binarizationData[ch,:] = data[ch,:] >= thresh

    binDataInt = binarizationData.astype(np.int)

    return binDataInt


def sampleEntropy(signal,m=5,r=0.2,tau=1):
    #Using m-1 for B and m for A rather than m and m+1.
    r=0.2*np.std(signal)
    B = 0
    A = 0
    N = len(signal)
    for i in range(N-m):
        Bi = 0 
        Ai = 0
        for j in range(N-m):
            if (np.max(np.abs(np.subtract(signal[range(i,i+((m-1)*tau),tau)],signal[range(j,j+((m-1)*tau),tau)])))<=r) and (i!=j):
                Bi += 1
            if (np.max(np.abs(np.subtract(signal[range(i,i+(m*tau),tau)],signal[range(j,j+(m*tau),tau)])))<=r) and (i!=j):
                Ai += 1
        B += Bi/(N-m-1)
        A += Ai/(N-m-1)
    B = B/(N-m)
    A = A/(N-m)
    sampEnt = -np.log(np.divide(A,B))
    
    pdb.set_trace()
    return sampEnt

def svdEntropy(signal,m,tau=1):
    N = len(signal)
    X = np.zeros([N-(m-1),m])
    for i in range(N-(m-1)):
        X[i,:] = signal[range(i,i+(m*tau),tau)]

    S = np.linalg.svd(X,full_matrices=True,compute_uv=False)
    sigmaNormed = np.divide(S,np.sum(S))
    svdEnt = -np.sum(np.multiply(sigmaNormed,np.log2(sigmaNormed)))

    return svdEnt

def lempel_ziv_complexityLOCAL(binary_sequence):
    """ Manual implementation of the Lempel-Ziv complexity.
    It is defined as the number of different substrings encountered as the stream is viewed from begining to the end.
    As an example:
    >>> s = '1001111011000010'
    >>> lempel_ziv_complexity(s)  # 1 / 0 / 01 / 1110 / 1100 / 0010
    6
    Marking in the different substrings the sequence complexity :math:`\mathrm{Lempel-Ziv}(s) = 6`: :math:`s = 1 / 0 / 01 / 1110 / 1100 / 0010`.
    - See the page https://en.wikipedia.org/wiki/Lempel-Ziv_complexity for more details.
    Other examples:
    >>> lempel_ziv_complexity('1010101010101010')  # 1 / 0 / 10
    3
    >>> lempel_ziv_complexity('1001111011000010000010')  # 1 / 0 / 01 / 1110 / 1100 / 0010 / 000 / 010
    7
    >>> lempel_ziv_complexity('100111101100001000001010')  # 1 / 0 / 01 / 1110 / 1100 / 0010 / 000 / 010 / 10
    8
    - Note: it is faster to give the sequence as a string of characters, like `'10001001'`, instead of a list or a numpy array.
    - Note: see this notebook for more details, comparison, benchmarks and experiments: https://Nbviewer.Jupyter.org/github/Naereen/Lempel-Ziv_Complexity/Short_study_of_the_Lempel-Ziv_complexity.ipynb
    - Note: there is also a Cython-powered version, for speedup, see :download:`lempel_ziv_complexity_cython.pyx`.
    """
    u, v, w = 0, 1, 1
    v_max = 1
    length = len(binary_sequence)
    complexity = 1
    while True:
        if binary_sequence[u + v - 1] == binary_sequence[w + v - 1]:
            v += 1
            if w + v >= length:
                complexity += 1
                break
        else:
            if v > v_max:
                v_max = v
            u += 1
            if u == w:
                complexity += 1
                w += v_max
                if w > length:
                    break
                else:
                    u = 0
                    v = 1
                    v_max = 1
            else:
                v = 1
    return complexity

def getSmallPSDparts(data,bands,srate=250,numPart=1):
    channelNum = np.shape(data)[0]
    samples = np.shape(data)[1]
    #bands = range(0,40,1)#[0,1,4,8,12,16,25,40]#

    numBands = len(bands)
    N = int(np.floor(samples/numPart))
    D = 0

    powers = np.zeros([channelNum,numPart,numBands-1])
    for ch in range(channelNum): #loop through channels
        for n in range(0,numPart):
            startInd = (n*(N-D))
            endInd = (n+1)*(N-D)-1 + D
            f, Pxx_den = signal.periodogram(np.reshape(data[ch,startInd:endInd],-1), srate)
            pow_array = []
            for b in range(numBands-1):
                lowerBound = f>=bands[b]
                upperBound = f<bands[b+1]

                band_ind = np.logical_and(lowerBound,upperBound)

                tot_pow = np.sum(Pxx_den)
                band_pow = np.sum(Pxx_den[band_ind])/tot_pow
            
                pow_array.append(band_pow)

            powers[ch][n][:] = pow_array

    return powers

def getSmallPSD(data,bands=[1,4,4,8,8,12,12,16,16,20,25,40],srate=250):
    channelNum = np.size(data,axis=0)
    #bands = range(0,41,1)
    numBands = len(bands)

    powers = np.zeros([channelNum,numBands-1])
    for ch in range(channelNum): #loop through channels
        f, Pxx_den = signal.periodogram(np.reshape(data[ch],-1), srate)
        pow_array = []
        for b in range(numBands-1):
            lowerBound = f>=bands[b]
            upperBound = f<bands[b+1]

            band_ind = np.logical_and(lowerBound,upperBound)

            tot_pow = np.sum(Pxx_den)
            band_pow = np.sum(Pxx_den[band_ind])/tot_pow
        
            pow_array.append(band_pow)

        powers[ch][:] = pow_array

    return powers

def getPSD(data,srate=250):
    channelNum = np.size(data,axis=0)
    numBands = 7
    bands = [1,4,4,8,8,12,12,16,16,20,25,40]

    powers = np.zeros([channelNum,numBands])
    for ch in range(channelNum): #loop through channels
        f, Pxx_den = signal.periodogram(np.reshape(data[ch],-1), srate)
        
        lower_lb = f>0
        lower_ub = f<1
        delta_lb = f>=bands[0]
        delta_ub = f<bands[1]
        theta_lb = f>=bands[2]
        theta_ub = f<bands[3]
        alpha_lb = f>=bands[4]
        alpha_ub = f<bands[5]
        mu_lb = f>=bands[6]
        mu_ub = f<bands[7]
        beta_lb = f>=bands[8]
        beta_ub = f<bands[9]
        gamma_lb = f>=bands[10]
        gamma_ub = f<bands[11]


        lower_ind = np.logical_and(lower_lb,lower_ub)
        delta_ind = np.logical_and(delta_lb,delta_ub)
        theta_ind = np.logical_and(theta_lb,theta_ub)
        alpha_ind = np.logical_and(alpha_lb,alpha_ub)
        mu_ind = np.logical_and(mu_lb,mu_ub)
        beta_ind = np.logical_and(beta_lb,beta_ub)
        gamma_ind = np.logical_and(gamma_lb,gamma_ub)

        tot_pow = np.sum(Pxx_den)
        lower_pow = np.sum(Pxx_den[lower_ind])/tot_pow
        delta_pow = np.sum(Pxx_den[delta_ind])/tot_pow
        theta_pow = np.sum(Pxx_den[theta_ind])/tot_pow
        alpha_pow = np.sum(Pxx_den[alpha_ind])/tot_pow
        mu_pow = np.sum(Pxx_den[mu_ind])/tot_pow
        beta_pow = np.sum(Pxx_den[beta_ind])/tot_pow
        gamma_pow = np.sum(Pxx_den[gamma_ind])/tot_pow
        pow_array = [lower_pow,delta_pow,theta_pow,alpha_pow,mu_pow,beta_pow,gamma_pow]
        
        powers[ch][:] = pow_array

    return powers

def getSmallPSDpartsAbs(data,bands,srate=250,numPart=1):
    channelNum = np.shape(data)[0]
    samples = np.shape(data)[1]
    #bands = range(0,40,1)#[0,1,4,8,12,16,25,40]#

    numBands = len(bands)
    N = int(np.floor(samples/numPart))
    D = 0

    powers = np.zeros([channelNum,numPart,numBands-1])
    for ch in range(channelNum): #loop through channels
        for n in range(0,numPart):
            startInd = (n*(N-D))
            endInd = (n+1)*(N-D)-1 + D
            f, Pxx_den = signal.periodogram(np.reshape(data[ch,startInd:endInd],-1), srate)
            pow_array = []
            for b in range(numBands-1):
                lowerBound = f>=bands[b]
                upperBound = f<bands[b+1]

                band_ind = np.logical_and(lowerBound,upperBound)

                #tot_pow = np.sum(Pxx_den)
                band_pow = np.sum(Pxx_den[band_ind])
            
                pow_array.append(band_pow)

            powers[ch][n][:] = pow_array

    return powers


def getSmallPSDAbs(data,bands=[1,4,4,8,8,12,12,16,16,20,25,40],srate=250):
    channelNum = np.size(data,axis=0)
    #bands = range(0,41,1)
    numBands = len(bands)

    powers = np.zeros([channelNum,numBands-1])
    for ch in range(channelNum): #loop through channels
        f, Pxx_den = signal.periodogram(np.reshape(data[ch],-1), srate)
        pow_array = []
        for b in range(numBands-1):
            lowerBound = f>=bands[b]
            upperBound = f<bands[b+1]

            band_ind = np.logical_and(lowerBound,upperBound)

            #tot_pow = np.sum(Pxx_den)
            band_pow = np.sum(Pxx_den[band_ind])
        
            pow_array.append(band_pow)

        powers[ch][:] = pow_array

    return powers

def getAllFeatures(data,features,bands,srate=250,numPart=1):
    channelNum = np.shape(data)[0]
    samples = np.shape(data)[1]
    numBands = len(bands)
    N = int(np.floor(samples/numPart))
    D = 0
    if 'all' in features:
        features = ['psdR','psdA','fft','spectral','entropy','lzc','time']#,'addTime']
    
    if ('psdR' in features):
        powersRel = np.zeros([channelNum,numBands-1,numPart])
    
    if ('psdA' in features):
        powersAbs = np.zeros([channelNum,numBands-1,numPart])
    
    if ('fft' in features):
        FE = np.zeros([channelNum,1,numPart])
        FED = np.zeros([channelNum,1,numPart])
        FEP = np.zeros([channelNum,1,numPart])
        FEPD = np.zeros([channelNum,1,numPart])
        #fourierEntBined = np.zeros([channelNum,1,numPart])
    
    if ('spectral' in features):
        PE = np.zeros([channelNum,1,numPart])
        SE = np.zeros([channelNum,1,numPart])
        #specEntBined = np.zeros([channelNum,1,numPart])
    
    if ('entropy' in features):
        ENT = np.zeros([channelNum,1,numPart])
        ENTP = np.zeros([channelNum,1,numPart])
        #dataEntBined = np.zeros([channelNum,1,numPart])

    if ('lzc' in features):
        LZC = np.zeros([channelNum,1,numPart])
    
    if ('time' in features):
        CL = np.zeros([channelNum,1,numPart])
        E = np.zeros([channelNum,1,numPart])
        NE = np.zeros([channelNum,1,numPart])
        SP = np.zeros([channelNum,1,numPart])
        mini = np.zeros([channelNum,1,numPart])
        maxi = np.zeros([channelNum,1,numPart])
        medi = np.zeros([channelNum,1,numPart])
        mean = np.zeros([channelNum,1,numPart])
        var = np.zeros([channelNum,1,numPart])
        std = np.zeros([channelNum,1,numPart])
        skew = np.zeros([channelNum,1,numPart])
        kurt = np.zeros([channelNum,1,numPart])
        inte = np.zeros([channelNum,1,numPart])
        sumi = np.zeros([channelNum,1,numPart])

    if ('addTime' in features):
        mobility = np.zeros([channelNum,1,numPart])
        complexity = np.zeros([channelNum,1,numPart])
        #sampEnt = np.zeros([channelNum,1,numPart])
        #svdEnt = np.zeros([channelNum,1,numPart])

    epsilon = 1e-40

    allFeats = np.empty((channelNum,0,numPart))

    for ch in range(channelNum): #loop through channels
        if ('fft' in features):
            dataDFT = np.fft.fft(data[ch,:]) #taking log2, avoid divide by zero
        
        for n in range(numPart):
            startInd = (n*(N-D))
            endInd = (n+1)*(N-D) + D #(n+1)*(N-D)-1 + D

            if ('psdR' in features) or ('psdA' in features) or ('spectral' in features):
                f, Pxx_den = signal.periodogram(np.reshape(data[ch,startInd:endInd],-1), srate)
                tot_pow = np.sum(Pxx_den)
            
            if ('psdR' in features) or ('psdA' in features): 
                pow_arrayRel = []
                pow_arrayAbs = []
                for b in range(numBands-1):
                    lowerBound = f>=bands[b]
                    upperBound = f<bands[b+1]
                    band_ind = np.logical_and(lowerBound,upperBound)
                    band_powRel = np.sum(Pxx_den[band_ind])/tot_pow
                    band_powAbs = np.sum(Pxx_den[band_ind])
                    pow_arrayRel.append(band_powRel)
                    pow_arrayAbs.append(band_powAbs)
            #pdb.set_trace()

            if ('psdR' in features): 
                powersRel[ch,:,n] = pow_arrayRel
            
            if ('psdA' in features): 
                powersAbs[ch,:,n] = pow_arrayAbs

            if ('fft' in features):
                FE[ch,:,n] = np.abs(-np.sum(np.multiply(np.add(dataDFT[startInd:endInd],epsilon),np.log2(np.add(dataDFT[startInd:endInd],epsilon)))))
                FED[ch,:,n] = np.abs(-np.sum(np.multiply(np.add(np.abs(dataDFT[startInd:endInd]),epsilon),np.log2(np.add(np.abs(dataDFT[startInd:endInd]),epsilon)))))
                fftNorm = np.add(np.divide(dataDFT[startInd:endInd],np.sum(dataDFT[startInd:endInd])),epsilon)
                FEP[ch,:,n] = np.abs(-np.sum(np.multiply(fftNorm,np.log2(fftNorm))))
                fftNormD = np.add(np.divide(np.abs(dataDFT[startInd:endInd]),np.sum(np.abs(dataDFT[startInd:endInd]))),epsilon)
                FEPD[ch,:,n] = -np.sum(np.multiply(fftNormD,np.log2(fftNormD)))
                
                #fftBins = np.histogram(np.add(np.abs(dataDFT[startInd:endInd]),epsilon),bins=20)
                #fftBinsNormed = stats.rv_histogram(fftBins)#np.add(np.divide(fftBins[0],np.sum(fftBins[0])),epsilon)
                #fourierEntBined[ch,:,n] = fftBinsNormed.entropy()#-np.sum(np.multiply(fftBinsNormed,np.log2(fftBinsNormed)))

            if ('spectral' in features):
                PE[ch,:,n] = np.abs(-np.sum(np.multiply(np.add(Pxx_den,epsilon),np.log2(np.add(Pxx_den,epsilon)))))
                specPowNorm = np.add(np.divide(Pxx_den,tot_pow),epsilon) #taking log2, avoid divide by zero
                SE[ch,:,n] = -np.sum(np.multiply(specPowNorm,np.log2(specPowNorm)))

                #specBins = np.histogram(np.add(Pxx_den,epsilon),bins=20)
                #specBinsNormed = stats.rv_histogram(specBins)#np.add(np.divide(specBins[0],np.sum(specBins[0])),epsilon)
                #specEntBined[ch,:,n] = specBinsNormed.entropy()#-np.sum(np.multiply(specBinsNormed,np.log2(specBinsNormed)))
            
            if ('entropy' in features):
                shiftedData = np.subtract(data[ch,startInd:endInd],np.min(np.min(data[:,startInd:endInd])))
                ENT[ch,:,n] = np.abs(-np.sum(np.multiply(np.add(shiftedData,epsilon),np.log2(np.add(shiftedData,epsilon)))))
                shiftedDataNorm = np.add(np.divide(shiftedData,np.sum(shiftedData)),epsilon)
                ENTP[ch,:,n] = -np.sum(np.multiply(shiftedDataNorm,np.log2(shiftedDataNorm)))

                #dataBins = np.histogram(np.add(data[ch,startInd:endInd],epsilon),bins=20)
                #dataBinsNormed = stats.rv_histogram(dataBins)#np.add(np.divide(dataBins[0],np.sum(dataBins[0])),epsilon)
                #dataEntBined[ch,:,n] = dataBinsNormed.entropy()#-np.sum(np.multiply(dataBinsNormed,np.log2(dataBinsNormed)))              

            if ('lzc' in features):   
                binSeq = ''.join(str(x) for x in convertBinarization(data[ch,startInd:endInd]))
                LZC[ch,:,n] = lempel_ziv_complexity(binSeq.decode('utf-8'))
            
            if ('time' in features):
                CL[ch,:,n] = np.sum(np.abs(np.subtract(data[ch,startInd:endInd-1],data[ch,startInd+1:endInd])))
                E[ch,:,n] = (1./N)*(np.sum(np.power(data[ch,startInd:endInd],2)))
                NE[ch,:,n]= (1./N)*(np.sum(np.subtract(np.power(data[ch,startInd+1:endInd-1],2),np.multiply(data[ch,startInd:endInd-2],data[ch,startInd+2:endInd]))))
                SP[ch,:,n]= (1./N)*(np.sum(np.power(data[ch,startInd:endInd],6)))
                mini[ch,:,n]= np.min(data[ch,startInd:endInd])
                maxi[ch,:,n] = np.max(data[ch,startInd:endInd])
                medi[ch,:,n]= np.median(data[ch,startInd:endInd])
                mean[ch,:,n] = np.mean(data[ch,startInd:endInd])
                var[ch,:,n]= np.var(data[ch,startInd:endInd])
                std[ch,:,n] = np.std(data[ch,startInd:endInd])
                skew[ch,:,n] = stats.skew(data[ch,startInd:endInd])
                kurt[ch,:,n] = stats.kurtosis(data[ch,startInd:endInd])
                inte[ch,:,n] = np.trapz(data[ch,startInd:endInd])
                sumi[ch,:,n] = np.sum(data[ch,startInd:endInd])

            if ('addTime' in features):
                a0 = np.var(data[ch,startInd:endInd])
                a1 = np.var(np.diff(data[ch,startInd:endInd],n=1))
                a2 = np.var(np.diff(data[ch,startInd:endInd],n=2))
                mobility[ch,:,n] = np.sqrt(np.divide(a1,a0))
                complexity[ch,:,n] = np.sqrt(np.subtract(np.divide(a2,a1),np.divide(a1,a0)))
                #sampEnt[ch,:,n] = sampleEntropy(data[ch,startInd:endInd],m=5,r=0.2)
                #svdEnt[ch,:,n] = svdEntropy(data[ch,startInd:endInd],m=5)
                #print 'Done ch:',ch,'pt:',n
            
    if ('psdR' in features): 
        allFeats = np.append(allFeats,powersRel,axis=1)

    if ('psdA' in features): 
        allFeats = np.append(allFeats,powersAbs,axis=1)

    if ('fft' in features):
        allFeats = np.append(allFeats,FE,axis=1)
        allFeats = np.append(allFeats,FED,axis=1)
        allFeats = np.append(allFeats,FEP,axis=1)
        allFeats = np.append(allFeats,FEPD,axis=1)
        #allFeats = np.append(allFeats,fourierEntBined,axis=1)


    if ('spectral' in features):
        allFeats = np.append(allFeats,PE,axis=1)
        allFeats = np.append(allFeats,SE,axis=1)
        #allFeats = np.append(allFeats,specEntBined,axis=1)

    if ('entropy' in features):
        allFeats = np.append(allFeats,ENT,axis=1)
        allFeats = np.append(allFeats,ENTP,axis=1)
        #allFeats = np.append(allFeats,dataEntBined,axis=1)


    if ('lzc' in features):
        allFeats = np.append(allFeats,LZC,axis=1)
  
    if ('time' in features):
        allFeats = np.append(allFeats,CL,axis=1)
        allFeats = np.append(allFeats,E,axis=1)
        allFeats = np.append(allFeats,NE,axis=1)
        allFeats = np.append(allFeats,SP,axis=1)
        allFeats = np.append(allFeats,mini,axis=1)
        allFeats = np.append(allFeats,maxi,axis=1)
        allFeats = np.append(allFeats,medi,axis=1)
        allFeats = np.append(allFeats,mean,axis=1)
        allFeats = np.append(allFeats,var,axis=1)
        allFeats = np.append(allFeats,std,axis=1)
        allFeats = np.append(allFeats,skew,axis=1)
        allFeats = np.append(allFeats,kurt,axis=1)
        allFeats = np.append(allFeats,inte,axis=1)
        allFeats = np.append(allFeats,sumi,axis=1)

    if ('addTime' in features):
        allFeats = np.append(allFeats,mobility,axis=1)
        allFeats = np.append(allFeats,complexity,axis=1)
        #allFeats = np.append(allFeats,sampEnt,axis=1)
        #allFeats = np.append(allFeats,svdEnt,axis=1)


    #allFeats = np.concatenate((powersRel,powersAbs,FE,PE,SE,ENT,ENTP,CL,E,NE,SP,LZC,mini,maxi,medi,mean,var,std,skew,kurt,inte,sumi),axis=1)

    return allFeats

from scipy import optimize
import pylab as pl
def oneFspectra(data,srate=100,numPart=1):
    hzCut = 1
    channelNum = np.shape(data)[0]
    samples = np.shape(data)[1]
    N = int(np.floor(samples/numPart))
    D = 0
    fitP = np.zeros([channelNum,2,numPart])

    fitFunc = lambda x,a,c: a*np.power(x+1e-40,-1)+c

    for ch in range(channelNum): #loop through channels
        for n in range(numPart):
            startInd = (n*(N-D))
            endInd = (n+1)*(N-D) + D #(n+1)*(N-D)-1 + D
            f, Pxx_den = signal.periodogram(np.reshape(data[ch,startInd:endInd],-1), srate)

            p,pcov = optimize.curve_fit(fitFunc,f[f>hzCut],Pxx_den[f>hzCut])
            #print p, np.sqrt(np.diag(pcov))

        fitP[ch,:,n] = p
        #pl.plot(f[f>hzCut],Pxx_den[f>hzCut],'b-', label='data')
        #pl.plot(f[f>hzCut], fitFunc(f[f>hzCut], *p), 'r-',label='fit: b=%5.3f, c=%5.3f' % tuple(p))
        #pl.xlabel('Frequency')
        #pl.ylabel('Periodogram')
        #pl.legend()
        #pl.show()

    return fitP    

def getRawEEG(eeg,srate=250):
    eegData, times = getEEGdata(eeg)

    if np.size(eegData,axis=0) != 19:
        #print np.shape(eegData)
        warnings.warn("19 Channels not found.",Warning)
        raise ValueError("19 Channels not found.")
        return eegData,times

    
    #pdb.set_trace()
    #print 'Pre - Min:',np.min(np.min(eegData)),' Max:',np.max(np.max(eegData))
    #eegData = np.multiply(eegData,(10**6)) #convert microVolts to Volts
    
    samplingFrequency = int(eeg.info['sfreq'])
    if samplingFrequency < 10:
        samplingFrequency = 1./(eeg.times[1]-eeg.times[0])
    if samplingFrequency < 10:
        raise ValueError('Bad Sampling Frequency:%f'%(samplingFrequency))
    #srateR = 1/(times[1]-times[0]) #time in seconds => Hz
    #print 'Sampled at:',srateR
    #if (np.shape(eegData)[1] < (samplingFrequency*60*(timeMin+1))):
    #    return eegData,samplingFrequency
    
    #eegDataRe,srate = reSampleData(eegData,srateR,250)
    eegDataRe = resampy.resample(eegData,samplingFrequency,srate,axis=1,filter='kaiser_fast')
    
    #filtData = idealFilter(eegDataRe,lowHz=0.5,highHz=50,srate=srate)#iirFiltFilt(eegData,lowHz=0.5,highHz=50,srate=srate,order=6)

    #refData = reRefdata(filtData)  

    #print 'Post - Min:',np.min(np.min(refData)),' Max:',np.max(np.max(refData))

    return eegDataRe,srate

def preProcessData(eeg,timeMin=16,srate=100):
    #pdb.set_trace()
    eegData, times = getEEGdata(eeg)

    if np.size(eegData,axis=0) != 19:
        #print np.shape(eegData)
        warnings.warn("19 Channels not found.",Warning)
        return eegData,times
    
    #pdb.set_trace()
    #print 'Pre - Min:',np.min(np.min(eegData)),' Max:',np.max(np.max(eegData))
    eegData = np.multiply(eegData,(10**6)) #convert microVolts to Volts
    samplingFrequency = int(eeg.info['sfreq'])
    if samplingFrequency < 10:
        samplingFrequency = 1./(eeg.times[1]-eeg.times[0])
    if samplingFrequency < 10:
        raise ValueError('Bad Sampling Frequency:%f'%(samplingFrequency))
    #srateR = 1/(times[1]-times[0]) #time in seconds => Hz
    #print 'Sampled at:',srateR
    if (np.shape(eegData)[1] < (samplingFrequency*60*(timeMin+1))):
        return eegData,samplingFrequency
    
    #eegDataRe,srate = reSampleData(eegData,srateR,250)
    eegDataRe = resampy.resample(eegData,samplingFrequency,srate,axis=1,filter='kaiser_fast')
    
    filtData = idealFilter(eegDataRe,lowHz=0.5,highHz=50,srate=srate)#iirFiltFilt(eegData,lowHz=0.5,highHz=50,srate=srate,order=6)

    refData = reRefdata(filtData)  

    #print 'Post - Min:',np.min(np.min(refData)),' Max:',np.max(np.max(refData))

    return refData,srate

def getEEGdata(eeg):
    data, times = eeg[:]
    channelsUni = eeg.ch_names
    #print 'EEGNames:',channelsUni
    #channelNames = [x.encode('UTF8') for x in channelsUni]
    tenTwenty = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','FZ','CZ','PZ']
    chNamesTT = ['EEG FP1','EEG FP2','EEG F3','EEG F4','EEG C3','EEG C4','EEG P3','EEG P4','EEG O1','EEG O2','EEG F7','EEG F8','EEG T3','EGG T4','EEG T5','EEG T6','EEG FZ','EEG CZ','EEG PZ']
    #Find EEG Channels
    eegChannelNum = []
    eegChannelNames = []
    for eegCh in tenTwenty:
        chNum = 0
        for ch in channelsUni:
            if eegCh in ch:
                eegChannelNum.append(chNum)
                eegChannelNames.append(eegCh)
                break
            chNum += 1
    ch=0
    numSamples = np.size(data,axis=1)
    eegData = np.zeros([len(eegChannelNum),numSamples])
    #EEGdata = np.zeros([len(eegChannelNum),int(np.ceil(numSamples*(250/srate)))])
    for i in eegChannelNum:
        eegData[ch,:] = data[i,:]
        ch+=1
    return eegData,times

def reSampleData(data,srate=250,desiredSRate=250):
    eegChannelNum = np.shape(data)[0]
    numSamples = np.shape(data)[1]

    if srate != desiredSRate:
        reSampledData = np.zeros([eegChannelNum,int(np.ceil(numSamples*(desiredSRate/srate)))])
        for k in range(eegChannelNum):
            reSampledData[k,:] = signal.resample(data[k,:],int(np.ceil(numSamples*(desiredSRate/srate))))
        srate = desiredSRate
        #print 'After:',np.size(eegData),'srate:',srate
        return reSampledData,srate
    else:
        return data,srate

def idealFilter(data, lowHz=0.5,highHz=50,srate=250):
    numCh = np.shape(data)[0]
    numSamples = np.shape(data)[1]
    filteredData = np.zeros(np.shape(data))
    W = fftpack.rfftfreq(numSamples,d=1./srate)
    for ch in range(numCh): #loop through channels
        dataDFT = fftpack.rfft(data[ch])
        dataDFT[(W<lowHz)] = 0
        dataDFT[(W>highHz)] = 0
        filteredData[ch,:] = fftpack.irfft(dataDFT)
    #pdb.set_trace()
    return filteredData

def iirFiltFilt(data,lowHz=1,highHz=40,srate=250,order=6):
    #srate = 1/(times[1]-times[0]) #time in seconds => Hz
    nyquistF = srate/2;
    #Manual Filter
    #N,Wn = signal.buttord([lowHz/nyquistF,highHz/nyquistF],[(lowHz-0.1)/nyquistF,(highHz+1)/nyquistF],1,30,)
    #b,a = signal.butter(N,Wn,btype='bandpass',output='ba')
    b,a = signal.iirfilter(6,[lowHz/nyquistF,highHz/nyquistF],btype='band',ftype='butter',output='ba')    
    numCh = np.size(data,axis=0)
    numSamples = np.size(data,axis=1)
    filtData = np.zeros([numCh,numSamples])
    refData = np.zeros([numCh,numSamples])
    
    for i in range(numCh):
        filtData[i] = signal.filtfilt(b,a,data[i])

    return filtData

def reRefdata(data):
    numCh = np.size(data,axis=0)
    numSamples = np.size(data,axis=1)
    timeAvg = np.mean(data,axis=0)
    refData = np.zeros([numCh,numSamples])
    for j in range(numSamples):
        refData[:,j] = data[:,j] - timeAvg[j]
    return refData

def EEGplots(eeg,mode="psd",srate=250):
    if mode=="multi":
        data, times = eeg[:]
        try:
            eeg.plot(n_channels=19,duration=20,bgcolor='w',color='k',lowpass=50,highpass=1,filtorder=8)
        except:
            print "Requires EDF EEG for multi-plot. Plotting single channel."
            mode = "single"

    if mode=="single":
        eegCh = eeg[6,:]
        times = np.divide(range(len(eegCh)),60.*srate)
        plt.plot(times,np.multiply(eegCh,(10**0)))#pow6
        plt.ylim([np.min(eegCh),np.max(eegCh)])
        plt.xlim([times[0],times[-1]])
        ticks = np.arange(min(times), max(times)+1, 1)
        plt.xticks(ticks)
        plt.xlabel('Time (min)')
        plt.ylabel('Signal Value (muV)')
        plt.title('EEG Signal of a Single Channel')
        plt.show()
    
    if mode=="psd":
        #srate = 1/(times[1]-times[0])
        f, Pxx_den = signal.periodogram(eeg[8], srate)
        plt.semilogy(f, Pxx_den)
        plt.ylim([1e-3, 1e3])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [muV^2/Hz]')
        plt.title('Power Spectrum Density of channel O1')
        plt.show()  

    if mode=="fft":
        # Number of samplepoints
        N = np.size(eeg,axis=1)
        # sample spacing
        T = 1.0 / srate
        
        yf = np.fft.fft(eeg[8])
        xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
        plt.plot(xf, 2.0/N * np.abs(yf[0:N/2]))
        plt.grid()
        plt.show()

def getBoxPlotData(allKeys,allLables,allPowers,COI,BOI):
    prevSubj = allKeys[0].split('_')[0]
    subjLabels = [allLables[0]]
    subjectsProc = [int(prevSubj)]
    allData = []
    curData = []
    sampleInd = 0
    for curSampleLabel in allKeys:
        curSubj = curSampleLabel.split('_')[0]
        #print 'cur',curSubj
        #print 'prev',prevSubj
        if curSubj == prevSubj:
            curData.append(allPowers[sampleInd][COI][BOI])
            sampleInd += 1
        else:
            sampleInd += 1
            if curData == []:
                continue
            subjLabels.append(allLables[sampleInd])
            subjectsProc.append(int(curSubj))
            prevSubj = curSubj
            allData.append(curData)
            curData = []
    
    #print 'NumberOfSubj:',np.shape(allData)[0]
    #print 'Prossesed Subj:',subjectsProc
    #print 'Genders:',subjLabels
    #print 'Subjects:',subjPlotted
    #print 'NumberOfSample:',np.shape(allData[subjPlotted[0]])[0]
    #print 'NumberOfSample:',np.shape(allData[subjPlotted[1]])[0]
    #print 'NumberOfSample:',np.shape(allData[subjPlotted[2]])[0]
    #print 'NumberOfSample:',np.shape(allData[subjPlotted[3]])[0]
    #print 'NumberOfSample:',np.shape(allData[subjPlotted[4]])[0]

    return allData,allLables,subjLabels,subjectsProc

def boxPlotTwo(allKeys,allLables,allPowers):
    #allKeys.append(key);; List of names eg. EDF file names.
    #allLables.append(label);; List of label eg. M/F
    #pow_array = [lower_pow,delta_pow,theta_pow,alpha_pow,mu_pow,beta_pow,gamma_pow]
    #powers[ch][:] = pow_array;; Can call getPSD()
    #allPowers[curSample][:][:] = powers
    allData1,subLabels1,subjLabels1,subjectsProc = getBoxPlotData(allKeys,allLables,allPowers,8,3)
    allData2,subLabels2,subjLabels2,subjPlotted2 = getBoxPlotData(allKeys,allLables,allPowers,4,1)


    subjPlotted = sorted(random.sample(range(len(subjectsProc)-1),5))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bp = ax.boxplot([np.multiply(allData1[subjPlotted[0]],100),np.multiply(allData1[subjPlotted[1]],100),np.multiply(allData1[subjPlotted[2]],100),np.multiply(allData1[subjPlotted[3]],100),np.multiply(allData1[subjPlotted[4]],100)])
    subB = 0
    for b in bp['boxes']:
        if subjLabels1[subjPlotted[subB]] == 0:
            b.set_color('blue')
        elif subjLabels1[subjPlotted[subB]] == 1:
            b.set_color('red')
        subB +=1
    red_patch = mpatches.Patch(color='red',label='Male')
    blue_patch = mpatches.Patch(color='blue',label='Female')
    plt.legend(handles=[red_patch,blue_patch])
    ax.set_ylabel('Percent of Signal Power')
    ax.set_xlabel('Subject')
    ax.set_title('Alpha band on channel O1')
    plt.show()


    fig = plt.figure()
    ax = fig.add_subplot(111)
    bp = ax.boxplot([np.multiply(allData2[subjPlotted[0]],100),np.multiply(allData2[subjPlotted[1]],100),np.multiply(allData2[subjPlotted[2]],100),np.multiply(allData2[subjPlotted[3]],100),np.multiply(allData2[subjPlotted[4]],100)])
    subB = 0
    for b in bp['boxes']:
        if subjLabels2[subjPlotted[subB]] == 0:
            b.set_color('blue')
        elif subjLabels2[subjPlotted[subB]] == 1:
            b.set_color('red')
        subB +=1
    red_patch = mpatches.Patch(color='red',label='Male')
    blue_patch = mpatches.Patch(color='blue',label='Female')
    plt.legend(handles=[red_patch,blue_patch])
    ax.set_ylabel('Percent of Signal Power')
    ax.set_xlabel('Subject')
    ax.set_title('Delta band on channel C3')
    plt.show()

def resultBoxPlot(fig,data,totalInst=None,ax=None,exclude=0):
    # Create a figure instance
    #fig = plt.figure(1, figsize=(9, 6))
    if ax == None:
        # Create an axes instance
        ax = fig.add_subplot(111)

    if totalInst != None:
        data = np.divide(data,totalInst/100.)
    
    # Create the boxplot
    bp = ax.boxplot(data,whis='range')#,whis='range'
    if np.shape(data)[1] == 31:
        ax.set_xticklabels(['Rel-Lower', 'Rel-Delta', 'Rel-Theta', 'Rel-Alpha', 'Rel-Mu', 'Rel-Beta', 'Rel-Gamma', 
            'Abs-Lower', 'Abs-Delta', 'Abs-Theta', 'Abs-Alpha', 'Abs-Mu', 'Abs-Beta', 'Abs-Gamma', 
            'Fourier-Entropy', 'Spectral-Entropy', 'Curve-Length', 'Energy', 'Nonlinear-Energy', 'Sixth-Power', 'LZC', 
            'Min', 'Max', 'Med', 'Mean', 'Var', 'Std', 'Skew', 'Kurtosis', 'Integral', 'Sum'],rotation='90')
    elif np.shape(data)[1] == 34:
        if exclude==0:
            ax.set_xticklabels(['Rel-Lower', 'Rel-Delta', 'Rel-Theta', 'Rel-Alpha', 'Rel-Mu', 'Rel-Beta', 'Rel-Gamma', 
                'Abs-Lower', 'Abs-Delta', 'Abs-Theta', 'Abs-Alpha', 'Abs-Mu', 'Abs-Beta', 'Abs-Gamma', 
                'Fourier-Entropy', 'Spectral-Entropy', 'Spectral-Entropy-Norm', 'Entropy', 'Entropy-Norm', 
                'Curve-Length', 'Energy', 'Nonlinear-Energy', 'Sixth-Power', 'LZC', 
                'Min', 'Max', 'Med', 'Mean', 'Var', 'Std', 'Skew', 'Kurtosis', 'Integral', 'Sum'],rotation='90')
        elif exclude==1:
            ax.set_xticklabels(['Rel-Lower', 'Rel-Delta', 'Rel-Theta', 'Rel-Alpha', 'Rel-Mu', 'Rel-Beta', 'Rel-Gamma', 
                'Abs-Lower', 'Abs-Delta', 'Abs-Theta', 'Abs-Alpha', 'Abs-Mu', 'Abs-Beta', 'Abs-Gamma', 
                'Fourier-Entropy', 'NormedFourier-Entropy', 'Fourier-Entropy-Norm', 'NormedFourier-Entropy-Norm', 'Spectral-Entropy', 'Spectral-Entropy-Norm', 
                'Entropy', 'Entropy-Norm', 'Curve-Length', 'Energy', 'Nonlinear-Energy', 'Sixth-Power', 'LZC', 
                'Min', 'Max', 'Med', 'Var', 'Std', 'Skew', 'Kurtosis'],rotation='90')
    elif np.shape(data)[1] == 37:
        ax.set_xticklabels(['Rel-Lower', 'Rel-Delta', 'Rel-Theta', 'Rel-Alpha', 'Rel-Mu', 'Rel-Beta', 'Rel-Gamma', 
            'Abs-Lower', 'Abs-Delta', 'Abs-Theta', 'Abs-Alpha', 'Abs-Mu', 'Abs-Beta', 'Abs-Gamma', 
            'Fourier-Entropy', 'NormedFourier-Entropy', 'Fourier-Entropy-Norm', 'NormedFourier-Entropy-Norm', 'Spectral-Entropy', 'Spectral-Entropy-Norm', 
            'Entropy', 'Entropy-Norm', 'Curve-Length', 'Energy', 'Nonlinear-Energy', 'Sixth-Power', 'LZC', 
            'Min', 'Max', 'Med', 'Mean', 'Var', 'Std', 'Skew', 'Kurtosis', 'Integral', 'Sum'],rotation='90')    
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    #fig.suptitle('Kruskal-Wallis Results of Intra-subject Variability of Features')
    ax.set_xlabel('Features')

    if totalInst != None:
        ax.set_ylabel('Percent with Consistent Features')

    #plt.show()
    return fig,ax

def butter_bandpass(lowcut,highcut,srate,order=6):
    nyquistF = 0.5*srate
    low = lowcut/nyquistF
    high = highcut/nyquistF
    b,a = signal.butter(order,[low,high],btype='band')
    return b,a

def butter_bandpass_filter(data,lowcut,highcut,srate,order=6):
    b,a = butter_bandpass(lowcut,highcut,srate,order=order)
    filtered = signal.lfilter(b,a,data)
    return filtered

def dataLoad3D(InputFileName,Features,Partitions,TimeMin,Threads,Write2File,FeatsNames,channels=[],features=[],prePend=''):
    dataLoad = np.load('/home/david/Documents/nedcProcessing/'+prePend+FeatsNames+'_'+str(Partitions)+'Parts_'+str(TimeMin)+'Min/'+FeatsNames+'_'+str(Partitions)+'Parts_'+str(TimeMin)+'Min'+'_'+InputFileName+'_Data'+'.npy',allow_pickle=True)
    labelsLoad = np.load('/home/david/Documents/nedcProcessing/'+prePend+FeatsNames+'_'+str(Partitions)+'Parts_'+str(TimeMin)+'Min/'+FeatsNames+'_'+str(Partitions)+'Parts_'+str(TimeMin)+'Min'+'_'+InputFileName+'_Labels'+'.npy',allow_pickle=True)

    if (len(channels)!=0):
        dataLoad = dataLoad[:,channels,:,:]
    if (len(features)!=0):
        dataLoad = dataLoad[:,:,features,:]

    dataLoad = np.reshape(dataLoad,(np.shape(dataLoad)[0],np.shape(dataLoad)[1]*np.shape(dataLoad)[2],np.shape(dataLoad)[3]))
    allData = dataLoad
    allLabels = labelsLoad
    
    return allData, allLabels


def dataLoad4D(InputFileName,Features,Partitions,TimeMin,Threads,Write2File,FeatsNames,channels=[],features=[],prePend=''):
    dataLoad = np.load('/home/david/Documents/nedcProcessing/'+prePend+FeatsNames+'_'+str(Partitions)+'Parts_'+str(TimeMin)+'Min/'+FeatsNames+'_'+str(Partitions)+'Parts_'+str(TimeMin)+'Min'+'_'+InputFileName+'_Data'+'.npy',allow_pickle=True)
    labelsLoad = np.load('/home/david/Documents/nedcProcessing/'+prePend+FeatsNames+'_'+str(Partitions)+'Parts_'+str(TimeMin)+'Min/'+FeatsNames+'_'+str(Partitions)+'Parts_'+str(TimeMin)+'Min'+'_'+InputFileName+'_Labels'+'.npy',allow_pickle=True)

    if (len(channels)!=0):
        dataLoad = dataLoad[:,channels,:,:]
    if (len(features)!=0):
        dataLoad = dataLoad[:,:,features,:]

    allData = dataLoad
    allLabels = labelsLoad

    return allData, allLabels

def mainTXT(allTXT):
    f = open(allTXT,'r')
    content = f.readlines()
    male = 0
    female = 0
    noGender = 0
    goodAge = 0
    noAge = 0
    age = []
    medFound = 0
    goodEDF = 0
    eegSysFound = 0
    valueErrorCount = 0
    noEEGerr = 0
    keyCount = np.zeros([5,1])# [0,0,0,0,0]
    specialCount = 0 #Keep count
    specialSubj = [] #Keep list
    genderNP = np.zeros([3,1])
    prevSubjName = 0
    subjGender = ''
    normalEEG = 0
    abnormalEEG = 0
    noImpressionEEG = 0
    
    val = Subject()
    '''
    val.male = male
    val.female = female
    val.noGender = noGender
    val.age = age
    val.goodAge = goodAge
    val.noAge = noAge
    val.medFound = medFound
    val.eegSysFound = eegSysFound
    val.keyCount = keyCount
    val.valueErrorCount = valueErrorCount
    val.specialSubj = specialSubj
    val.specialCount = specialCount
    val.noEEGerr = noEEGerr
    val.genderNP = genderNP
    val.prevSubjName = prevSubjName
    val.subjGender = subjGender
    '''
    
    #txtFiles = [x for x in content]
    for txtFiles in content:
        val.fileName = txtFiles
        val = singleTXTEval(val)
        male += val.male
        female += val.female
        noGender += val.noGender
        age += val.age #like append
        goodAge += val.goodAge
        noAge += val.noAge
        medFound += val.medFound
        eegSysFound += val.eegSysFound
        keyCount = keyCount + val.keyCount
        valueErrorCount += val.valueErrorCount
        noEEGerr += val.noEEGerr
        genderNP = genderNP + val.genderNP
        prevSubjName = val.prevSubjName
        normalEEG += val.normalEEG
        abnormalEEG += val.abnormalEEG
        noImpressionEEG += val.noImpressionEEG
    
    return male,female,noneGender,np.mean(age),goodAge,badAge,medFound,eegSysFound,keyCount,valueErrorCount,noEEGerr,genderNP,normalEEG,abnormalEEG,noImpressionEEG

def singleTXTEval(val):
    male = 0
    female = 0
    none = 0
    goodAge = 0
    badAge = 0
    age = []
    medFound = 0
    goodEDF = 0
    eegSysFound = 0
    normalEEG = 0
    abnormalEEG = 0
    noImpressionEEG = 0
    valueErrorCount = 0
    noEEGerr = 0
    keyCount = np.zeros([5,1])# [0,0,0,0,0]
    specialCount = 0 #Keep count
    specialSubj = [] #Keep list
    genderNP = np.zeros([3,1])
    subjEdfFiles = []
    singleTXTdata = []
    subjDates = []

    prevSubjName = val.prevSubjName

    txtErrorCount = 0
    numEDF = 0
    txtFileStr = val.fileName.split('/')
    name = txtFileStr[8]
    session = txtFileStr[9].split('s')[1]
    #textFile = txtFileDirectory+txtFile
    textFile = val.fileName.rstrip()
    
    newName = 0
    if int(name) != prevSubjName: #commment out if par.
       subjAge = 0
       subjGender = ''
       prevSubjName = int(name)
       newName = 1

    #Only on first session for repeated visits
    if newName==1: #int(txtFile.split('s')[1].split('.')[0].split('x')[0])==1:
        #print 'NewName:',int(name),'Session',session
        subjGender,subjAge,subjMed,subjEEGsys,subjKeywords,subjNormalState = readNotes(textFile)
        #Subject Gender
        if subjGender == 'male':
            txtGender = 'male'
            male = 1
        elif subjGender == 'female':
            txtGender = 'female'
            female = 1
        else:
            subjGender = 'noGender'
            txtGender = 'noGender'
            none = 1

        #Subject Age
        if (subjAge > 0) and (subjAge < 120):
            goodAge = 1
            age.append(subjAge)
            txtAge = subjAge
        else:
            badAge = 1
            txtAge = 0

        #Subject medications
        if len(subjMed) > 0:
            medFound = 1
            txtMed = subjMed
        else:
            subjMed.append('none')
            txtMed = ['none']

        #Subjects with 10-20 EEG Sys
        if subjEEGsys == '10-20':
            subjEEGsys = 'yes'
            txtEEGsys = 'yes'
            eegSysFound = 1
        else:
            subjEEGsys = 'noEEGsys'
            txtEEGsys = 'noEEGsys'

        #Subjects normality
        if subjNormalState == 0:
            normalEEG = 1
            txtNormalState = 0
        elif subjNormalState == 1:
            abnormalEEG = 1
            txtNormalState = 1
        else:
            noImpressionEEG = 1
            txtNormalState = 2


    else:
        subjGender = val.subjGender
        subjAge = val.age
        subjMed = val.medFound
        subjEEGsys = val.eegSysFound

        if val.male == 1:
            subjGender = 'male'
        elif val.female == 1:
            subjGender = 'female'
        elif val.noGender == 1:
            subjGender = 'noGender'

        #print 'Name:',int(name),'PrevName:',prevSubjName,'Session',session

        subjGenderNext,subjAgeNext,subjMedNext,subjEEGsysNext,subjKeywords,subjNormalStateNext = readNotes(textFile)

        #Subject Gender
        if subjGenderNext == 'male':
            if (subjGender == 'noGender') or (subjGender == ''):
                male = 1
            txtGender = 'male'
            subjGender = 'male'
        elif subjGenderNext == 'female':
            if (subjGender == 'noGender') or (subjGender == ''):
                female = 1
            txtGender = 'female'
            subjGender = 'female'
        else:
            if (subjGender == ''):
                none = 1
            txtGender = 'noGender'
            subjGender = 'noGender'

        #Subject Age
        if (subjAgeNext > 0) and (subjAgeNext < 120):
            if subjAge == 0:
                goodAge = 1
            age.append(subjAgeNext)
            txtAge = subjAgeNext
            subjAge = subjAgeNext
        else:
            txtAge = subjAge

        #Subject medications
        if len(subjMedNext) > 0:
            txtMed = subjMedNext
        else:
            subjMedNext.append('none')
            txtMed = ['none']

        #Subjects with 10-20 EEG Sys
        if subjEEGsysNext == '10-20':
            txtEEGsys = 'yes'
        else:
            txtEEGsys = 'noEEGsys'

        #Subjects normality
        if subjNormalStateNext == 0:
            normalEEG = 1
            txtNormalState = 0
        elif subjNormalStateNext == 1:
            abnormalEEG = 1
            txtNormalState = 1
        else:
            noImpressionEEG = 1
            txtNormalState = 2
    
    #Keep count of subjects with criteria found
    for k in range(len(subjKeywords)):
        if subjKeywords[k] != 'notFound':
            keyCount[k] += 1
    #Number of subjects with only last criteria
    if (subjKeywords[0] == 'notFound') and (subjKeywords[1] == 'notFound') and (subjKeywords[4] != 'notFound'):
        if int(session) == 1:
            specialCount += 1
            specialSubj.append(name+'_'+session)

    #Find all associated EDf files
    edfExt = val.fileName.split('.txt')[0]+'*.edf'
    for edfFile in glob.glob(edfExt):
        edfFile = edfFile.rstrip()
        subjEEG,errorValue=readEDF(edfFile) #subjEEG
        numEDF += 1
        if errorValue != '':
            valueErrorCount += 1
            txtErrorCount +=1
            continue
        #elif txtEEGsys == 'yes':
        else:
            errorValue = 'noError'
            noEEGerr += 1
            '''
            try:
                #subjEEGdata,eegChannelNames,srate = preProcessEEG(subjEEG) #subjPreProcEEG,subjChannelNames,subjeegChannelNum,
                refData,times = preProcessData(subjEEG)
                #print 'After:',np.size(subjEEGdata),'srate:',srate
                if subjGender == 'male':
                    genderNP[0] +=1
                elif subjGender == 'female':
                    genderNP[1] +=1
                elif (subjGender == 'noGender') or (subjGender == ''):
                    genderNP[2] +=1
            except MemoryError:
                print 'Data:',edfFile.split('/')[-1],'; Error.'
                continue
            try:
                psdOfData = getPSD(refData)
            except RuntimeWarning:
                print 'Data:',edfFile.split('/')[-1],'; PSD-Error.'
                continue
            '''
            #print 'Data:',np.shape(refData)
            subjEdfFiles.append(edfFile)
            singleTXTdata.append(subjEEG)
            dateCur = getDate(edfFile)
            subjDates.append(datetime.date(dateCur[0],dateCur[1],dateCur[2]))
            #if noEEGerr == 1:
            #    singleTXTdata = [refData]
            #else:
            #    singleTXTdata = [singleTXTdata,refData]
            #print np.shape(singleTXTdata)

    #printCSV(name,session,numEDF,txtEEGsys,txtAge,subjKeywords,txtMed,newName)
    val.male = male
    val.female = female
    val.noGender = none
    val.age = age
    val.goodAge = goodAge
    val.noAge = badAge
    val.medFound = medFound
    val.eegSysFound = eegSysFound
    val.keyCount = val.keyCount
    val.keywords = subjKeywords
    val.name = int(name)
    val.session = int(session)
    val.subjGender = subjGender
    val.subjMed = txtMed
    val.subjNormalState = txtNormalState
    val.normalEEG = normalEEG
    val.abnormalEEG = abnormalEEG
    val.noImpressionEEG = noImpressionEEG
    val.subjEdfFiles = subjEdfFiles
    val.singleTXTdata = singleTXTdata
    val.subjDates = subjDates
    return val

def singleTXTEvalCSV(txtFile,val,txtFileDirectory=DIRECTORY+'edfFiles/'):
    male = 0
    female = 0
    none = 0
    goodAge = 0
    badAge = 0
    age = []
    medFound = 0
    goodEDF = 0
    eegSysFound = 0
    valueErrorCount = 0
    noEEGerr = 0
    keyCount = np.zeros([5,1])# [0,0,0,0,0]
    specialCount = 0 #Keep count
    specialSubj = [] #Keep list
    genderNP = np.zeros([3,1])
    subjEdfFiles = []
    singleTXTdata = []

    prevSubjName = val[12]

    txtErrorCount = 0
    numEDF = 0
    txtFileStr = txtFile.split('_')
    name = txtFileStr[0]
    session = txtFileStr[1].split('s')[1].split('.')[0].split('x')[0]
    textFile = txtFileDirectory+txtFile
    textFile = textFile.rstrip()
    newName = 0
    if int(name) != prevSubjName: #commment out if par.
       subjAge = 0
       subjGender = ''
       prevSubjName = int(name)
       newName = 1

    #Only on first session for repeated visits
    if newName==1: #int(txtFile.split('s')[1].split('.')[0].split('x')[0])==1:
        #print 'NewName:',int(name),'Session',session
        subjGender,subjAge,subjMed,subjEEGsys,subjKeywords,subjNormalState = readNotes(textFile)
        #Subject Gender
        if subjGender == 'male':
            txtGender = 'male'
            male = 1
        elif subjGender == 'female':
            txtGender = 'female'
            female = 1
        else:
            subjGender = 'noGender'
            txtGender = 'noGender'
            none =1

        #Subject Age
        if (subjAge > 0) and (subjAge < 120):
            goodAge = 1
            age.append(subjAge)
            txtAge = subjAge
        else:
            badAge = 1
            txtAge = 0

        #Subject medications
        if len(subjMed) > 0:
            medFound = 1
            txtMed = subjMed
        else:
            subjMed.append('none')
            txtMed = ['none']

        #Subjects with 10-20 EEG Sys
        if subjEEGsys == '10-20':
            subjEEGsys = 'yes'
            txtEEGsys = 'yes'
            eegSysFound = 1
        else:
            subjEEGsys = 'noEEGsys'
            txtEEGsys = 'noEEGsys'

        #Subjects normality
        if subjNormalState == 0:
            normalEEG = 1
            txtNormalState = 0
        elif subjNormalState == 1:
            abnormalEEG = 1
            txtNormalState = 1
        else:
            noImpressionEEG = 1
            txtNormalState = 2

    else:
        subjGender = val[13]
        subjAge = val[3] #like append
        subjMed = val[6]
        subjEEGsys = val[7]

        if val[0] == 1:
            subjGender = 'male'
        elif val[1] == 1:
            subjGender = 'female'
        elif val[2] == 1:
            subjGender = 'noGender'

        #print 'Name:',int(name),'PrevName:',prevSubjName,'Session',session

        subjGenderNext,subjAgeNext,subjMedNext,subjEEGsysNext,subjKeywords,subjNormalStateNext = readNotes(textFile)

        #Subject Gender
        if subjGenderNext == 'male':
            if (subjGender == 'noGender') or (subjGender == ''):
                male = 1
            txtGender = 'male'
            subjGender = 'male'
        elif subjGenderNext == 'female':
            if (subjGender == 'noGender') or (subjGender == ''):
                female = 1
            txtGender = 'female'
            subjGender = 'female'
        else:
            if (subjGender == ''):
                none = 1
            txtGender = 'noGender'
            subjGender = 'noGender'

        #Subject Age
        if (subjAgeNext > 0) and (subjAgeNext < 120):
            if subjAge == 0:
                goodAge = 1
            age.append(subjAgeNext)
            txtAge = subjAgeNext
            subjAge = subjAgeNext
        else:
            txtAge = subjAge

        #Subject medications
        if len(subjMedNext) > 0:
            txtMed = subjMedNext
        else:
            subjMedNext.append('none')
            txtMed = ['none']

        #Subjects with 10-20 EEG Sys
        if subjEEGsysNext == '10-20':
            txtEEGsys = 'yes'
        else:
            txtEEGsys = 'noEEGsys'

        #Subjects normality
        if subjNormalStateNext == 0:
            normalEEG = 1
            txtNormalState = 0
        elif subjNormalStateNext == 1:
            abnormalEEG = 1
            txtNormalState = 1
        else:
            noImpressionEEG = 1
            txtNormalState = 2
    
    #Keep count of subjects with criteria found
    for k in range(len(subjKeywords)):
        if subjKeywords[k] != 'notFound':
            keyCount[k] += 1
    #Number of subjects with only last criteria
    if (subjKeywords[0] == 'notFound') and (subjKeywords[1] == 'notFound') and subjKeywords[4] != 'notFound':
        if int(session) == 1:
            specialCount += 1
            specialSubj.append(name+'_'+session)

    #Find all associated EDf files
    edfExt = txtFileDirectory+txtFile.split('.')[0]+'*.edf'
    '''
    for edfFile in glob.glob(edfExt):
        edfFile = edfFile.rstrip()
        subjEEG,errorValue=readEDF(edfFile) #subjEEG
        numEDF += 1
        if errorValue != '':
            valueErrorCount += 1
            txtErrorCount +=1
        elif txtEEGsys == 'yes':
        else:
            errorValue = 'noError'
            noEEGerr += 1
            ####
            try:
                #subjEEGdata,eegChannelNames,srate = preProcessEEG(subjEEG) #subjPreProcEEG,subjChannelNames,subjeegChannelNum,
                refData,times = preProcessData(subjEEG)
                #print 'After:',np.size(subjEEGdata),'srate:',srate
                if subjGender == 'male':
                    genderNP[0] +=1
                elif subjGender == 'female':
                    genderNP[1] +=1
                elif (subjGender == 'noGender') or (subjGender == ''):
                    genderNP[2] +=1
            except MemoryError:
                print 'Data:',edfFile.split('/')[-1],'; Error.'
                continue
            try:
                psdOfData = getPSD(refData)
            except RuntimeWarning:
                print 'Data:',edfFile.split('/')[-1],'; PSD-Error.'
                continue
            #print 'Data:',np.shape(refData)
            ###
            #subjEdfFiles.append(edfFile)
            #singleTXTdata.append(subjEEG)
            ###
            #if noEEGerr == 1:
            #    singleTXTdata = [refData]
            #else:
            #    singleTXTdata = [singleTXTdata,refData]
            #print np.shape(singleTXTdata)
    '''
    printCSV(name,session,numEDF,txtEEGsys,txtAge,txtGender,subjKeywords,txtMed,newName)
    return [male,female,none,age,goodAge,badAge,medFound,eegSysFound,keyCount,valueErrorCount,noEEGerr,genderNP,int(name),subjGender,txtMed,txtNormalState,normalEEG,abnormalEEG,noImpressionEEG,subjEdfFiles,singleTXTdata]


def getMedsListStr(txtMed):
    txtMedStr = ''
    for med in txtMed:
        if med == txtMed[-1]:
            txtMedStr = txtMedStr+med
        else:
            txtMedStr = txtMedStr+med+' '
    return txtMedStr

def getDate(fileName):
    curFileArray = fileName.split('/')[-1].split('.')[0].split('_')
    year = int(curFileArray[2])
    month = int(curFileArray[3])
    day = int(curFileArray[4])
    return [year,month,day]

def getAgeStr(txtAge):
    if len(txtAge) == 0:
        txtAgeStr = 'None'
    else:
        txtAgeStr = str(txtAge[0])
    return txtAgeStr

def printCSV(name,session,numEDF,txtEEGsys,txtAge,txtGender,subjKeywords,txtMed,newName):
    txtMedStr = getMedsListStr(txtMed)
    if newName == 1:
        print name,session,numEDF,txtEEGsys,txtAge,txtGender,subjKeywords[0],subjKeywords[1],subjKeywords[2],subjKeywords[3],subjKeywords[4],txtMedStr

def printMatrix(mat,varNames=None):

    if mat.ndim == 1:
        for i in mat:
            print i

    elif mat.ndim == 2:
        for i in range(np.shape(mat)[0]):
            for j in range(np.shape(mat)[1]):
                if (j==range(np.shape(mat)[1])[-1]):
                    print mat[i][j]
                else:
                    print mat[i][j],',',

    elif mat.ndim == 3:
        if (varNames == None) or (len(varNames) != np.shape(mat)[0]):
            varNames = range(np.shape(mat)[0])
        for i in range(np.shape(mat)[0]):
            print 'Variable',varNames[i],':'
            for j in range(np.shape(mat)[1]):
                for k in range(np.shape(mat)[2]):
                    if (k==range(np.shape(mat)[2])[-1]):
                        print mat[i][j][k]
                    else:
                        print mat[i][j][k],',',
            print '\n'

    else:
        print 'Dimensions are not 1,2 or 3.'

def getUniqueSubj(txtFile):
    #txtFile = str(sys.argv[1])
    f = open(txtFile,'r')
    content = f.readlines()
    subjCaptured = []
    for i in content:
        subj = i.split('/')[8]
        if subj not in subjCaptured:
            subjCaptured.append(subj)
    
    print len(subjCaptured)

if __name__ == '__main__':
    start = time.time()
    #directory = '/media/david/WD 2TB EXT/EEGcorpus/'
    allTXT = DIRECTORY+str(sys.argv[1])
    male,female,none,avgAge,goodAge,badAge,medFound,eegSysFound,keyCount,edfError,noError,genderNP,normalEEG,abnormalEEG,noImpressionEEG = mainTXT(allTXT)  
    #mainTXT(allTXT)
    end=time.time()
    print '\nMale Recordings:',male,'; Female Recordings:',female, '; Neither:',none
    print 'Male Data Points:',int(genderNP[0][0]),'; Female Data Points:',int(genderNP[1][0])
    print 'Ages Captured:',goodAge,'; No Age:',badAge,'; Mean Age:',avgAge
    print 'Medication Found:',medFound, '; EEG 10-20 Found:',eegSysFound,'; EDFErrorCount:',edfError,'/',noError
    print 'Keywords Found:',int(keyCount[0][0]),int(keyCount[1][0]),int(keyCount[2][0]),int(keyCount[3][0]),int(keyCount[4][0])
    print 'Normal EEG Found:',normalEEG,'; Abnormal EEG Found:',abnormalEEG,'; Neither:',noImpressionEEG
    print '\nTime Elapsed:',end-start,'\n'
