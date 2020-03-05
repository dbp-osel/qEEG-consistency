import numpy as np
from scipy import stats
import pdb
import time
import sys
from scipy.linalg import pinv
import datetime
import random
random.seed(11081992)
#import matplotlib.pyplot as plt
import pylab as pl

import topo_plot
from nedcTools import resultBoxPlot,dataLoad3D,dataLoad4D

from multiprocessing import Pool
from functools import partial

def defineParams():
    features = ['all']
    partitions = 128
    timeMin = 16
    threads = 1
    write2File = 0
    featsNames = ''
    textFile = 'allTextFiles'
    for x in features:
        featsNames = featsNames+x
    return features,partitions,timeMin,threads,write2File,featsNames,textFile

def generateTdist(data,labels,runs=1000):
    subjArray = getSubjArray(labels)
    #pdb.set_trace()
    subj1 = 0
    subj2 = 1

    n = np.shape(data)[-1]

    allT = np.zeros((np.shape(data)[1],np.shape(data)[2],runs))

    for c in range(np.shape(data)[1]):
        for v in range(np.shape(data)[2]):
            subjCaptured = []
            sigP = 0
            CV = []
            multiSession = 0
            goodDates = []
            badDates = []
            goodAges = []
            badAges = []

            for i in range(runs):
                randSubj1 = -1
                randSubj2 = -1
                while (subjArray[randSubj1] == subjArray[randSubj2]):
                    randSubj1 = random.randint(0,len(labels)-1)
                    randSubj2 = random.randint(0,len(labels)-1)

                #tScore = (np.mean(data[randSubj1,c,v,:])-np.mean(data[randSubj2,c,v,:]))/(np.std(data[randSubj1,c,v,:])/np.sqrt(n))
                tScore = (np.mean(data[randSubj1,c,v,:])-np.mean(data[randSubj2,c,v,:]))/(np.std(data[randSubj1,c,v,:])/np.sqrt(n))

                allT[c,v,i] = tScore

    return allT
    
def tScoreAnaly2D(data, labels):
    print('Data:',np.shape(data))
    
    subjArray = getSubjArray(labels)
    #pdb.set_trace()
    subj1 = 0
    subj2 = 1
    runs = 1000
    allT = generateTdist(data,labels,runs)

    print('T-distribution generated!')

    n = np.shape(data)[-1]

    allSigP = np.zeros((np.shape(data)[1],np.shape(data)[2]))
    allAvgDate = np.zeros((np.shape(data)[1],np.shape(data)[2],2))
    allAges = np.zeros((np.shape(data)[1],np.shape(data)[2],2))

    for c in range(np.shape(data)[1]):
        for v in range(np.shape(data)[2]):
            print 'Channel:',c,'Variable:',v
            subjCaptured = []
            sigP = 0
            multiSession = 0
            goodDates = []
            badDates = []
            goodAges = []
            badAges = []

            for i in range(len(labels)):
                curSubj = labels[i][0].split('_')[0]
                if curSubj in subjCaptured:
                    continue
                subjCaptured,dataCaptured = getDiffSessions(labels,subjArray,subjCaptured,curSubj)

                if len(dataCaptured) > 1:

                    multiSession += 1
                    #pdb.set_trace()
                    tScore = (np.mean(data[dataCaptured[subj2],c,v,:])-np.mean(data[dataCaptured[subj1],c,v,:]))/(np.std(data[dataCaptured[subj2],c,v,:])/np.sqrt(n))

                    percentAbove = np.divide(np.sum(i > tScore for i in allT[c,v,:]),runs/100.)
                    if percentAbove < 50:
                        percentAbove = 100-percentAbove
                    #print 'Percent Above:',percentAbove
                    meanAge = getMeanAge([labels[dataCaptured[subj1],3],labels[dataCaptured[subj2],3]])
                    if percentAbove > 95:
                        print 'Significant Percent Above:',percentAbove
                        sigP += 1
                        goodDates.append(getDateDiff([labels[dataCaptured[subj1],1],labels[dataCaptured[subj2],1]]))#append(getDateDiff(labels[dataCaptured[0:2],1]))
                        if meanAge > 0:
                            goodAges.append(meanAge)
                    else:
                        badDates.append(getDateDiff([labels[dataCaptured[subj1],1],labels[dataCaptured[subj2],1]]))#append(getDateDiff(labels[dataCaptured[0:2],1]))
                        if meanAge > 0:
                            badAges.append(meanAge)
            
            if len(goodDates) == 0:
                goodDates.append(0)
            if len(badDates) == 0:
                badDates.append(0)
            if len(goodAges) == 0:
                goodAges.append(0)
            if len(badAges) == 0:
                badAges.append(0)
            
            allSigP[c,v] = sigP
            allAvgDate[c,v,:] = [np.mean(goodDates),np.mean(badDates)]
            allAges[c,v,:] = [np.mean(goodAges),np.mean(badAges)]

    Partitions = n
    useRand = 0
    for c in range(np.shape(data)[1]):
        print 'Channels: ',c
        print 'Multi-Session Subjects:',multiSession,'/',len(set(subjArray))
        print 'More than 75% Significant:', sum(ii > 0.7*multiSession for ii in allSigP[c,:]),'/',len(allSigP[c,:])
        print 'Average Significant:', np.mean(allSigP[c,:]),'/',multiSession
        print 'Sig Date Diff: Mean-',np.mean(allAvgDate[c,np.nonzero(allAvgDate[c,:,0])]),'; SD-',np.std(allAvgDate[c,np.nonzero(allAvgDate[c,:,0])])
        print 'non-Sig Date Diff: Mean-',np.mean(allAvgDate[c,np.nonzero(allAvgDate[c,:,1])]),'; SD-',np.std(allAvgDate[c,np.nonzero(allAvgDate[c,:,1])])
        print 'Sig Ages: Mean-',np.mean(allAges[c,np.nonzero(allAges[c,:,0])]),'; SD-',np.std(allAges[c,np.nonzero(allAges[c,:,0])])
        print 'non-Sig Ages: Mean-',np.mean(allAges[c,np.nonzero(allAges[c,:,1])]),'; SD-',np.std(allAges[c,np.nonzero(allAges[c,:,1])])
    for v in range(np.shape(data)[2]):
        print 'Feature: ',v
        print 'Multi-Session Subjects:',multiSession,'/',len(set(subjArray))
        print 'More than 75% Significant:', sum(ii > 0.7*multiSession for ii in allSigP[:,v]),'/',len(allSigP[:,v])
        print 'Average Significant:', np.mean(allSigP[:,v]),'/',multiSession
        print 'Sig Date Diff: Mean-',np.mean(allAvgDate[np.nonzero(allAvgDate[:,v,0]),v]),'; SD-',np.std(allAvgDate[np.nonzero(allAvgDate[:,v,0]),v])
        print 'non-Sig Date Diff: Mean-',np.mean(allAvgDate[np.nonzero(allAvgDate[:,v,1]),v]),'; SD-',np.std(allAvgDate[np.nonzero(allAvgDate[:,v,1]),v])
        print 'Sig Ages: Mean-',np.mean(allAges[np.nonzero(allAges[:,v,0]),v]),'; SD-',np.std(allAges[np.nonzero(allAges[:,v,0]),v])
        print 'non-Sig Ages: Mean-',np.mean(allAges[np.nonzero(allAges[:,v,1]),v]),'; SD-',np.std(allAges[np.nonzero(allAges[:,v,1]),v])


    fig = pl.figure(figsize=(9, 6))
    fig,ax = resultBoxPlot(fig,allSigP,totalInst=multiSession)
    ax.set_ylim(ymin=0)
    fig.suptitle('t-Test Results of Intra-subject Variability of Features, %d parts (n=%d)'%(Partitions,multiSession))
    fig.subplots_adjust(left=0.09,bottom=0.33,right=0.96,top=0.93,wspace=0.2,hspace=0.2)
    fig.savefig('images/tScoreIntra%d-%d.eps'%(Partitions,useRand), format='eps', dpi=1000)

    fig = pl.figure(figsize=(9, 6))
    fig,ax = resultBoxPlot(fig,allAvgDate[:,:,0])
    fig.suptitle('t-Test Significant Days Between of Intra-subject Variability of Features, %d parts (n=%d)'%(Partitions,multiSession))
    ax.set_ylabel('Days Between Sessions')
    fig.subplots_adjust(left=0.09,bottom=0.33,right=0.96,top=0.93,wspace=0.2,hspace=0.2)
    fig.savefig('images/tScoresigDays%d-%d.eps'%(Partitions,useRand), format='eps', dpi=1000)

    fig = pl.figure(figsize=(9, 6))
    fig,ax = resultBoxPlot(fig,allAvgDate[:,:,1])
    fig.suptitle('t-Test Non-Significant Days Between of Intra-subject Variability of Features, %d parts (n=%d)'%(Partitions,multiSession))
    ax.set_ylabel('Days Between Sessions')
    fig.subplots_adjust(left=0.09,bottom=0.33,right=0.96,top=0.93,wspace=0.2,hspace=0.2)
    fig.savefig('images/tScorenonSigDays%d-%d.eps'%(Partitions,useRand), format='eps', dpi=1000)

    fig = pl.figure(figsize=(9, 6))
    fig,ax = resultBoxPlot(fig,allAges[:,:,0])
    fig.suptitle('t-Test Significant Ages of Intra-subject Variability of Features, %d parts (n=%d)'%(Partitions,multiSession))
    ax.set_ylabel('Age of Subject')
    fig.subplots_adjust(left=0.09,bottom=0.33,right=0.96,top=0.93,wspace=0.2,hspace=0.2)
    fig.savefig('images/tScoresigAges%d-%d.eps'%(Partitions,useRand), format='eps', dpi=1000)

    fig = pl.figure(figsize=(9, 6))
    fig,ax = resultBoxPlot(fig,allAges[:,:,1])
    fig.suptitle('t-Test Non-Significant Ages of Intra-subject Variability of Features, %d parts (n=%d)'%(Partitions,multiSession))
    ax.set_ylabel('Ages of Subject')
    fig.subplots_adjust(left=0.09,bottom=0.33,right=0.96,top=0.93,wspace=0.2,hspace=0.2)
    fig.savefig('images/tScorenonSigAges%d-%d.eps'%(Partitions,useRand), format='eps', dpi=1000)

    #print allSigP
    #print allAvgDate
    
    return allSigP,allAvgDate,allAges,multiSession

def generateKWdist(data,labels,runs=10,useRand=1,exclude=0):
    maxChanSigP = np.zeros((runs,np.shape(data)[2]))
    maxChanSigPLoc = np.zeros((runs,np.shape(data)[2]))
    minSigDate = np.zeros((runs,np.shape(data)[2]))
    minNonSigDate = np.zeros((runs,np.shape(data)[2]))

    channels = 19
    for r in range(runs):
        print "Run:", r
        print('Data:',np.shape(data))
        pool = Pool(processes=channels)
        parFunction = partial(parKWAnaly2D,data=data,labels=labels,useRand=useRand)
        results = pool.map(parFunction,range(np.shape(data)[1]))#,contentList)
        pool.close()
        pool.join()

        allSigP = results[0][0]
        allAvgDate = results[0][1]
        allAges = results[0][2]
        multiSession = results[0][3]
        
        for l in range(1,len(results)):
            allSigP = np.concatenate((allSigP,results[l][0]),axis=0)
            allAvgDate = np.concatenate((allAvgDate,results[l][1]),axis=0)
            allAges = np.concatenate((allAges,results[l][2]),axis=0)

        for v in range(np.shape(allSigP)[1]):
            maxChanSigP[r,v] = np.max(allSigP[:,v])
            maxChanSigPLoc[r,v] = np.argmax(allSigP[:,v])
            minSigDate[r,v] = np.max(allAvgDate[:,v,0])
            minNonSigDate[r,v] = np.max(allAvgDate[:,v,1])

    
    if useRand == 1:
        print 'non-Random Run:'
        print('Data:',np.shape(data))
        pool = Pool(processes=channels)
        parFunction = partial(parKWAnaly2D,data=data,labels=labels,useRand=0)
        results = pool.map(parFunction,range(np.shape(data)[1]))#,contentList)
        pool.close()
        pool.join()

        allSigP = results[0][0]
        allAvgDate = results[0][1]
        allAges = results[0][2]
        multiSession = results[0][3]
        
        for l in range(1,len(results)):
            allSigP = np.concatenate((allSigP,results[l][0]),axis=0)
            allAvgDate = np.concatenate((allAvgDate,results[l][1]),axis=0)
            allAges = np.concatenate((allAges,results[l][2]),axis=0)

    print 'All mean:',np.mean(np.ravel(allAvgDate),axis=0),'Std:',np.std(np.ravel(allAvgDate),axis=0),'n=',np.size(np.ravel(allAvgDate))
    print 'Good mean:',np.mean(np.ravel(allAvgDate[:,:,0]),axis=0),'Std:',np.std(np.ravel(allAvgDate[:,:,0]),axis=0),'n=',np.size(np.ravel(allAvgDate[:,:,0]))
    print 'Bad mean:',np.mean(np.ravel(allAvgDate[:,:,1]),axis=0),'Std:',np.std(np.ravel(allAvgDate[:,:,1]),axis=0),'n=',np.size(np.ravel(allAvgDate[:,:,1]))


    fig = pl.figure(figsize=(9, 6))
    fig,ax = resultBoxPlot(fig,allSigP,totalInst=multiSession,exclude=exclude)
    if useRand == 1:
        for r in range(runs):
            ax.plot(np.concatenate((np.array([0]),np.divide(maxChanSigP[r],multiSession/100.)),axis=0),'b.')
    ax.set_ylim(ymin=0)
    fig.suptitle('Kruskal-Wallis Results of Intra-subject Variability of Features, %d parts (n=%d)'%(np.shape(data)[-1],multiSession))
    fig.subplots_adjust(left=0.09,bottom=0.40,right=0.96,top=0.93,wspace=0.2,hspace=0.2)
    #fig.savefig('images/randResultsKWIntra%d-%d.eps'%(Partitions,useRand), format='eps', dpi=1000)
    
    fig = pl.figure(figsize=(9, 6))
    fig,ax = resultBoxPlot(fig,allAvgDate[:,:,0],exclude=exclude)
    #for r in range(runs):
    #    ax.plot(np.concatenate((np.array([0]),minSigDate[r]),axis=0),'b.')
    ax.set_ylim(ymin=0)
    fig.suptitle('Kruskal-Wallis Significant Days Between of Intra-subject Variability of Features, %d parts (n=%d)'%(Partitions,multiSession))
    ax.set_ylabel('Days Between Sessions')
    fig.subplots_adjust(left=0.09,bottom=0.33,right=0.96,top=0.93,wspace=0.2,hspace=0.2)
    #fig.savefig('images/RandResultsKWsigDays%d-%d.eps'%(Partitions,useRand), format='eps', dpi=1000)

    fig = pl.figure(figsize=(9, 6))
    fig,ax = resultBoxPlot(fig,allAvgDate[:,:,1],exclude=exclude)
    #for r in range(runs):
    #    ax.plot(np.concatenate((np.array([0]),minNonSigDate[r]),axis=0),'b.')
    ax.set_ylim(ymin=0)
    fig.suptitle('Kruskal-Wallis Non-Significant Days Between of Intra-subject Variability of Features, %d parts (n=%d)'%(Partitions,multiSession))
    ax.set_ylabel('Days Between Sessions')
    fig.subplots_adjust(left=0.09,bottom=0.40,right=0.96,top=0.93,wspace=0.2,hspace=0.2)
    #fig.savefig('images/randResultsKWnonSigDays%d-%d.eps'%(Partitions,useRand), format='eps', dpi=1000)
    '''
    featNamesL = ['Rel-Lower', 'Rel-Delta', 'Rel-Theta', 'Rel-Alpha', 'Rel-Mu', 'Rel-Beta', 'Rel-Gamma', 
                'Abs-Lower', 'Abs-Delta', 'Abs-Theta', 'Abs-Alpha', 'Abs-Mu', 'Abs-Beta', 'Abs-Gamma', 
                'Fourier-Entropy', 'NormedFourier-Entropy', 'Fourier-Entropy-Norm', 'NormedFourier-Entropy-Norm', 'Spectral-Entropy', 'Spectral-Entropy-Norm', 
                'Entropy', 'Entropy-Norm', 'Curve-Length', 'Energy', 'Nonlinear-Energy', 'Sixth-Power', 'LZC', 
                'Min', 'Max', 'Med', 'Var', 'Std', 'Skew', 'Kurtosis']
    for v in range(np.shape(allAvgDate)[1]):
        meanP = stats.kruskal(allAvgDate[:,v,0],allAvgDate[:,v,1])[1]#data[randSubj,c,v,:])[1]#data[dataCaptured[subj2],c,v,:])[1]#stats.kruskal(*data[dataCaptured,c,:,v])[1]#data[randSubj,c,v,:])[1]#
        print 'v:',featNamesL[v],'p-value:', meanP
    '''
    mask = np.concatenate((range(14),range(18,34)))

    #np.save('allSigPTopo.npy',np.transpose(np.divide(allSigP[:,mask],multiSession/100.)))
        
    #fig = pl.figure()
    #topo_plot.plotEEGData(fig,np.transpose(np.divide(allSigP[:,mask],multiSession/100.)))

    #fig = pl.figure()
    #topo_plot.plotEEGData(fig,np.transpose(np.divide(allSigP[:,mask],multiSession/1.)))

    #pl.show()

    return allSigP,allAvgDate,allAges,multiSession,maxChanSigP,maxChanSigPLoc


def parKWAnaly2D(channels, data, labels, useRand=0):
    #print('Data:',np.shape(data))
    subjArray = getSubjArray(labels)
    #pdb.set_trace()
    channelArray = [channels]
    channelNum = len(channelArray)
    subj1 = 0
    subj2 = 1
    Partitions = np.shape(data)[-1]
    allSigP = np.zeros((channelNum,np.shape(data)[2]))
    allAvgDate = np.zeros((channelNum,np.shape(data)[2],2))
    allAges = np.zeros((channelNum,np.shape(data)[2],2))

    for c in channelArray:
        for v in range(np.shape(data)[2]):
            subjCaptured = []
            sigP = 0
            CV = []
            multiSession = 0
            goodDates = []
            badDates = []
            goodAges = []
            badAges = []

            for i in range(len(labels)):
                curSubj = labels[i][0].split('_')[0]
                if curSubj in subjCaptured:
                    continue
                subjCaptured,dataCaptured = getDiffSessions(labels,subjArray,subjCaptured,curSubj)


                if len(dataCaptured) > 1:

                    if useRand == 1:
                        randSubj = -1
                        while (randSubj in subjCaptured) or (randSubj<0):
                            randSubj = random.randint(0,len(labels)-1) 
                        dataCaptured[subj2] = randSubj

                    multiSession += 1
                    #pdb.set_trace()
                    #print(labels[dataCaptured[0]][0]+','+labels[dataCaptured[1]][0])
                    ###control for number of samples
                    KWfactor = int(np.divide(np.shape(data)[3],8.))
                    KWmask = list([int(x*KWfactor) for x in range(8)])
                    #pdb.set_trace()

                    meanP = stats.kruskal(data[dataCaptured[subj1],c,v,KWmask],data[dataCaptured[subj2],c,v,KWmask])[1]
                    ###control for number of samples
                    
                    #meanP = stats.kruskal(data[dataCaptured[subj1],c,v,:],data[dataCaptured[subj2],c,v,:])[1]#data[randSubj,c,v,:])[1]#data[dataCaptured[subj2],c,v,:])[1]#stats.kruskal(*data[dataCaptured,c,:,v])[1]#data[randSubj,c,v,:])[1]#
                    #print 'Variable',v,'p-value on',len(dataCaptured),'subjects:',meanP
                    meanAge = getMeanAge([labels[dataCaptured[subj1],3],labels[dataCaptured[subj2],3]])
                    #print [labels[dataCaptured[subj1]],labels[dataCaptured[subj2]]]

                    if meanP > 0.05:
                        sigP += 1
                        goodDates.append(getDateDiff([labels[dataCaptured[subj1],1],labels[dataCaptured[subj2],1]]))#append(getDateDiff(labels[dataCaptured[0:2],1]))
                        if meanAge > 0:
                            goodAges.append(meanAge)
                    else:
                        badDates.append(getDateDiff([labels[dataCaptured[subj1],1],labels[dataCaptured[subj2],1]]))#append(getDateDiff(labels[dataCaptured[0:2],1]))
                        if meanAge > 0:
                            badAges.append(meanAge)
                    
                    #print 'Number of significant P-values',meanP
                    #print 'Coeff of Var for channel',c,'variable',v,':',CV[-1],'\n'
            if len(goodDates) == 0:
                goodDates.append(0)
            if len(badDates) == 0:
                badDates.append(0)
            if len(goodAges) == 0:
                goodAges.append(0)
            if len(badAges) == 0:
                badAges.append(0)
            
            #pdb.set_trace()

            allSigP[channelArray.index(c),v] = sigP
            allAvgDate[channelArray.index(c),v,:] = [np.mean(goodDates),np.mean(badDates)]
            allAges[channelArray.index(c),v,:] = [np.mean(goodAges),np.mean(badAges)]
    
    return allSigP,allAvgDate,allAges,multiSession

def generateKendallDist(data,labels,numSessions=2,runs=5,useRand=1,exclude=0):
    maxChanKendall = np.zeros((runs,np.shape(data)[2]))
    maxChanKendallLoc = np.zeros((runs,np.shape(data)[2]))

    for r in range(runs):
        print "Run:", r
        allData = data[:,:,:,random.randint(0,np.shape(data)[-1]-1)]

        if useRand == 0:
            tabledData = tableData(allData,allLabels,numSessions=numSessions)
        elif useRand == 1:
            tabledData = randTableData(allData,allLabels,numSessions=numSessions)
        
        print 'Tabled Data:',tabledData.shape
        allKW = np.zeros((np.shape(tabledData)[2],np.shape(tabledData)[3]))
        #allpVal = np.zeros((np.shape(allData)[2],np.shape(tabledData)[3]))
        #pdb.set_trace()

        for v in range(np.shape(tabledData)[3]):
            pool = Pool(processes=19)
            parFunction = partial(parKendallW,X=tabledData[:,:,:,v])
            results = pool.map(parFunction,range(19))#,contentList)
            pool.close()
            pool.join()

            allKW[:,v] = results

        for v in range(np.shape(allKW)[1]):
            maxChanKendall[r,v] = np.max(np.abs(allKW[:,v]))
            maxChanKendallLoc[r,v] = np.argmax(np.abs(allKW[:,v]))

    if useRand == 1:
        print 'non-Random Run:'
        allData = data[:,:,:,random.randint(0,np.shape(data)[-1]-1)]

        tabledData = tableData(allData,allLabels,numSessions=numSessions)
        
        print 'Tabled Data:',tabledData.shape
        allKW = np.zeros((np.shape(tabledData)[2],np.shape(tabledData)[3]))
        #allpVal = np.zeros((np.shape(allData)[2],np.shape(tabledData)[3]))
        #pdb.set_trace()

        for v in range(np.shape(tabledData)[3]):
            pool = Pool(processes=19)
            parFunction = partial(parKendallW,X=tabledData[:,:,:,v])
            results = pool.map(parFunction,range(19))#,contentList)
            pool.close()
            pool.join()

            allKW[:,v] = results

    # Create a figure instance
    fig = pl.figure(figsize=(9, 6))
    fig,ax = resultBoxPlot(fig,allKW,exclude=exclude)
    for r in range(runs):
        ax.plot(np.concatenate((np.array([0]),maxChanKendall[r]),axis=0),'b.')
    fig.suptitle("Kendall's W, %d parts with %d sessions (n=%d)"%(np.shape(data)[-1],np.shape(tabledData)[1],np.shape(tabledData)[0]))
    ax.set_ylabel('Correlation Coefficient')
    ax.set_ylim(ymin=0,ymax=1)
    fig.subplots_adjust(left=0.09,bottom=0.40,right=0.96,top=0.93,wspace=0.2,hspace=0.2)
    #fig.savefig('images/randResultsKendallW%d.eps'%(Partitions), format='eps', dpi=1000)
    #pl.show()
    return allKW,maxChanKendall,maxChanKendallLoc,np.shape(tabledData)[0]

def randTableData(data,labels,numSessions=3):
    subjArray = getSubjArray(labels)
    allDataCapt = []
    subjCaptured = []
    multiSession = 0

    for i in range(len(labels)-1):
        curSubj = labels[i][0].split('_')[0]
        if curSubj in subjCaptured:
            continue
        subjCaptured,dataCaptured = getDiffSessions(labels,subjArray,subjCaptured,curSubj)
        #print len(subjCaptured)
        if len(dataCaptured) >= numSessions:
            for s in range(1,numSessions):
                randSubj = -1
                while (randSubj in subjCaptured) or (randSubj<0):
                    randSubj = random.randint(0,len(labels)-1) 
                    dataCaptured[s] = randSubj
                    
            allDataCapt.append(data[dataCaptured[0:numSessions]])
            dataCaptured = []

    return np.array(allDataCapt)

def parKendallW(channel,X):
    X = X[:,:,channel]

    k = np.shape(X)[0]
    m = np.shape(X)[1]

    for i in range(m):
        X[:,i] = stats.rankdata(X[:,i],method='ordinal')
    #pdb.set_trace()

    sumSubj = np.sum(X,1)
    meanSumSubj = np.mean(sumSubj)
    devSq = sum(np.square(np.subtract(sumSubj,meanSumSubj)),0)
    #mean_X = np.mean(X)
    #SST = ((X-mean_X)**2).sum()
    W = 12*(devSq)/((m**2)*((k**3)-k))
    r = (m*W-1)/(m-1)
    
    chiSq = m*(k-1)*W
    df = k-1
    pval = stats.chi2.pdf(chiSq,df)

    #Alternate
    #W = 12*np.sum(np.square(sumSubj),0)/((m**2)*((k**3)-k)-3*((k+1)/(k-1)))
    #Wfried,pval = stats.friedmanchisquare(*X)
    #W = Wfried/(m*(k-1))
    #pdb.set_trace()
    return r

def parSpearman(channel,X):
    skew = stats.skew(X[:,:,channel])
    r = stats.spearmanr(X[:,:,channel])[0]
    print 'Channel:',channel,'; Skew:',skew,'r:',r
    return r

def getNormData(dataLoad,labelsLoad,normal=1):
    labelsN = []
    dataN = []
    labelsA = []
    dataA = []
    
    for n in range(len(labelsLoad)):
        if (labelsLoad[n][5] == 0):
            labelsN.append(labelsLoad[n,:])
            dataN.append(dataLoad[n,:])
        elif (labelsLoad[n][5] == 1):
            labelsA.append(labelsLoad[n,:])
            dataA.append(dataLoad[n,:])
    labelsN = np.array(labelsN)
    dataN = np.array(dataN)
    labelsA = np.array(labelsA)
    dataA = np.array(dataA)

    if normal == 1:
        return dataN,labelsN
    elif normal == 0:
        return dataA,labelsA

def kendallWold(X):
    m = np.shape(X)[0]
    k = np.shape(X)[1]
    pdb.set_trace()

    for i in range(m):
        X[:,i] = stats.rankdata(X[:,i],method='ordinal')
    #pdb.set_trace()

    sumSubj = np.sum(X,1)
    meanSumSubj = np.mean(sumSubj)
    devSq = sum(np.square(np.subtract(sumSubj,meanSumSubj)),0)
    #mean_X = np.mean(X)
    #SST = ((X-mean_X)**2).sum()
    W = 12*(devSq)/((m**2)*((k**3)-k))
    r = (m*W-1)/(m-1)
    chiSq = m*(k-1)*W
    df = k-1
    pval = stats.chi2.pdf(chiSq,df)

    #Alternate
    #W = 12*np.sum(np.square(sumSubj),0)/((m**2)*((k**3)-k)-3*((k+1)/(k-1)))
    #Wfried,pval = stats.friedmanchisquare(*X)
    #W = Wfried/(m*(k-1))
    #pdb.set_trace()

    return r,pval

def kendallW(X):
    k = np.shape(X)[0]
    m = np.shape(X)[1]

    for i in range(m):
        X[:,i] = stats.rankdata(X[:,i],method='ordinal')
    #pdb.set_trace()

    sumSubj = np.sum(X,1)
    meanSumSubj = np.mean(sumSubj)
    devSq = sum(np.square(np.subtract(sumSubj,meanSumSubj)),0)
    #mean_X = np.mean(X)
    #SST = ((X-mean_X)**2).sum()
    W = 12*(devSq)/((m**2)*((k**3)-k))
    r = (m*W-1)/(m-1)
    
    chiSq = m*(k-1)*W
    df = k-1
    pval = stats.chi2.pdf(chiSq,df)

    #Alternate
    #W = 12*np.sum(np.square(sumSubj),0)/((m**2)*((k**3)-k)-3*((k+1)/(k-1)))
    #Wfried,pval = stats.friedmanchisquare(*X)
    #W = Wfried/(m*(k-1))
    pdb.set_trace()

    return r,pval
def ICC_rep_anova(Y):
    '''
    the data Y are entered as a 'table' ie subjects are in rows and repeated
    measures in columns
    One Sample Repeated measure ANOVA
    Y = XB + E with X = [FaTor / Subjects]
    '''

    [nb_subjects, nb_conditions] = Y.shape
    dfc = nb_conditions - 1
    dfe = (nb_subjects - 1) * dfc
    dfr = nb_subjects - 1

    # Compute the repeated measure effect
    # ------------------------------------

    # Sum Square Total
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    # create the design matrix for the different levels
    x = np.kron(np.eye(nb_conditions), np.ones((nb_subjects, 1)))  # sessions
    x0 = np.tile(np.eye(nb_subjects), (nb_conditions, 1))  # subjects
    X = np.hstack([x, x0])

    # Sum Square Error
    predicted_Y = np.dot(np.dot(np.dot(X, pinv(np.dot(X.T, X))), X.T), Y.flatten('F'))
    residuals = Y.flatten('F') - predicted_Y
    SSE = (residuals ** 2).sum()

    residuals.shape = Y.shape

    MSE = SSE / dfe

    # Sum square session effect - between colums/sessions
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * nb_subjects
    MSC = SSC / dfc / nb_subjects

    session_effect_F = MSC / MSE

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    # ICC(3,1) = (mean square subjeT - mean square error) / (mean square subjeT + (k-1)*-mean square error)
    ICC = (MSR - MSE) / (MSR + dfc * MSE)

    e_var = MSE  # variance of error
    r_var = (MSR - MSE) / nb_conditions  # variance between subjects

    return ICC, r_var#, e_var, session_effect_F, dfc, dfe


def freqAnalyPerVar(data, labels):
    print('Data:',np.shape(data))
    subjArray = getSubjArray(labels)
    allSigP = np.zeros((np.shape(data)[1],1))
    allAvgDate = np.zeros((np.shape(data)[1],2))
    allAges = np.zeros((np.shape(data)[1],2))

    subj1 = 0
    subj2 = 1

    for v in range(np.shape(data)[1]):
        subjCaptured = []
        sigP = 0
        CV = []
        multiSession = 0
        goodDates = []
        badDates = []
        goodAges = []
        badAges = []
        
        for i in range(len(labels)-1):
            curSubj = labels[i][0].split('_')[0]
            if curSubj in subjCaptured:
                continue
            subjCaptured,dataCaptured = getDiffSessions(labels,subjArray,subjCaptured,curSubj)

            if len(dataCaptured) > 1:
                multiSession += 1
                #pdb.set_trace()
                meanP = stats.kruskal(data[dataCaptured[subj1],v,:],data[dataCaptured[subj2],v,:])[1]#stats.kruskal(*data[dataCaptured[0:2],v,:])[1]
                #meanP = stats.f_oneway(*data[dataCaptured[0:2],:,v])[1] # Assuming Normal
                #print 'Variable',v,'p-value on',len(dataCaptured),'subjects:',meanP
                meanAge = getMeanAge([labels[dataCaptured[subj1],3],labels[dataCaptured[subj2],3]])
                if meanP > 0.05:
                    sigP += 1
                    goodDates.append(getDateDiff([labels[dataCaptured[subj1],1],labels[dataCaptured[subj2],1]]))#append(getDateDiff(labels[dataCaptured[0:2],1]))
                    if meanAge > 0:
                        goodAges.append(meanAge)
                else:
                    badDates.append(getDateDiff([labels[dataCaptured[subj1],1],labels[dataCaptured[subj2],1]]))#append(getDateDiff(labels[dataCaptured[0:2],1]))
                    if meanAge > 0:
                        badAges.append(meanAge)

                if (np.min(np.min(data[dataCaptured[0:2],v,:],axis=1),axis=0) > 0):
                    meanPop = np.mean(data[dataCaptured[0:2],v,:])
                    stdPop = np.std(data[dataCaptured[0:2],v,:])
                    CV.append((stdPop/meanPop)*100.)

                #print 'Number of significant P-values',meanP
                #print 'Coeff of Var for variable',v,':',CV[-1],'\n'


        if len(CV) == 0:
            CV.append(0)
        if len(goodDates) == 0:
            goodDates.append(0)
        if len(badDates) == 0:
            badDates.append(0)
        if len(goodAges) == 0:
            goodAges.append(0)
        if len(badAges) == 0:
            badAges.append(0)

        allSigP[v] = sigP
        allAvgDate[v,:] = [np.mean(goodDates),np.mean(badDates)]
        allAges[v,:] = [np.mean(goodAges),np.mean(badAges)]
    
        #print 'Number of significant P-values',sigP,'of',multiSession
        #print 'Average sigP days:',np.mean(goodDates),'; Average non-sigP days:',np.mean(badDates)
        #print 'Coeff of Var for variable',v,':',np.mean(CV), 'with ',len(CV),' subjects\n'

    #print allSigP
    #print allAvgDate
    print 'Multi-Session Subjects:',multiSession,'/',len(set(subjArray))
    print 'More than 75% Significant:', sum(i > 0.75*multiSession for i in allSigP),'/',len(allSigP)
    print 'Average Significant:', np.mean(allSigP),'/',multiSession
    print 'Sig Date Diff: Mean-',np.mean(allAvgDate[np.nonzero(allAvgDate[:,0])]),'; SD-',np.std(allAvgDate[np.nonzero(allAvgDate[:,0])])
    print 'non-Sig Date Diff: Mean-',np.mean(allAvgDate[np.nonzero(allAvgDate[:,1])]),'; SD-',np.std(allAvgDate[np.nonzero(allAvgDate[:,1])])
    print 'Sig Ages: Mean-',np.mean(allAges[np.nonzero(allAges[:,0])]),'; SD-',np.std(allAges[np.nonzero(allAges[:,0])])
    print 'non-Sig Ages: Mean-',np.mean(allAges[np.nonzero(allAges[:,1])]),'; SD-',np.std(allAges[np.nonzero(allAges[:,1])]),'\n'

    return allSigP,allAvgDate,allAges,multiSession

def freqAnaly1D(data, labels):
    print('Data:',np.shape(data))
    #for v in range(np.shape(data)[2]):
    subjArray = getSubjArray(labels)

    subj1 = 0
    subj2 = 1

    if len(np.shape(data)) == 3:
        data = np.reshape(data,(np.shape(data)[0],np.shape(data)[1]*np.shape(data)[2]))
    print('Data:',np.shape(data))
    subjCaptured = []
    sigP = 0
    CV = []
    multiSession = 0
    goodDates = []
    badDates = []
    goodAges = []
    badAges = []

    for i in range(len(labels)-1):
        curSubj = labels[i][0].split('_')[0]
        if curSubj in subjCaptured:
            continue
        subjCaptured,dataCaptured = getDiffSessions(labels,subjArray,subjCaptured,curSubj)

        if len(dataCaptured) > 1:
            multiSession += 1
            #pdb.set_trace()
            meanP = stats.kruskal(data[dataCaptured[subj1],:],data[dataCaptured[subj2],:])[1]#stats.kruskal(*data[dataCaptured[0:2],:])[1]
            #print 'Variable',v,'p-value on',len(dataCaptured),'subjects:',meanP
            meanAge = getMeanAge([labels[dataCaptured[subj1],3],labels[dataCaptured[subj2],3]])
            if meanP > 0.05:
                sigP += 1
                goodDates.append(getDateDiff([labels[dataCaptured[subj1],1],labels[dataCaptured[subj2],1]]))#append(getDateDiff(labels[dataCaptured[0:2],1]))
                if meanAge > 0:
                    goodAges.append(meanAge)
            
            else:
                badDates.append(getDateDiff([labels[dataCaptured[subj1],1],labels[dataCaptured[subj2],1]]))#append(getDateDiff(labels[dataCaptured[0:2],1]))
                if meanAge > 0:
                    badAges.append(meanAge)

            if (np.min(np.min(data[dataCaptured[0:2],:],axis=0),axis=0)> 0):
                meanPop = np.mean(data[dataCaptured[0:2],:])
                stdPop = np.std(data[dataCaptured[0:2],:])
                CV.append((stdPop/meanPop)*100.)

            #print 'Number of significant P-values',meanP
            #print 'Coeff of Var for variable',v,':',CV[-1],'\n'
    if len(CV) == 0:
        CV.append(0)
    if len(goodDates) == 0:
        goodDates.append(0)
    if len(badDates) == 0:
        badDates.append(0)
    if len(goodAges) == 0:
        goodAges.append(0)
    if len(badAges) == 0:
        badAges.append(0)    

    print 'Number of significant P-values',sigP,'of',multiSession
    print 'Average sigP days:',np.mean(goodDates),'; Average non-sigP days:',np.mean(badDates)
    print 'Coeff of Var for variables:',np.mean(CV), 'with ',len(CV),' subjects\n'

    return sigP,goodDates,badDates,goodAges,badAges

def freqAnaly2D(data, labels, useRand=0,plots=1):
    print('Data:',np.shape(data))
    subjArray = getSubjArray(labels)
    #pdb.set_trace()
    subj1 = 0
    subj2 = 1
    Partitions = np.shape(data)[-1]
    allSigP = np.zeros((np.shape(data)[1],np.shape(data)[2]))
    allAvgDate = np.zeros((np.shape(data)[1],np.shape(data)[2],2))
    allAges = np.zeros((np.shape(data)[1],np.shape(data)[2],2))

    for c in range(np.shape(data)[1]):
        for v in range(np.shape(data)[2]):
            subjCaptured = []
            sigP = 0
            CV = []
            multiSession = 0
            goodDates = []
            badDates = []
            goodAges = []
            badAges = []

            for i in range(len(labels)):
                curSubj = labels[i][0].split('_')[0]
                if curSubj in subjCaptured:
                    continue
                subjCaptured,dataCaptured = getDiffSessions(labels,subjArray,subjCaptured,curSubj)


                if len(dataCaptured) > 1:

                    if useRand == 1:
                        randSubj = -1
                        while (randSubj in subjCaptured) or (randSubj<0):
                            randSubj = random.randint(0,len(labels)-1) 
                        dataCaptured[subj2] = randSubj

                    multiSession += 1
                    #pdb.set_trace()
                    #print(labels[dataCaptured[0]][0]+','+labels[dataCaptured[1]][0])
                    meanP = stats.kruskal(data[dataCaptured[subj1],c,v,:],data[dataCaptured[subj2],c,v,:])[1]#data[randSubj,c,v,:])[1]#data[dataCaptured[subj2],c,v,:])[1]#stats.kruskal(*data[dataCaptured,c,:,v])[1]#data[randSubj,c,v,:])[1]#
                    #print 'Variable',v,'p-value on',len(dataCaptured),'subjects:',meanP
                    meanAge = getMeanAge([labels[dataCaptured[subj1],3],labels[dataCaptured[subj2],3]])

                    if meanP > 0.05:
                        sigP += 1
                        goodDates.append(getDateDiff([labels[dataCaptured[subj1],1],labels[dataCaptured[subj2],1]]))#append(getDateDiff(labels[dataCaptured[0:2],1]))
                        if meanAge > 0:
                            goodAges.append(meanAge)
                    else:
                        badDates.append(getDateDiff([labels[dataCaptured[subj1],1],labels[dataCaptured[subj2],1]]))#append(getDateDiff(labels[dataCaptured[0:2],1]))
                        if meanAge > 0:
                            badAges.append(meanAge)
                    
                    if (np.min(np.min(data[dataCaptured[0:2],c,v,:],axis=1),axis=0) > 0):
                        meanPop = np.mean(data[dataCaptured[0:2],c,v,:])
                        stdPop = np.std(data[dataCaptured[0:2],c,v,:])
                        CV.append((stdPop/meanPop)*100.)
                    
                    #print 'Number of significant P-values',meanP
                    #print 'Coeff of Var for channel',c,'variable',v,':',CV[-1],'\n'
            if len(CV) == 0:
                CV.append(0)
            if len(goodDates) == 0:
                goodDates.append(0)
            if len(badDates) == 0:
                badDates.append(0)
            if len(goodAges) == 0:
                goodAges.append(0)
            if len(badAges) == 0:
                badAges.append(0)
            
            #pdb.set_trace()

            allSigP[c,v] = sigP
            allAvgDate[c,v,:] = [np.mean(goodDates),np.mean(badDates)]
            allAges[c,v,:] = [np.mean(goodAges),np.mean(badAges)]

            #print 'Number of significant P-values',sigP,'of',multiSession
            #print 'Average sigP days:',np.mean(goodDates),'; Average non-sigP days:',np.mean(badDates)
            #print 'Coeff of Var for channel',c,'variable',v,':',np.mean(CV), 'with ',len(CV),' subjects\n'
            #if np.mean(CV) == 0:
            #    pdb.set_trace()

    if plots == 1:
        for c in range(np.shape(data)[1]):
            print 'Channels: ',c
            print 'Multi-Session Subjects:',multiSession,'/',len(set(subjArray))
            print 'More than 75% Significant:', sum(ii > 0.7*multiSession for ii in allSigP[c,:]),'/',len(allSigP[c,:])
            print 'Average Significant:', np.mean(allSigP[c,:]),'/',multiSession
            print 'Sig Date Diff: Mean-',np.mean(allAvgDate[c,np.nonzero(allAvgDate[c,:,0])]),'; SD-',np.std(allAvgDate[c,np.nonzero(allAvgDate[c,:,0])])
            print 'non-Sig Date Diff: Mean-',np.mean(allAvgDate[c,np.nonzero(allAvgDate[c,:,1])]),'; SD-',np.std(allAvgDate[c,np.nonzero(allAvgDate[c,:,1])])
            print 'Sig Ages: Mean-',np.mean(allAges[c,np.nonzero(allAges[c,:,0])]),'; SD-',np.std(allAges[c,np.nonzero(allAges[c,:,0])])
            print 'non-Sig Ages: Mean-',np.mean(allAges[c,np.nonzero(allAges[c,:,1])]),'; SD-',np.std(allAges[c,np.nonzero(allAges[c,:,1])])
        for v in range(np.shape(data)[2]):
            print 'Feature: ',v
            print 'Multi-Session Subjects:',multiSession,'/',len(set(subjArray))
            print 'More than 75% Significant:', sum(ii > 0.7*multiSession for ii in allSigP[:,v]),'/',len(allSigP[:,v])
            print 'Average Significant:', np.mean(allSigP[:,v]),'/',multiSession
            print 'Sig Date Diff: Mean-',np.mean(allAvgDate[np.nonzero(allAvgDate[:,v,0]),v]),'; SD-',np.std(allAvgDate[np.nonzero(allAvgDate[:,v,0]),v])
            print 'non-Sig Date Diff: Mean-',np.mean(allAvgDate[np.nonzero(allAvgDate[:,v,1]),v]),'; SD-',np.std(allAvgDate[np.nonzero(allAvgDate[:,v,1]),v])
            print 'Sig Ages: Mean-',np.mean(allAges[np.nonzero(allAges[:,v,0]),v]),'; SD-',np.std(allAges[np.nonzero(allAges[:,v,0]),v])
            print 'non-Sig Ages: Mean-',np.mean(allAges[np.nonzero(allAges[:,v,1]),v]),'; SD-',np.std(allAges[np.nonzero(allAges[:,v,1]),v])


        fig = pl.figure(figsize=(9, 6))
        fig,ax = resultBoxPlot(fig,allSigP,totalInst=multiSession)
        ax.set_ylim(ymin=0)
        fig.suptitle('Kruskal-Wallis Results of Intra-subject Variability of Features, %d parts (n=%d)'%(Partitions,multiSession))
        fig.subplots_adjust(left=0.09,bottom=0.33,right=0.96,top=0.93,wspace=0.2,hspace=0.2)
        #fig.savefig('images/KWIntra%d-%d.eps'%(Partitions,useRand), format='eps', dpi=1000)

        fig = pl.figure(figsize=(9, 6))
        fig,ax = resultBoxPlot(fig,allAvgDate[:,:,0])
        fig.suptitle('Kruskal-Wallis Significant Days Between of Intra-subject Variability of Features, %d parts (n=%d)'%(Partitions,multiSession))
        ax.set_ylabel('Days Between Sessions')
        fig.subplots_adjust(left=0.09,bottom=0.33,right=0.96,top=0.93,wspace=0.2,hspace=0.2)
        #fig.savefig('images/KWsigDays%d-%d.eps'%(Partitions,useRand), format='eps', dpi=1000)

        fig = pl.figure(figsize=(9, 6))
        fig,ax = resultBoxPlot(fig,allAvgDate[:,:,1])
        fig.suptitle('Kruskal-Wallis Non-Significant Days Between of Intra-subject Variability of Features, %d parts (n=%d)'%(Partitions,multiSession))
        ax.set_ylabel('Days Between Sessions')
        fig.subplots_adjust(left=0.09,bottom=0.33,right=0.96,top=0.93,wspace=0.2,hspace=0.2)
        #fig.savefig('images/KWnonSigDays%d-%d.eps'%(Partitions,useRand), format='eps', dpi=1000)

        fig = pl.figure(figsize=(9, 6))
        fig,ax = resultBoxPlot(fig,allAges[:,:,0])
        fig.suptitle('Kruskal-Wallis Significant Ages of Intra-subject Variability of Features, %d parts (n=%d)'%(Partitions,multiSession))
        ax.set_ylabel('Age of Subject')
        fig.subplots_adjust(left=0.09,bottom=0.33,right=0.96,top=0.93,wspace=0.2,hspace=0.2)
        #fig.savefig('images/KWsigAges%d-%d.eps'%(Partitions,useRand), format='eps', dpi=1000)

        fig = pl.figure(figsize=(9, 6))
        fig,ax = resultBoxPlot(fig,allAges[:,:,1])
        fig.suptitle('Kruskal-Wallis Non-Significant Ages of Intra-subject Variability of Features, %d parts (n=%d)'%(Partitions,multiSession))
        ax.set_ylabel('Ages of Subject')
        fig.subplots_adjust(left=0.09,bottom=0.33,right=0.96,top=0.93,wspace=0.2,hspace=0.2)
        #fig.savefig('images/KWnonSigAges%d-%d.eps'%(Partitions,useRand), format='eps', dpi=1000)

        #print allSigP
        #print allAvgDate

    return allSigP,allAvgDate,allAges,multiSession

def getSubjArray(labels):
    subjArray = []
    for i in labels[:,0]:
        subjArray.append(i.split('_')[0])
    return np.array(subjArray)

def getDiffSessions(labels,subjArray,subjCaptured,subj):
    curSubjSess = []
    ind = np.where(subjArray == subj)[0]
    #pdb.set_trace()

    if len(ind)>1:
        for i in range(len(ind)-1):
            sessionNameCur = labels[ind[i]][0].split('_')
            sessionNameNext = labels[ind[i+1]][0].split('_')
            if (sessionNameCur[0] == sessionNameNext[0]) and (sessionNameCur[1] != sessionNameNext[1]):
                #pdb.set_trace()

                if subj not in subjCaptured:
                    #pdb.set_trace()
                    subjCaptured.append(subj)
                curSubjSess.append(ind[i])
                if (i == len(ind)-2):
                    curSubjSess.append(ind[i+1])
            
            elif (i>0):
                if (sessionNameCur[0] == labels[ind[i-1]][0].split('_')[0]) and (sessionNameCur[1] != labels[ind[i-1]][0].split('_')[1]):
                    curSubjSess.append(ind[i])

    return subjCaptured,curSubjSess

def getDateDiff(dates):
    #print 'Dates:',dates
    dateDiff = max(dates)-min(dates)
    #print 'Diff:',dateDiff, 'Days:', dateDiff.days
    return dateDiff.days

def getMeanAge(ages):

    if (len(ages[0]) + len(ages[1])) > 1:
        meanAge = np.mean([ages[0][0],ages[1][0]])
    elif len(ages[0]) > 0:
        meanAge = ages[0][0]
    elif len(ages[1]) > 0:
        meanAge = ages[1][0]
    else:
        meanAge = -1

    return meanAge

def tableData(data,labels,numSessions=3):
    subjArray = getSubjArray(labels)
    allDataCapt = []
    subjCaptured = []
    multiSession = 0

    for i in range(len(labels)-1):
        curSubj = labels[i][0].split('_')[0]
        if curSubj in subjCaptured:
            continue
        subjCaptured,dataCaptured = getDiffSessions(labels,subjArray,subjCaptured,curSubj)

        if len(dataCaptured) >= numSessions:
            allDataCapt.append(data[dataCaptured[0:numSessions]])
            dataCaptured = []

    return np.array(allDataCapt)

def dataSummary(data,labels):
    #allLabels.append([dataName,dateCur,val.subjGender,val.age,getMedsListStr(val.subjMed),val.subjNormalState])

    allNormal = 0
    ages = []
    normal = 0
    abnormal = 0
    noNA = 0
    male = 0
    female = 0
    noSex = 0
    
    subjArray = getSubjArray(labels)
    
    subjCaptured = []
    subjCapturedS = []
    subjCapturedA = []

    print 'Total Sessions:',len(subjArray)
    for i in range(len(subjArray)):
        curSubj = subjArray[i]

        if curSubj not in subjCaptured:
            subjCaptured.append(curSubj)
        
        #if curSubj not in subjCapturedS:
        if labels[i][2] == 'male':
            male += 1
            subjCapturedS.append(curSubj)        
        if labels[i][2] == 'female':
            female += 1
            subjCapturedS.append(curSubj)        
        if (labels[i][2] != 'male') and (labels[i][2] != 'female'):
            noSex += 1
    
        #if curSubj not in subjCapturedA:
        if len(labels[i][3])>0:
            ages.append(labels[i][3][0])
            subjCapturedA.append(curSubj)        


        if labels[i][5] == 0:
            normal += 1
        if labels[i][5] == 1:
            abnormal += 1
        if labels[i][5] == 2:
            noNA += 1

    print 'Males:',male,'; Female:',female,'; Neither:',noSex,' Total:',male+female+noSex
    print 'Age: Mean:',np.mean(ages),' SD:',np.std(ages),' IQR:',stats.iqr(ages),' n=',len(ages)
    print 'Normal:',normal,'; Abnormal:',abnormal,'; Neither:',noNA,' Total:',normal+abnormal+noNA
    print 'Unique Subjects Found:',len(np.unique(subjArray)),'\n'
    
    subjCaptured = []
    multiSession = 0
    allDates = []
    allAges = []
    allMale = 0
    allFemale = 0
    allNoSex = 0
    subj1 = 0
    subj2 = 1
    useRand = 0
    for i in range(len(labels)):
        curSubj = labels[i][0].split('_')[0]
        if curSubj in subjCaptured:
            continue
        subjCaptured,dataCaptured = getDiffSessions(labels,subjArray,subjCaptured,curSubj)


        if len(dataCaptured) > 1:
            if useRand == 1:
                randSubj = -1
                while (randSubj in subjCaptured) or (randSubj<0):
                    randSubj = random.randint(0,len(labels)-1) 
                dataCaptured[subj2] = randSubj

            multiSession += 1
            allAges.append(getMeanAge([labels[dataCaptured[subj1],3],labels[dataCaptured[subj2],3]]))
            allDates.append(getDateDiff([labels[dataCaptured[subj1],1],labels[dataCaptured[subj2],1]]))
            if labels[dataCaptured[subj1],2] == 'male':
                allMale += 1
            if labels[dataCaptured[subj1],2] == 'female':
                allFemale += 1
            if (labels[dataCaptured[subj1],2] != 'male') and (labels[dataCaptured[subj1],2] != 'female'):
                allNoSex += 1
    

    allAges = np.array(allAges)
    allDates = np.array(allDates)

    print 'Number of Multi-Session Subjects:',multiSession
    print 'Males:',allMale,'; Female:',allFemale,'; Neither:',allNoSex,' Total:',allMale+allFemale+allNoSex
    print 'Time between first and second visit (days): Median:',np.median(allDates),' SD:',np.std(allDates),' IQR:',stats.iqr(allDates),' n=',len(allDates)
    print 'Mean Ages of subjects with multiple sessions: Mean:',np.mean(allAges[allAges>0]),' SD:',np.std(allAges[allAges>0]),' IQR:',stats.iqr(allAges[allAges>0]),' n=',np.size(allAges[allAges>0])

def plotAll(allSigP,maxChanSigP,multiSession):
    exclude = 0

    fig = pl.figure(figsize=(15, 15))
    fig.suptitle('Kruskal-Wallis results of intra-subject variability across time partitions (n=%d)'%(multiSession))

    
    featsV = ['Rel-Lower', 'Rel-Delta', 'Rel-Theta', 'Rel-Alpha', 'Rel-Mu', 'Rel-Beta', 'Rel-Gamma', 
                'Abs-Lower', 'Abs-Delta', 'Abs-Theta', 'Abs-Alpha', 'Abs-Mu', 'Abs-Beta', 'Abs-Gamma', 
                'Spectral-Entropy', 'Spectral-Entropy-Norm', 
                'Entropy', 'Entropy-Norm', 'Curve-Length', 'Energy', 'Nonlinear-Energy', 'Sixth-Power', 'LZC', 'Min', 
                'Max', 'Med', 'Var', 'Std', 'Skew', 'Kurtosis']
    #pdb.set_trace()
    featsV = ['Mobility','Complexity']
    for v in range(len(featsV)):
        #pdb.set_trace()
        #ax = fig.add_subplot(2,2,curD+1)
        #fig,ax = resultBoxPlot(fig,allCV[curD],ax=ax,exclude=exclude)
        ax = fig.add_subplot(1,2,v+1)
        fig,ax = resultBoxPlot(fig,np.transpose(allSigP[:,:,v]),totalInst=multiSession,ax=ax,exclude=exclude)
        for r in range(np.shape(maxChanSigP)[1]):
            ax.plot(np.concatenate((np.array([0]),np.divide(np.transpose(maxChanSigP[:,r,v]),multiSession/100.)),axis=0),'b.')
        
        ax.set_ylim(ymin=0,ymax=100)        
        ax.set_xticklabels([ '0.25', '0.5', '1', '2'])
        ax.set_xlabel('Epoch time (minutes)')

        ax.set_ylabel('%s consistent'%('%'))
        #ax.set_ylim(ymax=0.3)

        if v < 24:
            ax.get_xaxis().set_visible(False)
        if v%6 != 0:
            ax.get_yaxis().set_visible(False)

        #ax.set_title('COV of Features on %d parts, each %0.2f minutes (n=%d)'%(np.shape(curData)[3],16./np.shape(curData)[3],subjNum))
        ax.set_title('%s'%(featsV[v]))

    fig.subplots_adjust(left=0.06,bottom=0.10,right=0.96,top=0.90,wspace=0.2,hspace=0.2)
    
    #fig.savefig('images/allKWresultsAdjust.eps', format='eps', dpi=1000)

    pl.show()

def plotAllKW(allSigP,maxChanSigP,multiSession):
    exclude = 0

    fig = pl.figure(figsize=(15, 15))
    fig.suptitle("Kendall's W results of intra-subject variability across time partitions (n=%d)"%(multiSession))

    
    featsV = ['Rel-Lower', 'Rel-Delta', 'Rel-Theta', 'Rel-Alpha', 'Rel-Mu', 'Rel-Beta', 'Rel-Gamma', 
                'Abs-Lower', 'Abs-Delta', 'Abs-Theta', 'Abs-Alpha', 'Abs-Mu', 'Abs-Beta', 'Abs-Gamma', 
                'Spectral-Entropy', 'Spectral-Entropy-Norm', 
                'Entropy', 'Entropy-Norm', 'Curve-Length', 'Energy', 'Nonlinear-Energy', 'Sixth-Power', 'LZC', 'Min', 
                'Max', 'Med', 'Var', 'Std', 'Skew', 'Kurtosis']
    #pdb.set_trace()
    featsV = ['Mobility','Complexity']

    for v in range(len(featsV)):
        #pdb.set_trace()
        #ax = fig.add_subplot(2,2,curD+1)
        #fig,ax = resultBoxPlot(fig,allCV[curD],ax=ax,exclude=exclude)
        ax = fig.add_subplot(1,2,v+1)
        fig,ax = resultBoxPlot(fig,np.transpose(allSigP[:,:,v]),ax=ax,exclude=exclude)
        for r in range(np.shape(maxChanSigP)[1]):
            ax.plot(np.concatenate((np.array([0]),np.transpose(maxChanSigP[:,r,v])),axis=0),'b.')
        
        ax.set_ylim(ymin=0,ymax=1)        
        ax.set_xticklabels([ '0.25', '0.5', '1', '2'])
        ax.set_xlabel('Epoch time (minutes)')

        ax.set_ylabel('Correlation Coefficient')
        #ax.set_ylim(ymax=0.3)

        if v < 24:
            ax.get_xaxis().set_visible(False)
        if v%6 != 0:
            ax.get_yaxis().set_visible(False)

        #ax.set_title('COV of Features on %d parts, each %0.2f minutes (n=%d)'%(np.shape(curData)[3],16./np.shape(curData)[3],subjNum))
        ax.set_title('%s'%(featsV[v]))

    fig.subplots_adjust(left=0.06,bottom=0.10,right=0.96,top=0.90,wspace=0.2,hspace=0.2)
    
    #fig.savefig('images/allKendallWresults.eps', format='eps', dpi=1000)

    pl.show()


if __name__ == '__main__':
    start = time.time()

    twoD = 0
    oneD = 0
    perVar = 0
    icc = 0
    kendallOne = 0

    tScore = 0

    generateDist = 0
    kruskal = 0
    kendall = 0
    
    Features,Partitions,TimeMin,Threads,Write2File,FeatsNames,InputFileName = defineParams()

    allData, allLabels = dataLoad4D(InputFileName,Features,1,TimeMin,Threads,Write2File,FeatsNames)
    allData,allLabels = getNormData(allData,allLabels,normal=1)
    dataSummary(allData,allLabels)

    if twoD == 1:
        allData, allLabels = dataLoad4D(InputFileName,Features,Partitions,TimeMin,Threads,Write2File,FeatsNames)
        allData,allLabels = getNormData(allData,allLabels)
        #pdb.set_trace()
        allSigP1,allAvgDate1,allAges1,multiSession1 = freqAnaly2D(allData, allLabels)
        allSigP2,allAvgDate2,allAges2,multiSession2 = freqAnaly2D(allData, allLabels,useRand=1)

        for v in range(np.shape(allSigP1)[1]):
            meanP = stats.kruskal(allSigP1[:,v],allSigP2[:,v])[1]
            print "Variable %d, p-value: %f"%(v,meanP)


        #fig = pl.figure()
        #topo_plot.plotEEGData(fig,np.transpose(np.divide(allSigP,multiSession/100.)))
        pl.show()

    
    if oneD == 1:
        allData, allLabels = dataLoad3D(InputFileName,Features,Partitions,TimeMin,Threads,Write2File,FeatsNames)
        allData,allLabels = getNormData(allData,allLabels)

        #pdb.set_trace()
        sigP,goodDates,badDates,goodAges,badAges = freqAnaly1D(allData, allLabels)


    if  perVar == 1:       
        allData, allLabels = dataLoad3D(InputFileName,Features,Partitions,TimeMin,Threads,Write2File,FeatsNames)
        allData,allLabels = getNormData(allData,allLabels)

        #pdb.set_trace()
        allSigP,allAvgDate,allAges,multiSession = freqAnalyPerVar(allData, allLabels)

    if icc == 1:
        allData, allLabels = dataLoad3D(InputFileName,Features,Partitions,TimeMin,Threads,Write2File,FeatsNames)
        allData,allLabels = getNormData(allData,allLabels)

        allData = allData[:,:,0]
        tabledData = tableData(allData,allLabels,numSessions=2)
        print 'Tabled Data:',tabledData.shape
        allICC = np.zeros((19,np.shape(tabledData)[2]/19))
        ICCVec = []
        allpVal = np.zeros((19,np.shape(tabledData)[2]/19))
        pValVec = []
        for v in range(np.shape(tabledData)[2]):
            ch, feat= divmod(v, np.shape(tabledData)[2]/19)
            allICC[ch,feat],allpVal[ch,feat] = ICC_rep_anova(tabledData[:,:,v])
            ICCVec.append(allICC[ch,feat])
            pValVec.append(allpVal[ch,feat])

            #print('Channel:',ch,'Feature:',feat)
            #print(allICC[ch,feat])
        #print allICC
        #print np.array(ICCVec).argsort()[-20:][::-1]
        #print allpVal
        #print np.array(pValVec).argsort()[-20:][::-1]

        # Create a figure instance
        fig = pl.figure(figsize=(9, 6))
        fig,ax = resultBoxPlot(fig,allICC)
        fig.suptitle("ICC")
        ax.set_ylabel('Correlation Coefficient')
        fig.subplots_adjust(left=0.09,bottom=0.33,right=0.96,top=0.93,wspace=0.2,hspace=0.2)
        fig.savefig('images/ICC%d.eps'%(Partitions), format='eps', dpi=1000)

        pl.show()

    if kendallOne == 1:
        allData, allLabels = dataLoad3D(InputFileName,Features,Partitions,TimeMin,Threads,Write2File,FeatsNames)
        allData,allLabels = getNormData(allData,allLabels)

        allData = allData[:,:,0]

        tabledData = tableData(allData,allLabels,numSessions=2)
        #tabledData = randTableData(allData,allLabels,numSessions=2)
        print 'Tabled Data:',tabledData.shape
        allKW = np.zeros((19,np.shape(tabledData)[2]/19))
        KWVec = []
        allpVal = np.zeros((19,np.shape(tabledData)[2]/19))
        pValVec = []
        #pdb.set_trace()

        for v in range(np.shape(tabledData)[2]):
            ch, feat= divmod(v, np.shape(tabledData)[2]/19)
            #allICC[ch,feat],allpVal[ch,feat] = ICC_rep_anova(tabledData[:,:,v])
            allKW[ch,feat],allpVal[ch,feat] = kendallW(tabledData[:,:,v])
            KWVec.append(allKW[ch,feat])
            pValVec.append(allpVal[ch,feat])
         
            #print('Channel:',ch,'Feature:',feat)
            #print(allICC[ch,feat])
        #print allKW
        #print np.array(KWVec).argsort()[-20:][::-1]
        #print allpVal
        #print np.array(pValVec).argsort()[-20:][::-1]        

        # Create a figure instance
        fig = pl.figure(figsize=(9, 6))
        fig,ax = resultBoxPlot(fig,allKW)
        fig.suptitle("Kendall's W, %d parts with %d sessions (n=%d)"%(Partitions,np.shape(tabledData)[1],np.shape(tabledData)[0]))
        ax.set_ylabel('Correlation Coefficient')
        fig.subplots_adjust(left=0.09,bottom=0.33,right=0.96,top=0.93,wspace=0.2,hspace=0.2)
        #fig.savefig('images/kendallW%d.eps'%(Partitions), format='eps', dpi=1000)
        pl.show()
        
        #topo_plot.plotEEGData(np.transpose(allKW))

    if tScore == 1:
        allData, allLabels = dataLoad4D(InputFileName,Features,Partitions,TimeMin,Threads,Write2File,FeatsNames)
        allData,allLabels = getNormData(allData,allLabels)
        #pdb.set_trace()
        allSigP1,allAvgDate1,allAges1,multiSession1 = tScoreAnaly2D(allData, allLabels)

    if generateDist == 1:
        #variableMask = np.concatenate((range(0,30),range(31,35)),axis=0)
        #variableMask = np.concatenate((range(0,14),range(18,30),range(31,35)),axis=0)
        if 'all' in Features:
            variableMaskL1 = np.concatenate((range(0,14),range(18,30),range(31,35)),axis=0)
            variableMask = variableMaskL1[np.concatenate((range(0,14),[15],range(17,30)))]
            featSaveStr = 'All'
        elif 'addTime' in Features:
            variableMask = range(2)
            featSaveStr = 'AddTime'

        if kruskal == 1:
            allData, allLabels = dataLoad4D(InputFileName,Features,8,TimeMin,Threads,Write2File,FeatsNames)
            allData,allLabels = getNormData(allData,allLabels)
            dataSummary(allData,allLabels)
            allData = allData[:,:,variableMask,:]
            allSigP1,allAvgDate1,allAges1,multiSession1,maxChanSigP1,maxChanSigP1Loc = generateKWdist(allData,allLabels,runs=100,useRand=1,exclude=1)
            

            allData, allLabels = dataLoad4D(InputFileName,Features,16,TimeMin,Threads,Write2File,FeatsNames)
            allData,allLabels = getNormData(allData,allLabels)
            dataSummary(allData,allLabels)
            allData = allData[:,:,variableMask,:]
            allSigP2,allAvgDate2,allAges2,multiSession2,maxChanSigP2,maxChanSigP2Loc = generateKWdist(allData,allLabels,runs=100,useRand=1,exclude=1)

            allData, allLabels = dataLoad4D(InputFileName,Features,32,TimeMin,Threads,Write2File,FeatsNames)
            allData,allLabels = getNormData(allData,allLabels)
            dataSummary(allData,allLabels)
            allData = allData[:,:,variableMask,:]
            allSigP3,allAvgDate3,allAges3,multiSession3,maxChanSigP3,maxChanSigP3Loc = generateKWdist(allData,allLabels,runs=100,useRand=1,exclude=1)

            allData, allLabels = dataLoad4D(InputFileName,Features,64,TimeMin,Threads,Write2File,FeatsNames)
            allData,allLabels = getNormData(allData,allLabels)
            dataSummary(allData,allLabels)
            allData = allData[:,:,variableMask,:]
            allSigP4,allAvgDate4,allAges4,multiSession4,maxChanSigP4,maxChanSigP4Loc = generateKWdist(allData,allLabels,runs=100,useRand=1,exclude=1)

            #pdb.set_trace()
            if Write2File == 1:
                np.save('figureGenFiles/allSigPdata{}New5.npy'.format(featSaveStr),np.array([allSigP4,allSigP3,allSigP2,allSigP1]))
                np.save('figureGenFiles/allSigPrand{}New5.npy'.format(featSaveStr),np.array([maxChanSigP4,maxChanSigP3,maxChanSigP2,maxChanSigP1]))
                np.save('figureGenFiles/allSigPrandLoc{}New5.npy'.format(featSaveStr),np.array([maxChanSigP4Loc,maxChanSigP3Loc,maxChanSigP2Loc,maxChanSigP1Loc]))

            #plotAll(np.array([allSigP4,allSigP3,allSigP2,allSigP1]),np.array([maxChanSigP4,maxChanSigP3,maxChanSigP2,maxChanSigP1]),multiSession1)


        if kendall == 1:

            allData, allLabels = dataLoad4D(InputFileName,Features,8,TimeMin,Threads,Write2File,FeatsNames)
            allData,allLabels = getNormData(allData,allLabels)
            dataSummary(allData,allLabels)
            allData = allData[:,:,variableMask,:]
            allSigP1,maxChanSigP1,maxChanSigP1Loc,multiSession1 = generateKendallDist(allData,allLabels,numSessions=2,runs=100,useRand=1,exclude=1)

            allData, allLabels = dataLoad4D(InputFileName,Features,16,TimeMin,Threads,Write2File,FeatsNames)
            allData,allLabels = getNormData(allData,allLabels)
            dataSummary(allData,allLabels)
            allData = allData[:,:,variableMask,:]
            allSigP2,maxChanSigP2,maxChanSigP2Loc,multiSession2 = generateKendallDist(allData,allLabels,numSessions=2,runs=100,useRand=1,exclude=1)

            allData, allLabels = dataLoad4D(InputFileName,Features,32,TimeMin,Threads,Write2File,FeatsNames)
            allData,allLabels = getNormData(allData,allLabels)
            dataSummary(allData,allLabels)
            allData = allData[:,:,variableMask,:]
            allSigP3,maxChanSigP3,maxChanSigP3Loc,multiSession3 = generateKendallDist(allData,allLabels,numSessions=2,runs=100,useRand=1,exclude=1)

            allData, allLabels = dataLoad4D(InputFileName,Features,64,TimeMin,Threads,Write2File,FeatsNames)
            allData,allLabels = getNormData(allData,allLabels)
            dataSummary(allData,allLabels)
            allData = allData[:,:,variableMask,:]
            allSigP4,maxChanSigP4,maxChanSigP4Loc,multiSession4 = generateKendallDist(allData,allLabels,numSessions=2,runs=100,useRand=1,exclude=1)

            #pdb.set_trace()
            if Write2File == 1:
                np.save('figureGenFiles/allKWdata{}New_zero.npy'.format(featSaveStr),np.array([allSigP4,allSigP3,allSigP2,allSigP1]))
                np.save('figureGenFiles/allKWrand{}New_zero.npy'.format(featSaveStr),np.array([maxChanSigP4,maxChanSigP3,maxChanSigP2,maxChanSigP1]))
                np.save('figureGenFiles/allKWrandLoc{}New_zero.npy'.format(featSaveStr),np.array([maxChanSigP4Loc,maxChanSigP3Loc,maxChanSigP2Loc,maxChanSigP1Loc]))

            #plotAllKW(np.array([allSigP4,allSigP3,allSigP2,allSigP1]),np.array([maxChanSigP4,maxChanSigP3,maxChanSigP2,maxChanSigP1]),multiSession1)

        #data = np.load('allKWdata.npy')
        #rand = np.load('allKWrand.npy')
        #plotAllKW(data,rand,419)

    end=time.time()
    print '\nTime Elapsed:',end-start,'\n'