#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" qEEG Consistency results generator 
Software to generate results presented in:

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
__copyright__ = 'No copyright - US Government, 2019 , qEEG Consistency'
__credits__ = ['David Nahmias']
__license__ = 'Public domain'
__version__ = '0.0.1'
__maintainer__ = 'David Nahmias'
__email__ = 'david.nahmias@fda.hhs.gov'
__status__ = 'alpha'


import numpy as np
from scipy import stats, signal
import pdb
import time
import sys
from scipy.linalg import pinv
import datetime
import random
#import matplotlib.pyplot as plt
import pylab as pl

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


import topo_plot
from nedcTools import resultBoxPlot,dataLoad3D,dataLoad4D,Subject,singleTXTEval,preProcessData
import nedcTools


from multiprocessing import Pool
from functools import partial
from scipy import optimize


def defineParams():
    features = ['addTime']
    partitions = 128
    timeMin = 16
    threads = 1
    write2File = 1
    featsNames = ''
    textFile = 'allTextFiles'
    for x in features:
        featsNames = featsNames+x
    return features,partitions,timeMin,threads,write2File,featsNames,textFile

def plotCOVcurves(allCV,subjNumT):
    fig = pl.figure(figsize=(13.5, 9))
    fig.suptitle('Average COV of absolute band powers',fontsize=26)
    ax = fig.add_subplot(1,1,1)#fig.add_subplot(1,2,v+1)

    featsV = ['Absolute Lower Power', 'Absolute Delta Power', 'Absolute Theta Power', 'Absolute Alpha Power', 'Absolute Mu Power', 'Absolute Beta Power', 'Absolute Gamma Power']
    avgCOV = np.mean(allCV,axis=1)
    #pdb.set_trace()
    #for v in range(len(featsV)):

    x = np.logspace(np.log2(7.5),np.log2(120),5,base=2)
    ax.plot(x,avgCOV,'o-',linewidth=3)
    pl.xticks(x, ('7.5', '15', '30', '60', '120'))

    #ax.plot(range(avgCOV.shape[0]),avgCOV,'o-',linewidth=3)
    #pl.xticks(np.arange(0,4.9,1), ('7.5', '15', '30', '60', '120'))
    
    pl.yticks(np.arange(0,0.31,step=0.03))
    #ax.set_xticklabels(['7.5', '15', '30', '60', '120'])
    #ax.get_xaxis().tick_bottom()
    ax.tick_params(labelsize=20) #12 for all 14 for single
    ax.set_title('',fontsize=24)
    ax.set_xlabel('Epoch time (s)',fontsize=24)
    ax.set_ylabel('Average COV',fontsize=24)
    ax.legend(featsV,fontsize=20)
    fig.subplots_adjust(left=0.11,bottom=0.08,right=0.99,top=0.90,wspace=0.2,hspace=0.2)
    
    #fitFunc = lambda x,a,c: a*np.power(x,-1)+c
    #p,pcov = optimize.curve_fit(fitFunc,x,np.mean(avgCOV,axis=1))



def singleCOVplot(allCV,subjNumT,featName,exclude=1):
    print np.shape(allCV)

    fig = pl.figure(figsize=(13.5, 9))
    fig.suptitle('Single COV across channels',fontsize=26)
    ax = fig.add_subplot(1,1,1)

    fig,ax = resultBoxPlot(fig,np.transpose(allCV),ax=ax,exclude=exclude)
    ax.set_xticklabels([ '7.5 sec', '15 sec', '30 sec', '60 sec', '120 sec'])
    ax.set_xlabel('Epoch time',fontsize=24)
    ax.set_ylabel('COV',fontsize=24)
    #ax.set_ylim(ymin=0)
    ax.set_title('Feature: %s'%(featName),fontsize=24)
    ax.tick_params(labelsize=20) #12 for all 14 for single

    #pl.xticks(np.arange(0,4.9,1), ('7.5', '15', '30', '60', '120'))
    pl.yticks(np.arange(0,0.31,step=0.03))
    fig.subplots_adjust(left=0.11,bottom=0.08,right=0.99,top=0.90,wspace=0.2,hspace=0.2)
    white_patch = mpatches.Patch(facecolor='white', edgecolor='k', label='COV across channels')
    ax.legend(handles=[white_patch],fontsize=20)


def plotCOV2D(allCV,subjNumT,exclude=1,makeSample=False):
    print np.shape(allCV)

    fig = pl.figure(figsize=(13.5, 9))
    fig.suptitle('COV across epoch lengths (n=%d)'%(subjNumT),fontsize=28)
    #pdb.set_trace()
    featsV = ['Rel-Lower', 'Rel-Delta', 'Rel-Theta', 'Rel-Alpha', 'Rel-Mu', 'Rel-Beta', 'Rel-Gamma', 
                'Abs-Lower', 'Abs-Delta', 'Abs-Theta', 'Abs-Alpha', 'Abs-Mu', 'Abs-Beta', 'Abs-Gamma', 
                'Spectral-Entropy', 
                'Entropy', 'Curve-Length', 'Energy', 'Sixth-Power', 'LZC', 
                'Max', 'Var','Mobility','Complexity']
    #featsV = ['Mobility','Complexity']

    if makeSample == True:
        indexSample = [2,9,19]
        featsSample = np.array(featsV)[indexSample].tolist()
        cvSample = allCV[:,:,indexSample]
        for v in range(len(featsSample)):
            dim1=1
            dim2=3

            ax = fig.add_subplot(dim1,dim2,v+1)#fig.add_subplot(1,2,v+1)
            fig,ax = resultBoxPlot(fig,np.transpose(cvSample[:,:,v]),ax=ax,exclude=exclude)
            ax.set_xticklabels([ '0.125', '0.25', '0.5', '1', '2'])
            ax.set_xlabel('Epoch time (min)',fontsize=16)

            if v%dim2 == 0:
                ax.set_ylabel('COV',fontsize=20)
            #ax.set_ylim(ymax=0.3)
            ax.set_ylim(ymin=0)

            if v < ((dim1*dim2)-dim2):
                ax.get_xaxis().set_visible(False)
            #if v%6 != 0:
            #    ax.get_yaxis().set_visible(False)

            #ax.set_title('COV of Features on %d parts, each %0.2f minutes (n=%d)'%(np.shape(curData)[3],16./np.shape(curData)[3],subjNum))
            ax.set_title('%s'%(featsSample[v]),fontsize=20)

        
        fig.subplots_adjust(left=0.05,bottom=0.10,right=0.99,top=0.90,wspace=0.25,hspace=0.2)
        

        fig = pl.figure(figsize=(13.5, 9))
        fig.suptitle('COV across epoch lengths (n=%d)'%(subjNumT),fontsize=28)


    for v in range(np.shape(allCV)[2]):
        #if (v!=2) and (v!=9) and (v!=19):
        #    continue
        
        dim1 = 6
        dim2 = 4
        #pdb.set_trace()
        #ax = fig.add_subplot(2,2,curD+1)
        #fig,ax = resultBoxPlot(fig,allCV[curD],ax=ax,exclude=exclude)
        if makeSample == True:
            dim1=1
            dim2=1
            ax = fig.add_subplot(dim1,dim2,1)
        else:
            ax = fig.add_subplot(dim1,dim2,v+1)#fig.add_subplot(1,2,v+1)
        
        fig,ax = resultBoxPlot(fig,np.transpose(allCV[:,:,v]),ax=ax,exclude=exclude)
        ax.set_xticklabels([ '7.5', '15', '30', '60', '120'])
        ax.set_xlabel('Epoch time (s)',fontsize=16)
        
        #ax.set_ylim(ymin=0,ymax=1)        

        '''
        bp = ax.boxplot(allCV[curD])
        
        ax.set_xticklabels(['Rel-Lower', 'Rel-Delta', 'Rel-Theta', 'Rel-Alpha', 'Rel-Mu', 'Rel-Beta', 'Rel-Gamma', 
        'Abs-Lower', 'Abs-Delta', 'Abs-Theta', 'Abs-Alpha', 'Abs-Mu', 'Abs-Beta', 'Abs-Gamma', 
        'Fourier-Entropy', 'Spectral-Entropy', 'Curve-Length', 'Energy', 'Nonlinear-Energy', 'Sixth-Power', 'LZC', 
        'Min', 'Max', 'Med', 'Mean', 'Var', 'Std', 'Skew', 'Kurtosis', 'Integral', 'Sum'],rotation='90')
        ax.get_xaxis().tick_bottom()
        ax.set_xlabel('Features')
        
        ax.set_ylabel('COV')
        ax.get_yaxis().tick_left()
        '''
        if v%dim2 == 0:
            ax.set_ylabel('COV',fontsize=20)
        #ax.set_ylim(ymax=0.3)
        ax.set_ylim(ymin=0)

        if v < ((dim1*dim2)-dim2):
            ax.get_xaxis().set_visible(False)
        #if v%6 != 0:
        #    ax.get_yaxis().set_visible(False)

        #ax.set_title('COV of Features on %d parts, each %0.2f minutes (n=%d)'%(np.shape(curData)[3],16./np.shape(curData)[3],subjNum))
        ax.set_title('%s'%(featsV[v]),fontsize=20)

        #fig.subplots_adjust(left=0.05,bottom=0.10,right=0.99,top=0.90,wspace=0.25,hspace=0.2)

        if makeSample == True:
            fig.subplots_adjust(left=0.08,bottom=0.07,right=0.96,top=0.90,wspace=0.2,hspace=0.2)
            fig = pl.figure(figsize=(13.5, 9))
            fig.suptitle('COV across epoch lengths (n=%d)'%(subjNumT),fontsize=28)

    #For widescreen
    #fig.subplots_adjust(left=0.06,bottom=0.10,right=0.96,top=0.90,wspace=0.2,hspace=0.2)
    
    #For tallscreen
    fig.subplots_adjust(left=0.08,bottom=0.07,right=0.96,top=0.90,wspace=0.2,hspace=0.2)


    sigDiff = np.zeros((np.shape(allCV)[0]-1,np.shape(allCV)[2]))
    for d in range(0,np.shape(allCV)[0]-1):
        for v in range(np.shape(allCV)[2]):
            sigP = 0
            #print allCV[d,:,v],allCV[d+1,:,v]
            try:
                if 0.05 < stats.kruskal(allCV[d,:,v],allCV[d+1,:,v])[1]:
                    sigP += 1
                sigDiff[d,v] = sigP
                #pdb.set_trace()
            except:
                print "Tried"

    print 'Mean CV:'
    print sigDiff
    #fig.savefig('images/COVstatAll.eps', format='eps', dpi=1000)
    '''
    fig, axs = plt.subplots(2,2)
    axs = axs.ravel()

    axs[curD].boxplot(allCV[curD])
    ax.set_xticklabels(['Rel-Lower', 'Rel-Delta', 'Rel-Theta', 'Rel-Alpha', 'Rel-Mu', 'Rel-Beta', 'Rel-Gamma', 
            'Abs-Lower', 'Abs-Delta', 'Abs-Theta', 'Abs-Alpha', 'Abs-Mu', 'Abs-Beta', 'Abs-Gamma', 
            'Fourier-Entropy', 'NormedFourier-Entropy', 'Fourier-Entropy-Norm', 'NormedFourier-Entropy-Norm', 'Spectral-Entropy', 'Spectral-Entropy-Norm', 
            'Entropy', 'Entropy-Norm', 'Curve-Length', 'Energy', 'Nonlinear-Energy', 'Sixth-Power', 'LZC', 
            'Min', 'Max', 'Med', 'Mean', 'Var', 'Std', 'Skew', 'Kurtosis', 'Integral', 'Sum'],rotation='90')  
    axs[curD].get_xaxis().tick_bottom()
    axs[curD].get_yaxis().tick_left()
    axs[curD].set_xlabel('Features')
    axs[curD].set_ylabel('COV')
    axs[curD].set_title('COV of Features on %d parts, each %f minutes (n=%d)'%(np.shape(curData)[3],16./np.shape(curData)[3],subjNum))
    pl.ylim(ymax=5)
    '''

    #fig.savefig('images/COVresultsAdjust.eps', format='eps', dpi=1000)

    #pl.show()

def plotAll(allSigP,maxChanSigP,multiSession,makeSample=False,thresholdOnly=False):
    exclude = 0
    figDim = [10,6.66]#[13.5,9] for single  #[15,15] for tall
    fig = pl.figure(figsize=(figDim[0], figDim[1])) #15,15
    fig.suptitle("Percent of consistent subjects'\n two sessions across channels",fontsize=24)

    
    featsV = ['Rel-Lower', 'Rel-Delta', 'Rel-Theta', 'Rel-Alpha', 'Rel-Mu', 'Rel-Beta', 'Rel-Gamma', 
                'Abs-Lower', 'Abs-Delta', 'Abs-Theta', 'Abs-Alpha', 'Abs-Mu', 'Abs-Beta', 'Abs-Gamma', 
                'Spectral Entropy', 
                'Entropy', 'Curve-Length', 'Energy', 'Nonlinear-Energy', 'Sixth-Power', 'LZC', 'Min', 
                'Max', 'Med', 'Var', 'Std', 'Skew', 'Kurtosis','Mobility','Complexity']

    featsV = ['Relative Lower Power', 'Relative Delta Power', 'Relative Theta Power', 'Relative Alpha Power', 'Relative Mu Power', 'Relative Beta Power', 'Relative Gamma Power', 
            'Absolute Lower Power', 'Absolute Delta$ Power', 'Absolute Theta Power', 'Absolute Alpha Power', 'Absolute Mu Power', 'Absolute Beta Power', 'Absolute Gamma Power', 
            'Spectral Entropy', 
            'Entropy', 'Curve Length', 'Energy', 'Non-linear Energy', 'Sixth Power', 'LZC', 'Minimum', 
            'Maximum', 'Median', 'Variance', 'SD', 'Skew', 'Kurtosis','Mobility','Complexity']

    fontSizes = [12,12,20,15,24]
    if makeSample == True:
        fontSizes = [20,24,24,24,26]

    #pdb.set_trace()
    '''
    if makeSample == True:
        indexSample = [10,26]
        featsSample = np.array(featsV)[indexSample].tolist()
        sigPSample = allSigP[:,:,indexSample]
        maxChanSample = maxChanSigP[:,:,indexSample]
        #print np.shape(sigPSample)
        #print np.shape(maxChanSample)
        for v in range(len(featsSample)):
            dim1=1
            dim2=2
            #pdb.set_trace()
            #ax = fig.add_subplot(2,2,curD+1)
            #fig,ax = resultBoxPlot(fig,allCV[curD],ax=ax,exclude=exclude)
            ax = fig.add_subplot(dim1,dim2,v+1)
            ax.tick_params(labelsize=12)

            fig,ax = resultBoxPlot(fig,np.transpose(sigPSample[:,:,v]),ax=ax,exclude=exclude)
            for r in range(np.shape(maxChanSample)[1]):
                ax.plot(np.concatenate((np.array([0]),np.transpose(maxChanSample[:,r,v])),axis=0),'b.')
            
            ax.set_ylim(ymin=0,ymax=1)        
            ax.set_xticklabels([ '0.25', '0.5', '1', '2'])
            ax.set_xlabel('Epoch time (min)',fontsize=12)

            ax.set_ylabel('%s consistent'%('%'),fontsize=20)
            #ax.set_ylim(ymax=0.3)

            if v < ((dim1*dim2)-dim2):
                ax.get_xaxis().set_visible(False)
            if v%dim2 != 0:
                ax.get_yaxis().set_visible(False)

            
            #ax.set_title('COV of Features on %d parts, each %0.2f minutes (n=%d)'%(np.shape(curData)[3],16./np.shape(curData)[3],subjNum))
            ax.set_title('%s'%(featsSample[v]),fontsize=15)

        fig.subplots_adjust(left=0.09,bottom=0.07,right=0.96,top=0.90,wspace=0.2,hspace=0.2)


        fig = pl.figure(figsize=(figDim[0], figDim[1]))
        fig.suptitle('Kruskal-Wallis results of intra-subject variability\n across epoch lengths (n=%d)'%(multiSession),fontsize=24)
    '''

    for v in range(len(featsV)):
        if makeSample == True:
            if (v != 10) and (v!= 26):
                continue
        dim1=5
        dim2=6
        #pdb.set_trace()
        #ax = fig.add_subplot(2,2,curD+1)
        #fig,ax = resultBoxPlot(fig,allCV[curD],ax=ax,exclude=exclude)
        if makeSample == True:
            dim1=1
            dim2=1
            ax = fig.add_subplot(dim1,dim2,1)
        else:
            ax = fig.add_subplot(dim1,dim2,v+1)
        
        ax.tick_params(labelsize=fontSizes[0]) #12 for all 14 for single

        fig,ax = resultBoxPlot(fig,np.transpose(allSigP[:,:,v]),totalInst=multiSession,ax=ax,exclude=exclude)
        for r in range(np.shape(maxChanSigP)[1]):
            ax.plot(np.concatenate((np.array([0]),np.divide(np.transpose(maxChanSigP[:,r,v]),multiSession/100.)),axis=0),'b.')
        if thresholdOnly == 1:
            #ax.plot(np.concatenate((np.array([0]),np.transpose(np.max(maxChanSigP[:,:,v],axis=1))),axis=0),'r_',markersize=20,markeredgewidth=5)
            ax.plot(np.concatenate((np.array([0]),np.array([50,50,50,50])),axis=0),'r_',markersize=160,markeredgewidth=3)
            ax.plot(np.concatenate((np.array([0]),np.array([75,75,75,75])),axis=0),'g_',markersize=160,markeredgewidth=3)
        
        ax.set_ylim(ymin=0,ymax=100)        
        ax.set_xticklabels([ '15 sec', '30 sec', '60 sec', '120 sec'])
        ax.set_xlabel('Epoch time',fontsize=fontSizes[1]) #20 for Single #12 for all

        ax.set_ylabel('%s subjects p>0.05'%('%'),fontsize=fontSizes[2]) #20 either way
        #ax.set_ylim(ymax=0.3)

        if v < ((dim1*dim2)-dim2):
            ax.get_xaxis().set_visible(False)
        if v%dim2 != 0:
            ax.get_yaxis().set_visible(False)

        #ax.set_title('COV of Features on %d parts, each %0.2f minutes (n=%d)'%(np.shape(curData)[3],16./np.shape(curData)[3],subjNum))
        ax.set_title('%s'%(featsV[v]),fontsize=fontSizes[3]) #20 for Single #15 for all

    
        if makeSample == True:
            fig.subplots_adjust(left=0.11,bottom=0.08,right=0.96,top=0.86,wspace=0.2,hspace=0.2)
            fig = pl.figure(figsize=(figDim[0], figDim[1]))
            fig.suptitle("Percent of consistent subjects'\n two sessions across channels",fontsize=24)
            white_patch = mpatches.Patch(facecolor='white', edgecolor='k', label='Intra-subject results across channels')
            blue_dot = mlines.Line2D([], [], color='white', markerfacecolor='blue', marker='o',
                          markersize=10, label='Inter-subject results (100 iterations)')
            red_bar = mlines.Line2D([], [], color='red',lw=3,
                          markersize=15, label='Low consistency threshold')
            green_bar = mlines.Line2D([], [], color='green',lw=3,
                          markersize=15, label='High consistency threshold')
            ax.legend(handles=[white_patch,blue_dot,green_bar,red_bar],fontsize=20,loc='lower left')
        

    fig.subplots_adjust(left=0.09,bottom=0.07,right=0.96,top=0.86,wspace=0.2,hspace=0.2)

    #fig.savefig('images/KWresultsAdjust.eps', format='eps', dpi=1000)

    #pl.show()

def plotAllKW(allSigP,maxChanSigP,multiSession,makeSample=False,thresholdOnly=False):
    exclude = 0
    figDim = [10,6.66]# for single  #[15,15] for tall

    fig = pl.figure(figsize=(figDim[0], figDim[1])) #15,15
    fig.suptitle("Correlation coefficients of subjects'\n two sessions across channels",fontsize=24)

    
    featsV = ['Rel-Lower', 'Rel-Delta', 'Rel-Theta', 'Rel-Alpha', 'Rel-Mu', 'Rel-Beta', 'Rel-Gamma', 
                'Abs-Lower', 'Abs-Delta', 'Abs-Theta', 'Abs-Alpha', 'Abs-Mu', 'Abs-Beta', 'Abs-Gamma', 
                'Spectral Entropy', 
                'Entropy', 'Curve-Length', 'Energy', 'Nonlinear-Energy', 'Sixth-Power', 'LZC', 'Min', 
                'Max', 'Med', 'Var', 'Std', 'Skew', 'Kurtosis','Mobility','Complexity']

    featsV = ['Relative Lower Power', 'Relative Delta Power', 'Relative Theta Power', 'Relative Alpha Power', 'Relative Mu Power', 'Relative Beta Power', 'Relative Gamma Power', 
            'Absolute Lower Power', 'Absolute Delta Power', 'Absolute Theta Power', 'Absolute Alpha Power', 'Absolute Mu Power', 'Absolute Beta Power', 'Absolute Gamma Power', 
            'Spectral Entropy', 
            'Entropy', 'Curve Length', 'Energy', 'Non-linear Energy', 'Sixth Power', 'LZC', 'Minimum', 
            'Maximum', 'Median', 'Variance', 'SD', 'Skew', 'Kurtosis','Mobility','Complexity']

    fontSizes = [12,12,20,15,24]
    if makeSample == True:
        fontSizes = [20,24,24,24,26]

    #pdb.set_trace()

    for v in range(len(featsV)):
        if makeSample == True:
            if (v != 10) and (v!= 26):
                continue
        dim1=5
        dim2=6
        #pdb.set_trace()
        #ax = fig.add_subplot(2,2,curD+1)
        #fig,ax = resultBoxPlot(fig,allCV[curD],ax=ax,exclude=exclude)

        if makeSample == True:
            dim1=1
            dim2=1
            ax = fig.add_subplot(dim1,dim2,1)
        else:
            ax = fig.add_subplot(dim1,dim2,v+1)

        ax.tick_params(labelsize=fontSizes[0])

        fig,ax = resultBoxPlot(fig,np.transpose(allSigP[:,:,v]),ax=ax,exclude=exclude)
        
        if thresholdOnly == True:
            #ax.plot(np.concatenate((np.array([0]),np.transpose(np.max(maxChanSigP[:,:,v],axis=1))),axis=0),'r_',markersize=20,markeredgewidth=5)
            ax.plot(np.concatenate((np.array([0]),np.array([0.22,0.22,0.22,0.22])),axis=0),'r_',markersize=160,markeredgewidth=3)
            ax.plot(np.concatenate((np.array([0]),np.array([0.5,0.5,0.5,0.5])),axis=0),'g_',markersize=160,markeredgewidth=3)

        else:
            for r in range(np.shape(maxChanSigP)[1]):
                ax.plot(np.concatenate((np.array([0]),np.transpose(maxChanSigP[:,r,v])),axis=0),'b.')
        
        ax.set_ylim(ymin=0,ymax=1)        
        ax.set_xticklabels([ '15 sec', '30 sec', '60 sec', '120 sec'])
        ax.set_xlabel('Epoch time',fontsize=fontSizes[1])

        ax.set_ylabel('Correlation coefficient', fontsize=fontSizes[2])
        #ax.set_ylim(ymax=0.3)

        if v < ((dim1*dim2)-dim2):
            ax.get_xaxis().set_visible(False)
        if v%dim2 != 0:
            ax.get_yaxis().set_visible(False)

        #ax.set_title('COV of Features on %d parts, each %0.2f minutes (n=%d)'%(np.shape(curData)[3],16./np.shape(curData)[3],subjNum))
        ax.set_title('%s'%(featsV[v]),fontsize=fontSizes[3])

        if makeSample == True:
            fig.subplots_adjust(left=0.11,bottom=0.08,right=0.96,top=0.86,wspace=0.2,hspace=0.2)
            fig = pl.figure(figsize=(figDim[0], figDim[1]))
            fig.suptitle("Correlation coefficients of subjects'\n two sessions across channels",fontsize=24)
            white_patch = mpatches.Patch(facecolor='white', edgecolor='k', label='C.C. results across channels')
            red_bar = mlines.Line2D([], [], color='red',lw=3,
                          markersize=15, label='Negligible threshold')
            green_bar = mlines.Line2D([], [], color='green',lw=3,
                          markersize=15, label='Moderate to high threshold')
            ax.legend(handles=[white_patch,green_bar,red_bar],fontsize=20,loc='upper right')

    fig.subplots_adjust(left=0.09,bottom=0.07,right=0.96,top=0.86,wspace=0.2,hspace=0.2)
    
    #fig.savefig('images/KendallWresultsAdjust.eps', format='eps', dpi=1000)

    #pl.show()
def makeSamplePlot(allSigP,multiSession,makeSample=False):
    figDim = [10,6.66]  #[15,15] for tall or 10 by 13
    fig = pl.figure(figsize=(figDim[0], figDim[1])) #[10,13]
    fig.suptitle('Kruskal-Wallis results for stationarity\n across varying epoch comparisons',fontsize=24)
    fontSizes = [12,12,20,15,24]
    if makeSample == True:
        fontSizes = [20,24,24,24,24]
    
    featsV = ['Highly Stationary Feature: Non-linear Energy','Somewhat Stationary Feature: LZC','Non-Stationary Feature: Spectral Entropy']
    #pdb.set_trace()


    for v in range(len(featsV)):
        dim1=3
        dim2=1
        #pdb.set_trace()
        #ax = fig.add_subplot(2,2,curD+1)
        #fig,ax = resultBoxPlot(fig,allCV[curD],ax=ax,exclude=exclude)
        if makeSample == True:
            dim1=1
            ax = fig.add_subplot(dim1,dim2,1)
        else:
            ax = fig.add_subplot(dim1,dim2,v+1)
        
        ax.tick_params(labelsize=fontSizes[0])

        fig,ax = resultBoxPlot(fig,np.transpose(allSigP[v,:,:]),totalInst=multiSession,ax=ax)
        
        ax.set_ylim(ymin=0,ymax=100)
        ax.set_xticklabels(['120 v. 60','120 v. 30','120 v. 15','120 v. 7.5','60 v. 30','60 v. 15','60 v. 7.5','30 v. 15','30 v. 7.5','15 v. 7.5'],rotation=45)
        #ax.set_xticklabels(['2 v. 1','2 v. 0.5','2 v. 0.25','2 v. 0.125','1 v. 0.5','1 v. 0.25','1 v. 0.125','0.5 v. 0.25','0.5 v. 0.125','0.25 v. 0.125'],rotation=45)
        #ax.set_xticklabels(['2 v. 1','2 v. 1/2','2 v. 1/4','2 v. 1/8','1 v. 1/2','1 v. 1/4','1 v. 1/8','1/2 v. 1/4','1/2 v. 1/8','1/4 v. 1/8'])
        #ax.set_xticklabels(['8 v. 16','8 v. 32','8 v. 64','8 v. 128','16 v. 32','16 v. 64','16 v. 128','32 v. 64','32 v. 128','64 v. 128'])
        #ax.set_xticklabels(['0.125','0.250','0.375','0.500','0.750','0.875','1.00','1.500','1.750','1.875'])
        ax.set_xlabel('Epoch times compared (in seconds)',fontsize=fontSizes[1])

        ax.set_ylabel('%s subjects p>0.05'%('%'),fontsize=fontSizes[2])
        #ax.set_ylim(ymax=0.3)

        if v < ((dim1*dim2)-dim2):
            ax.get_xaxis().set_visible(False)
        if v%dim2 != 0:
            ax.get_yaxis().set_visible(False)

        #ax.set_title('COV of Features on %d parts, each %0.2f minutes (n=%d)'%(np.shape(curData)[3],16./np.shape(curData)[3],subjNum))
        ax.set_title('%s'%(featsV[v]),fontsize=fontSizes[3])

        if makeSample == True:
            #fig.subplots_adjust(left=0.1,bottom=0.10,right=0.96,top=0.86,wspace=0.2,hspace=0.2)
            fig.subplots_adjust(left=0.11,bottom=0.14,right=0.96,top=0.88,wspace=0.2,hspace=0.2)

            fig = pl.figure(figsize=(figDim[0], figDim[1]))
            fig.suptitle('Kruskal-Wallis results for stationarity\n across varying epoch comparisons',fontsize=fontSizes[4])
            
            white_patch = mpatches.Patch(facecolor='white', edgecolor='k', label='Stationarity across channels')
            ax.legend(handles=[white_patch],fontsize=20)

    #fig.subplots_adjust(left=0.1,bottom=0.10,right=0.96,top=0.86,wspace=0.2,hspace=0.2)
    fig.subplots_adjust(left=0.11,bottom=0.14,right=0.96,top=0.88,wspace=0.2,hspace=0.2)

    #fig.savefig('images/KWstationarityExamples.eps', format='eps', dpi=1000)
    #fig.savefig('images/KWstationarityExamples.png', format='png')

    #pl.show()

def plotAllForwardResults(svmTest='gender',mode='Train',featNums=14,toPlot=True):
    allFeaturesOrig = np.array(['Rel-Lower', 'Rel-Delta', 'Rel-Theta', 'Rel-Alpha', 'Rel-Mu', 'Rel-Beta', 'Rel-Gamma', 
            'Abs-Lower', 'Abs-Delta', 'Abs-Theta', 'Abs-Alpha', 'Abs-Mu', 'Abs-Beta', 'Abs-Gamma', 
            'Fourier-Entropy', 'NormedFourier-Entropy', 'Fourier-Entropy-Norm', 'NormedFourier-Entropy-Norm', 'Spectral-Entropy-Raw', 'Spectral-Entropy', 
            'Entropy-Raw', 'Entropy', 'Curve-Length', 'Energy', 'Nonlinear-Energy', 'Sixth-Power', 'LZC', 
            'Min', 'Max', 'Med', 'Mean', 'Var', 'Std', 'Skew', 'Kurtosis', 'Integral', 'Sum'])

    allFeatures = np.array(['R-Lower', 'R-Delta', 'R-Theta', 'R-Alpha', 'R-Mu', 'R-Beta', 'R-Gamma', 
            'A-Lower', 'A-Delta', 'A-Theta', 'A-Alpha', 'A-Mu', 'A-Beta', 'A-Gamma', 
            'Fourier-Entropy', 'NormedFourier-Entropy', 'Fourier-Entropy-Norm', 'NormedFourier-Entropy-Norm', 'Spectral-Entropy-Raw', 'Spect-Ent', 
            'Entropy-Raw', 'Entropy', 'Curve-Len', 'Energy', 'Nonlinear-Energy', 'Sixth-Power', 'LZC', 
            'Min', 'Max', 'Med', 'Mean', 'Var', 'Std', 'Skew', 'Kurtosis', 'Integral', 'Sum'])

    allChannels = np.array(['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','FZ','CZ','PZ'])
    
    variableMask = np.concatenate((range(0,14),range(18,30),range(31,35)),axis=0)
    #variableMaskL1 = np.concatenate((range(0,14),range(18,30),range(31,35)),axis=0)
    #variableMask = variableMaskL1[np.concatenate((range(0,14),[15],range(17,30)))]
    
    allFeats = allFeatures[variableMask]

    allFeats = ['Relative $\ell$ Power', 'Relative $\delta$ Power', 'Relative $\\theta$ Power', 'Relative $\\alpha$ Power', 'Relative $\mu$ Power', 'Relative $\\beta$ Power', 'Relative $\gamma$ Power', 
                'Absolute $\ell$ Power', 'Absolute $\delta$ Power', 'Absolute $\\theta$ Power', 'Absolute $\\alpha$ Power', 'Absolute $\mu$ Power', 'Absolute $\\beta$ Power', 'Absolute $\gamma$ Power', 
                'Spectral Entropy-U', 'Spectral Entropy', 
                'Entropy-U', 'Entropy', 'Curve Length', 'Energy', 'Non-linear Energy', 'Sixth Power', 'LZC', 'Minimum', 
                'Maximum', 'Median', 'Variance', 'SD', 'Skew', 'Kurtosis']

    #pdb.set_trace()


    allVariables = []
    for c in allChannels:
        for f in allFeats:
            allVariables.append(c+'-'+f)
    allVariables  = np.array(allVariables)

    variableList = np.load('forwardSearchDocs/'+svmTest+'_max_allForward'+mode+'List_all_1Parts_16Min_allTextFiles.npy')
    
    variableOrder = allVariables[variableList]
    variableResults = np.load('forwardSearchDocs/'+svmTest+'_max_allForward'+mode+'AllResults_all_1Parts_16Min_allTextFiles.npy')
    
    singleResults = np.load('forwardSearchDocs/'+svmTest+'_max_allForward'+mode+'ListResults_all_1Parts_16Min_allTextFiles.npy')

    numVar = len(allVariables)
    
    '''
    savedResults = []
    curStartInd = 0
    for i in range(len(variableList)):
        curInd = curStartInd + variableList[i]
        print curStartInd,'+',variableList[i],'=',curInd
        savedResults.append(variableResults[curInd,:])
        curStartInd += numVar-i
    savedResults = np.array(savedResults)
    '''
    #pdb.set_trace()

    SFSresults = []
    if mode == 'Test':
        indexChoice = 1
    elif mode == 'Train':
        indexChoice = 0
    elif mode == 'Eval':
        indexChoice = 1
    curStartInd = 0
    for i in range(len(variableList)):
        curEndInd = curStartInd + numVar-i 
        curResArray = variableResults[curStartInd:curEndInd,indexChoice]
        curInd = np.argmax(curResArray)
        #print curStartInd,'+',curInd,'=',curStartInd+curInd
        SFSresults.append(variableResults[curStartInd+curInd,:])
        curStartInd += numVar-i


    #print subNormVariableOrder
    SFSresults = np.array(SFSresults)
    results = SFSresults
    #pdb.set_trace()
    #featNums = 14

    if toPlot == True:
        fig = pl.figure()
        fig.suptitle('Forward Selection on '+mode+' Dimension Influence on Classification Accuracies',fontsize=24)

        ax = fig.add_subplot(1,1,1)
        ax.tick_params(labelsize=20)
        ax.plot(range(1,featNums+1),results[:featNums],'-o')#marker='-.',color='k')
        
        #for i,txt in enumerate(subNormVariableOrder[:10]):
        #   ax.annotate(txt,(i+1,subNormVariableResults[i]),fontsize=18)
        
        ax.set_ylabel('%s accuracy'%('%'),fontsize=20)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
        if mode == 'Eval':
            ax.legend([svmTest+' Train Classification',svmTest+' Eval Classification',svmTest+' Test Classification'],loc=4,fontsize=18)

        else:
            ax.legend([svmTest+' Train Classification',svmTest+' Test Classification'],loc=4,fontsize=18)
        #ax.get_xaxis().set_visible(False)
        #ax.set_ylim(ymin=65,ymax=85)
        
        pl.xticks(np.arange(1, featNums+1, step=1),variableOrder[:featNums])

        ax.set_xlabel('Feature Added',fontsize=20)
        fig.subplots_adjust(left=0.04,bottom=0.07,right=0.98,top=0.95,wspace=0.2,hspace=0.2)

    return variableOrder,results

    #plt.show()

def plotBandChannelForwardResults(svmTest='gender',mode='Train',bandType='rel',featNums=14,toPlot=True):
    allFeaturesOrig = np.array(['Rel-Delta', 'Rel-Theta', 'Rel-Alpha', 'Rel-Mu', 'Rel-Beta', 'Rel-Gamma', 
            'Abs-Delta', 'Abs-Theta', 'Abs-Alpha', 'Abs-Mu', 'Abs-Beta', 'Abs-Gamma'])

    allFeatures = np.array(['R-Delta', 'R-Theta', 'R-Alpha', 'R-Mu', 'R-Beta', 'R-Gamma', 
            'A-Delta', 'A-Theta', 'A-Alpha', 'A-Mu', 'A-Beta', 'A-Gamma'])

    allChannels = np.array(['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','FZ','CZ','PZ'])
    
    if bandType == 'rel': 
        variableMask = range(0,6)
    elif bandType == 'abs':
        variableMask = range(6,12)


    #variableMaskL1 = np.concatenate((range(0,14),range(18,30),range(31,35)),axis=0)
    #variableMask = variableMaskL1[np.concatenate((range(0,14),[15],range(17,30)))]
    
    allFeats = allFeatures[variableMask]

    #allFeats = ['Relative $\delta$ Power', 'Relative $\\theta$ Power', 'Relative $\\alpha$ Power', 'Relative $\mu$ Power', 'Relative $\\beta$ Power', 'Relative $\gamma$ Power', 
    #            'Absolute $\delta$ Power', 'Absolute $\\theta$ Power', 'Absolute $\\alpha$ Power', 'Absolute $\mu$ Power', 'Absolute $\\beta$ Power', 'Absolute $\gamma$ Power']

    #pdb.set_trace()


    allVariables = []
    for c in allChannels:
        for f in allFeats:
            allVariables.append(c+'-'+f)
    allVariables  = np.array(allVariables)

    variableList = np.load('forwardSearchDocs/'+svmTest+'_max_'+bandType+'BandForward'+mode+'List_all_1Parts_16Min_allTextFiles.npy')
    
    variableOrder = allVariables[variableList]
    variableResults = np.load('forwardSearchDocs/'+svmTest+'_max_'+bandType+'BandForward'+mode+'AllResults_all_1Parts_16Min_allTextFiles.npy')
    
    singleResults = np.load('forwardSearchDocs/'+svmTest+'_max_'+bandType+'BandForward'+mode+'ListResults_all_1Parts_16Min_allTextFiles.npy')

    numVar = len(allVariables)
    
    '''
    savedResults = []
    curStartInd = 0
    for i in range(len(variableList)):
        curInd = curStartInd + variableList[i]
        print curStartInd,'+',variableList[i],'=',curInd
        savedResults.append(variableResults[curInd,:])
        curStartInd += numVar-i
    savedResults = np.array(savedResults)
    '''
    #pdb.set_trace()

    SFSresults = []
    if mode == 'Test':
        indexChoice = 1
    elif mode == 'Train':
        indexChoice = 0
    elif mode == 'Eval':
        indexChoice = 1
    curStartInd = 0
    for i in range(len(variableList)):
        curEndInd = curStartInd + numVar-i 
        curResArray = variableResults[curStartInd:curEndInd,indexChoice]
        curInd = np.argmax(curResArray)
        #print curStartInd,'+',curInd,'=',curStartInd+curInd
        SFSresults.append(variableResults[curStartInd+curInd,:])
        curStartInd += numVar-i


    #print subNormVariableOrder
    SFSresults = np.array(SFSresults)
    results = SFSresults
    #pdb.set_trace()
    #featNums = 14

    if toPlot == True:
        fig = pl.figure()
        fig.suptitle('Forward Selection on '+mode+' Dimension Influence on Classification Accuracies',fontsize=24)

        ax = fig.add_subplot(1,1,1)
        ax.tick_params(labelsize=20)
        ax.plot(range(1,featNums+1),results[:featNums],'-o')#marker='-.',color='k')
        
        #for i,txt in enumerate(subNormVariableOrder[:10]):
        #   ax.annotate(txt,(i+1,subNormVariableResults[i]),fontsize=18)
        
        ax.set_ylabel('%s accuracy'%('%'),fontsize=20)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
        if mode == 'Eval':
            ax.legend([svmTest+' Train Classification',svmTest+' Eval Classification',svmTest+' Test Classification'],loc=4,fontsize=18)

        else:
            ax.legend([svmTest+' Train Classification',svmTest+' Test Classification'],loc=4,fontsize=18)
        #ax.get_xaxis().set_visible(False)
        #ax.set_ylim(ymin=65,ymax=85)
        
        pl.xticks(np.arange(1, featNums+1, step=1),variableOrder[:featNums])

        ax.set_xlabel('Feature Added',fontsize=20)
        fig.subplots_adjust(left=0.04,bottom=0.07,right=0.98,top=0.95,wspace=0.2,hspace=0.2)

    return variableOrder,results

    #plt.show()

def plotBandForwardResults(svmTest='gender',mode='Train',bandType='rel',featNums=6,toPlot=True):
    allFeaturesOrig = np.array(['Rel-Delta', 'Rel-Theta', 'Rel-Alpha', 'Rel-Mu', 'Rel-Beta', 'Rel-Gamma', 
            'Abs-Delta', 'Abs-Theta', 'Abs-Alpha', 'Abs-Mu', 'Abs-Beta', 'Abs-Gamma'])

    allFeatures = np.array(['R-Delta', 'R-Theta', 'R-Alpha', 'R-Mu', 'R-Beta', 'R-Gamma', 
            'A-Delta', 'A-Theta', 'A-Alpha', 'A-Mu', 'A-Beta', 'A-Gamma'])

    allChannels = np.array(['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','FZ','CZ','PZ'])
    
    if bandType == 'rel': 
        variableMask = range(0,6)
    elif bandType == 'abs':
        variableMask = range(6,12)


    #variableMaskL1 = np.concatenate((range(0,14),range(18,30),range(31,35)),axis=0)
    #variableMask = variableMaskL1[np.concatenate((range(0,14),[15],range(17,30)))]
    
    allFeats = allFeatures[variableMask]

    #allFeats = ['Relative $\delta$ Power', 'Relative $\\theta$ Power', 'Relative $\\alpha$ Power', 'Relative $\mu$ Power', 'Relative $\\beta$ Power', 'Relative $\gamma$ Power', 
    #            'Absolute $\delta$ Power', 'Absolute $\\theta$ Power', 'Absolute $\\alpha$ Power', 'Absolute $\mu$ Power', 'Absolute $\\beta$ Power', 'Absolute $\gamma$ Power']

    #pdb.set_trace()


    allVariables = []
    for f in allFeats:
        allVariables.append(f)
    allVariables  = np.array(allVariables)

    variableList = np.load('forwardSearchDocs/'+svmTest+'_max_'+'channelsBandForward'+mode+'List_all_1Parts_16Min_allTextFiles.npy')

    variableOrder = allVariables[variableList-1] #subtract one from way numbers are saved by ommiting 'lower' band
    variableResults = np.load('forwardSearchDocs/'+svmTest+'_max_'+'channelsBandForward'+mode+'AllResults_all_1Parts_16Min_allTextFiles.npy')
    
    singleResults = np.load('forwardSearchDocs/'+svmTest+'_max_'+'channelsBandForward'+mode+'ListResults_all_1Parts_16Min_allTextFiles.npy')

    #pdb.set_trace()
    numVar = len(allVariables)
    
    for curF in range(len(allVariables)):
        print 'Accuracy for %s in %s band: %0.02f'%(svmTest,allVariables[curF],variableResults[curF,0]) 

    '''
    savedResults = []
    curStartInd = 0
    for i in range(len(variableList)):
        curInd = curStartInd + variableList[i]
        print curStartInd,'+',variableList[i],'=',curInd
        savedResults.append(variableResults[curInd,:])
        curStartInd += numVar-i
    savedResults = np.array(savedResults)
    '''
    #pdb.set_trace()

    SFSresults = []
    if mode == 'Test':
        indexChoice = 1
    elif mode == 'Train':
        indexChoice = 0
    elif mode == 'Eval':
        indexChoice = 1
    curStartInd = 0
    for i in range(len(variableList)):
        curEndInd = curStartInd + numVar-i 
        curResArray = variableResults[curStartInd:curEndInd,indexChoice]
        curInd = np.argmax(curResArray)
        #print curStartInd,'+',curInd,'=',curStartInd+curInd
        SFSresults.append(variableResults[curStartInd+curInd,:])
        curStartInd += numVar-i


    #print subNormVariableOrder
    SFSresults = np.array(SFSresults)
    results = SFSresults
    #pdb.set_trace()
    #featNums = 14

    if toPlot == True:
        fig = pl.figure()
        fig.suptitle('Forward Selection on '+mode+' Band Influence on Classification Accuracies',fontsize=24)

        ax = fig.add_subplot(1,1,1)
        ax.tick_params(labelsize=20)
        ax.plot(range(1,featNums+1),results[:featNums],'-o')#marker='-.',color='k')
        
        #for i,txt in enumerate(subNormVariableOrder[:10]):
        #   ax.annotate(txt,(i+1,subNormVariableResults[i]),fontsize=18)
        
        ax.set_ylabel('%s accuracy'%('%'),fontsize=20)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
        if mode == 'Eval':
            ax.legend([svmTest+' Train Classification',svmTest+' Eval Classification',svmTest+' Test Classification'],loc=4,fontsize=18)

        else:
            ax.legend([svmTest+' Train Classification',svmTest+' Test Classification'],loc=4,fontsize=18)
        #ax.get_xaxis().set_visible(False)
        #ax.set_ylim(ymin=65,ymax=85)
        
        pl.xticks(np.arange(1, featNums+1, step=1),variableOrder[:featNums])

        ax.set_xlabel('Feature Added',fontsize=20)
        fig.subplots_adjust(left=0.04,bottom=0.07,right=0.98,top=0.95,wspace=0.2,hspace=0.2)

    return variableOrder,results

    #plt.show()

def plotChannelForwardResults(svmTest='gender',mode='Test',featNums=14,toPlot=True):
    allFeaturesOrig = np.array(['Rel-Lower', 'Rel-Delta', 'Rel-Theta', 'Rel-Alpha', 'Rel-Mu', 'Rel-Beta', 'Rel-Gamma', 
            'Abs-Lower', 'Abs-Delta', 'Abs-Theta', 'Abs-Alpha', 'Abs-Mu', 'Abs-Beta', 'Abs-Gamma', 
            'Fourier-Entropy', 'NormedFourier-Entropy', 'Fourier-Entropy-Norm', 'NormedFourier-Entropy-Norm', 'Spectral-Entropy-Raw', 'Spectral-Entropy', 
            'Entropy-Raw', 'Entropy', 'Curve-Length', 'Energy', 'Nonlinear-Energy', 'Sixth-Power', 'LZC', 
            'Min', 'Max', 'Med', 'Mean', 'Var', 'Std', 'Skew', 'Kurtosis', 'Integral', 'Sum'])

    allFeatures = np.array(['R-Lower', 'R-Delta', 'R-Theta', 'R-Alpha', 'R-Mu', 'R-Beta', 'R-Gamma', 
            'A-Lower', 'A-Delta', 'A-Theta', 'A-Alpha', 'A-Mu', 'A-Beta', 'A-Gamma', 
            'Fourier-Entropy', 'NormedFourier-Entropy', 'Fourier-Entropy-Norm', 'NormedFourier-Entropy-Norm', 'Spectral-Entropy-Raw', 'Spect-Ent', 
            'Entropy-Raw', 'Entropy', 'Curve-Len', 'Energy', 'Nonlinear-Energy', 'Sixth-Power', 'LZC', 
            'Min', 'Max', 'Med', 'Mean', 'Var', 'Std', 'Skew', 'Kurtosis', 'Integral', 'Sum'])

    allChannels = np.array(['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','FZ','CZ','PZ'])
    
    variableMask = np.concatenate((range(0,14),range(18,30),range(31,35)),axis=0)
    #variableMaskL1 = np.concatenate((range(0,14),range(18,30),range(31,35)),axis=0)
    #variableMask = variableMaskL1[np.concatenate((range(0,14),[15],range(17,30)))]
    
    allFeats = allFeatures[variableMask]

    allFeats = ['Relative $\ell$ Power', 'Relative $\delta$ Power', 'Relative $\\theta$ Power', 'Relative $\\alpha$ Power', 'Relative $\mu$ Power', 'Relative $\\beta$ Power', 'Relative $\gamma$ Power', 
                'Absolute $\ell$ Power', 'Absolute $\delta$ Power', 'Absolute $\\theta$ Power', 'Absolute $\\alpha$ Power', 'Absolute $\mu$ Power', 'Absolute $\\beta$ Power', 'Absolute $\gamma$ Power', 
                'Spectral Entropy-U', 'Spectral Entropy', 
                'Entropy-U', 'Entropy', 'Curve Length', 'Energy', 'Non-linear Energy', 'Sixth Power', 'LZC', 'Minimum', 
                'Maximum', 'Median', 'Variance', 'SD', 'Skew', 'Kurtosis']

    #pdb.set_trace()


    allVariables = []
    for c in allChannels:
        for f in allFeats:
            allVariables.append(c+'-'+f)
    #allVariables  = np.array(allVariables)
    allVariables = allChannels

    variableList = np.load('forwardSearchDocs/'+svmTest+'_max_channelForward'+mode+'List_all_1Parts_16Min_allTextFiles.npy')
    
    variableOrder = allVariables[variableList]
    variableResults = np.load('forwardSearchDocs/'+svmTest+'_max_channelForward'+mode+'AllResults_all_1Parts_16Min_allTextFiles.npy')
    
    singleResults = np.load('forwardSearchDocs/'+svmTest+'_max_channelForward'+mode+'ListResults_all_1Parts_16Min_allTextFiles.npy')

    numVar = len(allVariables)
    
    '''
    savedResults = []
    curStartInd = 0
    for i in range(len(variableList)):
        curInd = curStartInd + variableList[i]
        print curStartInd,'+',variableList[i],'=',curInd
        savedResults.append(variableResults[curInd,:])
        curStartInd += numVar-i
    savedResults = np.array(savedResults)
    '''

    SFSresults = []
    if mode == 'Test':
        indexChoice = 1
    elif mode == 'Train':
        indexChoice = 0
    elif mode == 'Eval':
        indexChoice = 1
    curStartInd = 0
    for i in range(len(variableList)):
        curEndInd = curStartInd + numVar-i 
        curResArray = variableResults[curStartInd:curEndInd,indexChoice]
        curInd = np.argmax(curResArray)
        #print curStartInd,'+',curInd,'=',curStartInd+curInd
        SFSresults.append(variableResults[curStartInd+curInd,:])
        curStartInd += numVar-i


    #print subNormVariableOrder
    SFSresults = np.array(SFSresults)
    results = SFSresults
    #pdb.set_trace()
    #featNums = 14

    if toPlot == True:
        fig = pl.figure()
        fig.suptitle('Forward Selection on '+mode+' Channel Influence on Classification Accuracies',fontsize=24)

        ax = fig.add_subplot(1,1,1)
        ax.tick_params(labelsize=13)
        ax.plot(range(1,featNums+1),results[:featNums],'-o')#marker='-.',color='k')
        
        #for i,txt in enumerate(subNormVariableOrder[:10]):
        #   ax.annotate(txt,(i+1,subNormVariableResults[i]),fontsize=18)
        
        ax.set_ylabel('%s accuracy'%('%'),fontsize=20)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
        if mode == 'Eval':
            ax.legend([svmTest+' Train Classification',svmTest+' Eval Classification',svmTest+' Test Classification'],loc=4,fontsize=18)

        else:
            ax.legend([svmTest+' Train Classification',svmTest+' Test Classification'],loc=4,fontsize=18)
        #ax.get_xaxis().set_visible(False)
        #ax.set_ylim(ymin=65,ymax=85)
        
        pl.xticks(np.arange(1, featNums+1, step=1),variableOrder[:featNums])

        ax.set_xlabel('Feature Added',fontsize=20)
        fig.subplots_adjust(left=0.04,bottom=0.07,right=0.98,top=0.95,wspace=0.2,hspace=0.2)

    return variableOrder,results

def listFeaturesNames():
    allFeaturesOrig = np.array(['Rel-Lower', 'Rel-Delta', 'Rel-Theta', 'Rel-Alpha', 'Rel-Mu', 'Rel-Beta', 'Rel-Gamma', 
            'Abs-Lower', 'Abs-Delta', 'Abs-Theta', 'Abs-Alpha', 'Abs-Mu', 'Abs-Beta', 'Abs-Gamma', 
            'Fourier-Entropy', 'NormedFourier-Entropy', 'Fourier-Entropy-Norm', 'NormedFourier-Entropy-Norm', 'Spectral-Entropy-Raw', 'Spectral-Entropy', 
            'Entropy-Raw', 'Entropy', 'Curve-Length', 'Energy', 'Nonlinear-Energy', 'Sixth-Power', 'LZC', 
            'Min', 'Max', 'Med', 'Mean', 'Var', 'Std', 'Skew', 'Kurtosis', 'Integral', 'Sum'])

    allFeatures = np.array(['R-Lower', 'R-Delta', 'R-Theta', 'R-Alpha', 'R-Mu', 'R-Beta', 'R-Gamma', 
            'A-Lower', 'A-Delta', 'A-Theta', 'A-Alpha', 'A-Mu', 'A-Beta', 'A-Gamma', 
            'Fourier-Entropy', 'NormedFourier-Entropy', 'Fourier-Entropy-Norm', 'NormedFourier-Entropy-Norm', 'Spectral-Entropy-Raw', 'Spect-Ent', 
            'Entropy-Raw', 'Entropy', 'Curve-Len', 'Energy', 'Nonlinear-Energy', 'Sixth-Power', 'LZC', 
            'Min', 'Max', 'Med', 'Mean', 'Var', 'Std', 'Skew', 'Kurtosis', 'Integral', 'Sum'])

    allChannels = np.array(['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','FZ','CZ','PZ'])
    variableMask = np.concatenate((range(0,14),range(18,30),range(31,35)),axis=0)
    allFeats = allFeatures[variableMask]

    allVariables = []
    for c in allChannels:
        for f in allFeats:
            allVariables.append(c+'-'+f)
    allVariables  = np.array(allVariables)

    subNormVariableList = [408, 275, 366, 55, 177, 115, 478, 7, 8, 9, 10, 11, 12, 13, 14, 17, 19, 20, 21, 22]
    subNormVariableOrder = allVariables[subNormVariableList]
    subNormVariableResults = [75.28089887640449, 77.52808988764045, 79.02621722846442, 79.7752808988764, 80.1498127340824, 80.52434456928839, 80.89887640449439, 80.89887640449439, 80.89887640449439, 80.89887640449439, 80.89887640449439, 80.89887640449439, 80.89887640449439, 80.89887640449439, 80.89887640449439, 80.89887640449439, 80.89887640449439, 80.89887640449439, 80.89887640449439, 80.89887640449439]
    #print subNormVariableOrder

    normVariableList = [558, 272, 93, 90, 323, 91, 363, 555, 83, 298, 58, 148, 10, 77, 79, 207, 52, 17, 449, 47]
    normVariableOrder = allVariables[normVariableList]
    normVariableResults = [68.67749419953596, 74.36194895591647, 76.10208816705337, 77.84222737819026, 78.77030162412993, 79.23433874709977, 79.81438515081207, 80.2784222737819, 80.74245939675174, 80.97447795823666, 81.20649651972158, 81.32250580046404, 81.32250580046404, 81.32250580046404, 81.32250580046404, 81.4385150812065, 81.55452436194895, 81.55452436194895, 81.67053364269141, 81.67053364269141]
    #print normVariableOrder

    featNums = 14

    fig = pl.figure()
    fig.suptitle("Forward Selection on Dimensions' Influence on Classification Accuracies",fontsize=24)

    ax = fig.add_subplot(2,1,1)
    ax.tick_params(labelsize=13)
    ax.plot(range(1,featNums+1),subNormVariableResults[:featNums],'-o',color='k')#marker='-.',color='k')
    
    #for i,txt in enumerate(subNormVariableOrder[:10]):
    #   ax.annotate(txt,(i+1,subNormVariableResults[i]),fontsize=18)
    
    ax.set_ylabel('%s accuracy'%('%'),fontsize=20)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    ax.legend(['TUH Normal vs. Abnormal'],loc=4,fontsize=18)
    #ax.get_xaxis().set_visible(False)
    #ax.set_ylim(ymin=65,ymax=85)
    
    pl.xticks(np.arange(1, featNums+1, step=1),subNormVariableOrder[:featNums])

    ax = fig.add_subplot(2,1,2)
    ax.tick_params(labelsize=13)

    ax.plot(range(1,featNums+1),normVariableResults[:featNums],'-o',color='k')#marker='-.',color='k')
    
    #for i,txt in enumerate(normVariableOrder[:10]):
    #   ax.annotate(txt,(i+1,normVariableResults[i]))
    
    ax.set_ylabel('%s accuracy'%('%'),fontsize=20)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    ax.legend(['Normal vs. Abnormal'],loc=4, fontsize=18)
    #ax.get_xaxis().set_visible(False)
    #ax.set_ylim(ymin=65,ymax=85)

    pl.xticks(np.arange(1, featNums+1, step=1),normVariableOrder[:featNums])

    ax.set_xlabel('Feature Added',fontsize=20)
    fig.subplots_adjust(left=0.04,bottom=0.07,right=0.98,top=0.95,wspace=0.2,hspace=0.2)

    #plt.show()

def selectionResults():
    allChannels = np.array(['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','FZ','CZ','PZ'])
    bandNames = np.array(['Lower', 'Delta', 'Theta', 'Alpha', 'Mu', 'Beta', 'Gamma'])

    normCH = [10, 9, 12, 11, 8, 15, 5, 16, 6, 7, 17, 18, 3, 14, 1, 4, 2, 0, 13]
    normCHresAll = np.array([(75.2447192168985, 79.58236658932715), (79.63678516228748, 80.74245939675174), (82.83101494075218, 81.67053364269141), (81.27253992787223, 82.94663573085847), (82.25141679546626, 82.83062645011601), (82.84389489953632, 82.5986078886311), (83.73261205564143, 82.94663573085847), (85.02060793405461, 83.52668213457076), (85.92220504894384, 83.29466357308584), (82.26429675425038, 82.83062645011601), (82.45749613601237, 81.67053364269141), (82.95981452859351, 81.67053364269141), (83.39773312725399, 82.25058004640371), (84.0030911901082, 82.71461716937355), (84.36373003606388, 82.25058004640371), (84.96908809891808, 82.36658932714617), (85.49716640906749, 82.01856148491879), (85.93508500772798, 82.25058004640371), (86.2699639361154, 81.78654292343387)])
    normCHresults2 = normCHresAll[:,1].tolist()
    normCHresults = [79.23433874709977, 80.62645011600928, 82.25058004640371, 83.06264501160094, 82.83062645011601, 83.06264501160094, 83.17865429234338, 82.94663573085847, 82.94663573085847, 82.36658932714617, 82.13457076566125, 82.36658932714617, 82.5986078886311, 82.25058004640371, 82.25058004640371, 82.36658932714617, 81.90255220417633, 81.90255220417633, 80.8584686774942]
    
    subNormCH = [14, 13, 1, 16, 9, 0, 5, 11, 17, 3, 8, 15, 2, 6, 18, 4, 7, 10, 12]
    subNormCHresAll = np.array([(78.89634601043997, 79.40074906367042), (82.06562266964951, 82.39700374531836), (84.82475764354959, 82.39700374531836), (83.48247576435496, 82.39700374531836), (84.78747203579418, 82.39700374531836), (85.94332587621179, 81.64794007490637), (82.5876211782252, 80.1498127340824), (88.21774794929158, 81.27340823970037), (81.73005219985086, 80.89887640449439), (83.89261744966443, 82.39700374531836), (82.36390753169276, 81.27340823970037), (80.98434004474272, 80.89887640449439), (85.90604026845638, 80.89887640449439), (86.31618195376585, 81.64794007490637), (86.53989560029828, 80.89887640449439), (87.47203579418344, 81.64794007490637), (82.36390753169276, 80.1498127340824), (88.88888888888889, 79.02621722846442), (89.67188665175243, 78.65168539325843)])
    subnormCHresults2 = subNormCHresAll[:,1].tolist()
    subNormCHresults = [79.02621722846442, 81.64794007490637, 82.77153558052434, 82.77153558052434, 83.14606741573034, 82.39700374531836, 83.14606741573034, 82.02247191011236, 83.14606741573034, 82.39700374531836, 82.77153558052434, 82.02247191011236, 81.27340823970037, 80.1498127340824, 81.64794007490637, 80.89887640449439, 79.40074906367042, 78.65168539325843, 77.15355805243446]

    genderCH = [12, 16, 10, 15, 5, 1, 7, 6, 17, 8, 0, 13, 11, 14, 3, 18, 2, 9, 4]
    genderCHresAll = np.array([(62.06896551724138, 67.2316384180791), (69.71786833855799, 69.77401129943503), (73.00940438871473, 68.64406779661017), (77.61755485893417, 69.77401129943503), (81.50470219435736, 68.36158192090396), (84.1692789968652, 69.77401129943503), (77.86833855799372, 70.90395480225989), (89.34169278996865, 68.64406779661017), (73.04075235109718, 72.03389830508475), (74.20062695924764, 71.75141242937853), (75.23510971786834, 72.31638418079096), (76.42633228840126, 72.59887005649718), (77.3667711598746, 72.59887005649718), (77.86833855799372, 71.1864406779661), (78.84012539184953, 71.46892655367232), (79.68652037617555, 71.46892655367232), (80.78369905956113, 72.31638418079096), (93.22884012539186, 71.46892655367232), (82.31974921630093, 69.49152542372882)])
    genderCHresults2 = genderCHresAll[:,1].tolist()
    genderCHresults = [66.94915254237289, 68.64406779661017, 69.77401129943503, 70.62146892655367, 72.88135593220339, 71.75141242937853, 72.03389830508475, 71.46892655367232, 72.88135593220339, 73.44632768361582, 73.44632768361582, 73.44632768361582, 72.59887005649718, 72.03389830508475, 70.90395480225989, 71.46892655367232, 71.1864406779661, 69.2090395480226, 70.33898305084746]

    ageDiffCH = [0, 4, 7, 5, 6, 17, 2, 3, 12, 9, 11, 1, 8, 13, 15, 16, 18, 10, 14]
    ageDiffCHresAll = np.array([(90.4320987654321, 64.70588235294117), (94.1358024691358, 58.8235294117647), (95.37037037037037, 61.76470588235294), (92.90123456790124, 64.70588235294117), (92.5925925925926, 61.76470588235294), (94.44444444444444, 67.6470588235294), (94.44444444444444, 67.6470588235294), (94.44444444444444, 67.6470588235294), (94.75308641975309, 67.6470588235294), (92.5925925925926, 61.76470588235294), (95.06172839506173, 67.6470588235294), (96.29629629629629, 67.6470588235294), (93.51851851851852, 67.6470588235294), (93.51851851851852, 67.6470588235294), (93.51851851851852, 67.6470588235294), (93.51851851851852, 64.70588235294117), (93.51851851851852, 67.6470588235294), (93.51851851851852, 70.58823529411765), (93.51851851851852, 64.70588235294117)])
    ageDiffCHresults2 = ageDiffCHresAll[:,1].tolist()
    ageDiffCHresults = [67.6470588235294, 64.70588235294117, 64.70588235294117, 67.6470588235294, 67.6470588235294, 70.58823529411765, 67.6470588235294, 70.58823529411765, 70.58823529411765, 73.52941176470588, 70.58823529411765, 70.58823529411765, 70.58823529411765, 70.58823529411765, 70.58823529411765, 67.6470588235294, 67.6470588235294, 67.6470588235294, 67.6470588235294]
    
    lowAgeCH = [5, 12, 17, 14, 15, 6, 0, 7, 3, 9, 11, 4, 13, 8, 18, 10, 1, 2, 16]
    lowAgeCHresAll = np.array([(82.79702970297029, 64.77272727272727), (85.39603960396039, 71.5909090909091), (85.64356435643565, 76.13636363636364), (85.76732673267327, 81.81818181818181), (85.14851485148515, 80.68181818181819), (86.75742574257426, 84.0909090909091), (88.11881188118812, 81.81818181818181), (88.49009900990099, 79.54545454545455), (88.24257425742574, 79.54545454545455), (89.60396039603961, 78.4090909090909), (89.60396039603961, 84.0909090909091), (89.85148514851485, 81.81818181818181), (89.97524752475248, 77.27272727272727), (89.72772277227723, 79.54545454545455), (90.22277227722772, 77.27272727272727), (90.5940594059406, 76.13636363636364), (91.46039603960396, 76.13636363636364), (91.58415841584159, 71.5909090909091), (92.32673267326733, 70.45454545454545)])
    lowAgeCHresults2 = lowAgeCHresAll[:,1].tolist()
    lowAgeCHresults = [72.72727272727273, 75.0, 76.13636363636364, 80.68181818181819, 81.81818181818181, 84.0909090909091, 82.95454545454545, 80.68181818181819, 79.54545454545455, 78.4090909090909, 81.81818181818181, 80.68181818181819, 79.54545454545455, 77.27272727272727, 77.27272727272727, 77.27272727272727, 75.0, 73.86363636363636, 72.72727272727273]
    
    highAgeCH = [9, 17, 11, 6, 14, 7, 10, 18, 2, 3, 1, 4, 0, 8, 5, 15, 13, 12, 16]
    highAgeCHresAll = np.array([(71.46263910969793, 68.1159420289855), (77.66295707472177, 76.08695652173913), (82.03497615262322, 78.98550724637681), (80.84260731319554, 77.53623188405797), (87.51987281399046, 78.98550724637681), (89.1891891891892, 81.15942028985508), (89.98410174880763, 82.6086956521739), (85.05564387917329, 78.98550724637681), (86.24801271860095, 80.43478260869566), (86.80445151033386, 81.8840579710145), (94.83306836248013, 79.71014492753623), (87.83783783783784, 77.53623188405797), (83.38632750397456, 75.3623188405797), (90.77901430842607, 79.71014492753623), (91.09697933227345, 78.26086956521739), (98.25119236883943, 77.53623188405797), (90.22257551669317, 76.08695652173913), (90.69952305246423, 73.18840579710145), (91.41494435612083, 72.46376811594203)])
    highAgeCHresults2 = highAgeCHresAll[:,1].tolist()
    highAgeCHresults = [68.84057971014492, 73.91304347826087, 75.3623188405797, 79.71014492753623, 81.15942028985508, 80.43478260869566, 82.6086956521739, 81.8840579710145, 82.6086956521739, 81.15942028985508, 81.8840579710145, 81.15942028985508, 81.15942028985508, 81.15942028985508, 81.15942028985508, 78.98550724637681, 78.26086956521739, 77.53623188405797, 75.3623188405797]

    genderAllCH = [8, 18, 5, 6, 15, 10, 12, 3, 1, 4, 17, 2, 11, 14, 16, 7, 0, 9, 13]
    genderAllCHresAll = np.array([(59.19524889897237, 58.05288461538461), (61.66421993860937, 61.71875), (64.63365808087548, 62.01923076923077), (66.275190177499, 61.95913461538461), (69.88522621113039, 63.46153846153846), (73.2550380354998, 62.62019230769231), (75.75070065394368, 62.98076923076923), (78.74015748031496, 63.82211538461539), (80.89550246897105, 63.58173076923077), (73.75550513812892, 63.64182692307692), (67.92339516882424, 62.19951923076923), (69.35139463499266, 61.41826923076923), (78.27972774589617, 62.92067307692308), (79.38075537168024, 63.46153846153846), (80.7887361537435, 63.64182692307692), (81.8096890431069, 62.68028846153846), (72.28079540904845, 62.43990384615385), (72.78126251167757, 63.22115384615385), (73.1282530361671, 63.04086538461539)])
    genderAllCHresults2 = genderAllCHresAll[:,1].tolist()
    genderAllCHresults = [58.77403846153846, 61.59855769230769, 62.92067307692308, 62.43990384615385, 63.04086538461539, 63.10096153846154, 63.40144230769231, 64.54326923076923, 63.82211538461539, 63.22115384615385, 62.98076923076923, 63.04086538461539, 64.00240384615384, 63.34134615384615, 63.58173076923077, 62.74038461538461, 62.62019230769231, 62.62019230769231, 62.07932692307692]
    
    ageDiffAllCH = [17, 5, 4, 2, 3, 13, 18, 11, 0, 15, 1, 6, 8, 16, 7, 12, 14, 9, 10]
    ageDiffAllCHresAll = np.array([(84.01937046004842, 66.66666666666667), (85.47215496368038, 61.111111111111114), (88.25665859564165, 57.77777777777778), (86.19854721549636, 66.66666666666667), (85.71428571428571, 61.111111111111114), (87.65133171912834, 56.666666666666664), (88.98305084745763, 53.333333333333336), (85.83535108958837, 63.333333333333336), (87.409200968523, 61.111111111111114), (87.65133171912834, 57.77777777777778), (87.65133171912834, 57.77777777777778), (88.25665859564165, 57.77777777777778), (88.37772397094432, 57.77777777777778), (88.61985472154964, 58.888888888888886), (88.86198547215497, 57.77777777777778), (93.94673123486683, 56.666666666666664), (86.31961259079903, 57.77777777777778), (86.80387409200968, 56.666666666666664), (86.68280871670702, 56.666666666666664)])
    ageDiffAllCHresults2 = ageDiffAllCHresAll[:,1].tolist()
    ageDiffAllCHresults = [63.333333333333336, 68.88888888888889, 68.88888888888889, 67.77777777777777, 63.333333333333336, 64.44444444444444, 63.333333333333336, 64.44444444444444, 65.55555555555556, 62.22222222222222, 63.333333333333336, 58.888888888888886, 61.111111111111114, 62.22222222222222, 61.111111111111114, 56.666666666666664, 58.888888888888886, 58.888888888888886, 57.77777777777778]
    
    lowAgeAllCH = [14, 17, 12, 16, 1, 2, 4, 0, 9, 6, 18, 15, 8, 7, 11, 3, 5, 10, 13]
    lowAgeAllCHresAll = np.array([(74.09138110072689, 64.01869158878505), (80.73727933541018, 73.3644859813084), (84.3717549325026, 72.89719626168224), (88.11007268951194, 71.96261682242991), (80.01038421599169, 71.96261682242991), (86.3447559709242, 72.89719626168224), (83.2294911734164, 72.42990654205607), (83.12564901349948, 73.83177570093459), (84.26791277258567, 73.83177570093459), (85.98130841121495, 72.42990654205607), (86.60436137071652, 73.3644859813084), (86.86396677050882, 72.42990654205607), (88.00623052959502, 72.42990654205607), (88.47352024922118, 71.02803738317758), (88.11007268951194, 71.02803738317758), (89.25233644859813, 70.5607476635514), (84.89096573208722, 71.02803738317758), (90.08307372793354, 67.75700934579439), (85.66978193146417, 69.62616822429906)])
    lowAgeAllCHresults2 = lowAgeAllCHresAll[:,1].tolist()
    lowAgeAllCHresults = [66.35514018691589, 72.42990654205607, 74.29906542056075, 75.23364485981308, 74.76635514018692, 74.29906542056075, 73.3644859813084, 74.29906542056075, 74.29906542056075, 74.29906542056075, 73.83177570093459, 74.29906542056075, 74.29906542056075, 74.29906542056075, 73.83177570093459, 72.89719626168224, 71.02803738317758, 70.09345794392523, 70.5607476635514]
    
    highAgeAllCH = [6, 0, 11, 12, 1, 3, 14, 7, 18, 4, 5, 2, 13, 15, 16, 8, 9, 10, 17]
    highAgeAllCHresAll = np.array([(64.30125067971724, 65.98694942903752), (69.29490665216603, 69.16802610114192), (72.72068152981693, 71.28874388254486), (75.52111654884901, 73.4094616639478), (77.89559543230017, 73.89885807504078), (80.20663404023927, 74.22512234910278), (82.59923871669386, 73.6541598694943), (78.18560812035527, 72.34910277324633), (79.86224397317383, 73.16476345840131), (89.32390792097154, 71.77814029363785), (82.37266630415081, 73.4094616639478), (83.55084284937466, 73.5725938009788), (84.48432118905203, 73.16476345840131), (85.78031538879826, 73.08319738988581), (86.994743520029, 71.53344208809135), (87.4932028276237, 71.77814029363785), (79.53597969911183, 72.34910277324633), (80.06162769621172, 72.43066884176183), (81.20355265542868, 71.28874388254486)])
    highAgeAllCHresults2 = highAgeAllCHresAll[:,1].tolist()
    highAgeAllCHresults = [66.3132137030995, 69.9836867862969, 72.43066884176183, 73.4094616639478, 74.14355628058728, 74.14355628058728, 73.5725938009788, 73.81729200652529, 74.06199021207178, 74.14355628058728, 73.89885807504078, 73.98042414355628, 74.38825448613377, 74.30668841761828, 73.81729200652529, 73.81729200652529, 73.16476345840131, 73.08319738988581, 72.02283849918435]


    dilantinCH = [4, 6, 2, 1, 7, 0, 12, 8, 18, 11, 13, 16, 5, 14, 3, 9, 15, 17, 10]
    dilantinCHresAll = np.array([(55.092592592592595, 39.583333333333336), (71.29629629629629, 43.75), (78.35648148148148, 48.958333333333336), (83.10185185185185, 52.083333333333336), (88.31018518518519, 50.0), (70.7175925925926, 50.0), (55.6712962962963, 37.5), (74.30555555555556, 43.75), (85.87962962962963, 51.041666666666664), (70.60185185185185, 54.166666666666664), (70.94907407407408, 52.083333333333336), (71.75925925925925, 53.125), (69.79166666666667, 55.208333333333336), (63.888888888888886, 50.0), (63.19444444444444, 48.958333333333336), (66.55092592592592, 45.833333333333336), (75.0, 52.083333333333336), (77.31481481481481, 52.083333333333336), (85.76388888888889, 50.0)])
    dilantinCHresults2 = dilantinCHresAll[:,1].tolist()
    dilantinCHresults = [53.125, 50.0, 51.041666666666664, 57.291666666666664, 53.125, 50.0, 52.083333333333336, 51.041666666666664, 53.125, 54.166666666666664, 54.166666666666664, 52.083333333333336, 54.166666666666664, 57.291666666666664, 54.166666666666664, 51.041666666666664, 51.041666666666664, 47.916666666666664, 45.833333333333336]
    
    keppraCH = [18, 13, 8, 3, 1, 16, 4, 5, 12, 0, 7, 15, 11, 6, 10, 14, 17, 9, 2]
    keppraCHresAll = np.array([(55.092592592592595, 39.583333333333336), (71.29629629629629, 43.75), (78.35648148148148, 48.958333333333336), (83.10185185185185, 52.083333333333336), (88.31018518518519, 50.0), (70.7175925925926, 50.0), (55.6712962962963, 37.5), (74.30555555555556, 43.75), (85.87962962962963, 51.041666666666664), (70.60185185185185, 54.166666666666664), (70.94907407407408, 52.083333333333336), (71.75925925925925, 53.125), (69.79166666666667, 55.208333333333336), (63.888888888888886, 50.0), (63.19444444444444, 48.958333333333336), (66.55092592592592, 45.833333333333336), (75.0, 52.083333333333336), (77.31481481481481, 52.083333333333336), (85.76388888888889, 50.0)])
    keppraCHresults2 = keppraCHresAll[:,1].tolist()
    keppraCHresults = [60.90909090909091, 65.45454545454545, 62.72727272727273, 62.72727272727273, 66.36363636363636, 65.45454545454545, 63.63636363636363, 60.90909090909091, 62.72727272727273, 62.72727272727273, 65.45454545454545, 64.54545454545455, 66.36363636363636, 67.27272727272727, 63.63636363636363, 60.90909090909091, 61.81818181818182, 60.90909090909091, 53.63636363636363]

    ageDiffB = [3, 1, 2, 4, 0, 5, 6]
    ageDiffBresults = [67.6470588235294, 76.47058823529412, 79.41176470588235, 76.47058823529412, 76.47058823529412, 76.47058823529412, 79.41176470588235]

    dilantinB = [0, 6, 5, 2, 4, 1, 3]
    dilantinBresults = [55.208333333333336, 54.166666666666664, 53.125, 54.166666666666664, 54.166666666666664, 52.083333333333336, 48.958333333333336]

    normB = [3, 2, 1, 4, 5, 0, 6]
    normBresults = [72.2737819025522, 77.1461716937355, 79.11832946635731, 79.35034802784223, 79.23433874709977, 79.23433874709977, 78.53828306264501]

    subNormB = [1, 3, 0, 5, 6, 2, 4]
    subNormBresults = [76.02996254681648, 80.89887640449439, 79.7752808988764, 78.27715355805243, 79.02621722846442, 77.90262172284645, 77.90262172284645]

    genderB = [0, 3, 2, 6, 4, 5, 1]
    genderBresults = [63.27683615819209, 68.36158192090396, 70.33898305084746, 69.77401129943503, 68.07909604519774, 65.81920903954803, 62.429378531073446]


    #evalList(vType='channels',svmTest='norm',transform=['max'],lst=normCH)
    #evalList(vType='channels',svmTest='subNorm',transform=['max'],lst=subNormCH)
    #evalList(vType='channels',svmTest='gender',transform=['max'],lst=genderCH)
    #evalList(vType='channels',svmTest='ageDiff',transform=['max'],lst=ageDiffCH)
    #evalList(vType='channels',svmTest='lowAge',transform=['max'],lst=lowAgeCH)
    #evalList(vType='channels',svmTest='highAge',transform=['max'],lst=highAgeCH)

    #evalList(vType='channels',svmTest='gender',transform=['max'],lst=genderAllCH)
    #evalList(vType='channels',svmTest='ageDiff',transform=['max'],lst=ageDiffAllCH)
    #evalList(vType='channels',svmTest='lowAge',transform=['max'],lst=lowAgeAllCH)
    #evalList(vType='channels',svmTest='highAge',transform=['max'],lst=highAgeAllCH)    

    #evalList(vType='channels',svmTest='dilantin',transform=['max'],lst=dilantinCH)
    #evalList(vType='channels',svmTest='keppra',transform=['max'],lst=keppraCH)

    #evalList(vType='features',svmTest='ageDiff',transform=['max'],lst=ageDiffB)

    fig = pl.figure()
    fig.suptitle("Forward Selection on Channels' Influence on Classification Accuracies",fontsize=20)

    ax = fig.add_subplot(12,1,1)
    ax.plot(range(1,20),normCHresults,'-o',color='k')#marker='-.',color='k')
    for i,txt in enumerate(allChannels[normCH]):
        ax.annotate(txt,(i+1,normCHresults[i]),fontsize=16)
    ax.set_ylabel('%s accuracy'%('%'),fontsize=16)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    ax.legend(['Normal vs. Abnormal'],loc=4,fontsize=15)
    ax.get_xaxis().set_visible(False)
    #ax.set_ylim(ymin=50,ymax=100)

    ax = fig.add_subplot(12,1,2)
    ax.plot(range(1,20),subNormCHresults,'-o',color='k')#marker='-.',color='k')
    for i,txt in enumerate(allChannels[subNormCH]):
        ax.annotate(txt,(i+1,subNormCHresults[i]),fontsize=16)
    ax.set_ylabel('%s accuracy'%('%'),fontsize=16)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    ax.legend(['TUH Normal vs. Abnormal'],loc=4,fontsize=15)
    ax.get_xaxis().set_visible(False)
    #ax.set_ylim(ymin=50,ymax=100)

    ax = fig.add_subplot(12,1,3)
    ax.plot(range(1,20),genderAllCHresults,'-s',color='b')#marker='-.',color='k')
    for i,txt in enumerate(allChannels[genderAllCH]):
        ax.annotate(txt,(i+1,genderAllCHresults[i]),fontsize=16)
    ax.set_ylabel('%s accuracy'%('%'),fontsize=16)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    ax.legend(['Male vs. Female'],loc=4,fontsize=15)
    ax.get_xaxis().set_visible(False)
    #ax.set_ylim(ymin=50,ymax=100)

    ax = fig.add_subplot(12,1,4)
    ax.plot(range(1,20),genderCHresults,'-s',color='b')#marker='-.',color='k')
    for i,txt in enumerate(allChannels[genderCH]):
        ax.annotate(txt,(i+1,genderCHresults[i]),fontsize=16)
    ax.set_ylabel('%s accuracy'%('%'),fontsize=16)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    ax.legend(['Normal Male vs. Female'],loc=4,fontsize=15)
    ax.get_xaxis().set_visible(False)
    #ax.set_ylim(ymin=50,ymax=100)

    ax = fig.add_subplot(12,1,5)
    ax.plot(range(1,20),lowAgeAllCHresults,'-x',color='r')#marker='-.',color='k')
    for i,txt in enumerate(allChannels[lowAgeAllCH]):
        ax.annotate(txt,(i+1,lowAgeAllCHresults[i]),fontsize=16)
    ax.set_ylabel('%s accuracy'%('%'),fontsize=16)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    ax.legend(['Age<20 vs. others'],loc=4,fontsize=15)
    ax.get_xaxis().set_visible(False)
    #ax.set_ylim(ymin=50,ymax=100)

    ax = fig.add_subplot(12,1,6)
    ax.plot(range(1,20),lowAgeCHresults,'-x',color='r')#marker='-.',color='k')
    for i,txt in enumerate(allChannels[lowAgeCH]):
        ax.annotate(txt,(i+1,lowAgeCHresults[i]),fontsize=16)
    ax.set_ylabel('%s accuracy'%('%'),fontsize=16)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    ax.legend(['Normal Age<20 vs. others'],loc=1,fontsize=15)
    ax.get_xaxis().set_visible(False)
    #ax.set_ylim(ymin=50,ymax=100)

    ax = fig.add_subplot(12,1,7)
    ax.plot(range(1,20),highAgeCHresults,'-d',color='g')#marker='-.',color='k')
    for i,txt in enumerate(allChannels[highAgeCH]):
        ax.annotate(txt,(i+1,highAgeCHresults[i]),fontsize=16)
    ax.set_ylabel('%s accuracy'%('%'),fontsize=16)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    ax.legend(['Age>60 vs. others'],loc=4,fontsize=15)
    ax.get_xaxis().set_visible(False)
    #ax.set_ylim(ymin=50,ymax=100)

    ax = fig.add_subplot(12,1,8)
    ax.plot(range(1,20),highAgeAllCHresults,'-d',color='g')#marker='-.',color='k')
    for i,txt in enumerate(allChannels[highAgeAllCH]):
        ax.annotate(txt,(i+1,highAgeAllCHresults[i]),fontsize=16)
    ax.set_ylabel('%s accuracy'%('%'),fontsize=16)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    ax.legend(['Normal Age>60 vs. others'],loc=4,fontsize=15)
    ax.get_xaxis().set_visible(False)
    #ax.set_ylim(ymin=50,ymax=100)
    
    ax = fig.add_subplot(12,1,9)
    ax.plot(range(1,20),ageDiffAllCHresults,'-*',color='c')#marker='-.',color='k')
    for i,txt in enumerate(allChannels[ageDiffAllCH]):
        ax.annotate(txt,(i+1,ageDiffAllCHresults[i]),fontsize=16)
    ax.set_ylabel('%s accuracy'%('%'),fontsize=16)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    ax.legend(['Age<20 vs. Age>60'],loc=1,fontsize=15)
    ax.get_xaxis().set_visible(False)
    #ax.set_ylim(ymin=50,ymax=100)

    ax = fig.add_subplot(12,1,10)
    ax.plot(range(1,20),ageDiffCHresults,'-*',color='c')#marker='-.',color='k')
    for i,txt in enumerate(allChannels[ageDiffCH]):
        ax.annotate(txt,(i+1,ageDiffCHresults[i]),fontsize=16)
    ax.set_ylabel('%s accuracy'%('%'),fontsize=16)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    ax.legend(['Normal Age<20 vs. Age>60'],loc=4,fontsize=16)
    ax.get_xaxis().set_visible(False)
    #ax.set_ylim(ymin=50,ymax=100)

    ax = fig.add_subplot(12,1,11)
    ax.plot(range(1,20),dilantinCHresults,'-+',color='m')#marker='-.',color='k')
    for i,txt in enumerate(allChannels[dilantinCH]):
        ax.annotate(txt,(i+1,dilantinCHresults[i]),fontsize=16)
    ax.set_ylabel('%s accuracy'%('%'),fontsize=16)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    ax.legend(['Dilantin vs. none'],loc=1,fontsize=15)
    ax.get_xaxis().set_visible(False)
    #ax.set_ylim(ymin=50,ymax=100)

    ax = fig.add_subplot(12,1,12)
    ax.plot(range(1,20),keppraCHresults,'-+',color='m')#marker='-.',color='k')
    for i,txt in enumerate(allChannels[keppraCH]):
        ax.annotate(txt,(i+1,keppraCHresults[i]),fontsize=16)
    ax.set_ylabel('%s accuracy'%('%'),fontsize=16)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    ax.legend(['Keppra vs. none'],loc=4,fontsize=15)
    #ax.get_xaxis().set_visible(False)
    #ax.set_ylim(ymin=50,ymax=100)


    pl.xticks(np.arange(1, 20, step=1))
    ax.set_xlabel('Number of Channels Added',fontsize=18)
    fig.subplots_adjust(left=0.07,bottom=0.04,right=0.97,top=0.95,wspace=0.2,hspace=0.2)



    fig = pl.figure()
    fig.suptitle("Forward Selection on Band Powers' Influence on Classification Accuracies",fontsize=24)

    ax = fig.add_subplot(4,1,1)
    ax.plot(range(1,8),subNormBresults,'-o',color='k')#marker='-.',color='k')
    for i,txt in enumerate(bandNames[subNormB]):
        ax.annotate(txt,(i+1,subNormBresults[i]),fontsize=18)
    ax.set_ylabel('%s accuracy'%('%'),fontsize=20)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    ax.legend(['TUH Normal vs. Abnormal'],loc=1,fontsize=18)
    ax.get_xaxis().set_visible(False)
    #ax.set_ylim(ymin=50,ymax=100)

    ax = fig.add_subplot(4,1,2)
    ax.plot(range(1,8),genderBresults,'-s',color='b')#marker='-.',color='k')
    for i,txt in enumerate(bandNames[genderB]):
        ax.annotate(txt,(i+1,genderBresults[i]),fontsize=18)
    ax.set_ylabel('%s accuracy'%('%'),fontsize=20)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    ax.legend(['Normal Male vs. Female'],loc=1,fontsize=18)
    ax.get_xaxis().set_visible(False)
    #ax.set_ylim(ymin=50,ymax=100)

    ax = fig.add_subplot(4,1,3)
    ax.plot(range(1,8),ageDiffBresults,'-*',color='c')#marker='-.',color='k')
    for i,txt in enumerate(bandNames[ageDiffB]):
        ax.annotate(txt,(i+1,ageDiffBresults[i]),fontsize=18)
    ax.set_ylabel('%s accuracy'%('%'),fontsize=20)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    ax.legend(['Age<20 vs. Age>60'],loc=4,fontsize=18)
    ax.get_xaxis().set_visible(False)
    #ax.set_ylim(ymin=50,ymax=100)
    
    ax = fig.add_subplot(4,1,4)
    ax.plot(range(1,8),dilantinBresults,'-+',color='m')#marker='-.',color='k')
    for i,txt in enumerate(bandNames[dilantinB]):
        ax.annotate(txt,(i+1,dilantinBresults[i]),fontsize=18)
    ax.set_ylabel('%s accuracy'%('%'),fontsize=20)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    ax.legend(['Dilantin vs. none'],loc=1,fontsize=18)
    #ax.get_xaxis().set_visible(False)
    #ax.set_ylim(ymin=50,ymax=100)


    pl.xticks(np.arange(1, 8, step=1))
    ax.set_xlabel('Number of Band Powers Added',fontsize=20)
    fig.subplots_adjust(left=0.04,bottom=0.06,right=0.98,top=0.95,wspace=0.2,hspace=0.2)


    #pl.show()

def EEGplotsData(mode="single",timeMin=16):

    txtFile = '/media/david/Data1/NEDC/tuh_eeg/v1.0.0/edf/00000003/s001/00000003_s001_2002_12_31.txt'
    val = Subject()
    val.fileName = txtFile
    val = singleTXTEval(val)

    numOfEDF = len(val.singleTXTdata)
    for edfFileNum in range(numOfEDF):
        edfFile = val.subjEdfFiles[edfFileNum]
        subjEEG = val.singleTXTdata[edfFileNum]
        refData,srate = preProcessData(subjEEG,timeMin=timeMin,srate=100)
        if (np.shape(refData)[1] >= (srate*60*(timeMin+1))):


            if mode=="multi":
                eeg = subjEEG
                data, times = eeg[:]
                try:
                    fig = pl.gcf()
                    eeg.plot(n_channels=19,duration=20,bgcolor='w',color='b',lowpass=50,highpass=1,filtorder=8)
                except:
                    print "Requires EDF EEG for multi-plot. Plotting single channel."
                    mode = "single"

            if mode=="single":
                eeg = refData
                eegCh = eeg[6,:]
                times = np.divide(range(len(eegCh)),60.*srate)
                pl.plot(times[60*100:120*100],np.multiply(eegCh[60*100:120*100],(10**0)))#pow6
                pl.ylim([np.min(eegCh[60*100:120*100]),np.max(eegCh[60*100:120*100])])
                pl.xlim([times[60*100],times[120*100]])
                ticks = np.arange(min(times[60*100:120*100]), max(times[60*100:120*100])+1, 1)
                pl.xticks(ticks)
                pl.xlabel('Time (min)')
                pl.ylabel('Signal Value (muV)')
                pl.title('EEG Signal of a Single Channel')
                pl.show()
            
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


def displaySummaryRes(kruskal=1,kendall=1,stationarity=1):
    summaryRes=1

    featsV = ['Relative $\ell$ Power', 'Relative $\delta$ Power', 'Relative $\\theta$ Power', 'Relative $\\alpha$ Power', 'Relative $\mu$ Power', 'Relative $\\beta$ Power', 'Relative $\gamma$ Power', 
                'Absolute $\ell$ Power', 'Absolute $\delta$ Power', 'Absolute $\\theta$ Power', 'Absolute $\\alpha$ Power', 'Absolute $\mu$ Power', 'Absolute $\\beta$ Power', 'Absolute $\gamma$ Power', 
                'Spectral Entropy', 
                'Entropy', 'Curve Length', 'Energy', 'Non-linear Energy', 'Sixth Power', 'LZC', 'Minimum', 
                'Maximum', 'Median', 'Variance', 'SD', 'Skew', 'Kurtosis','Mobility','Complexity']
    allChannels = np.array(['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','FZ','CZ','PZ'])

    #pdb.set_trace()
    if summaryRes == 1 and kruskal == 1:
        data = np.load('figureGenFiles/allSigPdataAllNew.npy')
        dataTime = np.load('figureGenFiles/allSigPdataAddTimeNew.npy')
        dataR = np.load('figureGenFiles/allSigPrandAllNew.npy')
        dataRTime = np.load('figureGenFiles/allSigPrandAddTimeNew.npy')
        #mask = np.concatenate((range(0,14),[15],range(17,30)))
        #data = data[:,:,mask]
        data = np.concatenate((data,dataTime),axis=2)
        #dataR = dataR[:,:,mask]
        dataR = np.concatenate((dataR,dataRTime),axis=2)

        medianRes1 = np.multiply(np.divide(np.median(np.median(data,axis=1),axis=0),419),100)
        rangeRes1 = np.divide(np.multiply(np.divide(np.median(np.ptp(data,axis=1),axis=0),419),100),2)
        medianResCH1 = np.multiply(np.divide(np.median(data,axis=0),419),100)
        distMinRes1 = medianRes1 - np.multiply(np.divide(np.median(np.min(data,axis=1),axis=0),419),100)


        medianRand1 = np.multiply(np.divide(np.median(np.median(dataR,axis=1),axis=0),419),100)
        rangeRand1 = np.divide(np.multiply(np.divide(np.median(np.ptp(dataR,axis=1),axis=0),419),100),2)
        distMaxRand1 = np.multiply(np.divide(np.median(np.max(dataR,axis=1),axis=0),419),100) - medianRand1

        order = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,25,26,27,16,17,18,19,28,29]
        for i in order:
            print "{:.2f}".format(medianRes1[i]),'$\pm$',"{:.2f}".format(rangeRes1[i]),"({:.2f}".format(medianRand1[i]),'$\pm$',"{:.2f})".format(rangeRand1[i])
        
        for j in order:
            print allChannels[np.argsort(medianResCH1[:,j])[::-1]] 


    if summaryRes == 1 and kendall == 1:
        data = np.load('figureGenFiles/allKWdataAllNew.npy')
        dataTime = np.load('figureGenFiles/allKWdataAddTimeNew.npy')
        dataR = np.load('figureGenFiles/allKWrandAllNew.npy')
        dataRTime = np.load('figureGenFiles/allKWrandAddTimeNew.npy')
        #mask = np.concatenate((range(0,14),[15],range(17,30)))
        #data = data[:,:,mask]
        data = np.concatenate((data,dataTime),axis=2)
        #dataR = dataR[:,:,mask]
        dataR = np.concatenate((dataR,dataRTime),axis=2)

        medianRes2 = np.median(np.median(data,axis=1),axis=0)
        rangeRes2 = np.divide(np.median(np.ptp(data,axis=1),axis=0),2)
        medianResCH2 = np.median(data,axis=0)
        distMinRes2 = medianRes2 - np.median(np.min(data,axis=1),axis=0)

        medianRand2 = np.median(np.median(dataR,axis=1),axis=0)
        rangeRand2 = np.divide(np.median(np.ptp(dataR,axis=1),axis=0),2)
        distMaxRand2 = np.median(np.max(dataR,axis=1),axis=0) - medianRand2

        order = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,25,26,27,16,17,18,19,28,29]
        for i in order:
            print "{:.2f}".format(medianRes2[i]),'$\pm$',"{:.2f}".format(rangeRes2[i]),"({:.2f}".format(medianRand2[i]),'$\pm$',"{:.2f})".format(rangeRand2[i])

        for j in order:
            print allChannels[np.argsort(medianResCH2[:,j])[::-1]] 


    if summaryRes == 1 and stationarity == 1:
        data1 = np.load('figureGenFiles/KWstationarityExamplesAll.npy')
        data2 = np.load('figureGenFiles/KWstationarityExamplesAddTime.npy')
        data = np.concatenate((data1,data2),axis=0)
        medianRes3 = np.multiply(np.divide(np.median(np.median(data,axis=2),axis=1),4313),100)
        rangeRes3 = np.divide(np.multiply(np.divide(np.median(np.ptp(data,axis=2),axis=1),4313),100),2)


        order = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,25,26,27,16,17,18,19,28,29]
        for i in order:
            print "{:.2f}".format(medianRes3[i]),'$\pm$',"{:.2f}".format(rangeRes3[i])

        #stationarity=0

    if (summaryRes==1) and (kruskal==1) and (kendall==1)  and (stationarity==1):
        print '\n\n'
        order = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,25,26,27,16,17,18,19,28,29]
        for i in order:
            if medianRes3[i] > 99:
                color3 = 'green'
            elif medianRes3[i] > 1:
                color3 = 'yellow'
            else:
                color3 = 'red'

            if (medianRes2[i] - distMinRes2[i]) < (medianRand2[i] + distMaxRand2[i]):
                color2 = 'red'
            elif medianRes2[i]-distMinRes2[i] < 0.29:
                color2 = 'yellow'
            else:
                color2 = 'green'

            if (medianRes1[i] - distMinRes1[i]) > (medianRand1[i] + distMaxRand1[i]):
                color1 = 'green'
            elif (medianRand1[i] + distMaxRand1[i]) >= medianRes1[i]:
                color1 = 'red'
            else:
                color1 = 'yellow'


            #print '&',featsV[i],'&',"\cellcolor{{{}!50}}".format(color3),"{:.2f}".format(medianRes3[i]),'$\pm$',"{:.2f}".format(rangeRes3[i]),'&',"\cellcolor{{{}!50}}".format(color1),"{:.2f}".format(medianRes1[i]),'$\pm$',"{:.2f}".format(rangeRes1[i]),"({:.2f}".format(medianRand1[i]),'$\pm$',"{:.2f})".format(rangeRand1[i]),'&',"\cellcolor{{{}!50}}".format(color2),"{:.2f}".format(medianRes2[i]),'$\pm$',"{:.2f}".format(rangeRes2[i]),"({:.2f}".format(medianRand2[i]),'$\pm$',"{:.2f})".format(rangeRand2[i]),'\\\\'
            print '&',featsV[i],'&',"\cellcolor{{{}!50}}".format(color3),"{:.2f}".format(medianRes3[i]),'$\pm$',"{:.2f}".format(rangeRes3[i]),'&',"\cellcolor{{{}!50}}".format(color1),"{:.2f}".format(medianRes1[i]),'$-$',"{:.2f}".format(distMinRes1[i]),"({:.2f}".format(medianRand1[i]),'$+$',"{:.2f})".format(distMaxRand1[i]),'&',"\cellcolor{{{}!50}}".format(color2),"{:.2f}".format(medianRes2[i]),'$-$',"{:.2f}".format(distMinRes2[i]),"({:.2f}".format(medianRand2[i]),'$+$',"{:.2f})".format(distMaxRand2[i]),'\\\\'


def displaySummaryResIntra(kruskal=1,kendall=1,stationarity=1):
    summaryRes=1

    featsV = ['Relative $\ell$ Power', 'Relative $\delta$ Power', 'Relative $\\theta$ Power', 'Relative $\\alpha$ Power', 'Relative $\mu$ Power', 'Relative $\\beta$ Power', 'Relative $\gamma$ Power', 
                'Absolute $\ell$ Power', 'Absolute $\delta$ Power', 'Absolute $\\theta$ Power', 'Absolute $\\alpha$ Power', 'Absolute $\mu$ Power', 'Absolute $\\beta$ Power', 'Absolute $\gamma$ Power', 
                'Spectral Entropy', 
                'Entropy', 'Curve Length', 'Energy', 'Non-linear Energy', 'Sixth Power', 'LZC', 'Minimum', 
                'Maximum', 'Median', 'Variance', 'SD', 'Skew', 'Kurtosis','Mobility','Complexity']
    allChannels = np.array(['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','FZ','CZ','PZ'])

    #pdb.set_trace()
    if summaryRes == 1 and kruskal == 1:
        data = np.load('figureGenFiles/allSigPdataAllNew.npy')
        dataTime = np.load('figureGenFiles/allSigPdataAddTimeNew.npy')
        dataR = np.load('figureGenFiles/allSigPrandAllNew.npy')
        dataRTime = np.load('figureGenFiles/allSigPrandAddTimeNew.npy')
        #mask = np.concatenate((range(0,14),[15],range(17,30)))
        #data = data[:,:,mask]
        data = np.concatenate((data,dataTime),axis=2)
        #dataR = dataR[:,:,mask]
        dataR = np.concatenate((dataR,dataRTime),axis=2)

        medianRes1 = np.multiply(np.divide(np.median(np.median(data,axis=1),axis=0),419),100)
        rangeRes1 = np.divide(np.multiply(np.divide(np.median(np.ptp(data,axis=1),axis=0),419),100),2)
        medianResCH1 = np.multiply(np.divide(np.median(data,axis=0),419),100)
        distMinRes1 = medianRes1 - np.multiply(np.divide(np.median(np.min(data,axis=1),axis=0),419),100)


        medianRand1 = np.multiply(np.divide(np.median(np.median(dataR,axis=1),axis=0),419),100)
        rangeRand1 = np.divide(np.multiply(np.divide(np.median(np.ptp(dataR,axis=1),axis=0),419),100),2)
        distMaxRand1 = np.multiply(np.divide(np.median(np.max(dataR,axis=1),axis=0),419),100) - medianRand1

        order = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,25,26,27,16,17,18,19,28,29]
        for i in order:
            print "{:.2f}".format(medianRes1[i]),'$\pm$',"{:.2f}".format(rangeRes1[i]),"({:.2f}".format(medianRand1[i]),'$\pm$',"{:.2f})".format(rangeRand1[i])
        
        for j in order:
            print allChannels[np.argsort(medianResCH1[:,j])[::-1]] 


    if summaryRes == 1 and kendall == 1:
        data = np.load('figureGenFiles/allKWdataAllNew.npy')
        dataTime = np.load('figureGenFiles/allKWdataAddTimeNew.npy')
        dataR = np.load('figureGenFiles/allKWrandAllNew.npy')
        dataRTime = np.load('figureGenFiles/allKWrandAddTimeNew.npy')
        #mask = np.concatenate((range(0,14),[15],range(17,30)))
        #data = data[:,:,mask]
        data = np.concatenate((data,dataTime),axis=2)
        #dataR = dataR[:,:,mask]
        dataR = np.concatenate((dataR,dataRTime),axis=2)

        medianRes2 = np.median(np.median(data,axis=1),axis=0)
        rangeRes2 = np.divide(np.median(np.ptp(data,axis=1),axis=0),2)
        medianResCH2 = np.median(data,axis=0)
        distMinRes2 = medianRes2 - np.median(np.min(data,axis=1),axis=0)
        distMaxRes2 = np.median(np.max(data,axis=1),axis=0) - medianRes2


        medianRand2 = np.median(np.median(dataR,axis=1),axis=0)
        rangeRand2 = np.divide(np.median(np.ptp(dataR,axis=1),axis=0),2)
        distMaxRand2 = np.median(np.max(dataR,axis=1),axis=0) - medianRand2

        order = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,25,26,27,16,17,18,19,28,29]
        for i in order:
            print "{:.2f}".format(medianRes2[i]),'$\pm$',"{:.2f}".format(rangeRes2[i]),"({:.2f}".format(medianRand2[i]),'$\pm$',"{:.2f})".format(rangeRand2[i])

        for j in order:
            print allChannels[np.argsort(medianResCH2[:,j])[::-1]] 


    if summaryRes == 1 and stationarity == 1:
        data1 = np.load('figureGenFiles/KWstationarityExamplesAll.npy')
        data2 = np.load('figureGenFiles/KWstationarityExamplesAddTime.npy')
        data = np.concatenate((data1,data2),axis=0)
        medianRes3 = np.multiply(np.divide(np.median(np.median(data,axis=2),axis=1),4313),100)
        rangeRes3 = np.divide(np.multiply(np.divide(np.median(np.ptp(data,axis=2),axis=1),4313),100),2)


        order = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,25,26,27,16,17,18,19,28,29]
        for i in order:
            print "{:.2f}".format(medianRes3[i]),'$\pm$',"{:.2f}".format(rangeRes3[i])

        #stationarity=0

    if (summaryRes==1) and (kruskal==1) and (kendall==1)  and (stationarity==1):
        print '\n\n'
        order = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,25,26,27,16,17,18,19,28,29]
        for i in order:
            if medianRes3[i] > 99:
                color3 = 'green'
            elif medianRes3[i] > 1:
                color3 = 'yellow'
            else:
                color3 = 'red'

            if (medianRes2[i] + distMaxRes2[i]) < 0.3:
                color2 = 'red'
            elif medianRes2[i] - distMinRes2[i] < 0.3:
                color2 = 'yellow'
            else:
                color2 = 'green'

            if (medianRes1[i] - medianRand1[i]) < 5:
                color1 = 'red'
            elif (medianRes1[i] - medianRand1[i]) < 10:
                color1 = 'yellow'
            else:
                color1 = 'green'


            #print '&',featsV[i],'&',"\cellcolor{{{}!50}}".format(color3),"{:.2f}".format(medianRes3[i]),'$\pm$',"{:.2f}".format(rangeRes3[i]),'&',"\cellcolor{{{}!50}}".format(color1),"{:.2f}".format(medianRes1[i]),'$\pm$',"{:.2f}".format(rangeRes1[i]),"({:.2f}".format(medianRand1[i]),'$\pm$',"{:.2f})".format(rangeRand1[i]),'&',"\cellcolor{{{}!50}}".format(color2),"{:.2f}".format(medianRes2[i]),'$\pm$',"{:.2f}".format(rangeRes2[i]),"({:.2f}".format(medianRand2[i]),'$\pm$',"{:.2f})".format(rangeRand2[i]),'\\\\'
            print '&',featsV[i],'&',"\cellcolor{{{}!50}}".format(color3),"{:.2f}".format(medianRes3[i]),'$\pm$',"{:.2f}".format(rangeRes3[i]),'&',"\cellcolor{{{}!50}}".format(color1),"{:.2f}".format(medianRes1[i]-medianRand1[i]),'$\pm$',"{:.2f}".format(rangeRes1[i]),'&',"\cellcolor{{{}!50}}".format(color2),"{:.2f}".format(medianRes2[i]),'$\pm$',"{:.2f}".format(rangeRes2[i]),'\\\\'

def displaySummaryResIntraInter(kruskal=1,kendall=1,stationarity=1):
    summaryRes=1

    featsV = ['Relative $\ell$ Power', 'Relative $\delta$ Power', 'Relative $\\theta$ Power', 'Relative $\\alpha$ Power', 'Relative $\mu$ Power', 'Relative $\\beta$ Power', 'Relative $\gamma$ Power', 
                'Absolute $\ell$ Power', 'Absolute $\delta$ Power', 'Absolute $\\theta$ Power', 'Absolute $\\alpha$ Power', 'Absolute $\mu$ Power', 'Absolute $\\beta$ Power', 'Absolute $\gamma$ Power', 
                'Spectral Entropy', 
                'Entropy', 'Curve Length', 'Energy', 'Non-linear Energy', 'Sixth Power', 'LZC', 'Minimum', 
                'Maximum', 'Median', 'Variance', 'SD', 'Skew', 'Kurtosis','Mobility','Complexity']
    allChannels = np.array(['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','FZ','CZ','PZ'])

    #pdb.set_trace()
    if summaryRes == 1 and kruskal == 1:
        data = np.load('figureGenFiles/allSigPdataAllNew.npy')
        dataTime = np.load('figureGenFiles/allSigPdataAddTimeNew.npy')
        dataR = np.load('figureGenFiles/allSigPrandAllNew.npy')
        dataRTime = np.load('figureGenFiles/allSigPrandAddTimeNew.npy')
        #mask = np.concatenate((range(0,14),[15],range(17,30)))
        #data = data[:,:,mask]
        data = np.concatenate((data,dataTime),axis=2)
        #dataR = dataR[:,:,mask]
        dataR = np.concatenate((dataR,dataRTime),axis=2)

        medianRes1 = np.multiply(np.divide(np.median(np.median(data,axis=1),axis=0),419),100)
        rangeRes1 = np.divide(np.multiply(np.divide(np.median(np.ptp(data,axis=1),axis=0),419),100),2)
        medianResCH1 = np.multiply(np.divide(np.median(data,axis=0),419),100)
        distMinRes1 = medianRes1 - np.multiply(np.divide(np.median(np.min(data,axis=1),axis=0),419),100)


        medianRand1 = np.multiply(np.divide(np.median(np.median(dataR,axis=1),axis=0),419),100)
        rangeRand1 = np.divide(np.multiply(np.divide(np.median(np.ptp(dataR,axis=1),axis=0),419),100),2)
        distMaxRand1 = np.multiply(np.divide(np.median(np.max(dataR,axis=1),axis=0),419),100) - medianRand1

        order = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,25,26,27,16,17,18,19,28,29]
        for i in order:
            print "{:.2f}".format(medianRes1[i]),'$\pm$',"{:.2f}".format(rangeRes1[i]),"({:.2f}".format(medianRand1[i]),'$\pm$',"{:.2f})".format(rangeRand1[i])
        
        for j in order:
            print allChannels[np.argsort(medianResCH1[:,j])[::-1]] 


    if summaryRes == 1 and kendall == 1:
        data = np.load('figureGenFiles/allKWdataAllNew.npy')
        dataTime = np.load('figureGenFiles/allKWdataAddTimeNew.npy')
        dataR = np.load('figureGenFiles/allKWrandAllNew.npy')
        dataRTime = np.load('figureGenFiles/allKWrandAddTimeNew.npy')
        #mask = np.concatenate((range(0,14),[15],range(17,30)))
        #data = data[:,:,mask]
        data = np.concatenate((data,dataTime),axis=2)
        #dataR = dataR[:,:,mask]
        dataR = np.concatenate((dataR,dataRTime),axis=2)

        medianRes2 = np.median(np.median(data,axis=1),axis=0)
        rangeRes2 = np.divide(np.median(np.ptp(data,axis=1),axis=0),2)
        medianResCH2 = np.median(data,axis=0)
        distMinRes2 = medianRes2 - np.median(np.min(data,axis=1),axis=0)
        distMaxRes2 = np.median(np.max(data,axis=1),axis=0) - medianRes2


        medianRand2 = np.median(np.median(dataR,axis=1),axis=0)
        rangeRand2 = np.divide(np.median(np.ptp(dataR,axis=1),axis=0),2)
        distMaxRand2 = np.median(np.max(dataR,axis=1),axis=0) - medianRand2

        order = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,25,26,27,16,17,18,19,28,29]
        for i in order:
            print "{:.2f}".format(medianRes2[i]),'$\pm$',"{:.2f}".format(rangeRes2[i]),"({:.2f}".format(medianRand2[i]),'$\pm$',"{:.2f})".format(rangeRand2[i])

        for j in order:
            print allChannels[np.argsort(medianResCH2[:,j])[::-1]] 

        print 'Median CC:', np.median(medianRand2),'+/-',np.median(rangeRand2)
        print 'Max CC:',np.max(np.max(np.max(dataR,axis=1),axis=0),axis=0)

    if summaryRes == 1 and stationarity == 1:
        data1 = np.load('figureGenFiles/KWstationarityExamplesAll.npy')
        data2 = np.load('figureGenFiles/KWstationarityExamplesAddTime.npy')
        data = np.concatenate((data1,data2),axis=0)
        medianRes3 = np.multiply(np.divide(np.mean(np.median(data,axis=2),axis=1),4313),100)
        rangeRes3 = np.divide(np.multiply(np.divide(np.median(np.ptp(data,axis=2),axis=1),4313),100),2)


        order = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,25,26,27,16,17,18,19,28,29]
        for i in order:
            print "{:.2f}".format(medianRes3[i]),'$\pm$',"{:.2f}".format(rangeRes3[i])

        #stationarity=0

    if (summaryRes==1) and (kruskal==1) and (kendall==1)  and (stationarity==1):
        print '\n\n'
        order = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,25,26,27,16,17,18,19,28,29]
        for i in order:
            if medianRes3[i] > 99:
                color3 = 'green'
            elif medianRes3[i] > 1:
                color3 = 'yellow'
            else:
                color3 = 'red'

            if (medianRes2[i] - distMinRes2[i]) < (medianRand2[i] + distMaxRand2[i]):
                color2 = 'red'
            elif medianRes2[i] < 0.5:
                color2 = 'yellow'
            else:
                color2 = 'green'

            if (medianRes1[i]) < 50:
                color1 = 'red'
            elif (medianRes1[i]) < 75:
                color1 = 'yellow'
            else:
                color1 = 'green'

            if (medianRand1[i]) < 50:
                color0 = 'red'
            elif (medianRand1[i]) < 75:
                color0 = 'yellow'
            else:
                color0 = 'green'


            #print '&',featsV[i],'&',"\cellcolor{{{}!50}}".format(color3),"{:.2f}".format(medianRes3[i]),'$\pm$',"{:.2f}".format(rangeRes3[i]),'&',"\cellcolor{{{}!50}}".format(color1),"{:.2f}".format(medianRes1[i]),'$\pm$',"{:.2f}".format(rangeRes1[i]),"({:.2f}".format(medianRand1[i]),'$\pm$',"{:.2f})".format(rangeRand1[i]),'&',"\cellcolor{{{}!50}}".format(color2),"{:.2f}".format(medianRes2[i]),'$\pm$',"{:.2f}".format(rangeRes2[i]),"({:.2f}".format(medianRand2[i]),'$\pm$',"{:.2f})".format(rangeRand2[i]),'\\\\'
            print '&',featsV[i],'&',"\cellcolor{{{}!50}}".format(color3),"{:.2f}".format(medianRes3[i]),'$\pm$',"{:.2f}".format(rangeRes3[i]),'&',"\cellcolor{{{}!50}}".format(color1),"{:.2f}".format(medianRes1[i]),'$\pm$',"{:.2f}".format(rangeRes1[i]),'&',"\cellcolor{{{}!50}}".format(color0),"{:.2f}".format(medianRand1[i]),'$\pm$',"{:.2f}".format(rangeRand1[i]),'&',"\cellcolor{{{}!50}}".format(color2),"{:.2f}".format(medianRes2[i]),'$\pm$',"{:.2f}".format(rangeRes2[i]),'\\\\'

def convertTextToResults(svmTest = 'subNorm'):
    if svmTest == 'norm':
        lineStart = 0
        lineLimit = 112142
        lst = np.array([558, 272, 93, 90, 323, 91, 363, 555, 83, 298, 58, 148, 10, 77, 79, 207, 52, 17, 449, 47])
        results = np.array([68.67749419953596, 74.36194895591647, 76.10208816705337, 77.84222737819026,
         78.77030162412993, 79.23433874709977, 79.81438515081207, 80.2784222737819, 80.74245939675174, 
         80.97447795823666, 81.20649651972158, 81.32250580046404, 81.32250580046404, 81.32250580046404, 
         81.32250580046404, 81.4385150812065, 81.55452436194895, 81.55452436194895, 81.67053364269141, 
         81.67053364269141])
    
    if svmTest == 'subNorm':
        lineStart = 112143
        lineLimit = 224284
        lst = np.array([408, 275, 366, 55, 177, 115, 478, 7, 8, 9, 10, 11, 12, 13, 14, 17, 19, 20, 21, 22])
        results = np.array([75.28089887640449, 77.52808988764045, 79.02621722846442, 79.7752808988764, 
            80.1498127340824, 80.52434456928839, 80.89887640449439, 80.89887640449439, 80.89887640449439, 
            80.89887640449439, 80.89887640449439, 80.89887640449439, 80.89887640449439, 80.89887640449439, 
            80.89887640449439, 80.89887640449439, 80.89887640449439, 80.89887640449439, 80.89887640449439, 
            80.89887640449439])

    if svmTest == 'gender':
        lineStart = 224285
        lineLimit = 303203
        lst = np.array([225, 255, 482, 239, 9, 8, 10, 11, 12, 13, 186, 478, 70, 7])
        results = np.zeros((14,1))

    docName = 'forwardAllFeatures'
    transform = 'max'
    strLabel = 'est'
    FeatsNames = 'all'
    Partitions = 1
    TimeMin = 16
    InputFileName = 'allTextFiles'

    f = open(docName+'.txt','rU')
    content = f.readlines()

    totalExpResults = 0
    for i in range(len(lst)):
        totalExpResults += (19*30) - i


    allResults = np.zeros((totalExpResults,2))

    curResultNum = 0
    for l in range(lineStart,lineLimit):
        curLine = content[l].split(':')
        futureLine = content[l+4].split(':')
        if (curLine[0] == 'Correct Predictions, overall') and (futureLine[0] == 'Correct Predictions, overall'):
            getNumCur = content[l].split('(')[1].split(')')[0]
            getNumFuture = content[l+4].split('(')[1].split(')')[0]
            allResults[curResultNum,:] = [float(getNumCur),float(getNumFuture)]
            curResultNum += 1
    #pdb.set_trace()
    
    np.save('forwardSearchDocs/'+svmTest+'_'+transform+'_'+'allForwardT'+strLabel+'AllResults_'+FeatsNames+'_'+str(Partitions)+'Parts_'+str(TimeMin)+'Min_'+InputFileName,np.array(allResults))  
    np.save('forwardSearchDocs/'+svmTest+'_'+transform+'_'+'allForwardT'+strLabel+'ListResults_'+FeatsNames+'_'+str(Partitions)+'Parts_'+str(TimeMin)+'Min_'+InputFileName,np.array(results))  
    np.save('forwardSearchDocs/'+svmTest+'_'+transform+'_'+'allForwardT'+strLabel+'List_'+FeatsNames+'_'+str(Partitions)+'Parts_'+str(TimeMin)+'Min_'+InputFileName,np.array(lst))  


def checkChanOrder(kruskal=1,kendall=1):
    if checkChannelOrder == 1:
        order = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,25,26,27,16,17,18,19,28,29]
        if kruskal == 1:
            data = np.load('figureGenFiles/allSigPdataAllNew4.npy')
            dataTime = np.load('figureGenFiles/allSigPdataAddTimeNew4.npy')
            dataR = np.load('figureGenFiles/allSigPrandLocAllNew4.npy')
            dataRTime = np.load('figureGenFiles/allSigPrandLocAddTimeNew4.npy')

            kruskal= 0
            
        if kendall == 1:
            data = np.load('figureGenFiles/allKWdataAllNew4.npy')
            dataTime = np.load('figureGenFiles/allKWdataAddTimeNew4.npy')
            dataR = np.load('figureGenFiles/allKWrandLocAllNew4.npy')
            dataRTime = np.load('figureGenFiles/allKWrandLocAddTimeNew4.npy')
            
            kendall = 0

        data = np.concatenate((data,dataTime),axis=2)
        maxData = np.argmax(data,axis=1)
        dataR = np.concatenate((dataR,dataRTime),axis=2)

        for f in order:
            print 'Feature:',featsV[f]
            for e in range(maxData.shape[0]):
                print 'True Test:',allChannels[maxData[e,f]],'; Random:',
                curRandArray = dataR[e,:,f]
                uniqueCH = np.unique(curRandArray,return_counts=True)
                for c in range(len(uniqueCH[0])):
                    print allChannels[int(uniqueCH[0][c])],':',uniqueCH[1][c],',',
                print '\n',
            print '\n',



if __name__ == '__main__':
    start = time.time()
    stationarity = 0
    kruskal = 0
    kendall = 0
    cov = 0
    SFSall = 0
    SFSbandsChannels = 0
    SFSbands = 1
    SFSch = 0
    summaryRes = 0
    checkChannelOrder = 0
    #EEGplotsData(mode="multi")


    if kruskal == 1:
        #data = np.load('figureGenFiles/allSigPdataAdjust.npy')
        #dataTime = np.load('figureGenFiles/allSigPdataAdjustAddTime.npy')
        #dataR = np.load('figureGenFiles/allSigPrandAdjust.npy')
        #dataRTime = np.load('figureGenFiles/allSigPrandAdjustAddTime.npy')
        data = np.load('figureGenFiles/allSigPdataAllNew.npy')
        dataTime = np.load('figureGenFiles/allSigPdataAddTimeNew.npy')
        dataR = np.load('figureGenFiles/allSigPrandAllNew.npy')
        dataRTime = np.load('figureGenFiles/allSigPrandAddTimeNew.npy')

    if kendall == 1:
        #data = np.load('figureGenFiles/allKWdataAdjust.npy')
        #dataTime = np.load('figureGenFiles/allKWdataAdjustAddTime.npy')
        #dataR = np.load('figureGenFiles/allKWrandAdjust.npy')
        #dataRTime = np.load('figureGenFiles/allKWrandAdjustAddTime.npy')
        data = np.load('figureGenFiles/allKWdataAllNew.npy')
        dataTime = np.load('figureGenFiles/allKWdataAddTimeNew.npy')
        dataR = np.load('figureGenFiles/allKWrandAllNew.npy')
        dataRTime = np.load('figureGenFiles/allKWrandAddTimeNew.npy')

    if cov == 1:
        data = np.load('figureGenFiles/allCOVdataAll.npy')
        dataTime = np.load('figureGenFiles/allCOVdataAddTime.npy')

    if (kruskal == 1) or (kendall == 1): 
        if (data.shape[2]>28):
            mask = np.concatenate((range(0,14),[15],range(17,30)))
            data = data[:,:,mask]
            dataR = dataR[:,:,mask]
        data = np.concatenate((data,dataTime),axis=2)
        dataR = np.concatenate((dataR,dataRTime),axis=2)

    if cov == 1:
        mask = np.concatenate((range(0,14),[15],range(17,24)))
        data = data[:,:,mask]
        data = np.concatenate((data,dataTime),axis=2)

    
    featsV = ['Relative $\ell$ Power', 'Relative $\delta$ Power', 'Relative $\\theta$ Power', 'Relative $\\alpha$ Power', 'Relative $\mu$ Power', 'Relative $\\beta$ Power', 'Relative $\gamma$ Power', 
                'Absolute $\ell$ Power', 'Absolute $\delta$ Power', 'Absolute $\\theta$ Power', 'Absolute $\\alpha$ Power', 'Absolute $\mu$ Power', 'Absolute $\\beta$ Power', 'Absolute $\gamma$ Power', 
                'Spectral Entropy', 
                'Entropy', 'Curve Length', 'Energy', 'Non-linear Energy', 'Sixth Power', 'LZC', 'Minimum', 
                'Maximum', 'Median', 'Variance', 'SD', 'Skew', 'Kurtosis','Mobility','Complexity']
    allChannels = np.array(['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','FZ','CZ','PZ'])

    if summaryRes == 1:
        #displaySummaryRes(kruskal,kendall,stationarity)
        #displaySummaryResIntra(kruskal,kendall,stationarity)
        displaySummaryResIntraInter(kruskal,kendall,stationarity)

        kruskal = 0
        kendall = 0
        stationarity = 0

    if checkChannelOrder == 1:
        checkChanOrder(kruskal,kendall)
        kruskal = 0
        kendall = 0

    #if Write2File == 1:
    #np.save('allKWdataAddTime.npy',np.array([allSigP4,allSigP3,allSigP2,allSigP1]))
    #np.save('allKWrandAddTime.npy',np.array([maxChanSigP4,maxChanSigP3,maxChanSigP2,maxChanSigP1]))
    if stationarity == 1:
        data = np.load('figureGenFiles/KWstationarityExamples.npy')
        #Check all: [0,20,21]#[21,22,26]#[26,27,29]
        #data = data[(27,16,17),:]
        makeSamplePlot(data,4313,makeSample=True)
    if kruskal == 1:
        plotAll(data,dataR,419,makeSample=True,thresholdOnly=True)
    if kendall ==  1:
        plotAllKW(data,dataR,419,makeSample=True,thresholdOnly=True)
    if cov == 1:
        #plotCOV2D(data,4313,exclude=1,makeSample=True)
        singleCOVplot(data[:,:,9],4313,'Absolute Theta Power')
        plotCOVcurves(data[:,:,7:14],4313) #only Absolute bands
    if SFSall == 1:
        #selectionResults()
        #listFeaturesNames()
        #convertTextToResults()
        #plotAllForwardResults(svmTest='norm',mode='Train')
        #plotAllForwardResults(svmTest='subNorm',mode='Train')
        #plotAllForwardResults(svmTest='medNorm',mode='Train')
        #plotAllForwardResults(svmTest='gender',mode='Train')
        #plotAllForwardResults(svmTest='genderNorm',mode='Train')
        #plotAllForwardResults(svmTest='lowAge',mode='Train')
        #plotAllForwardResults(svmTest='lowAgeNorm',mode='Train')
        #plotAllForwardResults(svmTest='midAge',mode='Train')
        #plotAllForwardResults(svmTest='midAgeNorm',mode='Train')
        ###plotAllForwardResults(svmTest='highAge',mode='Train')
        #plotAllForwardResults(svmTest='highAgeNorm',mode='Train')
        ###plotAllForwardResults(svmTest='ageDiff',mode='Train')
        #plotAllForwardResults(svmTest='ageDiffNorm',mode='Train')
        #plotAllForwardResults(svmTest='dilantin',mode='Train')
        #plotAllForwardResults(svmTest='keppra',mode='Train')

        classes = ['norm','subNorm','medNorm','gender','genderNorm','lowAge','lowAgeNorm','highAge','highAgeNorm','ageDiff','ageDiffNorm','dilantin','keppra']
        namedClasses = ['Normal vs. Abnormal','TUH Normal vs. Abnormal', 'Medicated Normal vs. Abnormal', 'Male vs. Female',
                        'Normal Male vs. Female', 'Age$<20$ vs. others', 'Normal Age$<20$ vs. others', 'Age$>60$ vs. others', 'Normal Age$>60$ vs. others',
                        'Age$<10$ vs. Age$>60$', 'Normal Age$<10$ vs. Age$>60$', 'Taking Dilantin vs. No Medication','Taking Keppra vs. no medication']
        del namedClasses[2]
        del classes[2]
        mode = 'Train'
        featNums = 20
        fig = pl.figure()
        fig.suptitle('Forward Selection of '+mode+' Dimension Influence on Classification Accuracies',fontsize=24)
        ax = fig.add_subplot(1,1,1)
        #pl.xticks(np.arange(1, featNums+1, step=1),range(1,featNums+1))
        
        ax.set_xlim(1, featNums)
        dim=np.arange(1,featNums+1,1)
        pl.xticks(dim)

        ax.set_xlabel('Feature Number Added',fontsize=20)
        ax.tick_params(labelsize=20)
        c=0
        resultsAll = []
        for svmTest in classes:
            results = plotAllForwardResults(svmTest=svmTest,mode=mode,featNums=featNums,toPlot=False)

            ax.plot(range(1,featNums+1),results[1][:featNums,1],'-o')#marker='-.',color='k')    
            #for i,txt in enumerate(subNormVariableOrder[:10]):
            #   ax.annotate(txt,(i+1,subNormVariableResults[i]),fontsize=18)
            #print namedClasses[c], results[1][:,1]
            print namedClasses[c], results[1][-1,1]
            #print namedClasses[c],'&',' & '.join(results[0].tolist()),'\\\\ \hline' 
            c += 1
            resultsAll.append(results)
        print '\\textbf\{Classification\} & Feature 1 & Feature 2 & Feature 3 & Feature 4 & Feature 5 \\\\ \hline'
        for cl in range(len(namedClasses)):
            print namedClasses[cl],'&',' & '.join(resultsAll[cl][0][:int(featNums/2)].tolist()),'\\\\ \hline' 
        print '\hline'
        print '\\textbf\{Classification\} & Feature 6 & Feature 7 & Feature 8 & Feature 9 & Feature 10 \\\\ \hline'
        for cl in range(len(namedClasses)):
            print namedClasses[cl],'&',' & '.join(resultsAll[cl][0][int(featNums/2):featNums].tolist()),'\\\\ \hline' 

        ax.set_ylabel("Classification Accuracy (%s)"%('%'),fontsize=20)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
        ax.legend(namedClasses,loc='center left', bbox_to_anchor=(1, 0.5),fontsize=18)
        fig.subplots_adjust(left=0.05,bottom=0.07,right=0.71,top=0.95,wspace=0.2,hspace=0.2)

        #ax.get_xaxis().set_visible(False)
        #ax.set_ylim(ymin=65,ymax=85)


    if SFSbandsChannels == 1:
        #selectionResults()
        #listFeaturesNames()
        #convertTextToResults()
        #plotAllForwardResults(svmTest='norm',mode='Train')
        #plotAllForwardResults(svmTest='subNorm',mode='Train')
        #plotAllForwardResults(svmTest='medNorm',mode='Train')
        #plotAllForwardResults(svmTest='gender',mode='Train')
        #plotAllForwardResults(svmTest='genderNorm',mode='Train')
        #plotAllForwardResults(svmTest='lowAge',mode='Train')
        #plotAllForwardResults(svmTest='lowAgeNorm',mode='Train')
        #plotAllForwardResults(svmTest='midAge',mode='Train')
        #plotAllForwardResults(svmTest='midAgeNorm',mode='Train')
        ###plotAllForwardResults(svmTest='highAge',mode='Train')
        #plotAllForwardResults(svmTest='highAgeNorm',mode='Train')
        ###plotAllForwardResults(svmTest='ageDiff',mode='Train')
        #plotAllForwardResults(svmTest='ageDiffNorm',mode='Train')
        #plotAllForwardResults(svmTest='dilantin',mode='Train')
        #plotAllForwardResults(svmTest='keppra',mode='Train')

        classes = ['norm','subNorm','medNorm','gender','genderNorm','lowAge','lowAgeNorm','highAge','highAgeNorm','ageDiff','ageDiffNorm','dilantin','keppra']
        namedClasses = ['Normal vs. Abnormal','TUH Normal vs. Abnormal', 'Medicated Normal vs. Abnormal', 'Male vs. Female',
                        'Normal Male vs. Female', 'Age$<20$ vs. others', 'Normal Age$<20$ vs. others', 'Age$>60$ vs. others', 'Normal Age$>60$ vs. others',
                        'Age$<10$ vs. Age$>60$', 'Normal Age$<10$ vs. Age$>60$', 'Taking Dilantin vs. No Medication','Taking Keppra vs. no medication']
        del namedClasses[2]
        del classes[2]
        mode = 'Train'
        featNums = 20
        fig = pl.figure()
        fig.suptitle('Forward Selection of '+mode+' Dimension Influence on Classification Accuracies',fontsize=24)
        ax = fig.add_subplot(1,1,1)
        #pl.xticks(np.arange(1, featNums+1, step=1),range(1,featNums+1))
        
        ax.set_xlim(1, featNums)
        dim=np.arange(1,featNums+1,1)
        pl.xticks(dim)

        ax.set_xlabel('Feature Number Added',fontsize=20)
        ax.tick_params(labelsize=20)
        c=0
        resultsAll = []
        for svmTest in classes:
            results = plotBandChannelForwardResults(svmTest=svmTest,mode=mode,bandType='rel',featNums=featNums,toPlot=False)

            ax.plot(range(1,featNums+1),results[1][:featNums,1],'-o')#marker='-.',color='k')    
            #for i,txt in enumerate(subNormVariableOrder[:10]):
            #   ax.annotate(txt,(i+1,subNormVariableResults[i]),fontsize=18)
            #print namedClasses[c], results[1][:,1]
            print namedClasses[c], results[1][-1,1]
            #print namedClasses[c],'&',' & '.join(results[0].tolist()),'\\\\ \hline' 
            c += 1
            resultsAll.append(results)
        print '\\textbf\{Classification\} & Feature 1 & Feature 2 & Feature 3 & Feature 4 & Feature 5 \\\\ \hline'
        for cl in range(len(namedClasses)):
            print namedClasses[cl],'&',' & '.join(resultsAll[cl][0][:int(featNums/2)].tolist()),'\\\\ \hline' 
        print '\hline'
        print '\\textbf\{Classification\} & Feature 6 & Feature 7 & Feature 8 & Feature 9 & Feature 10 \\\\ \hline'
        for cl in range(len(namedClasses)):
            print namedClasses[cl],'&',' & '.join(resultsAll[cl][0][int(featNums/2):featNums].tolist()),'\\\\ \hline' 

        ax.set_ylabel("Classification Accuracy (%s)"%('%'),fontsize=20)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
        ax.legend(namedClasses,loc='center left', bbox_to_anchor=(1, 0.5),fontsize=18)
        fig.subplots_adjust(left=0.05,bottom=0.07,right=0.71,top=0.95,wspace=0.2,hspace=0.2)

        #ax.get_xaxis().set_visible(False)
        #ax.set_ylim(ymin=65,ymax=85)

    if SFSbands == 1:
        #selectionResults()
        #listFeaturesNames()
        #convertTextToResults()
        #plotAllForwardResults(svmTest='norm',mode='Train')
        #plotAllForwardResults(svmTest='subNorm',mode='Train')
        #plotAllForwardResults(svmTest='medNorm',mode='Train')
        #plotAllForwardResults(svmTest='gender',mode='Train')
        #plotAllForwardResults(svmTest='genderNorm',mode='Train')
        #plotAllForwardResults(svmTest='lowAge',mode='Train')
        #plotAllForwardResults(svmTest='lowAgeNorm',mode='Train')
        #plotAllForwardResults(svmTest='midAge',mode='Train')
        #plotAllForwardResults(svmTest='midAgeNorm',mode='Train')
        ###plotAllForwardResults(svmTest='highAge',mode='Train')
        #plotAllForwardResults(svmTest='highAgeNorm',mode='Train')
        ###plotAllForwardResults(svmTest='ageDiff',mode='Train')
        #plotAllForwardResults(svmTest='ageDiffNorm',mode='Train')
        #plotAllForwardResults(svmTest='dilantin',mode='Train')
        #plotAllForwardResults(svmTest='keppra',mode='Train')

        classes = ['norm','subNorm','medNorm','gender','genderNorm','lowAge','lowAgeNorm','highAge','highAgeNorm','ageDiff','ageDiffNorm','dilantin','keppra']
        namedClasses = ['Normal vs. Abnormal','TUH Normal vs. Abnormal', 'Medicated Normal vs. Abnormal', 'Male vs. Female',
                        'Normal Male vs. Female', 'Age$<20$ vs. others', 'Normal Age$<20$ vs. others', 'Age$>60$ vs. others', 'Normal Age$>60$ vs. others',
                        'Age$<10$ vs. Age$>60$', 'Normal Age$<10$ vs. Age$>60$', 'Taking Dilantin vs. No Medication','Taking Keppra vs. no medication']
        del namedClasses[2]
        del classes[2]
        mode = 'Train'
        if mode == 'Train':
            dataIndex = 0
        elif mode == 'Test':
            dataIndex = 1

        featNums = 6
        fig = pl.figure()
        fig.suptitle('Forward Selection of '+mode+' Band Influence on Classification Accuracies',fontsize=24)
        ax = fig.add_subplot(1,1,1)
        #pl.xticks(np.arange(1, featNums+1, step=1),range(1,featNums+1))
        
        ax.set_xlim(1, featNums)
        dim=np.arange(1,featNums+1,1)
        pl.xticks(dim)

        ax.set_xlabel('Band Number Added',fontsize=20)
        ax.tick_params(labelsize=20)
        c=0
        resultsAll = []
        for svmTest in classes:
            results = plotBandForwardResults(svmTest=svmTest,mode=mode,bandType='rel',featNums=featNums,toPlot=False)
            #pdb.set_trace()
            ax.plot(range(1,featNums+1),results[1][:featNums,dataIndex],'-o')#marker='-.',color='k')    
            #for i,txt in enumerate(subNormVariableOrder[:10]):
            #   ax.annotate(txt,(i+1,subNormVariableResults[i]),fontsize=18)
            #print namedClasses[c], results[1][:,1]
            print namedClasses[c], results[1][-1,dataIndex]
            #print namedClasses[c],'&',' & '.join(results[0].tolist()),'\\\\ \hline' 
            c += 1
            resultsAll.append(results)
        print '\\textbf\{Classification\} & Band 1 & Band 2 & Band 3 & Band 4 & Band 5 & Band 6 \\\\ \hline'
        for cl in range(len(namedClasses)):
            print namedClasses[cl],'&',' & '.join(resultsAll[cl][0][:featNums].tolist()),'\\\\ \hline' 
        print '\hline'

        ax.set_ylabel("Classification Accuracy (%s)"%('%'),fontsize=20)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
        ax.legend(namedClasses,loc='center left', bbox_to_anchor=(1, 0.5),fontsize=18)
        fig.subplots_adjust(left=0.05,bottom=0.07,right=0.71,top=0.95,wspace=0.2,hspace=0.2)

        #ax.get_xaxis().set_visible(False)
        ax.set_ylim(ymin=50,ymax=100)


    if SFSch == 1:
        #selectionResults()
        #listFeaturesNames()
        #convertTextToResults()
        #plotAllForwardResults(svmTest='norm',mode='Train')
        #plotAllForwardResults(svmTest='subNorm',mode='Train')
        #plotAllForwardResults(svmTest='medNorm',mode='Train')
        #plotAllForwardResults(svmTest='gender',mode='Train')
        #plotAllForwardResults(svmTest='genderNorm',mode='Train')
        #plotAllForwardResults(svmTest='lowAge',mode='Train')
        #plotAllForwardResults(svmTest='lowAgeNorm',mode='Train')
        #plotAllForwardResults(svmTest='midAge',mode='Train')
        #plotAllForwardResults(svmTest='midAgeNorm',mode='Train')
        ###plotAllForwardResults(svmTest='highAge',mode='Train')
        #plotAllForwardResults(svmTest='highAgeNorm',mode='Train')
        ###plotAllForwardResults(svmTest='ageDiff',mode='Train')
        #plotAllForwardResults(svmTest='ageDiffNorm',mode='Train')
        #plotAllForwardResults(svmTest='dilantin',mode='Train')
        #plotAllForwardResults(svmTest='keppra',mode='Train')

        classes = ['norm','subNorm','medNorm','gender','genderNorm','lowAge','lowAgeNorm','highAge','highAgeNorm','ageDiff','ageDiffNorm','dilantin','keppra']
        namedClasses = ['Normal vs. Abnormal','TUH Normal vs. Abnormal', 'Medicated Normal vs. Abnormal', 'Male vs. Female',
                        'Normal Male vs. Female', 'Age$<20$ vs. others', 'Normal Age$<20$ vs. others', 'Age$>60$ vs. others', 'Normal Age$>60$ vs. others',
                        'Age$<10$ vs. Age$>60$', 'Normal Age$<10$ vs. Age$>60$', 'Taking Dilantin vs. No Medication','Taking Keppra vs. no medication']
        #del namedClasses[7]
        
        mode = 'Train'
        featNums = 19
        fig = pl.figure()
        fig.suptitle('Forward Selection of '+mode+' Channel Influence on Classification Accuracies',fontsize=24)
        ax = fig.add_subplot(1,1,1)
        #pl.xticks(np.arange(1, featNums+1, step=1),range(1,featNums+1))
        #ax.xaxis.set_tick_params(labelsize=20)
        #ax.tick_params(axis='both',labelsize=20)
        ax.set_xlim(1, featNums)
        dim=np.arange(1,featNums+1,1)
        pl.xticks(dim)

        ax.set_xlabel('Channel Number Added',fontsize=20)
        ax.tick_params(axis='both',labelsize=13)

        c=0
        resultsAll = []
        for svmTest in classes:
            results = plotChannelForwardResults(svmTest=svmTest,mode=mode,featNums=featNums,toPlot=False)

            ax.plot(range(1,featNums+1),results[1][:featNums,1],'-o')#marker='-.',color='k')    
            #for i,txt in enumerate(subNormVariableOrder[:10]):
            #   ax.annotate(txt,(i+1,subNormVariableResults[i]),fontsize=18)
            #print namedClasses[c], results[1][:,1]
            print namedClasses[c], results[1][-1,1]
            #print namedClasses[c],'&',' & '.join(results[0].tolist()),'\\\\ \hline' 
            c += 1
            resultsAll.append(results)

        for cl in range(len(namedClasses)):
            print namedClasses[cl],'&',' & '.join(resultsAll[cl][0][:int(featNums/2)].tolist()),'\\\\ \hline' 
        print '\hline'
        for cl in range(len(namedClasses)):
            print namedClasses[cl],'&',' & '.join(resultsAll[cl][0][int(featNums/2):featNums].tolist()),'\\\\ \hline' 

        ax.set_ylabel("Classification Accuracy (%s)"%('%'),fontsize=20)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
        ax.legend(namedClasses,loc='center left', bbox_to_anchor=(1, 0.5),fontsize=18)
        fig.subplots_adjust(left=0.04,bottom=0.07,right=0.71,top=0.95,wspace=0.2,hspace=0.2)

        #ax.get_xaxis().set_visible(False)
        #ax.set_ylim(ymin=65,ymax=85)


    pl.show()


    end=time.time()
    print '\nTime Elapsed:',end-start,'\n'