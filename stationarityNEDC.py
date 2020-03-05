import numpy as np
from scipy import stats,spatial
from sklearn.metrics import mean_absolute_error
import pdb
import time
import sys
#import matplotlib.pyplot as plt
import pylab as pl
from nedcTools import resultBoxPlot,dataLoad3D,dataLoad4D
from multiprocessing import Pool
from functools import partial
import warnings
warnings.filterwarnings("error")

def defineParams(parts):
    features = ['all']
    partitions = parts
    timeMin = 16
    threads = 1
    write2File = 1
    featsNames = ''
    textFile = 'allTextFiles'
    for x in features:
        featsNames = featsNames+x
    return features,partitions,timeMin,threads,write2File,featsNames,textFile

def getNormData(dataLoad,labelsLoad):
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

    return dataN,labelsN
def zca_whitening_matrix(X):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=False) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
    #pdb.set_trace()
    return ZCAMatrix


def genFeatAnaly2D(data1, labels1, data2, labels2,testNormal=False,exclude=0):
    channels = 19
    print('Parts: ',np.shape(data1)[3],np.shape(data1))
    print('Parts: ',np.shape(data2)[3],np.shape(data2))
    smallParts = np.shape(data1)[3]
    bigParts = np.shape(data2)[3]
    pool = Pool(processes=channels)
    parFunction = partial(parFeatAnaly2D,data1=data1,labels1=labels1,data2=data2,labels2=labels2,testNormal=testNormal,exclude=exclude)
    results = pool.map(parFunction,range(np.shape(data1)[1]))#,contentList)
    pool.close()
    pool.join()

    allSigP = results[0][0]
    allSigPbig = results[0][1]
    allSigPsmall = results[0][2]
    multiSession = results[0][3]
    
    for l in range(1,len(results)):
        allSigP = np.concatenate((allSigP,results[l][0]),axis=0)
        allSigPbig = np.concatenate((allSigPbig,results[l][1]),axis=0)
        allSigPsmall = np.concatenate((allSigPsmall,results[l][2]),axis=0)

    # Create a figure instance
    fig = pl.figure(figsize=(9, 6))
    resultBoxPlot(fig,allSigP,totalInst=multiSession,exclude=exclude)
    fig.suptitle('Kruskal-Wallis Results of Stationarity Variability of Features, %d,%d parts (n=%d)'%(smallParts,bigParts,multiSession))
    fig.subplots_adjust(left=0.09,bottom=0.33,right=0.96,top=0.93,wspace=0.2,hspace=0.2)
    #fig.savefig('images/KWstat-%d-%d.eps'%(smallParts,bigParts), format='eps', dpi=1000)
    
    if testNormal:
        fig = pl.figure(figsize=(9, 6))
        resultBoxPlot(fig,allSigPbig,totalInst=multiSession,exclude=exclude)
        fig.suptitle('Shapiro Results of Stationarity Variability of Features, %d parts (n=%d)'%(bigParts,multiSession))
        fig.subplots_adjust(left=0.09,bottom=0.33,right=0.96,top=0.93,wspace=0.2,hspace=0.2)
        fig.savefig('images/shapiro%d.eps'%(bigParts), format='eps', dpi=1000)

        fig = pl.figure(figsize=(9, 6))
        resultBoxPlot(fig,allSigPsmall,totalInst=multiSession,exclude=exclude)
        fig.suptitle('Shapiro Results of Stationarity Variability of Features, %d parts (n=%d)'%(smallParts,multiSession))
        fig.subplots_adjust(left=0.09,bottom=0.33,right=0.96,top=0.93,wspace=0.2,hspace=0.2)
        fig.savefig('images/shapiro%d.eps'%(smallParts), format='eps', dpi=1000)
    #pl.show()

    return allSigP


def parFeatAnaly2D(channels, data1, labels1, data2, labels2,testNormal=False,exclude=0):
    channelArray = [channels]
    channelNum = len(channelArray)
    #print('Parts: ',np.shape(data1)[3],np.shape(data1))
    #print('Parts: ',np.shape(data2)[3],np.shape(data2))
    if len(labels1) < len(labels2):
        subjNum = len(labels1)
        smallLabel = labels1
        smallData = data1
        bigLabel = labels2
        bigData = data2
        smallParts = np.shape(data1)[-1]
        bigParts = np.shape(data2)[-1]
    else:
        subjNum = len(labels2)
        smallLabel = labels2
        smallData = data2
        bigLabel = labels1
        bigData = data1
        smallParts = np.shape(data2)[-1]
        bigParts = np.shape(data1)[-1]
    
    allSigP = np.zeros((channelNum,np.shape(smallData)[2]))
    allSigPbig = np.zeros((channelNum,np.shape(bigData)[2]))
    allSigPsmall = np.zeros((channelNum,np.shape(smallData)[2]))
    
    for c in channelArray:
        for v in range(np.shape(smallData)[2]):
            sigP = 0
            if testNormal:
                sigPs = 0
                sigPb = 0

            multiSession = 0
            #pdb.set_trace()

            for i in range(subjNum):
                sessionNameCur = smallLabel[i][0]
                ind = np.where(bigLabel[:,0] == sessionNameCur)

                if len(ind[0])>0:
                    multiSession += 1
                    j = ind[0][0]
                    #print(labels[dataCaptured[0]][0]+','+labels[dataCaptured[1]][0])

                    meanP = stats.kruskal(smallData[i,c,v,:],bigData[j,c,v,:])[1]#stats.kruskal(*data[dataCaptured,c,:,v])[1]
                    if meanP > 0.05: #for K-W >0.05 #11.34:#0.01:
                        sigP += 1
                    #c,v=0,19
                    #pdb.set_trace()

                    #print "Small Shape:",np.shape(smallData[i,c,v,:])," Big Shape:",np.shape(bigData[i,c,v,:])
                    if testNormal:
                        meanPs = stats.shapiro(smallData[i,c,v,:])[1]
                        meanPb = stats.shapiro(bigData[j,c,v,:])[1]
                        #print 'Variable',v,'p-value on',len(dataCaptured),'subjects:',meanP
                        if (meanPs < 0.05): #for K-W >0.05 #11.34:#0.01:
                            sigPs += 1
                        if (meanPb < 0.05): #for K-W >0.05 #11.34:#0.01:
                            sigPb += 1
                    
                    #print 'Number of significant P-values',meanP
                    #print 'Coeff of Var for channel',c,'variable',v,':',CV[-1],'\n'        

            allSigP[channelArray.index(c),v] = sigP

            if testNormal:
                allSigPbig[channelArray.index(c),v] = sigPb
                allSigPsmall[channelArray.index(c),v] = sigPs

    return allSigP,allSigPbig,allSigPsmall,multiSession


def covAnaly2D(*datum):
    exclude = 0
    if len(datum) % 2 !=0:
        exclude = datum[-1] 
        datum = datum[:-1]

    subjNumT = len(datum[1])
    numData = len(datum)
    fig = pl.figure(figsize=(13.5, 9))
    fig.suptitle('COV across epoch lengths (n=%d)'%(subjNumT))

    allCV = np.zeros((numData/2,np.shape(datum[0])[1],np.shape(datum[0])[2]))
    indices = np.zeros((numData/2,np.shape(datum[0])[1],np.shape(datum[0])[2],2))

    for d in range(0,numData,2):
        print('Parts: ',np.shape(datum[d])[3],np.shape(datum[d]))
        curData = datum[d]
        curLabel = datum[d+1]
        subjNum = len(curLabel)
        curD=d/2
        for c in range(np.shape(curData)[1]):
            for v in range(np.shape(curData)[2]):

                CV = []

                for i in range(subjNum):
                    if v == 27:
                        curData[i,c,v,:] = np.abs(curData[i,c,v,:])
                    try:
                        curData[i,c,v,:] = np.abs(np.log(curData[i,c,v,:]))
                    except RuntimeWarning:
                        print 'Channel:',c,'Variable:',v
                        print curData[i,c,v,:]
                    if (np.min(curData[i,c,v,:],axis=0) > 0):
                        meanPop = np.mean(curData[i,c,v,:])
                        stdPop = np.std(curData[i,c,v,:])
                        CV.append((stdPop/meanPop))
                        
                if len(CV) == 0:
                    print 'Channel:',c,'Variable:',v
                    CV.append(0)           

                allCV[curD,c,v] = np.mean(CV)
                #print 'Coeff of Var Parts',np.shape(curData)[3],'for channel',c,'variable',v,':',allCV[curD,c,v]

                if (curD+c+v == 0):
                    allRawCV = CV
                    indices[curD,c,v,0] = 0
                    indices[curD,c,v,1] = indices[curD,c,v,0]+len(CV)
                elif (c+v ==0):
                    allRawCV = np.concatenate((allRawCV,CV))
                    indices[curD,c,v,0] = indices[curD-1,-1,-1,1]
                    indices[curD,c,v,1] = indices[curD,c,v,0]+len(CV)
                elif (v == 0):
                    allRawCV = np.concatenate((allRawCV,CV))
                    indices[curD,c,v,0] = indices[curD,c-1,-1,1]
                    indices[curD,c,v,1] = indices[curD,c,v,0]+len(CV)
                else:
                    allRawCV = np.concatenate((allRawCV,CV))
                    indices[curD,c,v,0] = indices[curD,c,v-1,1]
                    indices[curD,c,v,1] = indices[curD,c,v,0]+len(CV)
        
            #pdb.set_trace()
    #featsV = ['Abs-Lower', 'Abs-Delta', 'Abs-Theta', 'Abs-Alpha', 'Abs-Mu', 'Abs-Beta', 'Abs-Gamma']
    featsV = ['Rel-Lower', 'Rel-Delta', 'Rel-Theta', 'Rel-Alpha', 'Rel-Mu', 'Rel-Beta', 'Rel-Gamma', 
                'Abs-Lower', 'Abs-Delta', 'Abs-Theta', 'Abs-Alpha', 'Abs-Mu', 'Abs-Beta', 'Abs-Gamma', 
                'Spectral-Entropy', 'Spectral-Entropy-Norm', 
                'Entropy', 'Entropy-Norm', 'Curve-Length', 'Energy', 'Sixth-Power', 'LZC', 
                'Max', 'Var']
    #featsV = ['Mobility','Complexity']
    for v in range(np.shape(curData)[2]):
        #pdb.set_trace()
        #ax = fig.add_subplot(2,2,curD+1)
        #fig,ax = resultBoxPlot(fig,allCV[curD],ax=ax,exclude=exclude)
        ax = fig.add_subplot(4,6,v+1)#fig.add_subplot(1,2,v+1)
        fig,ax = resultBoxPlot(fig,np.transpose(allCV[:,:,v]),ax=ax,exclude=exclude)
        ax.set_xticklabels([ '0.125', '0.25', '0.5', '1', '2'])
        ax.set_xlabel('Epoch time (minutes)')
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
        if v%6 == 0:
            ax.set_ylabel('COV')
        #ax.set_ylim(ymax=0.3)
        ax.set_ylim(ymin=0)

        if v < 18:
            ax.get_xaxis().set_visible(False)
        #if v%6 != 0:
        #    ax.get_yaxis().set_visible(False)

        #ax.set_title('COV of Features on %d parts, each %0.2f minutes (n=%d)'%(np.shape(curData)[3],16./np.shape(curData)[3],subjNum))
        ax.set_title('%s'%(featsV[v]))

        fig.subplots_adjust(left=0.05,bottom=0.10,right=0.99,top=0.90,wspace=0.25,hspace=0.2)

    sigDiff = np.zeros((numData/2-1,np.shape(curData)[2]))
    for d in range(0,numData/2-1):
        for v in range(np.shape(curData)[2]):
            sigP = 0
            catDat1 = allRawCV[int(indices[d,0,v,0]):int(indices[d,0,v,1])]
            catDat2 = allRawCV[int(indices[d+1,0,v,0]):int(indices[d+1,0,v,1])]
            for c in range(1,np.shape(curData)[1]):
                catDat1 = np.concatenate((catDat1,allRawCV[int(indices[d,c,v,0]):int(indices[d,c,v,1])]))
                catDat2 = np.concatenate((catDat2,allRawCV[int(indices[d+1,c,v,0]):int(indices[d+1,c,v,1])]))

                #if 0.05 < stats.kruskal(allRawCV[int(indices[d,c,v,0]):int(indices[d,c,v,1])],allRawCV[int(indices[d+1,c,v,0]):int(indices[d+1,c,v,1])])[1]:
                #    sigP += 1
            catDat1 = np.ravel(catDat1)
            catDat2 = np.ravel(catDat2) 
            #print 'Shape:', np.shape(catDat1), np.shape(catDat2)
            try:
                if 0.05 < stats.kruskal(catDat1,catDat2)[1]:
                    sigP += 1
                    #pdb.set_trace()
                sigDiff[d,v] = sigP
            except:
                print "Tried"
    print 'Raw CV:'
    print sigDiff

    sigDiff = np.zeros((numData/2-1,np.shape(curData)[2]))
    for d in range(0,numData/2-1):
        for v in range(np.shape(curData)[2]):
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
    pl.show()
    return allCV,subjNumT

def plotCOV2D(allCV,subjNumT,exclude=1):
    print np.shape(allCV)

    fig = pl.figure(figsize=(13.5, 9))
    fig.suptitle('COV across epoch lengths (n=%d)'%(subjNumT))
    #pdb.set_trace()
    featsV = ['Rel-Lower', 'Rel-Delta', 'Rel-Theta', 'Rel-Alpha', 'Rel-Mu', 'Rel-Beta', 'Rel-Gamma', 
                'Abs-Lower', 'Abs-Delta', 'Abs-Theta', 'Abs-Alpha', 'Abs-Mu', 'Abs-Beta', 'Abs-Gamma', 
                'Spectral-Entropy', 'Spectral-Entropy-Norm', 
                'Entropy', 'Entropy-Norm', 'Curve-Length', 'Energy', 'Sixth-Power', 'LZC', 
                'Max', 'Var']
    #featsV = ['Mobility','Complexity']
    for v in range(np.shape(allCV)[2]):
        #pdb.set_trace()
        #ax = fig.add_subplot(2,2,curD+1)
        #fig,ax = resultBoxPlot(fig,allCV[curD],ax=ax,exclude=exclude)
        ax = fig.add_subplot(4,6,v+1)#fig.add_subplot(1,2,v+1)
        fig,ax = resultBoxPlot(fig,np.transpose(allCV[:,:,v]),ax=ax,exclude=exclude)
        ax.set_xticklabels([ '0.125', '0.25', '0.5', '1', '2'])
        ax.set_xlabel('Epoch time (minutes)')
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
        if v%6 == 0:
            ax.set_ylabel('COV')
        #ax.set_ylim(ymax=0.3)
        ax.set_ylim(ymin=0)

        if v < 18:
            ax.get_xaxis().set_visible(False)
        #if v%6 != 0:
        #    ax.get_yaxis().set_visible(False)

        #ax.set_title('COV of Features on %d parts, each %0.2f minutes (n=%d)'%(np.shape(curData)[3],16./np.shape(curData)[3],subjNum))
        ax.set_title('%s'%(featsV[v]))

        fig.subplots_adjust(left=0.05,bottom=0.10,right=0.99,top=0.90,wspace=0.25,hspace=0.2)

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
    pl.show()


def featAnaly1D(data1, labels1, data2, labels2):
    print('Parts: ',np.shape(data1)[1],np.shape(data1))
    print('Parts: ',np.shape(data2)[1],np.shape(data2))

    if len(labels1) < len(labels2):
        subjNum = len(labels1)
        smallLabel = labels1
        smallData = data1
        bigLabel = labels2
        bigData = data2
        smallParts = np.shape(data1)[-1]
        bigParts = np.shape(data2)[-1]
    else:
        subjNum = len(labels2)
        smallLabel = labels2
        smallData = data2
        bigLabel = labels1
        bigData = data1
        smallParts = np.shape(data2)[-1]
        bigParts = np.shape(data1)[-1]

    allSigP = np.zeros((np.shape(smallData)[1],1))
    allCVSmall = np.zeros((np.shape(smallData)[1],1))
    allCVBig = np.zeros((np.shape(smallData)[1],1))

    for v in range(np.shape(smallData)[1]):
        sigP = 0
        CVsmall = []
        CVbig = []
        multiSession = 0
        #pdb.set_trace()

        for i in range(subjNum):
            sessionNameCur = smallLabel[i][0]
            ind = np.where(bigLabel[:,0] == sessionNameCur)

            if len(ind[0])>0:
                multiSession += 1
                j = ind[0][0]
                #print(labels[dataCaptured[0]][0]+','+labels[dataCaptured[1]][0])
                meanP = stats.kruskal(smallData[i,v,:],bigData[j,v,:])[1]#stats.kruskal(*data[dataCaptured,c,:,v])[1]
                #print 'Variable',v,'p-value on',len(dataCaptured),'subjects:',meanP
                if meanP > 0.05:#11.34:#0.01:
                    sigP += 1
                
                combData = np.concatenate((smallData[i,v,:],bigData[j,v,:]),axis=0)

                if (np.min(combData,axis=0) > 0):
                    meanPop1 = np.mean(smallData[i,v,:])
                    stdPop1 = np.std(smallData[i,v,:])
                    CVsmall.append((stdPop1/meanPop1)*100.)
                    meanPop2 = np.mean(bigData[j,v,:])
                    stdPop2 = np.std(bigData[j,v,:])
                    CVbig.append((stdPop2/meanPop2)*100.)

                #print 'Number of significant P-values',meanP
                #print 'Coeff of Var for channel',c,'variable',v,':',CV[-1],'\n'
                            #print 'Coeff of Var for channel',c,'variable',v,':',CV[-1],'\n'
        if len(CVsmall) == 0:
            CVsmall.append(0)
        if len(CVbig) == 0:
            CVbig.append(0)

        allSigP[v] = sigP
        allCVSmall[v] = np.mean(CVsmall)
        allCVBig[v] = np.mean(CVbig)

        print 'Number of significant P-values',sigP,'of',multiSession
        print 'Coeff of Var Parts ',smallParts,' variable',v,':',np.mean(CVsmall)
        print 'Coeff of Var Parts ',bigParts,' variable',v,':',np.mean(CVbig),'\n'


    return allSigP,allCVSmall,allCVBig


def featAnaly2D(data1, labels1, data2, labels2,testNormal=False,exclude=0):

    print('Parts: ',np.shape(data1)[3],np.shape(data1))
    print('Parts: ',np.shape(data2)[3],np.shape(data2))
    if len(labels1) < len(labels2):
        subjNum = len(labels1)
        smallLabel = labels1
        smallData = data1
        bigLabel = labels2
        bigData = data2
        smallParts = np.shape(data1)[-1]
        bigParts = np.shape(data2)[-1]
    else:
        subjNum = len(labels2)
        smallLabel = labels2
        smallData = data2
        bigLabel = labels1
        bigData = data1
        smallParts = np.shape(data2)[-1]
        bigParts = np.shape(data1)[-1]
    
    allSigP = np.zeros((np.shape(smallData)[1],np.shape(smallData)[2]))
    allSigPbig = np.zeros((np.shape(bigData)[1],np.shape(bigData)[2]))
    allSigPsmall = np.zeros((np.shape(smallData)[1],np.shape(smallData)[2]))
    allCVSmall = np.zeros((np.shape(smallData)[1],np.shape(smallData)[2]))
    allCVBig = np.zeros((np.shape(smallData)[1],np.shape(smallData)[2]))
    
    for c in range(np.shape(smallData)[1]):
        for v in range(np.shape(smallData)[2]):
            sigP = 0
            if testNormal:
                sigPs = 0
                sigPb = 0
            CVsmall = []
            CVbig = []
            multiSession = 0
            #pdb.set_trace()

            for i in range(subjNum):
                sessionNameCur = smallLabel[i][0]
                ind = np.where(bigLabel[:,0] == sessionNameCur)

                if len(ind[0])>0:
                    multiSession += 1
                    j = ind[0][0]
                    #print(labels[dataCaptured[0]][0]+','+labels[dataCaptured[1]][0])

                    meanP = stats.kruskal(smallData[i,c,v,:],bigData[j,c,v,:])[1]#stats.kruskal(*data[dataCaptured,c,:,v])[1]
                    if meanP > 0.05: #for K-W >0.05 #11.34:#0.01:
                        sigP += 1
                    #c,v=0,19
                    #pdb.set_trace()

                    #print "Small Shape:",np.shape(smallData[i,c,v,:])," Big Shape:",np.shape(bigData[i,c,v,:])
                    if testNormal:
                        meanPs = stats.shapiro(smallData[i,c,v,:])[1]
                        meanPb = stats.shapiro(bigData[j,c,v,:])[1]
                        #print 'Variable',v,'p-value on',len(dataCaptured),'subjects:',meanP
                        if (meanPs < 0.05): #for K-W >0.05 #11.34:#0.01:
                            sigPs += 1
                        if (meanPb < 0.05): #for K-W >0.05 #11.34:#0.01:
                            sigPb += 1

                    combData = np.concatenate((smallData[i,c,v,:],bigData[j,c,v,:]),axis=0)

                    if (np.min(combData,axis=0) > 0):
                        meanPop1 = np.mean(smallData[i,c,v,:])
                        stdPop1 = np.std(smallData[i,c,v,:])
                        CVsmall.append((stdPop1/meanPop1)*100.)
                        meanPop2 = np.mean(bigData[j,c,v,:])
                        stdPop2 = np.std(bigData[j,c,v,:])
                        CVbig.append((stdPop2/meanPop2)*100.)
                    
                    #print 'Number of significant P-values',meanP
                    #print 'Coeff of Var for channel',c,'variable',v,':',CV[-1],'\n'
            if len(CVsmall) == 0:
                CVsmall.append(0)
            if len(CVbig) == 0:
                CVbig.append(0)            

            allSigP[c,v] = sigP
            allCVSmall[c,v] = np.mean(CVsmall)
            allCVBig[c,v] = np.mean(CVbig)
            if testNormal:
                allSigPbig[c,v] = sigPb
                allSigPsmall[c,v] = sigPs

            print 'Number of significant P-values',sigP,'of',multiSession
            if testNormal:
                print 'Number of significant P-values',sigPs,'of',multiSession,' Parts ', smallParts
                print 'Number of significant P-values',sigPb,'of',multiSession,' Parts ', bigParts

            print 'Coeff of Var Parts ',smallParts,' for channel',c,'variable',v,':',np.mean(CVsmall)
            print 'Coeff of Var Parts ',bigParts,' for channel',c,'variable',v,':',np.mean(CVbig),'\n'

    # Create a figure instance
    fig = pl.figure(figsize=(9, 6))
    resultBoxPlot(fig,allSigP,totalInst=multiSession,exclude=exclude)
    fig.suptitle('Kruskal-Wallis Results of Stationarity Variability of Features, %d,%d parts (n=%d)'%(smallParts,bigParts,multiSession))
    fig.subplots_adjust(left=0.09,bottom=0.33,right=0.96,top=0.93,wspace=0.2,hspace=0.2)
    #fig.savefig('images/KWstat-%d-%d.eps'%(smallParts,bigParts), format='eps', dpi=1000)
    
    if testNormal:
        fig = pl.figure(figsize=(9, 6))
        resultBoxPlot(fig,allSigPbig,totalInst=multiSession)
        fig.suptitle('Shapiro Results of Stationarity Variability of Features, %d parts (n=%d)'%(bigParts,multiSession))
        fig.subplots_adjust(left=0.09,bottom=0.33,right=0.96,top=0.93,wspace=0.2,hspace=0.2)
        fig.savefig('images/shapiro%d.eps'%(bigParts), format='eps', dpi=1000)

        fig = pl.figure(figsize=(9, 6))
        resultBoxPlot(fig,allSigPsmall,totalInst=multiSession)
        fig.suptitle('Shapiro Results of Stationarity Variability of Features, %d parts (n=%d)'%(smallParts,multiSession))
        fig.subplots_adjust(left=0.09,bottom=0.33,right=0.96,top=0.93,wspace=0.2,hspace=0.2)
        fig.savefig('images/shapiro%d.eps'%(smallParts), format='eps', dpi=1000)
    pl.show()

    return allSigP,allCVSmall,allCVBig

def distanceMetric1D(data1, labels1, data2, labels2):
    print('Parts: ',np.shape(data1)[1],np.shape(data1))
    print('Parts: ',np.shape(data2)[1],np.shape(data2))
    littleParti = np.shape(data1)[-1]
    largeParti = np.shape(data2)[-1] 
    everyElem = [x for x in range(largeParti) if x%(largeParti/littleParti) == 0]

    if len(labels1) < len(labels2):
        subjNum = len(labels1)
        smallLabel = labels1
        smallData = data1
        bigLabel = labels2
        bigData = data2
        smallParts = np.shape(data1)[-1]
        bigParts = np.shape(data2)[-1]
        smallerLength = 0
    else:
        subjNum = len(labels2)
        smallLabel = labels2
        smallData = data2
        bigLabel = labels1
        bigData = data1
        smallParts = np.shape(data2)[-1]
        bigParts = np.shape(data1)[-1]
        smallerLength = 1


    allL2 = np.zeros((np.shape(smallData)[1],1))
    allCos = np.zeros((np.shape(bigData)[1],1))
    allMAE = np.zeros((np.shape(smallData)[1],1))

    for v in range(np.shape(smallData)[1]):
        #smallData[:,v,:] = np.divide(np.subtract(smallData[:,v,:],np.mean(smallData[:,v,:],axis=0)),np.multiply(np.std(smallData[:,v,:],axis=0),3))
        #bigData[:,v,:] = np.divide(np.subtract(bigData[:,v,:],np.mean(bigData[:,v,:],axis=0)),np.multiply(np.std(bigData[:,v,:],axis=0),3))
        
        L2d = []
        cosD = []
        MAE = []
        multiSession = 0
        #pdb.set_trace()

        for i in range(subjNum):
            sessionNameCur = smallLabel[i][0]
            ind = np.where(bigLabel[:,0] == sessionNameCur)

            if len(ind[0])>0:
                multiSession += 1
                j = ind[0][0]
                #print(labels[dataCaptured[0]][0]+','+labels[dataCaptured[1]][0])
                if smallerLength == 0:
                    L2d.append(np.linalg.norm(np.subtract(smallData[i,v,:],bigData[j,v,everyElem]))) #stats.kruskal(*data[dataCaptured,c,:,v])[1]
                    cosD.append(spatial.distance.cosine(smallData[i,v,:],bigData[j,veveryElem]))
                    MAE.append(mean_absolute_error(smallData[i,v,:],bigData[j,v,everyElem]))
                elif smallerLength == 1:
                    L2d.append(np.linalg.norm(np.subtract(smallData[i,v,everyElem],bigData[j,v,:]))) #stats.kruskal(*data[dataCaptured,c,:,v])[1]
                    cosD.append(spatial.distance.cosine(smallData[i,v,everyElem],bigData[j,v,:]))
                    MAE.append(mean_absolute_error(smallData[i,v,everyElem],bigData[j,v,:]))

                meanL2 = np.mean(L2d)
                meanCos = np.mean(cosD)
                meanMAE = np.mean(MAE)


        print 'L2 Parts ',smallParts,',',bigParts,' variable',v,':',meanL2
        print 'Cosine Parts ',smallParts,',',bigParts,' variable',v,':',meanCos
        print 'MAE Parts ',smallParts,',',bigParts,' variable',v,':',meanMAE,'\n'

        allL2[v] = meanL2
        allCos[v] = meanCos
        allMAE[v] = meanMAE

    return allL2,allCos,allMAE

def distanceMetric2D(data1, labels1, data2, labels2):
    print('Parts: ',np.shape(data1)[3],np.shape(data1))
    print('Parts: ',np.shape(data2)[3],np.shape(data2))
    littleParti = np.shape(data1)[-1]
    largeParti = np.shape(data2)[-1] 
    everyElem = [x for x in range(largeParti) if x%(largeParti/littleParti) == 0]

    if len(labels1) < len(labels2):
        subjNum = len(labels1)
        smallLabel = labels1
        smallData = data1
        bigLabel = labels2
        bigData = data2
        smallParts = np.shape(data1)[-1]
        bigParts = np.shape(data2)[-1]
        smallerLength = 0
    else:
        subjNum = len(labels2)
        smallLabel = labels2
        smallData = data2
        bigLabel = labels1
        bigData = data1
        smallParts = np.shape(data2)[-1]
        bigParts = np.shape(data1)[-1]
        smallerLength = 1

    allL2 = np.zeros((np.shape(smallData)[1],np.shape(smallData)[2]))
    allCos = np.zeros((np.shape(bigData)[1],np.shape(bigData)[2]))
    allMAE = np.zeros((np.shape(smallData)[1],np.shape(smallData)[2]))

    for c in range(np.shape(smallData)[1]):
        for v in range(np.shape(smallData)[2]):
            #smallData[:,c,v,:] = np.divide(np.subtract(smallData[:,c,v,:],np.mean(smallData[:,c,v,:],axis=0)),np.multiply(np.std(smallData[:,c,v,:],axis=0),3))
            #bigData[:,c,v,:] = np.divide(np.subtract(bigData[:,c,v,:],np.mean(bigData[:,c,v,:],axis=0)),np.multiply(np.std(bigData[:,c,v,:],axis=0),3))
            #zcaSmall = zca_whitening_matrix(smallData[:,c,v,:])
            #smallData[:,c,v,:] = np.dot(smallData[:,c,v,:],zcaSmall)
            #zcaBig = zca_whitening_matrix(bigData[:,c,v,:])
            #bigData[:,c,v,:] = np.dot(bigData[:,c,v,:],zcaBig)

            #pdb.set_trace()
            L2d = []
            cosD = []
            MAE = []
            multiSession = 0
            #pdb.set_trace()

            for i in range(subjNum):
                sessionNameCur = smallLabel[i][0]
                ind = np.where(bigLabel[:,0] == sessionNameCur)

                if len(ind[0])>0:
                    multiSession += 1
                    j = ind[0][0]
                    #print(labels[dataCaptured[0]][0]+','+labels[dataCaptured[1]][0])
                    if smallerLength == 0:
                        L2d.append(np.linalg.norm(np.subtract(smallData[i,c,v,:],bigData[j,c,v,everyElem]))) #stats.kruskal(*data[dataCaptured,c,:,v])[1]
                        cosD.append(spatial.distance.cosine(smallData[i,:,c,v,:],bigData[j,c,v,everyElem]))
                        MAE.append(mean_absolute_error(smallData[i,c,v,:],bigData[j,c,v,everyElem]))
                    elif smallerLength == 1:
                        L2d.append(np.linalg.norm(np.subtract(smallData[i,c,v,everyElem],bigData[j,c,v,:]))) #stats.kruskal(*data[dataCaptured,c,:,v])[1]
                        cosD.append(spatial.distance.cosine(smallData[i,c,v,everyElem],bigData[j,c,v,:]))
                        MAE.append(mean_absolute_error(smallData[i,c,v,everyElem],bigData[j,c,v,:]))
                    #print 'Variable',v,'p-value on',len(dataCaptured),'subjects:',meanP
                    
                    if smallerLength == 0:
                        allDataCur = np.concatenate((smallData[i,c,v,:],bigData[j,c,v,everyElem]),axis=0)
                    elif smallerLength == 1:
                        allDataCur = np.concatenate((smallData[i,c,v,everyElem],bigData[j,c,v,:]),axis=0)
                    
                    curRange = np.max(allDataCur)-np.min(allDataCur)
                    L2d[-1] = np.divide(L2d[-1],curRange)
                    cosD[-1] = np.divide(cosD[-1],curRange)
                    MAE[-1] = np.divide(MAE[-1],curRange)


            meanL2 = np.mean(L2d)
            meanCos = np.mean(cosD)
            meanMAE = np.mean(MAE)

            #print 'Number of significant P-values',meanP
            #print 'Coeff of Var for channel',c,'variable',v,':',CV[-1],'\n'
            
            #if smallerLength == 0:
            #    allDataCur = np.concatenate((smallData[:,c,v,:],bigData[:,c,v,everyElem]),axis=0)
            #elif smallerLength == 1:
            #    allDataCur = np.concatenate((smallData[:,c,v,everyElem],bigData[:,c,v,:]),axis=0)
            
            #curSTD = np.std(np.std(allDataCur,axis=0),axis=0)
            #meanL2 = np.divide(meanL2,curSTD)
            #meanCos = np.divide(meanCos,curSTD)
            #meanMAE = np.divide(meanMAE,curSTD)

            #curRange = np.max(np.max(allDataCur,axis=0),axis=0)-np.min(np.min(allDataCur,axis=0),axis=0)
            #meanL2 = np.divide(meanL2,curRange)
            #meanCos = np.divide(meanCos,curRange)
            #meanMAE = np.divide(meanMAE,curRange)


            print 'L2 Parts ',smallParts,',',bigParts,' for channel',c,' variable',v,':',meanL2
            print 'Cosine Parts ',smallParts,',',bigParts,' for channel',c,' variable',v,':',meanCos
            print 'MAE Parts ',smallParts,',',bigParts,' for channel',c,' variable',v,':',meanMAE,'\n'

            allL2[c,v] = meanL2
            allCos[c,v] = meanCos
            allMAE[c,v] = meanMAE

    print "Pre-Range Transform"
    fig = pl.figure(figsize=(9, 6))
    fig,ax = resultBoxPlot(fig,allL2)
    fig.suptitle('L2 Variability of Features, %d,%d parts (n=%d)'%(smallParts,bigParts,multiSession))
    ax.set_ylabel('L2 Distance')
    fig.subplots_adjust(left=0.09,bottom=0.33,right=0.96,top=0.93,wspace=0.2,hspace=0.2)
    fig.savefig('images/L2-%d-%d.eps'%(smallParts,bigParts), format='eps', dpi=1000)

    fig = pl.figure(figsize=(9, 6))
    fig,ax = resultBoxPlot(fig,allCos)
    fig.suptitle('COS Variability of Features, %d,%d parts (n=%d)'%(smallParts,bigParts,multiSession))
    ax.set_ylabel('COS Distance')
    fig.subplots_adjust(left=0.09,bottom=0.33,right=0.96,top=0.93,wspace=0.2,hspace=0.2)
    fig.savefig('images/Cos-%d-%d.eps'%(smallParts,bigParts), format='eps', dpi=1000)

    fig = pl.figure(figsize=(9, 6))
    fig,ax = resultBoxPlot(fig,allMAE)
    fig.suptitle('MAE Variability of Features, %d,%d parts (n=%d)'%(smallParts,bigParts,multiSession))
    ax.set_ylabel('MAE Distance')
    fig.subplots_adjust(left=0.09,bottom=0.33,right=0.96,top=0.93,wspace=0.2,hspace=0.2)
    fig.savefig('images/MAE-%d-%d.eps'%(smallParts,bigParts), format='eps', dpi=1000)

    pl.show()

    return allL2,allCos,allMAE

def makeSamplePlot(allSigP,multiSession):

    fig = pl.figure(figsize=(10, 13))
    fig.suptitle('Example Kruskal-Wallis results for stationarity across varying epoch comparisons (n=%d)'%(multiSession))

    
    featsV = ['Highly Stationary Feature: Nonlinear-Energy','Somewhat Stationary Feature: LZC','Non-Stationary Feature: Spectral-Entropy']
    #pdb.set_trace()
    for v in range(len(featsV)):
        #pdb.set_trace()
        #ax = fig.add_subplot(2,2,curD+1)
        #fig,ax = resultBoxPlot(fig,allCV[curD],ax=ax,exclude=exclude)
        ax = fig.add_subplot(3,1,v+1)
        fig,ax = resultBoxPlot(fig,np.transpose(allSigP[v,:,:]),totalInst=multiSession,ax=ax)
        
        ax.set_ylim(ymin=0,ymax=100)
        ax.set_xticklabels(['2 v. 1','2 v. 0.5','2 v. 0.25','2 v. 0.125','1 v. 0.5','1 v. 0.25','1 v. 0.125','0.5 v. 0.25','0.5 v. 0.125','0.25 v. 0.125'])
        #ax.set_xticklabels(['2 v. 1','2 v. 1/2','2 v. 1/4','2 v. 1/8','1 v. 1/2','1 v. 1/4','1 v. 1/8','1/2 v. 1/4','1/2 v. 1/8','1/4 v. 1/8'])
        #ax.set_xticklabels(['8 v. 16','8 v. 32','8 v. 64','8 v. 128','16 v. 32','16 v. 64','16 v. 128','32 v. 64','32 v. 128','64 v. 128'])
        #ax.set_xticklabels(['0.125','0.250','0.375','0.500','0.750','0.875','1.00','1.500','1.750','1.875'])
        ax.set_xlabel('Epoch times compared (minutes)')

        ax.set_ylabel('%s consistent'%('%'))
        #ax.set_ylim(ymax=0.3)

        if v < 2:
            ax.get_xaxis().set_visible(False)
        if v%1 != 0:
            ax.get_yaxis().set_visible(False)

        #ax.set_title('COV of Features on %d parts, each %0.2f minutes (n=%d)'%(np.shape(curData)[3],16./np.shape(curData)[3],subjNum))
        ax.set_title('%s'%(featsV[v]))

    fig.subplots_adjust(left=0.06,bottom=0.10,right=0.96,top=0.90,wspace=0.2,hspace=0.2)
    fig.savefig('images/KWstationarityExamples.eps', format='eps', dpi=1000)
    fig.savefig('images/KWstationarityExamples.png', format='png')

    #pl.show()

if __name__ == '__main__':
    start = time.time()
    analy2D = 0
    analy1D = 0
    kw = 0
    dist = 0
    
    parKW = 0
    cov2D = 0

    sampleParKW = 0
    sampleParKWAll = 1
    
    parts1 = 8
    parts2 = 16
    
    Features,Partitions,TimeMin,Threads,Write2File,FeatsNames,InputFileName = defineParams(parts1)
    
    if parts2 < parts1:
        parts1,parts2 = parts2,parts1
    
    if analy2D == 1:
        allData1,allLabels1 = dataLoad4D(InputFileName,Features,parts1,TimeMin,Threads,Write2File,FeatsNames,)
        allData2,allLabels2 = dataLoad4D(InputFileName,Features,parts2,TimeMin,Threads,Write2File,FeatsNames,)
        allData1,allLabels1 = getNormData(allData1,allLabels1)
        allData2,allLabels2 = getNormData(allData2,allLabels2)
        
        #pdb.set_trace()

        if kw == 1:
            allSigP,allCVSmall,allCVBig = featAnaly2D(allData1, allLabels1, allData2, allLabels2, exclude = 1)
        if dist == 1:
            allL2,allCos,allMAE = distanceMetric2D(allData1, allLabels1, allData2, allLabels2)

    if analy1D == 1:
        allData1,allLabels1 = dataLoad3D(InputFileName,Features,parts1,TimeMin,Threads,Write2File,FeatsNames)
        allData2,allLabels2 = dataLoad3D(InputFileName,Features,parts2,TimeMin,Threads,Write2File,FeatsNames)
        allData1,allLabels1 = getNormData(allData1,allLabels1)
        allData2,allLabels2 = getNormData(allData2,allLabels2)

        #pdb.set_trace()

        if kw == 1:
            allSigP,allCVSmall,allCVBig = featAnaly1D(allData1, allLabels1, allData2, allLabels2)
        if dist ==1:
            allL2,allCos,allMAE = distanceMetric1D(allData1, allLabels1, allData2, allLabels2)

    if cov2D == 1:
        allCV = np.load('allCOVdataAll.npy')
        plotCOV2D(allCV,4313,exclude=1)

        parts1 = 128
        parts2 = 64
        parts3 = 32
        parts4 = 16
        parts5 = 8
        variableMask = np.concatenate((range(0,14),range(18,25),range(26,27),range(28,29),range(31,32)),axis=0)#range(0,30)#np.concatenate((range(0,7),range(7,14)),axis=0)#(range(0,30),range(31,35)),axis=0)
        #variableMask = range(2)

        allData1,allLabels1 = dataLoad4D(InputFileName,Features,parts1,TimeMin,Threads,Write2File,FeatsNames)
        allData2,allLabels2 = dataLoad4D(InputFileName,Features,parts2,TimeMin,Threads,Write2File,FeatsNames)
        allData3,allLabels3 = dataLoad4D(InputFileName,Features,parts3,TimeMin,Threads,Write2File,FeatsNames)
        allData4,allLabels4 = dataLoad4D(InputFileName,Features,parts4,TimeMin,Threads,Write2File,FeatsNames)
        allData5,allLabels5 = dataLoad4D(InputFileName,Features,parts5,TimeMin,Threads,Write2File,FeatsNames)
        
        allData1,allLabels1 = getNormData(allData1,allLabels1)
        allData2,allLabels2 = getNormData(allData2,allLabels2)
        allData3,allLabels3 = getNormData(allData3,allLabels3)
        allData4,allLabels4 = getNormData(allData4,allLabels4)
        allData5,allLabels5 = getNormData(allData5,allLabels5)

        allData1 = allData1[:,:,variableMask,:]
        allData2 = allData2[:,:,variableMask,:]
        allData3 = allData3[:,:,variableMask,:]
        allData4 = allData4[:,:,variableMask,:]
        allData5 = allData5[:,:,variableMask,:]

        allCOVres = covAnaly2D(allData1,allLabels1,allData2,allLabels2,allData3,allLabels3,allData4,allLabels4,allData5,allLabels5,1)
        if Write2File == 1:
            np.save('allCOVdataAll.npy',allCOVres)


    if parKW == 1:

        variableMask = np.concatenate((range(0,30),range(31,35)),axis=0)
        #variableMask = range(2)

        allData1,allLabels1 = dataLoad4D(InputFileName,Features,parts1,TimeMin,Threads,Write2File,FeatsNames,)
        allData2,allLabels2 = dataLoad4D(InputFileName,Features,parts2,TimeMin,Threads,Write2File,FeatsNames,)
        allData1,allLabels1 = getNormData(allData1,allLabels1)
        allData2,allLabels2 = getNormData(allData2,allLabels2)
        
        allData1 = allData1[:,:,variableMask,:]
        allData2 = allData2[:,:,variableMask,:]

        allSigP = genFeatAnaly2D(allData1, allLabels1, allData2, allLabels2, exclude = 1)
    
    if sampleParKW == 1:
        parts1 = 8#128
        parts2 = 16#64
        parts3 = 32
        parts4 = 64#16
        parts5 = 128#8
        variableMask = np.concatenate(([24],[26],[19]),axis=0)#range(0,30)#np.concatenate((range(0,7),range(7,14)),axis=0)#(range(0,30),range(31,35)),axis=0)
        #nonlinear energy, LZC, spectral entropy
        #variableMask = range(2)

        allData1,allLabels1 = dataLoad4D(InputFileName,Features,parts1,TimeMin,Threads,Write2File,FeatsNames)
        allData2,allLabels2 = dataLoad4D(InputFileName,Features,parts2,TimeMin,Threads,Write2File,FeatsNames)
        allData3,allLabels3 = dataLoad4D(InputFileName,Features,parts3,TimeMin,Threads,Write2File,FeatsNames)
        allData4,allLabels4 = dataLoad4D(InputFileName,Features,parts4,TimeMin,Threads,Write2File,FeatsNames)
        allData5,allLabels5 = dataLoad4D(InputFileName,Features,parts5,TimeMin,Threads,Write2File,FeatsNames)
        
        allData1,allLabels1 = getNormData(allData1,allLabels1)
        allData2,allLabels2 = getNormData(allData2,allLabels2)
        allData3,allLabels3 = getNormData(allData3,allLabels3)
        allData4,allLabels4 = getNormData(allData4,allLabels4)
        allData5,allLabels5 = getNormData(allData5,allLabels5)

        allData1 = allData1[:,:,variableMask,:]
        allData2 = allData2[:,:,variableMask,:]
        allData3 = allData3[:,:,variableMask,:]
        allData4 = allData4[:,:,variableMask,:]
        allData5 = allData5[:,:,variableMask,:]

        allSigP1 = genFeatAnaly2D(allData1, allLabels1, allData2, allLabels2, exclude = 1)
        allSigP2 = genFeatAnaly2D(allData1, allLabels1, allData3, allLabels3, exclude = 1)
        allSigP3 = genFeatAnaly2D(allData1, allLabels1, allData4, allLabels4, exclude = 1)
        allSigP4 = genFeatAnaly2D(allData1, allLabels1, allData5, allLabels5, exclude = 1)
        
        allSigP5 = genFeatAnaly2D(allData2, allLabels2, allData3, allLabels3, exclude = 1)
        allSigP6 = genFeatAnaly2D(allData2, allLabels2, allData4, allLabels4, exclude = 1)
        allSigP7 = genFeatAnaly2D(allData2, allLabels2, allData5, allLabels5, exclude = 1)
        
        allSigP8 = genFeatAnaly2D(allData3, allLabels3, allData4, allLabels4, exclude = 1)
        allSigP9 = genFeatAnaly2D(allData3, allLabels3, allData5, allLabels5, exclude = 1)
        
        allSigP10 = genFeatAnaly2D(allData4, allLabels4, allData5, allLabels5, exclude = 1)

        #feat0 = [allSigP10[:,0],allSigP8[:,0],allSigP9[:,0],allSigP5[:,0],allSigP6[:,0],allSigP7[:,0],allSigP1[:,0],allSigP2[:,0],allSigP3[:,0],allSigP4[:,0]]
        #feat1 = [allSigP10[:,1],allSigP8[:,1],allSigP9[:,1],allSigP5[:,1],allSigP6[:,1],allSigP7[:,1],allSigP1[:,1],allSigP2[:,1],allSigP3[:,1],allSigP4[:,1]]
        #feat2 = [allSigP10[:,2],allSigP8[:,2],allSigP9[:,2],allSigP5[:,2],allSigP6[:,2],allSigP7[:,2],allSigP1[:,2],allSigP2[:,2],allSigP3[:,2],allSigP4[:,2]]
        
        feat0 = [allSigP1[:,0],allSigP2[:,0],allSigP3[:,0],allSigP4[:,0],allSigP5[:,0],allSigP6[:,0],allSigP7[:,0],allSigP8[:,0],allSigP9[:,0],allSigP10[:,0]]
        feat1 = [allSigP1[:,1],allSigP2[:,1],allSigP3[:,1],allSigP4[:,1],allSigP5[:,1],allSigP6[:,1],allSigP7[:,1],allSigP8[:,1],allSigP9[:,1],allSigP10[:,1]]
        feat2 = [allSigP1[:,2],allSigP2[:,2],allSigP3[:,2],allSigP4[:,2],allSigP5[:,2],allSigP6[:,2],allSigP7[:,2],allSigP8[:,2],allSigP9[:,2],allSigP10[:,2]]

        #pdb.set_trace()

        #feat2 = [allSigP1[2],allSigP2[2],allSigP3[2],allSigP4[2],allSigP5[2],allSigP6[2],allSigP7[2],allSigP8[2],allSigP9[2],allSigP10[2]]

        np.save('figureGenFiles/KWstationarityExamples.npy',np.array([feat0,feat1,feat2]))


        makeSamplePlot(np.array([feat0,feat1,feat2]),4313)

    if sampleParKWAll == 1:
        par128 = False

        parts1 = 8#128
        parts2 = 16#64
        parts3 = 32
        parts4 = 64#16
        parts5 = 128#8
        #variableMask = np.concatenate((range(0,14),[15],range(17,30)))#np.concatenate(([24],[26],[19]),axis=0)#range(0,30)#np.concatenate((range(0,7),range(7,14)),axis=0)#(range(0,30),range(31,35)),axis=0)
        variableMaskL1 = np.concatenate((range(0,14),range(18,30),range(31,35)),axis=0)

        variableMask = variableMaskL1[np.concatenate((range(0,14),[15],range(17,30)))]
        #nonlinear energy, LZC, spectral entropy
        #['Rel-Lower', 'Rel-Delta', 'Rel-Theta', 'Rel-Alpha', 'Rel-Mu', 'Rel-Beta', 'Rel-Gamma', 
        #'Abs-Lower', 'Abs-Delta', 'Abs-Theta', 'Abs-Alpha', 'Abs-Mu', 'Abs-Beta', 'Abs-Gamma', 
        #'Fourier-Entropy', 'NormedFourier-Entropy', 'Fourier-Entropy-Norm', 'NormedFourier-Entropy-Norm', 'Spectral-Entropy', 'Spectral-Entropy-Norm', 
        #'Entropy', 'Entropy-Norm', 'Curve-Length', 'Energy', 'Nonlinear-Energy', 'Sixth-Power', 'LZC', 
        #'Min', 'Max', 'Med', 'Mean', 'Var', 'Std', 'Skew', 'Kurtosis', 'Integral', 'Sum']

        #['Rel-Lower', 'Rel-Delta', 'Rel-Theta', 'Rel-Alpha', 'Rel-Mu', 'Rel-Beta', 'Rel-Gamma', 
        #'Abs-Lower', 'Abs-Delta', 'Abs-Theta', 'Abs-Alpha', 'Abs-Mu', 'Abs-Beta', 'Abs-Gamma', 
        #'Spectral-Entropy', 'Spectral-Entropy-Norm', 
        #'Entropy', 'Entropy-Norm', 'Curve-Length', 'Energy', 'Nonlinear-Energy', 'Sixth-Power', 'LZC', 
        #'Min', 'Max', 'Med', 'Var', 'Std', 'Skew', 'Kurtosis']

        #['Rel-Lower', 'Rel-Delta', 'Rel-Theta', 'Rel-Alpha', 'Rel-Mu', 'Rel-Beta', 'Rel-Gamma', 
        #'Abs-Lower', 'Abs-Delta', 'Abs-Theta', 'Abs-Alpha', 'Abs-Mu', 'Abs-Beta', 'Abs-Gamma', 
        #'Spectral-Entropy-Norm', 
        #Entropy-Norm', 'Curve-Length', 'Energy', 'Nonlinear-Energy', 'Sixth-Power', 'LZC', 
        #'Min', 'Max', 'Med', 'Var', 'Std', 'Skew', 'Kurtosis']
        #variableMask = range(2)

        allData1,allLabels1 = dataLoad4D(InputFileName,Features,parts1,TimeMin,Threads,Write2File,FeatsNames)
        allData2,allLabels2 = dataLoad4D(InputFileName,Features,parts2,TimeMin,Threads,Write2File,FeatsNames)
        allData3,allLabels3 = dataLoad4D(InputFileName,Features,parts3,TimeMin,Threads,Write2File,FeatsNames)
        allData4,allLabels4 = dataLoad4D(InputFileName,Features,parts4,TimeMin,Threads,Write2File,FeatsNames)
        allData5,allLabels5 = dataLoad4D(InputFileName,Features,parts5,TimeMin,Threads,Write2File,FeatsNames)
        
        allData1,allLabels1 = getNormData(allData1,allLabels1)
        allData2,allLabels2 = getNormData(allData2,allLabels2)
        allData3,allLabels3 = getNormData(allData3,allLabels3)
        allData4,allLabels4 = getNormData(allData4,allLabels4)
        allData5,allLabels5 = getNormData(allData5,allLabels5)

        allData1 = allData1[:,:,variableMask,:]
        allData2 = allData2[:,:,variableMask,:]
        allData3 = allData3[:,:,variableMask,:]
        allData4 = allData4[:,:,variableMask,:]
        allData5 = allData5[:,:,variableMask,:]

        allSigP1 = genFeatAnaly2D(allData1, allLabels1, allData2, allLabels2, exclude = 1)
        allSigP2 = genFeatAnaly2D(allData1, allLabels1, allData3, allLabels3, exclude = 1)
        allSigP3 = genFeatAnaly2D(allData1, allLabels1, allData4, allLabels4, exclude = 1)
        if par128:
            allSigP4 = genFeatAnaly2D(allData1, allLabels1, allData5, allLabels5, exclude = 1)
        else:
            allSigP4 = parFeatAnaly2D(0, allData1, allLabels1, allData5, allLabels5)[0]
            for c in range(1,19):
                print '4:',c
                allSigP4 = np.concatenate((allSigP4,parFeatAnaly2D(c, allData1, allLabels1, allData5, allLabels5)[0]),axis=0)

        allSigP5 = genFeatAnaly2D(allData2, allLabels2, allData3, allLabels3, exclude = 1)
        allSigP6 = genFeatAnaly2D(allData2, allLabels2, allData4, allLabels4, exclude = 1)
        if par128:
            allSigP7 = genFeatAnaly2D(allData2, allLabels2, allData5, allLabels5, exclude = 1)
        else:
            allSigP7 = parFeatAnaly2D(0, allData2, allLabels2, allData5, allLabels5)[0]
            for c in range(1,19):
                print '7:',c
                allSigP7 = np.concatenate((allSigP7,parFeatAnaly2D(c, allData2, allLabels2, allData5, allLabels5)[0]),axis=0)
        
        allSigP8 = genFeatAnaly2D(allData3, allLabels3, allData4, allLabels4, exclude = 1)
        if par128:
            allSigP9 = genFeatAnaly2D(allData3, allLabels3, allData5, allLabels5, exclude = 1)
        else:
            allSigP9 = parFeatAnaly2D(0, allData3, allLabels3, allData5, allLabels5)[0]
            for c in range(1,19):
                print '9:',c
                allSigP9 = np.concatenate((allSigP9,parFeatAnaly2D(c, allData3, allLabels3, allData5, allLabels5)[0]),axis=0)
        
        if par128:
            allSigP10 = genFeatAnaly2D(allData4, allLabels4, allData5, allLabels5, exclude = 1)
        else:
            allSigP10 = parFeatAnaly2D(0, allData4, allLabels4, allData5, allLabels5)[0]
            for c in range(1,19):
                print '10:',c
                allSigP10 = np.concatenate((allSigP10,parFeatAnaly2D(c, allData4, allLabels4, allData5, allLabels5)[0]),axis=0)
        
        #feat0 = [allSigP10[:,0],allSigP8[:,0],allSigP9[:,0],allSigP5[:,0],allSigP6[:,0],allSigP7[:,0],allSigP1[:,0],allSigP2[:,0],allSigP3[:,0],allSigP4[:,0]]
        #feat1 = [allSigP10[:,1],allSigP8[:,1],allSigP9[:,1],allSigP5[:,1],allSigP6[:,1],allSigP7[:,1],allSigP1[:,1],allSigP2[:,1],allSigP3[:,1],allSigP4[:,1]]
        #feat2 = [allSigP10[:,2],allSigP8[:,2],allSigP9[:,2],allSigP5[:,2],allSigP6[:,2],allSigP7[:,2],allSigP1[:,2],allSigP2[:,2],allSigP3[:,2],allSigP4[:,2]]
        allFeatResults = []
        for f in range(np.shape(allSigP1)[1]):
            curFeat = [allSigP1[:,f],allSigP2[:,f],allSigP3[:,f],allSigP4[:,f],allSigP5[:,f],allSigP6[:,f],allSigP7[:,f],allSigP8[:,f],allSigP9[:,f],allSigP10[:,f]]
            allFeatResults.append(curFeat)

        #pdb.set_trace()

        #feat2 = [allSigP1[2],allSigP2[2],allSigP3[2],allSigP4[2],allSigP5[2],allSigP6[2],allSigP7[2],allSigP8[2],allSigP9[2],allSigP10[2]]

        np.save('figureGenFiles/KWstationarityExamplesAll.npy',np.array(allFeatResults))


        #makeSamplePlot(np.array([feat0,feat1,feat2]),4313)


    end=time.time()
    print '\nTime Elapsed:',end-start,'\n'