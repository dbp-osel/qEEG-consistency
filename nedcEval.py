from nedcTools import scatterEEGdataMATLAB,toNumeric,readEDF,readNotes,reSampleData,preProcessData,reRefdata,iirFiltFilt,getPSD,mainTXT,singleTXTEval,singleTXTEvalCSV
from nedcTools import getMedsListStr,getAgeStr,EEGplots,getEEGdata,getCurveLength,getEnergy,getNonlinearEnergy,getSpectralEntropy,getSixPower,getSecondThirdFeat,getSmallPSD,getSmallPSDparts
from nedcTools import getWaveletTransform,getLZC,getAllFeatures,oneFspectra,getRawEEG
from nedcTools import Subject
import nedcTools
import numpy as np
import sys
from time import time,sleep
import warnings
from scipy.io import savemat
import pywt
import pdb
from multiprocessing import Pool
from functools import partial
import os


warnings.filterwarnings("error")
warnings.simplefilter(action='ignore', category=FutureWarning)

#warnings.filterwarnings(action='ignore')
#import matlab.engine
#eng = matlab.engine.start_matlab('-nojvm')
def defineParameters():
    features = ['raw']#['addTime']
    partitions = 1
    timeMin = 5
    threads = 1
    write2File = 1
    featsNames = ''
    for x in features:
        featsNames = featsNames+x
    return features,partitions,timeMin,threads,write2File,featsNames

def mainPar(content,directory,timeMin,features,partitions,write2File,fileOutput,fileOutputSumm):
    allData = []
    allLabels = []
    allTimes = []

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
    
    EDFErrors = 0
    for txtFiles in content:
        if txtFiles.rstrip().split('.')[3] == 'edf':
            val.fileName = txtFiles.split('_t')[0]+'.txt'
        else:
            val.fileName = txtFiles
        #print val.fileName
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
        
        if val.fileName.rstrip().split('.')[3] == 'edf':
            subjEEG,errorValue=readEDF(txtFiles)
            val.singleTXTdata = [subjEEG]

        numOfEDF = len(val.singleTXTdata)
        for edfFileNum in range(numOfEDF):

            edfFile = val.subjEdfFiles[edfFileNum]
            subjEEG = val.singleTXTdata[edfFileNum]

            i = edfFile.split('_t')[1].split('.')[0]
            try:
                name = val.name
                session = val.session
                dateCur = val.subjDates[edfFileNum]
                #subjEEGdata,eegChannelNames,srate = preProcessEEG(subjEEG) #subjPreProcEEG,subjChannelNames,subjeegChannelNum,
                #refData,srate = preProcessData(subjEEG,timeMin=timeMin,srate=100)
                if 'unproc' in features:
                    refData,srate = getRawEEG(subjEEG,srate=250)
                else:
                    refData,srate = preProcessData(subjEEG,timeMin=timeMin,srate=100)

                #srate = 250#1/(times[1]-times[0])
                #sampledData,srate = reSampleData(refData,srate,100) 
                #print 'After:',np.size(subjEEGdata),'srate:',srate
                if val.subjGender == 'male':
                    genderNP[0] +=1
                elif val.subjGender == 'female':
                    genderNP[1] +=1
                elif (val.subjGender == 'noGender') or (val.subjGender == ''):
                    genderNP[2] +=1

            except MemoryError:
                printString = 'Error Subj# '+str(val.name)+' Session# '+str(val.session)+' EDF# '+str(i)+'; Memory-Error.'
                if write2File == 1:
                    fileOutput.write(printString+os.linesep)
                else:
                    print printString
                EDFErrors += 1
                continue
            except ValueError:
                printString = 'Error Subj# '+str(val.name)+' Session# '+str(val.session)+' EDF# '+str(i)+'; Unpack-Error.'
                if write2File == 1:
                    fileOutput.write(printString+os.linesep)
                else:
                    print printString                
                EDFErrors += 1
                continue
            except Warning:
                printString = 'Error Subj# '+str(val.name)+' Session# '+str(val.session)+' EDF# '+str(i)+'; Channel-Error.'
                if write2File == 1:
                    fileOutput.write(printString+os.linesep)
                else:
                    print printString
                EDFErrors += 1
                continue
            #eeg1.plot(n_channels=19,duration=30,bgcolor='w',color='k',lowpass=50,highpass=1,filtorder=8)
            
            #EEGplots(refData,mode="psd",srate=250)
            #EEGplots(refData,mode="fft",srate=250)
            
            if (np.shape(refData)[1] < (srate*60*(timeMin+1))) and ('unproc' not in features):
                printString = 'Error Subj# '+str(val.name)+' Session# '+str(val.session)+' EDF# '+str(i)+'; Not Enough Data, Length: %0.2f'%(np.shape(refData)[1]/(srate*60.))
                #if write2File == 1:
                #    fileOutput.write(printString+os.linesep)
                #else:
                #    print printString
                continue
            
            #eegDataBuild.EEGplots(refData[:,0:srate*60*timeMin],'single')
            #pdb.set_trace()

            if 'raw' in features:
                if write2File == 1:
                    dataName = str(name)+'_'+'S'+str(session)+'_'+str(i)
                    saveLabel = np.array([dataName,dateCur,val.subjGender,val.age,getMedsListStr(val.subjMed),val.subjNormalState,val.keywords])
                    saveData = refData[:,srate*60:srate*60*(timeMin+1)]
                    toSave = [saveLabel,saveData]
                    np.save('/media/david/Data1/pyEDF_'+str(timeMin)+'/'+dataName+'.npy',toSave)
                else:
                    allData.append(refData[:,srate*60:srate*60*(timeMin+1)])

            elif 'unproc' in features:
                if write2File == 1:
                    dataName = str(name)+'_'+'S'+str(session)+'_'+str(i)
                    saveLabel = np.array([dataName,dateCur,val.subjGender,val.age,getMedsListStr(val.subjMed),val.subjNormalState])
                    saveData = refData#[:,srate*60:srate*60*(timeMin+1)]
                    toSave = [saveLabel,saveData]
                    try:
                        np.save('/media/david/Data1/unProcPyEDF/'+dataName+'.npy',toSave)
                    except SystemError:
                        printString = 'Error Subj# '+str(val.name)+' Session# '+str(val.session)+' EDF# '+str(i)+'; System-Error.'
                        if write2File == 1:
                            fileOutput.write(printString+os.linesep)
                        else:
                            print printString                
                        EDFErrors += 1
                        continue
                else:
                    allData.append(refData[:,srate*60:srate*60*(timeMin+1)])

            else:
                try:
                    if 'oneF' in features:
                        dataFeatures = oneFspectra(refData[:,srate*60:srate*60*(timeMin+1)],srate,partitions)
                    elif '' not in features:
                        refData = refData[:,srate*60:srate*60*(timeMin+1)]
                        bands = [0,1,4,8,12,16,25,40]
                        dataFeatures = getAllFeatures(refData,features,bands,srate,partitions)
                except RuntimeWarning:
                    printString = 'Error Subj# '+str(val.name)+' Session# '+str(val.session)+' EDF# '+str(i)+'; Runtime-Error'
                    if write2File == 1:
                        fileOutput.write(printString+os.linesep)
                    else:
                        print printString
                    EDFErrors += 1
                    continue
                except MemoryError:
                    printString = 'Error Subj# '+str(val.name)+' Session# '+str(val.session)+' EDF# '+str(i)+'; MemoryProcessing-Error'
                    if write2File == 1:
                        fileOutput.write(printString+os.linesep)
                    else:
                        print printString
                    EDFErrors += 1
                    continue
                            
                allData.append(dataFeatures) #[CL,E,NE,SE,SP,mini,maxi,medi,mean,var,std,skew,kurt,inte,sumi]
            
            #pdb.set_trace()

            #print 'PSD:',np.shape(psdOfData)
            dataName = str(name)+'_'+'S'+str(session)+'_'+str(i)
            #np.save('/pyDataEDF/'+dataName+'.npy',refData)
            #curEEGdata,eegTimes = getEEGdata(subjEEG)
            ##savemat(directory+'/matDataEDF/'+dataName,mdict={'EEG': curEEGdata})
            allLabels.append([dataName,dateCur,val.subjGender,val.age,getMedsListStr(val.subjMed),val.subjNormalState])
            allDataShapeStr = ''
            for s in np.shape(allData):
                allDataShapeStr = allDataShapeStr+'['+str(s)+']'
            #if classN != val[15]:
            printString = 'Saved Subj# '+str(val.name)+' Session# '+str(val.session)+' EDF# '+str(i)+' Date '+dateCur.isoformat()+': ['+str(np.shape(refData)[0])+','+str(np.shape(refData)[1])+']'+'; '+allDataShapeStr+'; Gender: '+str(val.subjGender)+'; Age: '+getAgeStr(val.age)+'; Meds: '+getMedsListStr(val.subjMed)+'; Normality Found: '+str(val.subjNormalState)+'; Keywords: '+' '.join(val.keywords)

            if write2File == 1:
                fileOutput.write(printString+os.linesep)
            else:
                print printString
                print 


            #pdb.set_trace()

    labelArray = np.array(allLabels,dtype=object)

    if len(age) == 0:
        ageMean = 0
    else:
        ageMean = np.mean(age)

    if write2File == 1:
        fileOutputSumm.write(os.linesep+'Male Recordings: '+str(male)+'; Female Recordings: '+str(female)+'; Neither: '+str(noGender)+os.linesep)
        fileOutputSumm.write('Male Data Points: '+str(int(genderNP[0][0]))+'; Female Data Points: '+str(int(genderNP[1][0]))+'; Neither Data Points: '+str(int(genderNP[2][0]))+os.linesep)
        fileOutputSumm.write('Ages Captured: '+str(goodAge)+'; No Age: '+str(noAge)+'; Mean Age: '+str(ageMean)+os.linesep)
        fileOutputSumm.write('Medication Found:'+str(medFound)+'; EEG 10-20 Found:'+str(eegSysFound)+'; EDFErrorCount: '+str(valueErrorCount)+'/'+str(noEEGerr)+'; Other EDF Errors: '+str(EDFErrors)+os.linesep)
        fileOutputSumm.write('Keywords Found: '+str(int(keyCount[0][0]))+' '+str(int(keyCount[1][0]))+' '+str(int(keyCount[2][0]))+' '+str(int(keyCount[3][0]))+' '+str(int(keyCount[4][0]))+os.linesep)
        fileOutputSumm.write('Normal EEG Found: '+str(normalEEG)+'; Abnormal EEG Found: '+str(abnormalEEG)+'; Neither: '+str(noImpressionEEG)+os.linesep)
    else:
        print(os.linesep+'Male Recordings: '+str(male)+'; Female Recordings: '+str(female)+'; Neither: '+str(noGender)+os.linesep)
        print('Male Data Points: '+str(int(genderNP[0][0]))+'; Female Data Points: '+str(int(genderNP[1][0]))+'; Neither Data Points: '+str(int(genderNP[2][0]))+os.linesep)
        print('Ages Captured: '+str(goodAge)+'; No Age: '+str(noAge)+'; Mean Age: '+str(ageMean)+os.linesep)
        print('Medication Found:'+str(medFound)+'; EEG 10-20 Found:'+str(eegSysFound)+'; EDFErrorCount: '+str(valueErrorCount)+'/'+str(noEEGerr)+'; Other EDF Errors: '+str(EDFErrors)+os.linesep)
        print('Keywords Found: '+str(int(keyCount[0][0]))+' '+str(int(keyCount[1][0]))+' '+str(int(keyCount[2][0]))+' '+str(int(keyCount[3][0]))+' '+str(int(keyCount[4][0]))+os.linesep)
        print('Normal EEG Found: '+str(normalEEG)+'; Abnormal EEG Found: '+str(abnormalEEG)+'; Neither: '+str(noImpressionEEG)+os.linesep)

    return [allData,labelArray]


def main(directory,txtFile,threads,timeMin,features,partitions,write2File,fileOutput,fileOutputSumm):
    f = open(txtFile,'r')
    content = f.readlines()
    if threads == 1:
        results = mainPar(content,directory,timeMin,features,partitions,write2File,fileOutput,fileOutputSumm)
        totalResultsData = results[0]
        totalResultsLabels = results[1]
    else:
        entries = np.shape(content)[0]
        perThread = entries/threads
        #pdb.set_trace()
        contentList = []
        curStartInd = 0
        for i in range(threads):
            curEndInd = (i+1)*perThread-1
            if (i+1  == threads):
                contentList.append(content[curStartInd:])
                continue

            while (content[curEndInd].split('_')[0] == content[curEndInd+1].split('_')[0]):
                curEndInd += 1
                if curEndInd == len(content)-1:
                    break

            contentList.append(content[curStartInd:curEndInd])
            curStartInd = curEndInd+1

        #pdb.set_trace()

        pool = Pool(processes=threads)
        parFunction = partial(mainPar,directory=directory,timeMin=timeMin,features=features,partitions=partitions,write2File=write2File,fileOutput=fileOutput,fileOutputSumm=fileOutputSumm)
        results = pool.map(parFunction,contentList)
        pool.close()
        pool.join()

        nonZeroResults = []
        for r in range(threads):
            if len(results[r][0])>0:
                nonZeroResults.append(results[r])

        if len(nonZeroResults)>0:
            totalResultsData = nonZeroResults[0][0]
            totalResultsLabels = nonZeroResults[0][1]
            if len(nonZeroResults)>1:
                for l in range(1,len(nonZeroResults)):
                    totalResultsData = np.concatenate((totalResultsData,nonZeroResults[l][0]),axis=0)
                    totalResultsLabels = np.concatenate((totalResultsLabels,nonZeroResults[l][1]),axis=0)
        else:
            totalResultsData = []
            totalResultsLabels = []

    multiSessionSubj = getDiffSessions(totalResultsLabels)
    shapeStr = ''
    for s in np.shape(totalResultsData):
        shapeStr = shapeStr+'['+str(s)+']'
    if write2File == 1:
        fileOutputSumm.write('Number of Multi-Session subjects: '+str(len(multiSessionSubj))+os.linesep)
        fileOutputSumm.write('Data Shape: '+shapeStr+os.linesep)
    else:
        print('Number of Multi-Session subjects: '+str(len(multiSessionSubj))+os.linesep)
        print('Data Shape: '+shapeStr+os.linesep)

    #print 'Label Shape:',np.shape(totalResultsLabels)
    return totalResultsData,totalResultsLabels
    
    #featsNames = ''
    #for x in features:
    #    featsNames = featsNames+x
    #if Write2File == 1:
    #    np.save(featsNames+'_'+str(partitions)+'Parts_'+str(timeMin)+'Min'+'_'+InputFileName+'_Data',totalResultsData)
    #    np.save(featsNames+'_'+str(partitions)+'Parts_'+str(timeMin)+'Min'+'_'+InputFileName+'_Labels',totalResultsLabels)


def getDiffSessions(labels):
    subjCaptured = []
    for i in range(len(labels)-1):
        sessionNameCur = labels[i][0].split('_')
        sessionNameNext = labels[i+1][0].split('_')
        if sessionNameCur[0] in subjCaptured:
            continue

        if (sessionNameCur[0] == sessionNameNext[0]) and (sessionNameCur[1] != sessionNameNext[1]):
            subjCaptured.append(sessionNameCur[0])
    
    return subjCaptured

if __name__ == '__main__':
    start = time()
    features,partitions,timeMin,threads,write2File,featsNames = defineParameters()

    directory='/media/david/Data1/NEDC/tuh_eeg/v1.0.0/' #'/media/david/Data1/EEGCorpus06/' #'/media/david/WD 2TB EXT/EEGcorpus/' #'/media/david/FA324B89324B4A39/'#
    inputFileName = str(sys.argv[1]).split('.')[0]

    if os.path.exists('/home/david/Documents/nedcProcessing/'+featsNames+'_'+str(partitions)+'Parts_'+str(timeMin)+'Min/'+str(sys.argv[1])):
        basePath = '/home/david/Documents/nedcProcessing/'+featsNames+'_'+str(partitions)+'Parts_'+str(timeMin)+'Min/'+featsNames+'_'+str(partitions)+'Parts_'+str(timeMin)+'Min'+'_'+inputFileName
        txtFile = '/home/david/Documents/nedcProcessing/'+featsNames+'_'+str(partitions)+'Parts_'+str(timeMin)+'Min/'+str(sys.argv[1])
    else:
        basePath = '/home/david/Documents/nedcProcessing/'+featsNames+'_'+str(partitions)+'Parts_'+str(timeMin)+'Min'+'_'+inputFileName
        txtFile = str(sys.argv[1])

    if write2File == 1:
        fileOutput = open(basePath+'Out.txt','w')
        fileOutputSumm = open(basePath+'Summ.txt','w')
    else:
        fileOutput = ''
        fileOutputSumm = ''
        
    totalResultsData,totalResultsLabels = main(directory=directory,txtFile=txtFile,threads=threads,timeMin=timeMin,features=features,partitions=partitions,write2File=write2File,fileOutput=fileOutput,fileOutputSumm=fileOutputSumm) #modes: psd,band,time
    
    if write2File == 1:
        np.save(basePath+'_Data',totalResultsData)
        np.save(basePath+'_Labels',totalResultsLabels)

    end = time()
    if write2File == 1:
        fileOutputSumm.write(os.linesep+'Time Elapsed: '+str(end-start)+os.linesep)
    else:
        print(os.linesep+'Time Elapsed: '+str(end-start)+os.linesep)
    #pdb.set_trace()