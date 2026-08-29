import tensorflow.compat.v1 as tf
import numpy as np
import glob
import pickle
import util
import os,sys
from PhaseNetworkClassFlexible import PhaseNetwork

#system="walking_hindlimb"
system="walking_forelimb"
exercise="walking"

loadName="01_24Rats_pose_correct_Rat3toRat40_RT_15_2.-1"
testName="01_24Rats_pose_correct_Rat3toRat40_RT_15_2.-1_test"







if(system=="walking_hindlimb"):
	import setting03 as setting
elif(system=="walking_forelimb"):
	import setting04 as setting

D=setting.sphereCount*2+setting.scalarCount	#*2 because of AZEQ
gpuID=-1
startsess=703
numSess=827#598#860
testingSubjectList=[]
a = np.arange(startsess, numSess)
for i in a:
    testingSubjectList.append('rat'+str(i)+'_')#testingSubjectList.append('rat'+str(i))->before 03/01/2019#try beware
print(testingSubjectList)

prePhaseColorList=[]
b = np.arange(1,numSess)#np.arange(0,92)
for i2 in b:
    prePhaseColorList.append('#1f77b4','#2ca02c')
##prepare training data
regressWing=setting.regressWing
predictionGap=setting.predictionGap	 

windowWing=setting.windowWing
windowSize=1+windowWing*2

######################################

src=system+"\\d1pkl_new_state\\"+exercise+"\\WorldFrame\\"#beware try
dst=system+"\\d2model_new_state\\"+exercise+"\\"+testName+"\\"



sphereCount=setting.sphereCount
cropHead=setting.cropHead
cropTail=setting.cropTail

fileList=[]
for subjectName in testingSubjectList:
	#print("subjectName")
	#print(subjectName)
	fileList+=glob.glob(src+subjectName+'*.pkl', recursive=False)


recordList=[]
#firstVisualIndexList=[]#LP
#pp=1
for aFile in fileList:
	d=pickle.load(open(aFile,'rb'))
	pose=d['data']
	'''
	if(pp==1):
		print("aFile")
		print(aFile)
		print("d['data']")
		print(d['data'])
		pp=2
    '''
	recordList.append(pose[:,cropHead:pose.shape[1]-cropTail])
	'''
	if(pp==1):
		print("d['data']")
		print(d['data'])
		pp=2
	'''
#calculate the rotation matrix for each sphere
allTestData=np.hstack(recordList)

rotationMatrixList=[]

for s in range(setting.sphereCount):	

	selectedData=allTestData[s*3:s*3+3,:]
	tmpMean=np.mean(selectedData,axis=1)
	zAxis=tmpMean/np.linalg.norm(tmpMean)	#after rotation, this point will become z-axis

	pcaResult=util.PCA(selectedData-np.reshape(zAxis,[3,1]))

	firstPC=pcaResult.eigVecs[:,0]

	yAxis=np.cross(zAxis,firstPC)
	yAxis=yAxis/np.linalg.norm(yAxis)	#added later (22/4/2018)

	xAxis=np.cross(yAxis,zAxis)	#after rotation, this point will become x-axis	

	#form a rotation matrix ( modified = R*original )
	R=np.stack([xAxis,yAxis,zAxis], axis=0)	#stack row by row
	rotationMatrixList.append(R)

#project each sphere in training data into 2D
convertedRecordList=[]
for i in range(len(recordList)):
	n,m=recordList[i].shape
	converted=util.convertOriginalPoseByAzeqProjection(rotationMatrixList,recordList[i])
	convertedRecordList.append(converted)

#find a global scaling parameter (for each dimension)
tmp=np.concatenate(convertedRecordList, axis=1)

#mean = np.mean(tmp, axis=1, keepdims=True)#LP cut this line 03/01/2019
#sd = np.std(tmp-mean, axis=1, keepdims=True)#LP cut this line 03/01/2019
#LP add this below section 03/01/2019
d=pickle.load(open(dst+'preprocessParameters.pkl','rb'))#preprocessParameters.pkl#preprocessParameters2.pkl
'''
print(d["meanForPhaseExtraction"])
print(d["sdForPhaseExtraction"])

print(d["wMean"])
print(d["wSD"])
'''
mean = d["meanForPhaseExtraction"]
sd = d["sdForPhaseExtraction"]
#
'''
print("mean0")
print(mean)
print("sd0")
print(sd)
'''
os.makedirs(dst,exist_ok=True)
#pickle.dump(rotationMatrixList,open(dst+"rotationMatrixList.pkl",'wb'))
#mean.dump(dst+"meanForPhaseExtraction.dat")
#sd.dump(dst+"sdForPhaseExraction.dat")
pickle.dump(testingSubjectList,open(dst+'testingSubjectList.pkl','wb'))


pickle.dump({
	'regressWing':regressWing,
	'predictionGap':predictionGap,
	'rotationMatrixList':rotationMatrixList,
	'meanForPhaseExtraction':mean,
	'sdForPhaseExtraction':sd,
	'windowWing':windowWing
},open(dst+'preprocessParameters2.pkl','wb'))

###########################################################

sessionFile=system+"\\d2model_new_state\\"+exercise+"\\"+loadName+"\\"+"phaseModel.ckpt"

#logDirectory=dst+'/logGD/'
imageFolder=dst+'/img/'

#os.makedirs(logDirectory,exist_ok=True)
os.makedirs(imageFolder,exist_ok=True)



#scale=np.load(scaleFile)
#expandedScale=util.expandScale(scale,sphereCount)

#recordList = []
#for inputFileName in fileList:
#	aRecord=np.load(inputFileName)
#	aRecord=aRecord[:,cropHead:aRecord.shape[1]-cropTail]
#	recordList.append(aRecord)

print("mean")
print(mean)
print("sd")
print(sd)

standardizedList = []
for aRecord in convertedRecordList:
	#standardizedList.append(aRecord*expandedScale)	#coupled dimensions will have variance of 1 (3 times larger than previous experiments)
	standardizedList.append((aRecord-mean)/sd)	#adjust to have zero mean and unit variance in all dimensions#LP just for find avg trajec
	#print("standardizedList")
	#print(standardizedList)

#print("len(aRecord)"+str(len(aRecord)))
#print("len(standardizedList)"+str(len(standardizedList)))

nowPoseList=[]
nextPoseList=[]
nowHeadList=[]
nextHeadList=[]

srcColorList=[]
sessionLengthList=[]

for j,aScaled in enumerate(standardizedList):
	m=aScaled.shape[1]
	#print("m:"+str(m))
	allPose=aScaled[:,regressWing+regressWing:m]	#(D,m-2*regressWing)
	allHead=util.getRegressHeadingDirectionWithSphere(aScaled, sphereCount, regressWing=regressWing)	#(D,m-2*regressWing)
	#generate all possible window
	allPoseWindow=[]
	allHeadWindow=[]
	for i in range(m-2*regressWing-windowSize+1):
		allPoseWindow.append(allPose[:,i:i+windowSize])
		allHeadWindow.append(allHead[:,i:i+windowSize])
		
	#print("allPoseWindow"+str(allPoseWindow))
	nowPoseList+=allPoseWindow[:-predictionGap]
	nextPoseList+=allPoseWindow[predictionGap:]
	nowHeadList+=allHeadWindow[:-predictionGap]
	nextHeadList+=allHeadWindow[predictionGap:]
	#print("nowPoseList"+str(nowPoseList))

	thisSessionLength=len(allPoseWindow[:-predictionGap])
	
	#print('allPoseWindow'+str(allPoseWindow))
	#print('predictionGap'+str(predictionGap))
	#print('m'+str(m))
	#print('windowSize'+str(windowSize))
	#print('thisSessionLength:'+str(thisSessionLength))
	#print('prePhaseColorList[j//2]:'+str(prePhaseColorList[j//2]))
	#print('j:'+str(j))
	#print('j//2:'+str(j//2))
	#print('j, aScaled')
	#print(j, aScaled)
	
	srcColorList+=[prePhaseColorList[j%(numSess-1)]]*thisSessionLength#[prePhaseColorList[j//2]]*thisSessionLength#at first
	#print("srcColorList")
	#print(srcColorList)
	sessionLengthList.append(thisSessionLength)

pairCount=len(nowPoseList)	#this is equal to the sum of sessionLengthList
#print("pairCount"+str(pairCount))

#build a matrix for subsetDistributionPenalty
#print(sessionLengthList)
sessionCount=len(sessionLengthList)
#print("sessionCount"+str(sessionCount))
sessionSegmentMatrix=np.zeros([pairCount,sessionCount],dtype=float)	#designed to be multiplied with phaseXY	(each column is one session)
sessionFirstIndex=0
for i in range(sessionCount):
	sessionSegmentMatrix[sessionFirstIndex:sessionFirstIndex+sessionLengthList[i],i]=1
	sessionFirstIndex+=sessionLengthList[i]

#these two must be equal
#print(len(srcColorList))
#print(len(nowPoseList))

nowState=np.zeros([2,D,windowSize,pairCount]) #first half
nextState=np.zeros([2,D,windowSize,pairCount]) #second half

centerPose=np.zeros([D,pairCount])
#to correct x,y LP#####
copycenterPose=np.zeros([D,pairCount])

#######################
for i in range(pairCount):
	nowState[:,:,:,i]=np.stack([nowPoseList[i],nowHeadList[i]],axis=0)
	nextState[:,:,:,i]=np.stack([nextPoseList[i],nextHeadList[i]],axis=0)
	
	centerPose[:,i]=nowPoseList[i][:,windowSize-1]
	copycenterPose[:,i]=nowPoseList[i][:,windowSize-1]#LPto correct x,y LP#
	
	if(i==0):
		print("nowPoseList[i]")
		print(nowPoseList[i])
		print("nowHeadList[i]")
		print(nowHeadList[i])
		
	

#to correct x,y LP#####
copycenterPose=(centerPose*sd) + mean
#print("copycenterPose[:,0]")
#print(copycenterPose[:,0])
if system=="walking_hindlimb":
    #forelimb

    xF1AvgPose=np.zeros([1,pairCount])
    xF2AvgPose=np.zeros([1,pairCount])
    yF1AvgPose=np.zeros([1,pairCount])
    yF2AvgPose=np.zeros([1,pairCount])

    #hindlimb

    xH1AvgPose=np.zeros([1,pairCount])
    xH2AvgPose=np.zeros([1,pairCount])
    yH1AvgPose=np.zeros([1,pairCount])
    yH2AvgPose=np.zeros([1,pairCount])
    AvgPoseArray=np.array([xF1AvgPose,yF1AvgPose,xF2AvgPose,yF2AvgPose,xH1AvgPose,yH1AvgPose,xH2AvgPose,yH2AvgPose])
    AvgPoseArrayString=np.array(['xF1AvgPose','yF1AvgPose','xF2AvgPose','yF2AvgPose','xH1AvgPose','yH1AvgPose','xH2AvgPose','yH2AvgPose'])#'xF1AvgPose','yF1AvgPose','xF2AvgPose','yF2AvgPose',
    for ifile in range(0,8):
        file = open(dst+str(AvgPoseArrayString[ifile])+'.txt','w')
        #print(ifile)
        for i in range(0,pairCount):
                #print('ifile'+str(ifile))
                AvgPoseArray[ifile][0,i]=copycenterPose[ifile,i]
                #print('copycenterPose[ifile,i]'+str(copycenterPose[ifile,i]))
                file.write("%5.4f\n" % AvgPoseArray[ifile][0,i])
        file.close()
else:
    xF1AvgPose=np.zeros([1,pairCount])
    xF2AvgPose=np.zeros([1,pairCount])
    yF1AvgPose=np.zeros([1,pairCount])
    yF2AvgPose=np.zeros([1,pairCount])
    AvgPoseArray=np.array([xF1AvgPose,yF1AvgPose,xF2AvgPose,yF2AvgPose])
    AvgPoseArrayString=np.array(['xF1AvgPose','yF1AvgPose','xF2AvgPose','yF2AvgPose'])
    for ifile in range(0,4):
        file = open(dst+str(AvgPoseArrayString[ifile])+'.txt','w')
        for i in range(0,pairCount):
                AvgPoseArray[ifile][0,i]=copycenterPose[ifile,i]
                file.write("%5.4f\n" % AvgPoseArray[ifile][0,i])
        file.close()

#######################
pn=PhaseNetwork(D, windowWing, gpuID)
#print("nowState")
#print(nowState)#LP 03/01/2019

with pn.sess.graph.as_default():
    saver = tf.train.Saver()
    saver.restore(pn.sess, sessionFile)
    testphaseRad,testprePhase=pn.sess.run([pn.phaseRad,pn.prePhase],feed_dict={pn.inputPlace:nowState})
    testphase=testphaseRad%(2*np.pi)	#this is cleaner 
    #util.savePhaseRainbowPlot(centerPose,testphase, imageFolder+'r1_'+str(step)+'.png', fig=None)
    #util.savePhaseRainbowPlot2(centerPose,testphase, imageFolder+'r2_'+str(step)+'.png', fig=None)
    

    tf.train.write_graph(pn.sess.graph.as_graph_def(), dst, 'testModel.pbtxt')#LPadd 26/12
    testOutDict={
        'phasePredice':testphase,
        'centerPose':centerPose,
        'wMean':mean,
        'wSD':sd
    }
    pickle.dump(testOutDict,open(dst+'testOutDict.pkl','wb'))
    print('testOutDict.pkl saved')

import plotly
import plotly.plotly as py
import plotly.graph_objs as go


dataWithPhase=np.vstack([centerPose,testphase])
#DwithPhase=E+1

dataNow=[]
#x=list(range(nowState.shape[1])),
for i in range(0,D+1):
	dataNow.append(go.Scatter(
	x=list(range(nowState.shape[3])),
	y=dataWithPhase[i,:]
	#y=testDataWithPhase[i,:]
	))
        
#Lukpla's code
file = open(dst+'dataWithPhase_solvebug.txt','w') 
#output_file = open(dst+"phaseOutput_bin","wb")
for i in range(0,pairCount):
	file.write("%5.4f\n" % testphase[i]) #phaseRad is tensor??
	#pickle.dump(testphase[i],output_file)
    
file.close() 
output_file = open(dst+"phaseOutput_bin.bin","wb")
pickle.dump(testphase,output_file)
output_file.close()

'''
#forelimb
file1 = open(dst+'ximF1_centerpose.txt','w') 
for i in range(0,pairCount):
	file1.write("%5.4f\n" % centerPose[0,i]) #phaseRad is tensor??
file1.close() 
file2 = open(dst+'yimF1_centerpose.txt','w') 
for i in range(0,pairCount):
	file2.write("%5.4f\n" % centerPose[1,i]) #phaseRad is tensor??
file2.close() 
file3 = open(dst+'ximF2_centerpose.txt','w') 
for i in range(0,pairCount):
	file3.write("%5.4f\n" % centerPose[2,i]) #phaseRad is tensor??
file3.close() 
file4 = open(dst+'yimF2_centerpose.txt','w') 
for i in range(0,pairCount):
	file4.write("%5.4f\n" % centerPose[3,i]) #phaseRad is tensor??
file4.close() 

#hindlimb
file1 = open(dst+'ximH1_centerpose.txt','w') 
for i in range(0,pairCount):
	file1.write("%5.4f\n" % centerPose[4,i]) #phaseRad is tensor??
file1.close() 
file2 = open(dst+'yimH1_centerpose.txt','w') 
for i in range(0,pairCount):
	file2.write("%5.4f\n" % centerPose[5,i]) #phaseRad is tensor??
file2.close() 
file3 = open(dst+'ximH2_centerpose.txt','w') 
for i in range(0,pairCount):
	file3.write("%5.4f\n" % centerPose[6,i]) #phaseRad is tensor??
file3.close() 
file4 = open(dst+'yimH2_centerpose.txt','w') 
for i in range(0,pairCount):
	file4.write("%5.4f\n" % centerPose[7,i]) #phaseRad is tensor??
file4.close() 
'''
plotly.offline.plot(dataNow, filename=dst+'phaseAgainstTime_test.html')

        

