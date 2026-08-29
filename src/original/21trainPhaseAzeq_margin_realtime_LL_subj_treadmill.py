import tensorflow.compat.v1 as tf
#import tensorflow.compat.v1 as tf#try
#tf.disable_v2_behavior()#try
import numpy as np
import random
import os,sys
import glob
#import pickle
import time

import util
#import setting

import matplotlib
import csv
import pickle5 as pickle

if os.name != 'nt':
	matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import re

#import phaseNetwork as pn
from PhaseNetworkClassFlexible import PhaseNetwork

def is_float(string):
    """ True if given string is float else False"""
    try:
        return float(string)
    except ValueError:
        return False
RatAmount=24
filepath = 'C:\\Users\\Dollaporn Anopas\\Desktop\\SI\\Research\\Rehab\\Program\\matlab\\sumsampleFold4.dat'
data = []
with open(filepath, 'r') as f:
    RatEndNumberPlusOne = []
    d = f.readlines()
    for i in d:
        k = i.rstrip().split(",")
        data.append([float(i) if is_float(i) else i for i in k]) 

data = np.array(data, dtype='O')
print(data)
y = np.concatenate([np.array(i) for i in data])
for i in range(0,RatAmount):
    print(i)
    print(int(y[i]))
    RatEndNumberPlusOne.append(int(y[i]))
print(RatEndNumberPlusOne)
'''
prePhaseColorList=[]

with open('C:\\Users\\Dollaporn Anopas\\Desktop\\SI\\Research\\Rehab\\Program\\Backup\\Walking_LLLTrain\\d1pkl\\walking_left\\sumsample.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    RatEndNumberPlusOne = []
    for row in readCSV:
        sampleSameRatEnd_plusone = row[0]
        RatEndNumberPlusOne.append(int(float(sampleSameRatEnd_plusone)))
with open('C:\\Users\\Dollaporn Anopas\\Desktop\\SI\\Research\\Rehab\\Program\\Backup\\\\Walking_LLLTrain\\d1pkl\\walking_left\\sumtrial.csv') as csvfile2:
    readCSV2 = csv.reader(csvfile2, delimiter=',')
    EndRatSessionPlusOne = []
    for row2 in readCSV2:
        sessSameRatEnd_plusone = row2[0]
        EndRatSessionPlusOne.append(int(sessSameRatEnd_plusone))
colorCode = np.array(['#1f77b4','#eee8aa','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf','#030303','#419388','#0efff8','#f80eff','#fff80e','#800eff','#ff805b','#cea15c','#cebe5c','#ff7f50','#ffe4b5','#00fa9a','#ffffe0','#008080','#000080','#da70d6','#ff69b4','#fff5ee','#696969'])#np.array(['#00C5CD','#0000FF','#FF4040','#8A2BE2','#FF9912','#7FFF00','#FF1493','#00BFFF','#FFD700','#ADFF2F','#FF6A6A','#CDC673','#668B8B','#8B636C','#800080','#8B0000','#EEB4B4','436EEE','8B4C39','#8E8E38','#C67171','#388E83','#FF8247','#87CEEB','#6959CD','#6C7B8B','#FFA54F','#008080','#FEE1FF','#40E0D0'])#,'#00C5CD','#EE82EE','#8B2252','#8B7E66','#FFFF00','#483D8B','#FFD39B','#F0F8FF','#E3CF57','#EEAD0E','#6E8B3D','#8A3324'])
RatAmount=30
SessionAmount=1118

for iSession in range(0,RatAmount):
	if (iSession == 0): 
		Range = np.arange(1,EndRatSessionPlusOne[iSession])
	else:
		Range = np.arange(EndRatSessionPlusOne[iSession-1],EndRatSessionPlusOne[iSession])
	print(Range)
	for i in Range:
		prePhaseColorList.append(colorCode[iSession])
	print(EndRatSessionPlusOne[iSession])
print(Range)
'''
FTest=[1,6,11,16,21,26,31,37]
Fold1=[2,7,12,17,22,27,32,38]
Fold2=[3,8,13,18,23,28,33,40]
Fold3=[4,9,14,19,24,29,34,41]
Fold4=[5,10,15,20,25,30,35,42]
#Foldname=[1,2,3,4]
stackdir_loop = []
for k in range(1,4):
    for j in range(8):
        if k == 1:
            dir_loop = 'Subj' + str(Fold4[j]) + '\csvOP'
            stackdir_loop.append(dir_loop)
        if k == 2:
            dir_loop = 'Subj' + str(Fold1[j]) + '\csvOP'
            stackdir_loop.append(dir_loop)
        if k == 3:
            dir_loop = 'Subj' + str(Fold2[j]) + '\csvOP'
            stackdir_loop.append(dir_loop)
        # if k == 4:
        #     dir_loop = 'Fold'+str(k)+'\Subj' + str(Fold4[j]) + '\csvOP'
        #     stackdir_loop.append(dir_loop)
print(stackdir_loop)
colorCode = np.array(['#1f77b4','#eee8aa','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf','#030303','#419388','#0efff8','#f80eff','#fff80e','#800eff','#ff805b','#cea15c','#cebe5c','#ff7f50','#ffe4b5','#00fa9a','#ffffe0','#008080','#000080','#da70d6','#ff69b4','#fff5ee','#696969','#00C5CD','#0000FF'])#np.array(['#00C5CD','#0000FF','#FF4040','#8A2BE2','#FF9912','#7FFF00','#FF1493','#00BFFF','#FFD700','#ADFF2F','#FF6A6A','#CDC673','#668B8B','#8B636C','#800080','#8B0000','#EEB4B4','436EEE','8B4C39','#8E8E38','#C67171','#388E83','#FF8247','#87CEEB','#6959CD','#6C7B8B','#FFA54F','#008080','#FEE1FF','#40E0D0'])#,'#00C5CD','#EE82EE','#8B2252','#8B7E66','#FFFF00','#483D8B','#FFD39B','#F0F8FF','#E3CF57','#EEAD0E','#6E8B3D','#8A3324'])

#stack = []
stackendsess=[]
stackbeginsess=[]
numsess=[]
count = 0
for i in stackdir_loop:
    print(i)
    count = 0
    stack=[]
    for path in os.scandir(i):
        #print(path)
        if path.is_file():
            count += 1
            collectnumber=path.name
            z = re.findall(r'T(\d+)',collectnumber)
            print('z')
            print(z)
            stack.append(z)
            print(stack)
            if (count==1):
                x = re.findall(r'T(\d+)', collectnumber)
        #print('z')
        #print(z)
        #stack.append(z)
    #stackbeginsess.append(x)
    #stackendsess.append(z) 
    aList = list(stack)
    print('aList')
    print(aList)
    flat_list = [item for sublist in aList for item in sublist]
    res = [eval(i) for i in flat_list]
    print('res')
    print(res)
    stackbeginsess.append(min(res))
    stackendsess.append(max(res))
    numsess.append(int(count/2))
print('stackbeginsess')
print(stackbeginsess)
print('stackendsess')
print(stackendsess)
aList = list(stack)
flat_list = [item for sublist in aList for item in sublist]
res = [eval(i) for i in flat_list]
##aListstackbeginsess = list(stackbeginsess)
##flat_liststackbeginsess = [item for sublist in aListstackbeginsess for item in sublist]
##resstackbeginsess = [eval(i) for i in flat_liststackbeginsess]
##aListstackendsess = list(stackendsess)
##flat_liststackendsess = [item for sublist in aListstackendsess for item in sublist]
##resstackendsess = [eval(i) for i in flat_liststackendsess]
#print('numsess')
#print(numsess)
#print('res')
#print(res)
#print('resstackbeginsess')
#print(resstackbeginsess)
#print('resstackendsess')
#print(resstackendsess)
mini=min(res)
maxi=max(res)
#print(mini)
#print(maxi)
print('len(stackbeginsess)')
print(len(stackbeginsess))
prePhaseColorList=[]
for iSession in range(0,len(stackbeginsess)):#range(0,RatAmount):
	#Range = np.arange(resstackbeginsess[iSession],resstackendsess[iSession]+1)
	Range = np.arange(stackbeginsess[iSession],stackendsess[iSession]+1)
	for i in Range:
		prePhaseColorList.append(colorCode[iSession])
	print(stackbeginsess[iSession])
	print(stackendsess[iSession])
print(Range)
print(prePhaseColorList)
SessionAmount=len(prePhaseColorList)
#with open("prePhaseColorList.txt", "w") as text_file:
	#print(prePhaseColorList, file=text_file)

if(len(sys.argv)==1):
	#setting
	system="Fold4"

	exercise="walking_left"


	repeat=False

	gpuID=-1
	crossValidateNo=-1
	trainingSubjectList=[]
	'''
	aa = np.arange(1, SessionAmount)
	for ii in aa:
		trainingSubjectList.append('rat'+str(ii)+'_')

	trainingSubjectList_img=[]

	aa_img = np.arange(1, SessionAmount)
	for ii_img in aa_img:
		trainingSubjectList_img.append('rat'+str(ii_img)+'_')
	'''
    #aa = np.arange(1, SessionAmount)
	for aa in range(0,len(stackbeginsess)):
		print('len(stackbeginsess)')
		print(len(stackbeginsess))
		for ii in range(stackbeginsess[aa],stackendsess[aa]+1):
		    trainingSubjectList.append('rat'+str(ii)+'_')
	#print(trainingSubjectList)

	trainingSubjectList_img=[]
	for aa_img in range(0,len(stackbeginsess)):
		for ii_img in range(stackbeginsess[aa_img],stackendsess[aa_img]+1):
		    trainingSubjectList_img.append('rat'+str(ii_img)+'_')
	print(trainingSubjectList_img)
	os.makedirs("C:\\Users\\Dollaporn Anopas\\Desktop\\SI\\Research\\Rehab\\Program\\Fold4\\d2model",exist_ok=True)
	os.makedirs("C:\\Users\\Dollaporn Anopas\\Desktop\\SI\\Research\\Rehab\\Program\\Fold4\\d2model\\"+exercise,exist_ok=True)
	os.makedirs("C:\\Users\\Dollaporn Anopas\\Desktop\\SI\\Research\\Rehab\\Program\\Fold4\\d2model\\"+exercise+"\\test03AllRT."+str(crossValidateNo),exist_ok=True)
	trainName="test03AllRT."+str(crossValidateNo)#forelimb 01_24Rats_pose_correct_norm_RT8_all13#01_24Rats_pose_correct_norm_RT_goodData_14_1ratfull_clean_5.
#I think my code do not need this (LP)
'''
else:	#take info from external input
	system=sys.argv[1]
	exercise=sys.argv[2]			#'comb', 'rect', or 'tab'
	crossValidateNo=int(sys.argv[3])	# 0,1,2,3,4,5
	gpuID=int(sys.argv[4])			# 0 or 1

	if(len(sys.argv)==6):
		if(sys.argv[5]=="repeat"):
			repeat=True
		else:
			repeat=False
	else:
		repeat=False

	#trainingSubjectList will be different depending on the crossValidateNo
	allSubjects=[]
	
	aa1 = np.arange(1, SessionAmount)#np.arange(1, 860)
	for ii1 in aa1:
		allSubjects.append('rat'+str(ii1))

	#print("len(allSubjects)")
	#print(len(allSubjects))

	trainingSubjectList=[]
	testingSubjectList=[]
	for i in range(len(allSubjects)):
		if(i//2 != crossValidateNo):
			trainingSubjectList.append(allSubjects[i])
		else:
			testingSubjectList.append(allSubjects[i])

	#trainName="01_."+str(crossValidateNo)
	trainName="test03AllRT."+str(crossValidateNo)#forelimb 01_24Rats_pose_correct_norm_RT8_all13#01_24Rats_pose_correct_norm_RT_goodData_14_1ratfull_clean_5.
'''
if(system=="Fold4"):
	import setting06 as setting
elif(system=="walking_forelimb"):
	import setting04 as setting

#print(trainName)
#print("train:")
#print(trainingSubjectList)
if(crossValidateNo!=-1):
	print("test")
	print(testingSubjectList)

#exit()

#minimumPhaseProgressRad=5*np.pi/180
regressWing=setting.regressWing
predictionGap=setting.predictionGap	 

windowWing=setting.windowWing
windowSize=1+windowWing*2

######################################
src=system+"\\d1pkl\\"+exercise
src_img=system+"\\d1pkl\\"+exercise
#src=system+"\\d1pkl\\"+exercise#LP
dst=system+"\\d2model\\"+exercise+"\\"+trainName+"\\"

sphereCount=setting.sphereCount
cropHead=setting.cropHead
cropTail=setting.cropTail
#print('src')
#print(src)
fileList=[]
for subjectName in trainingSubjectList:
	fileList+=glob.glob(src+"\\"+subjectName+'*.pkl', recursive=False)
	#print(fileList)


recordList=[]
#firstVisualIndexList=[]#LP
#print('fileList')
#print(fileList)
for aFile in fileList:
	d=pickle.load(open(aFile,'rb'))

	pose=d['data']
	recordList.append(pose[:,cropHead:pose.shape[1]-cropTail])
	#print("len(recordList)")
	#print(recordList)
	#print(len(recordList))
	#firstVisualIndexList.append(d['firstVisualIndex'])	#before crop#LP
	#print(dataList[-1].shape)
#print("fileList")
#print(fileList)
#print("len(fileList)")
#print(len(fileList))
#calculate the rotation matrix for each sphere
allTrainData=np.hstack(recordList)
#print("len(allTrainData)")
#print(allTrainData)
#print(len(allTrainData))

fileList_img=[]
for subjectName_img in trainingSubjectList_img:
	fileList_img+=glob.glob(src_img+"\\"+subjectName_img+'*.pkl', recursive=False)
#print("fileList_img"+str(fileList_img))

recordList_img=[]
for aFile_img in fileList_img:
	d_img=pickle.load(open(aFile_img,'rb'))
	pose_img=d_img['data']
	recordList_img.append(pose_img[:,cropHead:pose_img.shape[1]-cropTail])
	#print("recordList_img" + str(recordList_img))

allTrainData_img=np.hstack(recordList_img)



rotationMatrixList=[]

for s in range(setting.sphereCount):	

	selectedData=allTrainData[s*3:s*3+3,:]
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

convertedRecordList_img=[]
for i in range(len(recordList_img)):
	n,m=recordList_img[i].shape
	converted_img=util.convertOriginalPoseByAzeqProjection(rotationMatrixList,recordList_img[i])
	convertedRecordList_img.append(converted_img)
#find a global scaling parameter (for each dimension)
#print("convertedRecordList_img")
#print(convertedRecordList_img)
tmp=np.concatenate(convertedRecordList, axis=1)

mean = np.mean(tmp, axis=1, keepdims=True)
sd = np.std(tmp-mean, axis=1, keepdims=True)

os.makedirs(dst,exist_ok=True)
#pickle.dump(rotationMatrixList,open(dst+"rotationMatrixList.pkl",'wb'))
#mean.dump(dst+"meanForPhaseExtraction.dat")
#sd.dump(dst+"sdForPhaseExraction.dat")
pickle.dump(trainingSubjectList,open(dst+'trainingSubjectList.pkl','wb'))

if(crossValidateNo!=-1):
	pickle.dump(testingSubjectList,open(dst+'testingSubjectList.pkl','wb'))	#testing will be a bit easier

pickle.dump({
	'regressWing':regressWing,
	'predictionGap':predictionGap,
	'rotationMatrixList':rotationMatrixList,
	'meanForPhaseExtraction':mean,
	'sdForPhaseExtraction':sd,
	'windowWing':windowWing
},open(dst+'preprocessParameters.pkl','wb'))

###########################################################

sessionFile=dst+"phaseModel.ckpt"

#logDirectory=dst+'/logGD/'
imageFolder=dst+'/img/'
imageFolderRat=dst+'/img/'+'/rat/'

if not os.path.exists(dst+'checkpoint'): #os.path.exists(folder) or
	startNewTraining=True
	print("new training")
else:
	startNewTraining=False
	print("continue training")

#os.makedirs(logDirectory,exist_ok=True)
os.makedirs(imageFolder,exist_ok=True)
os.makedirs(imageFolderRat,exist_ok=True)

#D=np.load(fileList[0]).shape[0]
D=setting.sphereCount*2+setting.scalarCount	#*2 because of AZEQ


standardizedList = []
for aRecord in convertedRecordList:
	#standardizedList.append(aRecord*expandedScale)	#coupled dimensions will have variance of 1 (3 times larger than previous experiments)
	standardizedList.append((aRecord-mean)/sd)	#adjust to have zero mean and unit variance in all dimensions

print("len(aRecord)"+str(len(aRecord)))
print("len(standardizedList)"+str(len(standardizedList)))

nowPoseList=[]
nextPoseList=[]
nowHeadList=[]
nextHeadList=[]

srcColorList=[]
sessionLengthList=[]


for j,aScaled in enumerate(standardizedList):
	m=aScaled.shape[1]
	#print("m:"+str(m))
	allPose=aScaled[:,regressWing+regressWing:m]	#(D,m-2*regressWing)#LP
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
	
	srcColorList+=[prePhaseColorList[j%len(prePhaseColorList)]]*thisSessionLength#[prePhaseColorList[j//2]]*thisSessionLength#at first
	

	#print("srcColorList")
	#print(srcColorList)
	sessionLengthList.append(thisSessionLength)

srcColorList_rat=[]
for iColor in range(0,RatAmount):
	# srcColorList_rat.append(prePhaseColorList[EndRatSessionPlusOne[iColor]-2])
	srcColorList_rat.append(colorCode[iColor])
print('srcColorList_rat'+str(srcColorList_rat))

pairCount=len(nowPoseList)	#this is equal to the sum of sessionLengthList
#print("pairCount"+str(pairCount))

#build a matrix for subsetDistributionPenalty
#print(sessionLengthList)
sessionCount=len(sessionLengthList)
print("sessionCount"+str(sessionCount))
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
#ratPose=np.zeros([D,4754])
for i in range(pairCount):
	nowState[:,:,:,i]=np.stack([nowPoseList[i],nowHeadList[i]],axis=0)
	nextState[:,:,:,i]=np.stack([nextPoseList[i],nextHeadList[i]],axis=0)
	
	centerPose[:,i]=nowPoseList[i][:,windowSize-1]#LP
	
	#if i < 4754:
		#ratPose[:,i]=nowPoseList[i][:,windowSize-1]
	





#prepare feedDict for training
#firstLayerFeed=np.hstack([nowState,nextState])

#####for img#####
standardizedList_img = []
for aRecord_img in convertedRecordList_img:
	standardizedList_img.append(aRecord_img)	#adjust to have zero mean and unit variance in all dimensions

print("len(aRecord_img)"+str(len(aRecord_img)))
print("len(standardizedList_img)"+str(len(standardizedList_img)))
#print("standardizedList_img"+str(standardizedList_img))

nowPoseList_img=[]
nextPoseList_img=[]
nowHeadList_img=[]
nextHeadList_img=[]

srcColorList_img=[]
sessionLengthList_img=[]
for j_img,aScaled_img in enumerate(standardizedList_img):

	m_img=aScaled_img.shape[1]
	#print("m_img"+str(m_img))


	allPose_img=aScaled_img[:,regressWing+regressWing:m_img]	#(D,m-2*regressWing)#LP
	allHead_img=util.getRegressHeadingDirectionWithSphere(aScaled_img, sphereCount, regressWing=regressWing)	#(D,m-2*regressWing)

	#generate all possible window
	allPoseWindow_img=[]
	allHeadWindow_img=[]
	for i in range(m_img-2*regressWing-windowSize+1):
		allPoseWindow_img.append(allPose_img[:,i:i+windowSize])
		allHeadWindow_img.append(allHead_img[:,i:i+windowSize])
		

	nowPoseList_img+=allPoseWindow_img[:-predictionGap]
	nextPoseList_img+=allPoseWindow_img[predictionGap:]
	nowHeadList_img+=allHeadWindow_img[:-predictionGap]
	nextHeadList_img+=allHeadWindow_img[predictionGap:]


	thisSessionLength_img=len(allPoseWindow_img[:-predictionGap])

	
	srcColorList_img+=[prePhaseColorList[j_img%len(prePhaseColorList)]]*thisSessionLength_img# I think this is not what I want
	sessionLengthList_img.append(thisSessionLength_img)

   

pairCount_img=len(nowPoseList_img)	#this is equal to the sum of sessionLengthList

#build a matrix for subsetDistributionPenalty
#print(sessionLengthList_img)
sessionCount_img=len(sessionLengthList_img)
print("sessionCount_img"+str(sessionCount_img))
sessionSegmentMatrix_img=np.zeros([pairCount_img,sessionCount_img],dtype=float)	#designed to be multiplied with phaseXY	(each column is one session)
sessionFirstIndex_img=0
for i in range(sessionCount_img):
	sessionSegmentMatrix_img[sessionFirstIndex_img:sessionFirstIndex_img+sessionLengthList_img[i],i]=1
	sessionFirstIndex_img+=sessionLengthList_img[i]

nowState_img=np.zeros([2,D,windowSize,pairCount_img]) #first half
nextState_img=np.zeros([2,D,windowSize,pairCount_img]) #second half

centerPose_img=np.zeros([D,pairCount_img])

for i in range(pairCount_img):
	nowState_img[:,:,:,i]=np.stack([nowPoseList_img[i],nowHeadList_img[i]],axis=0)
	nextState_img[:,:,:,i]=np.stack([nextPoseList_img[i],nextHeadList_img[i]],axis=0)
	
	centerPose_img[:,i]=nowPoseList_img[i][:,windowSize-1]#LP

print('BeforePhaseNetwork')
pn=PhaseNetwork(D, windowWing, gpuID)

startTime=time.time()
print('start timing:')

with pn.sess.graph.as_default():
	saver = tf.train.Saver()
	#tf.compat.v1.train.Saver()#try

	while(True):

		if(startNewTraining):
			pn.sess.run(pn.initializer)
			
		else:
			saver.restore(pn.sess, sessionFile)

		current_global_step=tf.train.global_step(pn.sess, pn.global_step_tensor)

		#feedDict={pn.firstLayerPlace:firstLayerFeed}
		feedDict={
			pn.inputPlace:nowState,
			pn.inputPlace2:nextState,
			#pn.sessionSegmentMatrix:sessionSegmentMatrix	#experimental#LP#invole maxdistributionpenalty
		}

		print ('start cost', pn.sess.run([pn.cost,pn.speedPenalty,pn.badDistributionPenalty,pn.singularityPenalty,pn.marginPenalty],feed_dict=feedDict))#LP

		fig0=plt.figure(figsize=(8, 8))#reuse fig it will save memory
		fig=[]
		for ifig in range(0,RatAmount):
			fig.append(plt.figure(figsize=(8, 8)))
		figRatF1=[]
		for ifigRatF1 in range(0,RatAmount):
			figRatF1.append(plt.figure(figsize=(8, 8)))
		figRatF2=[]
		for ifigRatF2 in range(0,RatAmount):
			figRatF2.append(plt.figure(figsize=(8, 8)))
		print(nowState.shape)
		print(nextState.shape)
		print(sessionSegmentMatrix.shape)

		for step in range(current_global_step,current_global_step+10001):
			_ = pn.sess.run([pn.train],feed_dict=feedDict)

			if step%50 == 0:
				currentCostList=pn.sess.run([pn.cost,pn.speedPenalty,pn.badDistributionPenalty,pn.singularityPenalty,pn.marginPenalty],feed_dict=feedDict)#LP
				print(step, currentCostList)

			if step%500==0:
				save_path = saver.save(pn.sess, sessionFile)	#save sess	
			
			if (step<20000 and step%1000==0) or (step<2000 and step%250==0) or (step<250 and step%50==0) or step%2000==0:
				#save the distribution of phaseXY as an image
				prePhase_distribution,phaseXY_distribution,prePhaseMargin=pn.sess.run([pn.prePhase,pn.phaseXY,pn.prePhaseMargin],feed_dict={pn.inputPlace:nowState})

				
				util.saveMarginPlot(prePhase_distribution[:,0:RatEndNumberPlusOne[0]],phaseXY_distribution[:,0:RatEndNumberPlusOne[0]],prePhaseMargin[:,0:RatEndNumberPlusOne[0]],imageFolderRat+'Rat1_prephase_'+str(step)+'.png',fig[0], srcColorList_rat[0])
				#print('len(prephase_distribution)')
				#print(len(prePhase_distribution))
				for ipic in range(1,RatAmount):
					util.saveMarginPlot(prePhase_distribution[:,RatEndNumberPlusOne[ipic-1]:RatEndNumberPlusOne[ipic]],phaseXY_distribution[:,RatEndNumberPlusOne[ipic-1]:RatEndNumberPlusOne[ipic]],prePhaseMargin[:,RatEndNumberPlusOne[ipic-1]:RatEndNumberPlusOne[ipic]],imageFolderRat+'Rat'+str(ipic+1)+'_prephase_'+str(step)+'.png',fig[ipic], srcColorList_rat[ipic])


				#LP
				phaseRad,prePhase=pn.sess.run([pn.phaseRad,pn.prePhase],feed_dict={pn.inputPlace:nowState})
				phase=phaseRad%(2*np.pi)	#this is cleaner 
				phaseRat=[]
				phaseRat.append(phaseRad[0:RatEndNumberPlusOne[0]]%(2*np.pi))
				for iphaseRat in range(1,RatAmount):
					phaseRat.append(phaseRad[RatEndNumberPlusOne[iphaseRat-1]:RatEndNumberPlusOne[iphaseRat]]%(2*np.pi))
				util.saveRatPhaseRainbowPlot(centerPose_img[:,0:RatEndNumberPlusOne[0]],phaseRat[0], imageFolderRat+'Trajec_Rat1_H2_'+str(step)+'.png', figRatF1[0])
				for iTrajec in range(1,RatAmount):
					util.saveRatPhaseRainbowPlot(centerPose_img[:,RatEndNumberPlusOne[iTrajec-1]:RatEndNumberPlusOne[iTrajec]],phaseRat[iTrajec], imageFolderRat+'Trajec_Rat'+str(iTrajec+1)+'_H2_'+str(step)+'.png', figRatF1[iTrajec])
			if step%2000==0:
				print("CV:"+str(crossValidateNo))
				print("Time(min):",(time.time()-startTime)/60)

			if(repeat and step==1000 and currentCostList[0]>0.005):
				break
				
			if(repeat and step==3000 and currentCostList[0]>0.003):
				break

		if(not repeat):
			break
		else:
			if(currentCostList[0]>0.002):
				continue
			else:
				break

	save_path = saver.save(pn.sess, sessionFile)	#save sess

print("Time(min):",(time.time()-startTime)/60)
print("Train size (consecutive pair):", nowState.shape[-1])
print(system)
print(exercise)
print(crossValidateNo)
