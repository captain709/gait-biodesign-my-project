# -*- coding: utf-8 -*-
#"""
#Created on Mon Oct 30 15:37:38 2023


#"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import random
import os,sys
import glob
import pickle5 as pickle
import seaborn as sns
import time

import util
#import setting

import matplotlib
import csv

if os.name != 'nt':
	matplotlib.use('Agg') 
import matplotlib.pyplot as plt

#import phaseNetwork as pn
from PhaseNetworkClassFlexible import PhaseNetwork

prePhaseColorList=[]

# with open('C:\\Users\\Dell Precision 3660\\Desktop\\data\\sampleSameRatEnd_plusone_LF.csv') as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     RatEndNumberPlusOne = []
#     for row in readCSV:
#         sampleSameRatEnd_plusone = row[0]
#         RatEndNumberPlusOne.append(int(float(sampleSameRatEnd_plusone)))
# with open('C:\\Users\\Dell Precision 3660\\Desktop\\data\\sessSameRatEnd_plusone_LF.csv') as csvfile2:
#     readCSV2 = csv.reader(csvfile2, delimiter=',')
#     EndRatSessionPlusOne = []
#     for row2 in readCSV2:
#         sessSameRatEnd_plusone = row2[0]
#         EndRatSessionPlusOne.append(int(sessSameRatEnd_plusone))
# RatEndNumberPlusOne = 1
        # print(len(EndRatSessionPlusOne))
        # print(len(RatEndNumberPlusOne))
RatEndNumberPlusOne = [11255,22507,33795,45319,56546,
                        67782,79042,90259,101469,112658,
                        123898,135078,146312,157528,168766,
                        179948,191153,202312,213489,224638,
                        235854,247119,258291,269403,280647,
                        291701,302810,314613,327809,338962,
                        350183,361379,372571,383733,394939,
                        406102,417275,428484,439647,450934,
                        462139,473526,484703,495913,507218,
                        518486,529697,540967,552192,563451,
                        574630,585873,597045,608208,619443,
                        630532,641683,652964,664094,675275,
                        686455,697654,709035,720255,731509,
                        742747,753945,765134,776295,787500,
                        798688,809959,821123,832316,841105,
                        852184,863208,874419,885644,896898,
                        901823,903186,906214,913643,918432,
                        922183,927521,931112,936102,938671,
                        949812,961031,972227,983400,994550,
                        1006092,1017213,1029651,1040928,1052563,
                        1063732,1074882,1086038,1097249,1108428,
                        1119585,1130734,1141961,1153153,1164379,
                        1175534,1186663,1197800,1208951,1220096,
                        1230980,1242183,1253334,1264460,1275594,
                        1286752,1297898,1309063,1320276,1331443,
                        1342599,1354057,1365222,1375235,1385242,
                        1396264,1407414,1418602,1426045,1431252,
                        1440905,1451793,1462559,1473725,1484841,
                        1495983,1507457,1518635,1529833,1540984,
                        1552144,1563440,1574663,1585795,1596974,
                        1608122,1619281,1630286,1641191,1652155,
                        1663095,1674122,1685610,1695484,1698243,
                        1709230,1720212,1731255,1742350,1753383,
                        1764591,1774068,1785374,1796469,1807594,
                        1818660,1829699,1840135,1851126,1862984,
                        1873963,1876390,1883143,1887008,1898040,
                        1909052,1920006,1925460,1931247,1936784,1942820]
# ## RatEndNumberPlusOne = [11077,21710,32812,43952,55056,66078]
# RatEndNumberPlusOne = [11077,21710,32812,43952,55240,66344,77366] #
EndRatSessionPlusOne = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,95,100,105,110,115,120,125,130,136,141,146,151,156,162,167,172,179]

# EndRatSessionPlusOne =[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14],
#                         [15,16,17,18,19],[20,21,22,23,24],[25,26,27,28,29],
#                         [30,31,32,33,34],[35,36,37,38,39],[40,41,42,43,44],
#                         [45,46,47,48,49],[50,51,52,53,54],[55,56,57,58,59],
#                         [60,61,62,63,64],[65,66,67,68,69],[70,71,72,73,74],
#                         [75,76,77,78,79],[80,81,82,83,84,85,86,87,88,89],[90,91,92,93,94],
#                         [95,96,97,98,99],[100,101,102,103,104],[105,106,107,108,109],
#                         [110,111,112,113,114],[115,116,117,118,119],[120,121,122,123,124],
#                         [125,126,127,128,129],[130,131,132,133,134,135],[136,137,138,139,140],
#                         [141,142,143,144,145],[146,147,148,149,150],[151,152,153,154,155],
#                         [156,157,158,159,160,161],[162,163,164,165,166],[167,168,169,170,171],
#                         [172,173,174,175,176,177,178],[179,180,181,182,183,184,185]
                        
# EndRatSessionPlusOne =  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,
#                         15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,
#                         30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,
#                         45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,
#                         60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,
#                         75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,
#                         95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,
#                         110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,
#                         125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,
#                         141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,
#                         156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,
#                         172,173,174,175,176,177,178,179,180,181,182,183,184,185]
# EndRatSessionPlusOne = [1,6,11,16,21,26,31,36,41,46,51,56,61,66,71,76,81,91,96,101,106,111,116,121,126,131,137,142,147,152,157,163,168,173,180] 

# EndRatSessionPlusOne = [1,1,1,1,1,
#                         2,2,2,2,2,
#                         3,3,3,3,3,
#                         4,4,4,4,4,
#                         5,5,5,5,5,
#                         6,6,6,6,6,
#                         7,7,7,7,7,
#                         8,8,8,8,8,
#                         9,9,9,9,9,
#                         10,10,10,10,10,
#                         11,11,11,11,11,
#                         12,12,12,12,12,
#                         13,13,13,13,13,
#                         14,14,14,14,14,
#                         15,15,15,15,15,
#                         16,16,16,16,16,  
#                         17,17,17,17,17,
#                         17,17,17,17,17,
#                         18,18,18,18,18, 	
#                         19,19,19,19,19, 	
#                         20,20,20,20,20, 	
#                         21,21,21,21,21, 	
#                         22,22,22,22,22, 	
#                         23,23,23,23,23, 	
#                         24,24,24,24,24, 	
#                         25,25,25,25,25, 	
#                         26,26,26,26,26,26, 	
#                         27,27,27,27,27,		
#                         28,28,28,28,28,		
#                         29,29,29,29,29,		
#                         30,30,30,30,30,		
#                         31,31,31,31,31,31,	
#                         32,32,32,32,32,		
#                         33,33,33,33,33,		
#                         34,34,34,34,34,34,34,	
#                         35,35,35,35,35,35,35]	

#EndRatSessionPlusOne = list(range(1, 187))
type_walk = 'treadmill' #over_ground
Nameofdata = 'Foot_left' #Foot_right,Foot_left
colorCode = np.array(['#eee8aa','#1f77b4','#1f77b4','#ff7f0e','#2ca02c',
                      '#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f',
                      '#bcbd22','#17becf','#030303','#419388','#0efff8',
                      '#f80eff','#fff80e','#800eff','#ff805b','#cea15c',
                      '#cebe5c','#ff7f50','#ffe4b5','#00fa9a','#ffffe0',
                      '#008080','#000080','#da70d6','#ff69b4','#fff5ee',
                      '#696969','#800000','#7fffd4','#adff2f','#808000'])
# colorCode = np.array(['#eee8aa','#1f77b4','#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f'])
RatAmount=35#10
SessionAmount=187 #176 #703 #actually amount of file#this value start from one then need to plus one

print(len(EndRatSessionPlusOne))
print(len(RatEndNumberPlusOne))
## Assign color to each subject
for iSession in range(RatAmount):
	if iSession == 0:
		Range = np.arange(1,EndRatSessionPlusOne[iSession])
	else:
 		Range = np.arange(EndRatSessionPlusOne[iSession-1],EndRatSessionPlusOne[iSession])
	for i in Range:
		prePhaseColorList.append(colorCode[iSession])


#print(prePhaseColorList)
print("len(prePhaseColorList)")
print(len(prePhaseColorList))
'''
with open("prePhaseColorList.txt", "w") as text_file:
	print(prePhaseColorList, file=text_file)
'''
if(len(sys.argv)==1):
	#setting
	system="walking_left"

	exercise="walking"

	repeat=False

	gpuID=-1
	crossValidateNo=-1
	trainingSubjectList=[]
	
	aa = np.arange(1, SessionAmount)#np.arange(1, 860)
	for ii in aa:
		trainingSubjectList.append(type_walk+str(ii)+'_walking_left')
        
	trainingSubjectList_img=[]

	aa_img = np.arange(1, SessionAmount)
	for ii_img in aa_img:
# 		trainingSubjectList_img.append(type_walk+str(ii_img)+'_walking_right')#trainingSubjectList_img.append('rat'+str(ii_img)+'_worldnotnorm_')

		trainingSubjectList_img.append(type_walk+str(ii_img)+'_walking_left')#trainingSubjectList_img.append('rat'+str(ii_img)+'_worldnotnorm_')	
	
	trainName="test01_24Rats_pose_correct_all_RT_15_2_"+str(crossValidateNo)#forelimb 01_24Rats_pose_correct_norm_RT8_all13#01_24Rats_pose_correct_norm_RT_goodData_14_1ratfull_clean_5.

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
		allSubjects.append(type_walk+str(ii1))

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
	trainName="test01_24Rats_pose_correct_all_RT_15_2_"+str(crossValidateNo)#forelimb 01_24Rats_pose_correct_norm_RT8_all13#01_24Rats_pose_correct_norm_RT_goodData_14_1ratfull_clean_5.

if(system=="walking_left"):
	import setting06 as setting
elif(system=="walking_right"):
	import setting06 as setting

#print(trainName)
#print("train:")s
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
# src=system+"\\d1pkl_new_state\\"+exercise+"\\WorldFrame\\"
# src_img=system+"\\d1pkl_new_state\\"+exercise+"\\WorldFrame\\"#"\\Notnorm_goodDataRatfull\\"

# src= "C:\\Users\\NAPATSAWAn\\OneDrive - Srinakharinwirot University\\Desktop\\data\\data_train\\over_ground\\Foot_right\\"
# src = "C:\\Users\\Dell Precision 3660\\Desktop\\data\\data_train\\treadmill\\Foot_left\\" 
# src="C:\\Users\\Dell Precision 3660\\Desktop\data\\csvOP_T\\pkl\\walking_left\\LeftFoot\\"
src = "../data/Training_data/pkl/walking_left/LeftFoot/"


# src="C:\\Users\\Dell Precision 3660\\Desktop\\data\\csvOP_T\\pkl\\walking_right\\RightFoot\\"
# src_img="C:\\Users\\NAPATSAWAn\\OneDrive - Srinakharinwirot University\\Desktop\\data\\data_train\\over_ground\\Foot_right\\"  #"\\Notnorm_goodDataRatfull\\"
# src_img="C:\\Users\\Dell Precision 3660\\Desktop\\data\\csvOP_T\\pkl\\walking_left\\LeftFoot\\"
src_img = "../data/Training_data/pkl/walking_left/LeftFoot/"
# src_img="C:\\Users\\Dell Precision 3660\\Desktop\\data\\csvOP_T\\pkl\\walking_right\\RightFoot\\"
# src_img = "C:\\Users\\Dell Precision 3660\\Desktop\\data\\data_train\\treadmill\\Foot_left\\" 
# dst=system+"\\d2model_new_state\\"+exercise+"\\"+trainName+"\\"

# dst="C:\\Users\\Dell Precision 3660\\Desktop\\data\\csvOP_T\\pkl\\walking_left\\d2model_new_state_FootL\\"+exercise
dst = "../data/Training_data/pkl/walking_left/d2model_new_state_FootL/"+exercise
# dst="C:\\Users\\Dell Precision 3660\\Desktop\\data\\csvOP_T\\pkl\\walking_right\\d2model_new_state_FootR\\"+exercise
# dst="C:\\Users\\Dell Precision 3660\\Desktop\\data\\data_train\\treadmill\\d2model_new_state_Foot_test\\"+exercise

sphereCount=setting.sphereCount
cropHead=setting.cropHead
cropTail=setting.cropTail 

fileList=[]
for subjectName in trainingSubjectList:
	fileList+=glob.glob(src+subjectName+'*.pkl', recursive=False)
	print(subjectName)


recordList=[]
#firstVisualIndexList=[]#LP
for aFile in fileList:
	d=pickle.load(open(aFile,'rb')) ###  
	
	pose=d['data']
	recordList.append(pose[:,cropHead:pose.shape[1]-cropTail]) 
	#print("recordList" + str(recordList))
	#print("recordList"+aFile)
allTrainData=np.hstack(recordList)

fileList_img=[]
for subjectName_img in trainingSubjectList_img:
	fileList_img+=glob.glob(src_img+subjectName_img+'*.pkl', recursive=False)
	print("fileList_img"+str(fileList_img))
#print("fileList_img"+str(fileList_img))

print('d')
recordList_img=[]
#print('fileList_img'+str(fileList_img))
for aFile_img in fileList_img:
	print("aFile_img"+aFile_img)
	d_img=pickle.load(open(aFile_img,'rb'))
# 	print("aFile_img"+aFile_img)
	pose_img=d_img['data']
	recordList_img.append(pose_img[:,cropHead:pose_img.shape[1]-cropTail])
	#print("recordList_img" + str(recordList_img))
	print("recordList_img" + aFile_img)
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


## DEBUG
# print(rotationMatrixList)

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

## saving traiing subjects' names list
pickle.dump(trainingSubjectList,open(dst+'trainingSubjectList.pkl','wb'))


## if the test -> saving testing subjecs list for testing too
if(crossValidateNo!=-1):
	pickle.dump(testingSubjectList,open(dst+'testingSubjectList.pkl','wb'))	#testing will be a bit easier

## Preprocessing config
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
	
	srcColorList+=[prePhaseColorList[j%len(prePhaseColorList)]]*thisSessionLength#[prePhaseColorList[j//2]]*thisSessionLength#at first

	#print("srcColorList")
	#print(srcColorList)
	sessionLengthList.append(thisSessionLength)

srcColorList_rat=[]
for iColor in range(0,RatAmount):
	srcColorList_rat.append(prePhaseColorList[EndRatSessionPlusOne[iColor]-2])
print('srcColorList_rat'+str(srcColorList_rat))

pairCount=len(nowPoseList)	#this is equal to the sum of sessionLengthList
#print("pairCount"+str(pairCount))

#build a matrix for subsetDistributionPenalty
sessionCount=len(sessionLengthList)
print("sessionCount"+str(sessionCount))
sessionSegmentMatrix=np.zeros([pairCount,sessionCount],dtype=float)	#designed to be multiplied with phaseXY	(each column is one session)
sessionFirstIndex=0
for i in range(sessionCount):
	sessionSegmentMatrix[sessionFirstIndex:sessionFirstIndex+sessionLengthList[i],i]=1
	sessionFirstIndex+=sessionLengthList[i]


nowState=np.zeros([2,D,windowSize,pairCount]) #first half
nextState=np.zeros([2,D,windowSize,pairCount]) #second half

centerPose=np.zeros([D,pairCount])
#ratPose=np.zeros([D,4754])
for i in range(pairCount):

	nowState[:,:,:,i]=np.stack([nowPoseList[i],nowHeadList[i]],axis=0)
	nextState[:,:,:,i]=np.stack([nextPoseList[i],nextHeadList[i]],axis=0)
	
	centerPose[:,i]=nowPoseList[i][:,windowSize-1] #LP
	'''
	if i < 4754:
		ratPose[:,i]=nowPoseList[i][:,windowSize-1]
	'''
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
print(sessionLengthList_img)
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
# print('Dow')

### saving the data
np.save("data/[REF]nowState.npy", nowState)
np.save("data/[REF]nextState.npy", nextState)
np.save("data/[REF]centerPose.npy", centerPose)

pn=PhaseNetwork(D, windowWing, gpuID)

startTime=time.time()


print('start timing:')

with pn.sess.graph.as_default():
	saver = tf.train.Saver()
	
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

		#print ('start cost', pn.sess.run([pn.cost,pn.speedPenalty,pn.badDistributionPenalty,pn.singularityPenalty],feed_dict=feedDict))
		#print ('start cost', pn.sess.run([pn.cost,pn.speedPenalty,pn.maxDistributionPenalty,pn.singularityPenalty,pn.marginPenalty],feed_dict=feedDict))#LP
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
				#currentCostList=pn.sess.run([pn.cost,pn.speedPenalty,pn.badDistributionPenalty,pn.singularityPenalty],feed_dict=feedDict)
				#currentCostList=pn.sess.run([pn.cost,pn.speedPenalty,pn.maxDistributionPenalty,pn.singularityPenalty,pn.marginPenalty],feed_dict=feedDict)#LP
				currentCostList=pn.sess.run([pn.cost,pn.speedPenalty,pn.badDistributionPenalty,pn.singularityPenalty,pn.marginPenalty],feed_dict=feedDict)#LP
				print(step, currentCostList)

			if step%500==0:
				save_path = saver.save(pn.sess, sessionFile)	#save sess	
			
			#if step in [0,1,2,4,8,16] or (step<1000 and step%32==0) or (step<20000 and step%1000==0) or step%2000==0:
			if (step<20000 and step%1000==0) or (step<2000 and step%250==0) or (step<250 and step%50==0) or step%2000==0:
				#save the distribution of phaseXY as an image
				#prePhase_distribution,phaseXY_distribution=pn.sess.run([pn.prePhase,pn.phaseXY],feed_dict={pn.inputPlace:nowState})
				prePhase_distribution,phaseXY_distribution,prePhaseMargin=pn.sess.run([pn.prePhase,pn.phaseXY,pn.prePhaseMargin],feed_dict={pn.inputPlace:nowState})
				
				#util.savePrePhaseAndPhasePlot(prePhase_distribution,phaseXY_distribution,imageFolder+str(step)+'.png',fig, srcColorList)
				util.saveMarginPlot(prePhase_distribution,phaseXY_distribution,prePhaseMargin,imageFolder+str(step)+'.png',fig0, srcColorList)
				
				util.saveMarginPlot(prePhase_distribution[:,0:RatEndNumberPlusOne[0]],phaseXY_distribution[:,0:RatEndNumberPlusOne[0]],prePhaseMargin[:,0:RatEndNumberPlusOne[0]],imageFolderRat+'Rat1_prephase_'+str(step)+'.png',fig[0], srcColorList_rat[0])
				
				for ipic in range(1,RatAmount):
					util.saveMarginPlot(prePhase_distribution[:,RatEndNumberPlusOne[ipic-1]:RatEndNumberPlusOne[ipic]],phaseXY_distribution[:,RatEndNumberPlusOne[ipic-1]:RatEndNumberPlusOne[ipic]],prePhaseMargin[:,RatEndNumberPlusOne[ipic-1]:RatEndNumberPlusOne[ipic]],imageFolderRat+'Rat'+str(ipic+1)+'_prephase_'+str(step)+'.png',fig[ipic], srcColorList_rat[ipic])

				#util.saveScatterPlot(phaseXY_distribution,imageFolder+str(step)+'.png',fig)

				#LP
				phaseRad,prePhase=pn.sess.run([pn.phaseRad,pn.prePhase],feed_dict={pn.inputPlace:nowState})
				phase=phaseRad%(2*np.pi)	#this is cleaner 
				#print("phase:"+str(phase[1:4755]))

				phaseRat=[]
				phaseRat.append(phaseRad[0:RatEndNumberPlusOne[0]]%(2*np.pi))
				for iphaseRat in range(1,RatAmount):
					phaseRat.append(phaseRad[RatEndNumberPlusOne[iphaseRat-1]:RatEndNumberPlusOne[iphaseRat]]%(2*np.pi))

				util.saveRatPhaseRainbowPlot(centerPose_img[:,0:RatEndNumberPlusOne[0]],phaseRat[0], imageFolderRat+'Trajec_Rat1_F1_'+str(step)+'.png', figRatF1[0])
# 				util.saveRatPhaseRainbowPlot2(centerPose_img[:,0:RatEndNumberPlusOne[0]],phaseRat[0], imageFolderRat+'Trajec_Rat1_F2_'+str(step)+'.png', figRatF2[0])
				for iTrajec in range(1,RatAmount):
 					util.saveRatPhaseRainbowPlot(centerPose_img[:,RatEndNumberPlusOne[iTrajec-1]:RatEndNumberPlusOne[iTrajec]],phaseRat[iTrajec], imageFolderRat+'Trajec_Rat'+str(iTrajec+1)+'_F1_'+str(step)+'.png', figRatF1[iTrajec])
# 					util.saveRatPhaseRainbowPlot2(centerPose_img[:,RatEndNumberPlusOne[iTrajec-1]:RatEndNumberPlusOne[iTrajec]],phaseRat[iTrajec], imageFolderRat+'Trajec_Rat'+str(iTrajec+1)+'_F2_'+str(step)+'.png', figRatF2[iTrajec])
						#util.saveRatPhaseRainbowPlot3(centerPose_img[:,RatEndNumberPlusOne[iTrajec-1]:RatEndNumberPlusOne[iTrajec]],phaseRat[iTrajec], imageFolderRat+'Trajec_Rat'+str(iTrajec+1)+'_H1_'+str(step)+'.png', figRatH1[iTrajec])
						#util.saveRatPhaseRainbowPlot4(centerPose_img[:,RatEndNumberPlusOne[iTrajec-1]:RatEndNumberPlusOne[iTrajec]],phaseRat[iTrajec], imageFolderRat+'Trajec_Rat'+str(iTrajec+1)+'_H2_'+str(step)+'.png', figRatH2[iTrajec])
				
				util.savePhaseRainbowPlot(centerPose_img,phase, imageFolder+'RatAll_F1_'+str(step)+'.png', fig=None)
# 				util.savePhaseRainbowPlot2(centerPose_img,phase, imageFolder+'RatAll_F2_'+str(step)+'.png', fig=None)
				

				#util.savePhaseRainbowPlot3(centerPose_img,phase, imageFolder+'Rat1_H1_'+str(step)+'.png', fig=None)
				#util.savePhaseRainbowPlot4(centerPose_img,phase, imageFolder+'Rat1_H2_'+str(step)+'.png', fig=None) #right hand
				#
			
			if step%2000==0:
				print("CV:"+str(crossValidateNo))
				print("Time(min):",(time.time()-startTime)/60)

			#if step==5000:
			#	_ = pn.sess.run([pn.assignSecondAp])
			#	print("***** Ap has beed updated. *****")

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
print(Nameofdata)