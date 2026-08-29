import tensorflow.compat.v1 as tf
import numpy as np
import random
import os,sys
import glob
import pickle
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

with open('C:\\Users\\Lukpla\\Desktop\\phD\\Neural_network\\Python\\LukplaCode\\walking_forelimb\\d1pkl_new_state\\walking\\WorldFrame\\sampleSameRatEnd_plusone.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    RatEndNumberPlusOne = []
    for row in readCSV:
        sampleSameRatEnd_plusone = row[0]
        RatEndNumberPlusOne.append(int(float(sampleSameRatEnd_plusone)))
with open('C:\\Users\\Lukpla\\Desktop\\phD\\Neural_network\\Python\\LukplaCode\\walking_forelimb\\d1pkl_new_state\\walking\\WorldFrame\\sessSameRatEnd_plusone.csv') as csvfile2:
    readCSV2 = csv.reader(csvfile2, delimiter=',')
    EndRatSessionPlusOne = []
    for row2 in readCSV2:
        sessSameRatEnd_plusone = row2[0]
        EndRatSessionPlusOne.append(int(sessSameRatEnd_plusone))
colorCode = np.array(['#1f77b4','#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf','#030303','#419388','#0efff8','#f80eff','#fff80e','#800eff','#ff805b','#cea15c','#cebe5c','#ff7f50','#ffe4b5','#00fa9a','#ffffe0','#008080','#000080','#da70d6','#ff69b4','#fff5ee','#696969','#800000','7fffd4','#adff2f','#808000','#eee8aa'])
RatAmount=12#10
SessionAmount=827#703#actually amount of file#this value start from one then need to plus one

for iSession in range(0,RatAmount):
	if (iSession == 0): 
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
	system="walking_forelimb"

	exercise="walking"


	repeat=False

	gpuID=-1
	crossValidateNo=-1
	trainingSubjectList=[]
	
	aa = np.arange(1, SessionAmount)#np.arange(1, 860)
	for ii in aa:
		trainingSubjectList.append('rat'+str(ii)+'_')

	trainingSubjectList_img=[]

	aa_img = np.arange(1, SessionAmount)
	for ii_img in aa_img:
		trainingSubjectList_img.append('rat'+str(ii_img)+'_')#trainingSubjectList_img.append('rat'+str(ii_img)+'_worldnotnorm_')

	
	
	trainName="test01_24Rats_pose_correct_all_RT_15_2."+str(crossValidateNo)#forelimb 01_24Rats_pose_correct_norm_RT8_all13#01_24Rats_pose_correct_norm_RT_goodData_14_1ratfull_clean_5.

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
	trainName="test01_24Rats_pose_correct_all_RT_15_2."+str(crossValidateNo)#forelimb 01_24Rats_pose_correct_norm_RT8_all13#01_24Rats_pose_correct_norm_RT_goodData_14_1ratfull_clean_5.

if(system=="walking_hindlimb"):
	import setting03 as setting
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
src=system+"\\d1pkl_new_state\\"+exercise+"\\WorldFrame\\"
src_img=system+"\\d1pkl_new_state\\"+exercise+"\\WorldFrame\\"#"\\Notnorm_goodDataRatfull\\"
#src=system+"\\d1pkl\\"+exercise#LP
dst=system+"\\d2model_new_state\\"+exercise+"\\"+trainName+"\\"

sphereCount=setting.sphereCount
cropHead=setting.cropHead
cropTail=setting.cropTail

fileList=[]
for subjectName in trainingSubjectList:
	fileList+=glob.glob(src+subjectName+'*.pkl', recursive=False)


recordList=[]
#firstVisualIndexList=[]#LP
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
	fileList_img+=glob.glob(src_img+subjectName_img+'*.pkl', recursive=False)
#print("fileList_img"+str(fileList_img))

recordList_img=[]
for aFile_img in fileList_img:
	d_img=pickle.load(open(aFile_img,'rb'))
	pose_img=d_img['data']
	recordList_img.append(pose_img[:,cropHead:pose_img.shape[1]-cropTail])
	#print("recordList_img" + str(recordList_img))

allTrainData_img=np.hstack(recordList_img)

## Please explain Hilbert Transform

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
	srcColorList_rat.append(prePhaseColorList[EndRatSessionPlusOne[iColor]-2])
print('srcColorList_rat'+str(srcColorList_rat))

pairCount=len(nowPoseList)	#this is equal to the sum of sessionLengthList
#print("pairCount"+str(pairCount))

#build a matrix for subsetDistributionPenalty
print(sessionLengthList)
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
				util.saveRatPhaseRainbowPlot2(centerPose_img[:,0:RatEndNumberPlusOne[0]],phaseRat[0], imageFolderRat+'Trajec_Rat1_F2_'+str(step)+'.png', figRatF2[0])
				for iTrajec in range(1,RatAmount):
					util.saveRatPhaseRainbowPlot(centerPose_img[:,RatEndNumberPlusOne[iTrajec-1]:RatEndNumberPlusOne[iTrajec]],phaseRat[iTrajec], imageFolderRat+'Trajec_Rat'+str(iTrajec+1)+'_F1_'+str(step)+'.png', figRatF1[iTrajec])
					util.saveRatPhaseRainbowPlot2(centerPose_img[:,RatEndNumberPlusOne[iTrajec-1]:RatEndNumberPlusOne[iTrajec]],phaseRat[iTrajec], imageFolderRat+'Trajec_Rat'+str(iTrajec+1)+'_F2_'+str(step)+'.png', figRatF2[iTrajec])
						#util.saveRatPhaseRainbowPlot3(centerPose_img[:,RatEndNumberPlusOne[iTrajec-1]:RatEndNumberPlusOne[iTrajec]],phaseRat[iTrajec], imageFolderRat+'Trajec_Rat'+str(iTrajec+1)+'_H1_'+str(step)+'.png', figRatH1[iTrajec])
						#util.saveRatPhaseRainbowPlot4(centerPose_img[:,RatEndNumberPlusOne[iTrajec-1]:RatEndNumberPlusOne[iTrajec]],phaseRat[iTrajec], imageFolderRat+'Trajec_Rat'+str(iTrajec+1)+'_H2_'+str(step)+'.png', figRatH2[iTrajec])
				
				util.savePhaseRainbowPlot(centerPose_img,phase, imageFolder+'RatAll_F1_'+str(step)+'.png', fig=None)
				util.savePhaseRainbowPlot2(centerPose_img,phase, imageFolder+'RatAll_F2_'+str(step)+'.png', fig=None)
				

				#util.savePhaseRainbowPlot3(centerPose_img,phase, imageFolder+'Rat1_H1_'+str(step)+'.png', fig=None)
				#util.savePhaseRainbowPlot4(centerPose_img,phase, imageFolder+'Rat1_H2_'+str(step)+'.png', fig=None)
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
'''
import plotly
import plotly.plotly as py
import plotly.graph_objs as go


dataWithPhaseRat1=np.vstack([centerPose_img[:,0:24166],phase[0:24166]])
dataWithPhaseRat2=np.vstack([centerPose_img[:,24166:33853],phase[24166:33853]])
dataWithPhaseRat3=np.vstack([centerPose_img[:,33853:36411],phase[33853:36411]])
dataWithPhaseRat4=np.vstack([centerPose_img[:,36411:46947],phase[36411:46947]])
dataWithPhaseRat5=np.vstack([centerPose_img[:,46947:48168],phase[46947:48168]])
dataWithPhaseRat6=np.vstack([centerPose_img[:,48168:48810],phase[48168:48810]])
dataWithPhaseRat7=np.vstack([centerPose_img[:,48810:51985],phase[48810:51985]])
dataWithPhaseRat8=np.vstack([centerPose_img[:,51985:54070],phase[51985:54070]])
dataWithPhaseRat9=np.vstack([centerPose_img[:,54070:61300],phase[54070:61300]])

#DwithPhase=E+1
print('dataWithPhaseRat1'+str(dataWithPhaseRat1))
dataNowRat1=[]
#x=list(range(nowState.shape[1])),
#print('dataWithPhaseRat1'+str(dataWithPhaseRat1))
for i in range(0,24165):
	dataNowRat1.append(go.Scatter(
	x=list(range(dataWithPhaseRat1.shape[1])),
	y=dataWithPhaseRat1[:,i]
	#y=testDataWithPhase[i,:]
	))

dataNowRat2=[]
#x=list(range(nowState.shape[1])),
print(dataWithPhaseRat2.shape[1])
for i in range(0,dataWithPhaseRat2.shape[1]):#for i in range(24166,33852):
	dataNowRat2.append(go.Scatter(
	x=list(range(dataWithPhaseRat2.shape[1])),
	y=dataWithPhaseRat2[:,i]
	#y=testDataWithPhase[i,:]
	))

dataNowRat3=[]
#x=list(range(nowState.shape[1])),
for i in range(0,dataWithPhaseRat3.shape[1]):#for i in range(33853,36410):
	dataNowRat3.append(go.Scatter(
	x=list(range(dataWithPhaseRat3.shape[1])),
	y=dataWithPhaseRat3[:,i]
	#y=testDataWithPhase[i,:]
	))

dataNowRat4=[]
#x=list(range(nowState.shape[1])),
for i in range(0,dataWithPhaseRat4.shape[1]):#for i in range(36411,46946):
	dataNowRat4.append(go.Scatter(
	x=list(range(dataWithPhaseRat4.shape[1])),
	y=dataWithPhaseRat4[:,i]
	#y=testDataWithPhase[i,:]
	))

dataNowRat5=[]
#x=list(range(nowState.shape[1])),
for i in range(0,dataWithPhaseRat5.shape[1]):#for i in range(46947,48167):
	dataNowRat5.append(go.Scatter(
	x=list(range(dataWithPhaseRat5.shape[1])),
	y=dataWithPhaseRat5[:,i]
	#y=testDataWithPhase[i,:]
	))

dataNowRat6=[]
#x=list(range(nowState.shape[1])),
for i in range(0,dataWithPhaseRat6.shape[1]):#for i in range(48168,48809):
	dataNowRat6.append(go.Scatter(
	x=list(range(dataWithPhaseRat6.shape[1])),
	y=dataWithPhaseRat6[:,i]
	#y=testDataWithPhase[i,:]
	))

dataNowRat7=[]
#x=list(range(nowState.shape[1])),
for i in range(0,dataWithPhaseRat7.shape[1]):#for i in range(48810,51984):
	dataNowRat7.append(go.Scatter(
	x=list(range(dataWithPhaseRat7.shape[1])),
	y=dataWithPhaseRat7[:,i]
	#y=testDataWithPhase[i,:]
	))

dataNowRat8=[]
#x=list(range(nowState.shape[1])),
for i in range(0,dataWithPhaseRat8.shape[1]):#for i in range(51985,54069):
	dataNowRat8.append(go.Scatter(
	x=list(range(dataWithPhaseRat8.shape[1])),
	y=dataWithPhaseRat8[:,i]
	#y=testDataWithPhase[i,:]
	))

dataNowRat9=[]
#x=list(range(nowState.shape[1])),
for i in range(0,dataWithPhaseRat9.shape[1]):#for i in range(54070,61299):
	dataNowRat9.append(go.Scatter(
	x=list(range(dataWithPhaseRat9.shape[1])),
	y=dataWithPhaseRat9[:,i]
	#y=testDataWithPhase[i,:]
	))

plotly.offline.plot(dataNowRat1, filename=dst+'Rat1_phaseAgainstTime_test.html')
plotly.offline.plot(dataNowRat2, filename=dst+'Rat2_phaseAgainstTime_test.html')
plotly.offline.plot(dataNowRat3, filename=dst+'Rat3_phaseAgainstTime_test.html')
plotly.offline.plot(dataNowRat4, filename=dst+'Rat4_phaseAgainstTime_test.html')
plotly.offline.plot(dataNowRat5, filename=dst+'Rat5_phaseAgainstTime_test.html')
plotly.offline.plot(dataNowRat6, filename=dst+'Rat6_phaseAgainstTime_test.html')
plotly.offline.plot(dataNowRat7, filename=dst+'Rat7_phaseAgainstTime_test.html')
plotly.offline.plot(dataNowRat8, filename=dst+'Rat8_phaseAgainstTime_test.html')
plotly.offline.plot(dataNowRat9, filename=dst+'Rat9_phaseAgainstTime_test.html')
print("Done")
'''