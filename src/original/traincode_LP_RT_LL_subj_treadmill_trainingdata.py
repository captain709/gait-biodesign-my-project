import tensorflow.compat.v1 as tf
#import tensorflow.compat.v1 as tf#try
import numpy as np
import glob
#import pickle
import pickle5 as pickle
import util
import os,sys
from PhaseNetworkClassFlexible import PhaseNetwork
import matplotlib
if os.name != 'nt':
	matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import re
system="Fold4"

exercise="walking_right"#"walking_right"

#loadName="01_24Rats_pose_correct_Rat3toRat40_RT_state_15.-1"
#testName="01_24Rats_pose_correct_Rat3toRat40_RT_state_15.-1_test"
loadName="test04AllRT.-1"#"test04AllRT.-1"
testName="test04AllRT.-1_test"#"test04AllRT.-1_test"
testFolder=system+"\\d2model\\"+exercise+"\\"+testName+"\\"
os.makedirs(testFolder,exist_ok=True)



if(system=="Fold4"):
	import setting06 as setting
elif(system=="Fold4"):
	import setting06 as setting

D=setting.sphereCount*2+setting.scalarCount	#*2 because of AZEQ
gpuID=-1

stackendsess=[]
stackbeginsess=[]
numsess=[]
count = 0
FTest=[1,6,11,16,21,26,31,37]
Fold1=[2,7,12,17,22,27,32,38]
Fold2=[3,8,13,18,23,28,33,40]
Fold3=[4,9,14,19,24,29,34,41]
Fold4=[5,10,15,20,25,30,35,42]
Fold1Train=[2,7,12,17,22,27,32,38,3,8,13,18,23,28,33,40,4,9,14,19,24,29,34,41]
Fold2Train=[3,8,13,18,23,28,33,40,4,9,14,19,24,29,34,41,5,10,15,20,25,30,35,42]
Fold3Train=[4,9,14,19,24,29,34,41,5,10,15,20,25,30,35,42,1,6,11,16,21,26,31,37]
Fold4Train=[5,10,15,20,25,30,35,42,2,7,12,17,22,27,32,38,3,8,13,18,23,28,33,40]
#Foldname=[1,2,3,4]
stackdir_loop = []
for k in range(4,5):
    for j in range(24):
        dir_loop = 'C:\\Users\\Dollaporn Anopas\\Desktop\\SI\\Research\\Rehab\\Program'+'\\Subj' + str(Fold4Train[j]) + '\\d1pkl\\walking_right\\'
        stackdir_loop.append(dir_loop)
print(stackdir_loop)
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
            z = re.findall(r't(\d+)',collectnumber)
            #print('z')
            #print(z)
            stack.append(z)
            #print(stack)
            if (count==1):
                x = re.findall(r't(\d+)', collectnumber)
        #print('z')
        #print(z)
        #stack.append(z)
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

startsess=1#1#1117
numSess=389#1000#1118#1605
testingSubjectList=[]
for z in range(0,len(stackbeginsess)):
    a = np.arange(stackbeginsess[z],stackendsess[z]+1)
    for i in a:
        testingSubjectList.append('rat'+str(i)+'_')#('newLSubjT'+str(i))#testingSubjectList.append('rat'+str(i))->before 03/01/2019
print('testingSubjectList')
print(testingSubjectList)

trainingSubjectList_img=[]
for zz in range(0,len(stackbeginsess)):
    aa_img = np.arange(stackbeginsess[zz],stackendsess[zz]+1)
    for ii_img in aa_img:
        trainingSubjectList_img.append('rat'+str(ii_img)+'_')#('newLSubjT'+str(i))#beware
print('trainingSubjectList_img')
print(trainingSubjectList_img)
#print(res)
prePhaseColorList=[]
b = np.arange(1,numSess)#(startsess,numSess)#np.arange(0,92)
for i2 in b:
	prePhaseColorList.append('#1f77b4')
##prepare training data
regressWing=setting.regressWing
predictionGap=setting.predictionGap	 

windowWing=setting.windowWing
windowSize=1+windowWing*2

######################################

src=system+"\\d1pkl\\"+exercise+"\\"
src_img=system+"\\d1pkl\\"+exercise+"\\"
prc=system+"\\d2model\\"+exercise+"\\"+loadName+"\\"
dst=system+"\\d2model\\"+exercise+"\\"+testName+"\\"



sphereCount=setting.sphereCount
cropHead=setting.cropHead
cropTail=setting.cropTail

fileList=[]
for subjectName in testingSubjectList:
	#print("subjectName")
	#print(subjectName)
	fileList+=glob.glob(src+subjectName+'*.pkl', recursive=False)
	#print(src+subjectName)


recordList=[]
#print(fileList)
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

allTestData_img=np.hstack(recordList_img)

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

convertedRecordList_img=[]
for i in range(len(recordList_img)):
	n,m=recordList_img[i].shape
	converted_img=util.convertOriginalPoseByAzeqProjection(rotationMatrixList,recordList_img[i])
	convertedRecordList_img.append(converted_img)

#find a global scaling parameter (for each dimension)
tmp=np.concatenate(convertedRecordList, axis=1)

#mean = np.mean(tmp, axis=1, keepdims=True)#LP cut this line 03/01/2019
#sd = np.std(tmp-mean, axis=1, keepdims=True)#LP cut this line 03/01/2019
#LP add this below section 03/01/2019
d=pickle.load(open(prc+'preprocessParameters.pkl','rb'))#preprocessParameters.pkl#preprocessParameters2.pkl
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

sessionFile=system+"\\d2model\\"+exercise+"\\"+loadName+"\\"+"phaseModel.ckpt"

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


if exercise=="walking_right":
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
	AvgPoseArray=np.array([xH1AvgPose,yH1AvgPose,xH2AvgPose,yH2AvgPose])
	AvgPoseArrayString=np.array(['xH2AvgPose','yH2AvgPose'])#'xF1AvgPose','yF1AvgPose','xF2AvgPose','yF2AvgPose',
	for ifile in range(0,2):
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
	AvgPoseArrayString=np.array(['xH1AvgPose','yH1AvgPose',])
	for ifile in range(0,2):
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
	testphaseRad,testprePhase,testprePhaseMargin=pn.sess.run([pn.phaseRad,pn.prePhase,pn.prePhaseMargin],feed_dict={pn.inputPlace:nowState})
	testphase=testphaseRad%(2*np.pi)	#this is cleaner 
	#util.savePhaseRainbowPlot(centerPose,testphase, imageFolder+'r1_'+str(step)+'.png', fig=None)
	#util.savePhaseRainbowPlot2(centerPose,testphase, imageFolder+'r2_'+str(step)+'.png', fig=None)
	fig0=plt.figure(figsize=(8, 8))
	util.saveMarginPlot(testprePhase,testprePhase,testprePhaseMargin,imageFolder+'test'+'.png',fig0, '#1f77b4')
	#print(len(centerPose_img))
	#print(len(testphase))
	##util.savePhaseRainbowPlot(centerPose_img,testphase, imageFolder+'Trajec_SubjAll_H1_'+'test'+'.png', fig=None)->previous code on18/11/2021
	util.savePhaseRainbowPlot(copycenterPose,testphase, imageFolder+'Trajec_SubjAll_HR_'+'test'+'.png', fig=None)#correct or not?
	#util.savePhaseRainbowPlot2(centerPose_img,testphase, imageFolder+'Trajec_RatAll_H2_'+'test'+'.png', fig=None)

	tf.train.write_graph(pn.sess.graph.as_graph_def(), dst, 'testModel.pbtxt')#LPadd 26/12
	testOutDict={
		'phasePredice':testphase,
		'centerPose':centerPose,
		'wMean':mean,
		'wSD':sd
	}
	pickle.dump(testOutDict,open(dst+'testOutDict.pkl','wb'))
	print('testOutDict.pkl saved')

#import plotly
#import plotly.plotly as py
import chart_studio
import chart_studio.plotly as py
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
#plotly.offline.plot(dataNow, filename=dst+'phaseAgainstTime_test.html')


		

