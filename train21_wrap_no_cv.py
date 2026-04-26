## Import realated libs

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


import numpy as np
import pandas as pd
import time

import random
import os, sys
import glob
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import csv

from PhaseNetworkClassFlexible import PhaseNetwork
import util as util
import pipeline as pipeline


DATA_DIR = "../data/Training_data/pkl/walking_left/LeftFoot"
DATA_ROOT = "../data"
print(len(glob.glob(os.path.join(DATA_DIR, "*.pkl"))))

# EndSessionPlusOne = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,95,100,105,110,115,120,125,130,136,141,146,151,156,162,167,172,179]
EndSessionPlusOne = []
with open(os.path.join(DATA_ROOT, 'sessSameRatEnd_plusone_1.csv')) as csvfile2:
	readCSV2 = csv.reader(csvfile2, delimiter=',')
	EndRatSessionPlusOne = []
	end_session_dict = {}
	for i, row2 in enumerate(readCSV2):
		sessSameRatEnd_plusone = row2[0]
		EndRatSessionPlusOne.append(int(sessSameRatEnd_plusone))
		end_session_dict[i] = int(sessSameRatEnd_plusone)



dataset = pipeline.GaitPhasingDataset(
    data_root=DATA_ROOT,
    data_dir=DATA_DIR,
    img_dir=DATA_DIR,
    end_session_plus_one_path="../data/EndSessionPlusOne.csv",
    windowWing=3,
    regressWing=3,
    predictionGap=1,
    rotationMatrixList=None,
    meanForPhaseExtraction=None,
    sdForPhaseExtraction=None,
)


gpuID=-1
crossValidateNo=-1
system="walking_left"
exercise="walking"
SubjectAmount = dataset.SubjectAmount
startNewTraining = True
sessionFile="../data/test_new_pipe/" + f"{dataset.trainName}phaseModel.ckpt"
imageFolder = "../data/test_new_pipe/img/"
imageFolderRat = "../data/test_new_pipe/img/rat/"


## dataset 
nowState = dataset.AllData["XY"]
nextState = dataset.AllData["XY2"]
sessionSegmentMatrix = dataset.AllData["SessionSegmentMatrix"]
centerPose = dataset.AllData["centerPose"]
srcColorList = dataset.AllData["srcColorList"]
srcColorList_rat = dataset.srcColorList_subject

nowState_img = dataset.AllData_img["XY"]
nextState_img = dataset.AllData_img["XY2"]
sessionSegmentMatrix_img = dataset.AllData_img["SessionSegmentMatrix"]
centerPose_img = dataset.AllData_img["centerPose"]
srcColorList_img = dataset.AllData_img["srcColorList"]


EndNumberPlusOne = dataset.EndNumberPlusOne
EndSessionPlusOne = dataset.EndSessionPlusOne


# Training Args
repeat = False


pn=PhaseNetwork(2, dataset.windowWing, gpuID)


startTime=time.time()


print('start timing:')

with pn.sess.graph.as_default():
	print("Start session")
	saver = tf.train.Saver()
	
	
	while True:
		
		if(startNewTraining):
			
			print("Model Initiation ...")
			pn.sess.run(pn.initializer)
			print("Model Initiation Done!")

			
			
		else:
			saver.restore(pn.sess, sessionFile)
		print("Start Feeding")
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
		for ifig in range(0,SubjectAmount):
			fig.append(plt.figure(figsize=(8, 8)))
		figRatF1=[]
		for ifigRatF1 in range(0,SubjectAmount):
			figRatF1.append(plt.figure(figsize=(8, 8)))
		figRatF2=[]
		for ifigRatF2 in range(0,SubjectAmount):
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
				
				util.saveMarginPlot(prePhase_distribution[:,0:EndNumberPlusOne[0]],phaseXY_distribution[:,0:EndNumberPlusOne[0]],prePhaseMargin[:,0:EndNumberPlusOne[0]],imageFolderRat+'Rat1_prephase_'+str(step)+'.png',fig[0], srcColorList_rat[0])
				
				for ipic in range(1,SubjectAmount):
					util.saveMarginPlot(prePhase_distribution[:,EndNumberPlusOne[ipic-1]:EndNumberPlusOne[ipic]],phaseXY_distribution[:,EndNumberPlusOne[ipic-1]:EndNumberPlusOne[ipic]],prePhaseMargin[:,EndNumberPlusOne[ipic-1]:EndNumberPlusOne[ipic]],imageFolderRat+'Rat'+str(ipic+1)+'_prephase_'+str(step)+'.png',fig[ipic], srcColorList_rat[ipic])

				#util.saveScatterPlot(phaseXY_distribution,imageFolder+str(step)+'.png',fig)

				#LP
				phaseRad,prePhase=pn.sess.run([pn.phaseRad,pn.prePhase],feed_dict={pn.inputPlace:nowState})
				phase=phaseRad%(2*np.pi)	#this is cleaner 
				#print("phase:"+str(phase[1:4755]))

				phaseRat=[]
				phaseRat.append(phaseRad[0:EndNumberPlusOne[0]]%(2*np.pi))
				for iphaseRat in range(1,SubjectAmount):
					phaseRat.append(phaseRad[EndNumberPlusOne[iphaseRat-1]:EndNumberPlusOne[iphaseRat]]%(2*np.pi))

				util.saveRatPhaseRainbowPlot(centerPose_img[:,0:EndNumberPlusOne[0]],phaseRat[0], imageFolderRat+'Trajec_Rat1_F1_'+str(step)+'.png', figRatF1[0])
# 				util.saveRatPhaseRainbowPlot2(centerPose_img[:,0:RatEndNumberPlusOne[0]],phaseRat[0], imageFolderRat+'Trajec_Rat1_F2_'+str(step)+'.png', figRatF2[0])
				for iTrajec in range(1,SubjectAmount):
 					util.saveRatPhaseRainbowPlot(centerPose_img[:,EndNumberPlusOne[iTrajec-1]:EndNumberPlusOne[iTrajec]],phaseRat[iTrajec], imageFolderRat+'Trajec_Rat'+str(iTrajec+1)+'_F1_'+str(step)+'.png', figRatF1[iTrajec])
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
# print(Nameofdata)