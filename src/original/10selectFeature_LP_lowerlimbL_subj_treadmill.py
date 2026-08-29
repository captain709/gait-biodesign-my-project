#objective
#read ExportSessionProto and dump as numpy data
#copy the whole structure inside d0ExportSessionProto

#import FeatureExport_pb2 as pb
import numpy as np
import math
import glob
import os
import csv
#import os.path
import pickle
import re
#from readcsv import reader, Relative_Forelimb_im1_x, Relative_Forelimb_im1_y, Relative_Forelimb_im2_x, Relative_Forelimb_im2_y, Relative_Hindlimb_im1_x, Relative_Hindlimb_im1_y, Relative_Hindlimb_im2_x, Relative_Hindlimb_im2_y
#import csv

#import setting
gofolder=os.path.join('Subj42','csvOP')#d0ExportSessionProto_recheck
dstfolder1=os.path.join('Subj42',os.path.join('d1pkl','walking_left'))
dstfolder2=os.path.join('Subj42',os.path.join('d1pkl','walking_right'))
#Ratfile=list(range(1, 45))#1117#1605
count = 0
dir_path = 'Subj42\csvOP'
#x[0] = 0
#y[0] = 0
#x[1] = 0
#y[1] = 0
print('here')
print(len([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))]))
stack = []
for path in os.scandir(dir_path):
    if path.is_file():
        count += 1
        #print(path.name)
        collectnumber=path.name
        #numb=[int(s) for s in collectnumber if s.isdigit()]
        #print(numb)
        #collect = re.findall('[0-9]+', collectnumber)
        #print(collectnumber)
        #print(int(collectnumber))
        #print(x[0])
        #print(y[0])
        #if int(collectnumber) < x[1]
        #    x = collectnumber
        #if int(collectnumber) > y[1]
        #    y = collectnumber
        #z = re.findall('[0-9]+',collectnumber)
        z = re.findall(r'T(\d+)',collectnumber)
        stack.append(z)
        #print(stack)
        if (count==1):
            x = re.findall('[0-9]+', collectnumber)
        #else:
            #y = re.findall('[0-9]+', collectnumber)
        #print(int(x[1]))
aList = list(stack)
#print(type(aList))
#print(aList)
flat_list = [item for sublist in aList for item in sublist]
res = [eval(i) for i in flat_list]
#print(res)
mini=min(res)
maxi=max(res)
#print(mini)
#print(maxi)
#print('*************')
#print(collect)
#print(int(x[1]),int(y[1]))
print('file count:', count/2)
endf=int(count/2)
#Ratfile=list(range(1, endf))#1117#1605
Ratfile = list(range(mini, maxi+1))

exercise = "walking_right"
for system in ["Subj42"]:#,"walking_forelimb"#"walking_hindlimb"

	#if(system=="01kinectAlone"):
	#	import setting01 as setting
	#elif(system=="02kinect+"):
	#	import setting02 as setting
	#elif(system=="03mocap"):
	#	import setting02 as setting
	if(exercise=="walking_left"):
		import setting06 as setting
	elif(exercise=="walking_right"):
		import setting06 as setting

	src=system+"\\csvOP\\"
	dst=system+"\\d1pkl\\"

	os.makedirs(dst,exist_ok=True)

	D=(setting.sphereCount*3+setting.scalarCount)+1
	#targetFiles=glob.glob(src+'**/*.ExportSessionProto', recursive=True)
	#targetFiles=glob.glob(src+'**/*.csv', recursive=True)


	for iRat in Ratfile:

		f1 = open(os.path.join(gofolder,'newRSubj42T'+str(iRat)+'.csv'), "rt")#new4RF1_norm_rat#Rat5_relativePositionForelimb1_norm_
		f2 = open(os.path.join(gofolder,'newLSubj42T'+str(iRat)+'.csv'), "rt")#new4RF2_norm_rat#Rat5_relativePositionForelimb2_norm_


		reader_Forelimb_im1 = csv.reader(f1)
		reader_Forelimb_im2 = csv.reader(f2)


		Relative_Forelimb_im1_x = []
		Relative_Forelimb_im1_y = []

		Relative_Forelimb_im2_x = []
		Relative_Forelimb_im2_y = []


		included_cols_Forelimb_im1_x = [0]
		included_cols_Forelimb_im1_y = [1]
		included_cols_Forelimb_im2_x = [0]
		included_cols_Forelimb_im2_y = [1]


		ans4 = np.zeros((2,len(Relative_Forelimb_im1_x)))

		
		for rowForelimbIm1 in reader_Forelimb_im1:
			Relative_Forelimb_im1_x.append(list(rowForelimbIm1[i] for i in included_cols_Forelimb_im1_x))
			Relative_Forelimb_im1_y.append(list(rowForelimbIm1[i] for i in included_cols_Forelimb_im1_y))
		for rowForelimbIm2 in reader_Forelimb_im2:
			Relative_Forelimb_im2_x.append(list(rowForelimbIm2[i] for i in included_cols_Forelimb_im2_x))
			Relative_Forelimb_im2_y.append(list(rowForelimbIm2[i] for i in included_cols_Forelimb_im2_y))

		if(exercise=="walking_left"):

			Relative_Forelimb_im2_x=np.reshape(Relative_Forelimb_im2_x,[-1])
			Relative_Forelimb_im2_y=np.reshape(Relative_Forelimb_im2_y,[-1])


			ans4 = np.vstack((Relative_Forelimb_im2_x,Relative_Forelimb_im2_y))

			print(ans4)

		else:
			Relative_Forelimb_im1_x=np.reshape(Relative_Forelimb_im1_x,[-1])
			Relative_Forelimb_im1_y=np.reshape(Relative_Forelimb_im1_y,[-1])


			ans4 = np.vstack((Relative_Forelimb_im1_x,Relative_Forelimb_im1_y))

			print(ans4)
		if(exercise=="walking_left"):
			with open(os.path.join(dstfolder1,('rat'+str(iRat)+'_walking_TLL_left.pkl')), 'wb') as fpickle1:#with open(os.path.join(dstfolder1,('rat'+str(iRat)+'_walking_normal_hindlimb.pkl')), 'wb') as fpickle1:
				#print("pklfile1")
				d={
				'data': ans4,
				}
				pklFile=pickle.dump(d,fpickle1,pickle.HIGHEST_PROTOCOL)      
		else:
			with open(os.path.join(dstfolder2,('rat'+str(iRat)+'_walking_TLL_right.pkl')), 'wb') as fpickle2:#with open(os.path.join(dstfolder2,('rat'+str(iRat)+'_walking_normal_forelimb.pkl')), 'wb') as fpickle2:
				#print("pklfile2")
				d={
				'data': ans4,
				}
				pklFile=pickle.dump(d,fpickle2,pickle.HIGHEST_PROTOCOL)  




print('done')


	