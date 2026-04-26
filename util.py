import random
import math
import numpy as np
from random import randint
from copy import deepcopy
import os
import pandas as pd
import matplotlib
# %matplotlib notebook
if os.name != 'nt':
	matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

import pickle

from scipy.signal import hilbert
import scipy.signal as signal
import scipy.stats as stats
from scipy.signal import find_peaks
from mpl_toolkits import mplot3d

def detect_peaks(data, threshold=None, distance=5):
    """
    Detect peaks in the data using scipy's find_peaks and dynamically choose threshold if not provided.
    
    Parameters:
    data (numpy array): The input data to detect peaks from.
    threshold (float): The minimum height required for a peak to be detected.
    distance (int): Minimum horizontal distance between neighboring peaks.
    
    Returns:
    peaks (numpy array): The indices of the peaks in the data.
    properties (dict): Properties of the detected peaks, like heights.
    """
    # If no threshold is provided, dynamically set it based on data distribution (mean + std)
    if threshold is None:
        threshold = np.mean(data) + np.std(data)
    
    # Detect peaks
    peaks, properties = signal.find_peaks(data, height=threshold, distance=distance)
    
    return peaks, properties


def choose_threshold(data, method="mean_std", percentile=90):
    """
    Dynamically choose a threshold based on the data and the method provided.

    Parameters:
    data (numpy array): The input data to determine the threshold.
    method (str): The method to choose the threshold, options are:
                  - "mean_std" (mean + standard deviation)
                  - "mad" (Median Absolute Deviation)
                  - "percentile" (based on a specified percentile)
                  - "otsu" (Otsu's method for bimodal distributions)
    percentile (int): Used if method is "percentile". Default is 90.
    
    Returns:
    threshold (float): The chosen threshold.
    """
    if method == "mean_std":
        threshold = np.mean(data) + np.std(data)
    elif method == "mad":
        median = np.median(data)
        mad = np.median(np.abs(data - median))  # Median Absolute Deviation
        threshold = median + 3 * mad  # Adjust the multiplier as needed
    elif method == "percentile":
        threshold = np.percentile(data, percentile)
    elif method == "otsu":
        hist, bin_edges = np.histogram(data, bins=256)
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]
        mean1 = np.cumsum(hist * bin_mids) / weight1
        mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
        inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        index_of_max_variance = np.argmax(inter_class_variance)
        threshold = bin_mids[:-1][index_of_max_variance]
    else:
        raise ValueError("Invalid method. Choose from 'mean_std', 'mad', 'percentile', 'otsu'.")
    
    return threshold


def detect_peaks(data, threshold=None, distance=5, threshold_method="mean_std", percentile=90):
    """
    Detect peaks in the data using scipy's find_peaks and dynamically choose threshold if not provided.
    
    Parameters:
    data (numpy array): The input data to detect peaks from.
    threshold (float): The minimum height required for a peak to be detected.
    distance (int): Minimum horizontal distance between neighboring peaks.
    threshold_method (str): The method to choose the threshold ('mean_std', 'mad', 'percentile', 'otsu').
    percentile (int): Percentile used if threshold_method is 'percentile'.
    
    Returns:
    peaks (numpy array): The indices of the peaks in the data.
    properties (dict): Properties of the detected peaks, like heights.
    """
    # If no threshold is provided, dynamically select it based on the chosen method
    if threshold is None:
        threshold = choose_threshold(data, method=threshold_method, percentile=percentile)
    
    # Detect peaks
    peaks, properties = signal.find_peaks(data, height=threshold, distance=distance)
    
    return peaks, properties


def get_between_peak(x: np.array, peaks, return_n_peak=False):

    cycle_list = []


    for count_peak, peak in enumerate(peaks):

        if count_peak == 0:

            cycle = x[0: peaks[count_peak]]

        elif count_peak == len(peaks) - 1:

            cycle = x[peaks[count_peak]: ]

        else:
            cycle = x[peaks[count_peak - 1]: peaks[count_peak]]


        # appending the signal to list of cycles

        cycle_list.append(cycle)

    if return_n_peak:

        return cycle_list, len(peaks)
    
    else:

        return cycle_list


def interpolate_signal_window(input_signal, target_length):
    """
    Interpolates a signal window to a specified length.

    Parameters:
    input_signal (numpy.ndarray): The input signal window to be interpolated.
    target_length (int): The desired length of the interpolated signal.

    Returns:
    numpy.ndarray: The interpolated signal with the specified length.
    """
    original_length = len(input_signal)
    if original_length == target_length:
        return input_signal

    # Original indices
    original_indices = np.arange(original_length)

    # Target indices
    target_indices = np.linspace(0, original_length - 1, target_length)

    # Interpolate using numpy.interp
    interpolated_signal = np.interp(target_indices, original_indices, input_signal)

    return interpolated_signal

def hilbert_transform_with_phase_color(data):

    # fs = 1000  # Sampling frequency (Hz)
    # duration = 1.0  # Duration of the signal (seconds)
    # t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Generate a test signal (e.g., a sine wave with frequency modulation)
    # carrier_freq = 50  # Carrier frequency (Hz)
    # mod_freq = 5  # Modulation frequency (Hz)
    # mod_index = 0.5  # Modulation index

    # Modulated signal
    # signal = np.sin(2 * np.pi * carrier_freq * t + mod_index * np.sin(2 * np.pi * mod_freq * t))

    # Compute the analytic signal using the Hilbert transform
    analytic_signal = signal.hilbert(data)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))

    # Reduce phase to the range [0, 2π)
    instantaneous_phase_mod_2pi = np.mod(instantaneous_phase, 2 * np.pi)

    # Instantaneous frequency
    # instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs

    # colorized
    # bins = np.linspace(0, 2* np.pi, 8)
    # color_index = np.digitize(instantaneous_phase_mod_2pi, bins)
    # colorized = [UNI_PALETTE[color - 1] for color in color_index]
    colorized = instantaneous_phase / np.pi

    return instantaneous_phase, colorized


def phase_sync_index(x, y):

    phase_x = np.unwrap(np.angle(signal.hilbert(x)))
    phase_y = np.unwrap(np.angle(signal.hilbert(y)))

    return np.abs(np.mean(np.exp(1j  * (phase_x-phase_y))))

def mean_phase_difference(x, y):
    
    phase_x = np.unwrap(np.angle(signal.hilbert(x)))
    phase_y = np.unwrap(np.angle(signal.hilbert(y)))
    
    return np.mean(phase_x-phase_y)

def phase_sync_corr(df):
    """
    Compute a matrix of phase synchronization indices between each pair of columns in df.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the signals (each column is a signal).
    
    Returns:
        pd.DataFrame: A symmetric DataFrame where each element (i,j) is the phase 
                      synchronization index between column i and column j.
    """
    cols = df.columns
    ps_index_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    
    for col1 in cols:
        for col2 in cols:
            ps_index_matrix.loc[col1, col2] = phase_sync_index(df[col1].values, df[col2].values)
    
    return ps_index_matrix

def phase_diff_corr(df):
    """
    Compute a matrix of phase synchronization indices between each pair of columns in df.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the signals (each column is a signal).
    
    Returns:
        pd.DataFrame: A symmetric DataFrame where each element (i,j) is the phase 
                      synchronization index between column i and column j.
    """
    cols = df.columns
    ps_index_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    
    for col1 in cols:
        for col2 in cols:
            ps_index_matrix.loc[col1, col2] = mean_phase_difference(df[col1].values, df[col2].values)
    
    return ps_index_matrix



def expandScale(scale,sphereCount):
	ans=np.zeros([scale.shape[0]+sphereCount*2,1])
	for s in range(sphereCount):
		ans[s*3+0,0]=scale[s]
		ans[s*3+1,0]=scale[s]
		ans[s*3+2,0]=scale[s]
	
	ans[sphereCount*3:,0]=scale[sphereCount:,0]

	return ans

def projectToPlane(planeNormal:np.ndarray,vector:np.ndarray,keepSize=False):
	unitPlaneNormal=planeNormal/np.linalg.norm(planeNormal,axis=0,keepdims=True)	#normalize to a unit vector
	toBeRemoved = unitPlaneNormal*(np.sum(unitPlaneNormal*vector,axis=0,keepdims=True))
	projection=vector-toBeRemoved
	if keepSize:
		originalSize=np.linalg.norm(vector,axis=0,keepdims=True)
		return projection/np.linalg.norm(projection,axis=0,keepdims=True)*originalSize	#more like rotation to the plane
	else:
		return projection #simple projection to the plane


def regressForSlope(y):	#only odd number
	l=len(y)
	#x is generated (assume equal time interval), zero at the center, meanX=0
	#x is something like [-3,-2,-1,0,1,2,3]
	x=np.array(range(l))-(l-1)/2
	avgY=np.average(y)
	'''	
	if(u==1):
		print("avgY:"+str(avgY))
		print("np.dot(y-avgY,x)/np.dot(x,x)"+str(np.dot(y-avgY,x)/np.dot(x,x)))
	'''
	
	return np.dot(y-avgY,x)/np.dot(x,x)

#follow Andrew's convention
def getRegressHeadingDirectionWithSphere(x, sphereCount, regressWing=3):	#each row = 1 sequence in one dimension
	regressLength=regressWing*2+1
	D,W = x.shape
	r=np.zeros([D,W-regressWing*2])
	#u=1
	for d in range(D):
		for i in range(W-regressWing*2):
			r[d,i]=regressForSlope(x[d,i:i+regressLength])
			#u=2
			'''
			if(u==1):
				print("x[d,i:i+regressLength]")
				print(x[d,i:i+regressLength])
				print("r[d,i]")
				print(r[d,i])
				u=2
				'''
	#correct sphere dimensions by rotate them to be orthogonal to the current radius
	#these are not unit sphere
	for s in range(sphereCount):
		for i in range(W-regressWing*2):
			midPoint=x[s*3:s*3+3,[i+regressWing]]
			heading=r[s*3:s*3+3,[i]]
			r[s*3:s*3+3,[i]]=projectToPlane(midPoint,heading,keepSize=True)
	
	#normalization
	#uu=1
	for i in range(W-regressWing*2):
		#print("r[:,i]"+str(r[:,i]))
		#print("np.linalg.norm(r[:,i])"+str(np.linalg.norm(r[:,i])))
		r[:,i]=r[:,i]/np.linalg.norm(r[:,i])
		'''
		if(uu==1):
			#print("W"+str(W)+"i"+str(i)+"W-regressWing*2"+str(W-regressWing*2))
			print("r[:,i]/np.linalg.norm(r[:,i])"+str(r[:,i]))
			uu=2
		'''	
			

	return r

#follow Andrew's convention
def getRegressHeadingDirection(x, regressWing=3):	#each row = 1 sequence in one dimension
	regressLength=regressWing*2+1
	D,W = x.shape
	r=np.zeros([D,W-regressWing*2])
	#regress each individual dimension
	for d in range(D):
		for i in range(W-regressWing*2):
			r[d,i]=regressForSlope(x[d,i:i+regressLength])
	
	#normalization
	for i in range(W-regressWing*2):
		r[:,i]=r[:,i]/np.linalg.norm(r[:,i])

	return r

#follow Andrew's convention, not tested
def addOrthogonalNoiseNDWithSphere(currentPatches,nextPatches,velocityPatches,sphereCount,radius,minSD=0.04,repeat=1):
	#first, add noise in all direction. Then, project it to a specific hyperplane
	#preLength=np.linalg.norm(currentPatches-previousPatches,axis=1)
	postLength=np.linalg.norm(nextPatches-currentPatches,axis=0,keepdims=True)
	noiseSize=postLength/2	
	#noiseSize=0.5*np.minimum(preLength,postLength)	#1SD of noise 
												#(if it move fast, noise can be large. if move slow, noise should be small)
												#just don't want the moving path to get overwhelmed by these noise

	noiseSize=np.maximum(noiseSize,minSD)	#noise standard deviation cannot be too small


	noisyPoseList=[]
	noiseDirectionList=[]
	noiseMagnitudeList=[]

	print('generating orthogonal noise...')
	for i in range(repeat):
		#print('round '+str(i)+'/'+str(repeat))
		#noise = np.random.normal(0,1,size=currentPatches.shape)*noiseSize	#noise in all direction
		noise = np.random.randn(currentPatches.shape[0],currentPatches.shape[1])*noiseSize	#noise in all direction

		#now must project this noise into hyperplanes
		#first, get unit normal vector
		#jumpVector=nextPatches-previousPatches
		jumpVector=velocityPatches
		normalVector=jumpVector/np.linalg.norm(jumpVector,axis=0,keepdims=True)

		#second, project noise to that unit normal vector
		toBeRemoved=np.sum(noise*normalVector,axis=0,keepdims=True)*normalVector
		projectedNoise=noise-toBeRemoved	

		#the projectedNoise need spherical correction (currentPatches+projectedNoise must be on sphere)
		noisyPose=currentPatches+projectedNoise
		surfaceDistance=np.zeros([sphereCount,currentPatches.shape[1]])
		for s in range(sphereCount):
			noisyPose[s*3:s*3+3,:]*=radius[s]/np.linalg.norm(noisyPose[s*3:s*3+3,:],axis=0)	#project them to sphere surface

			
			#noiseMagnitude must calculate with distance on a sphere surface as one dimension
			
			#first, find spherical surface distance from currentPatches to noisyPose
			unitNoisyPose=noisyPose[s*3:s*3+3,:]/radius[s]
			unitOriginal=currentPatches[s*3:s*3+3,:]/radius[s]
			dotProduct=np.sum(unitNoisyPose*unitOriginal,axis=0,keepdims=True)
			angle=np.arccos(np.minimum(1,np.maximum(-1,dotProduct)))	#distance in a unit sphere
			surfaceDistance[s,:]=radius[s]*angle

		otherDistance=projectedNoise[sphereCount*3:,:]

		mixedDistance=np.vstack([surfaceDistance,otherDistance])	#one row for one sphere (surface distance, positive only), one row for one scalar dimension (can be negative) 
		noiseMagnitude=np.linalg.norm(mixedDistance,axis=0,keepdims=True)
		unitMixedDistance=mixedDistance/np.linalg.norm(mixedDistance,axis=0,keepdims=True)

		#noiseDirection must be tangent to the sphere, at the noisy spot, and point away from the original point, and has unit norm
		roughNoiseDirection=noisyPose-currentPatches
		noiseDirection=np.zeros(noisyPose.shape)
		for s in range(sphereCount):
			tangentDirection=projectToPlane(currentPatches[s*3:s*3+3,:],roughNoiseDirection[s*3:s*3+3,:],keepSize=False)	#just want direction
			tangentDirection/=np.linalg.norm(tangentDirection,axis=0,keepdims=True)
			noiseDirection[s*3:s*3+3,:]=tangentDirection*unitMixedDistance[[s],:]
		noiseDirection[sphereCount*3:,:]=unitMixedDistance[sphereCount:,:]

		noisyPoseList.append(noisyPose)			  #done (not tested yet)
		noiseDirectionList.append(noiseDirection) #done (not tested yet)
		noiseMagnitudeList.append(noiseMagnitude) #done (not tested yet)

	return np.hstack(noisyPoseList), np.hstack(noiseDirectionList), np.hstack(noiseMagnitudeList)
	#return currentPatches+projectedNoise, noiseDirection, noiseMagnitude

#follow Andrew's convention, not tested
def addHeadNoiseNDWithSphere(nowPose,nowHead,sphereCount,noiseSD=0.5,repeat=1):	#sd=0.5 is about 30 degrees deviation
	
	noisyHeadList=[]
	noiseDirectionList=[]
	noiseMagnitudeList=[]
	for i in range(repeat):
		#in this high dimensional case, I add noise to the unit direction, and project it back to a hypersphere using normalization
		noise = np.random.randn(nowHead.shape[0],nowHead.shape[1])*noiseSD
		tmp = nowHead+noise 	#add random noise in all direction
		noisyPatches = tmp/np.linalg.norm(tmp,axis=0,keepdims=True)	#projected back to the unit hypersphere

		#fix heading direction of spherical dimension (must be tangent to sphere surface)
		for s in range(sphereCount):
			noisyPatches[s*3:s*3+3,:]=projectToPlane(nowPose[s*3:s*3+3,:],noisyPatches[s*3:s*3+3,:],keepSize=True)


		#direction of noise in heading direction must be orthogonal to the noisy unit vector
		nonOrthoVec = noisyPatches-nowHead
		#project nonOrthoVec to the direction of noisyPatches
		toBeRemoved = np.sum(nonOrthoVec*noisyPatches,axis=0,keepdims=True)*noisyPatches #a short vector in noisyPatches direction
		orthoVec = nonOrthoVec-toBeRemoved

		noiseDirection = orthoVec/np.linalg.norm(orthoVec,axis=0,keepdims=True)

		noisyHeadList.append(noisyPatches)
		noiseDirectionList.append(noiseDirection)
		noiseMagnitudeList.append(np.linalg.norm(nonOrthoVec,axis=0,keepdims=True))

	return np.hstack(noisyHeadList), np.hstack(noiseDirectionList), np.hstack(noiseMagnitudeList) 

def randomOuterPoseWithSphere(sphereCount,radius,scalarRange,count):
	#scalarRange does not contain sphere-related dimension

	outerPose=np.zeros([sphereCount*3+scalarRange.shape[0],count])

	#random possible pose
	#spherical data need Gaussian random and normalize to sphere
	for s in range(sphereCount):
		r=np.random.randn(3,count)
		r=r/np.linalg.norm(r,axis=0,keepdims=True)*radius[s]
		outerPose[s*3:s*3+3,:]=r
	
	#scalar dimension need uniform random
	outerPose[sphereCount*3:,:]=np.random.rand(scalarRange.shape[0],count)*(scalarRange[:,[1]]-scalarRange[:,[0]])+scalarRange[:,[0]]

	return outerPose

def searchForNearestNeighborAndDistance(q,nowPose,sphereCount,radius):
	count=q.shape[1]
	#search for the nearest neighbor
	nearest=np.zeros([nowPose.shape[0],count])
	distance=np.zeros([1,count])
	for i in range(count):
		p=q[:,[i]]
		sumSquareDistance=np.zeros([1,nowPose.shape[1]])	#to compare
		for s in range(sphereCount):
			dot=np.sum(p[s*3:s*3+3,:]*nowPose[s*3:s*3+3,:],axis=0,keepdims=True)/(radius[s]*radius[s])
			angle=np.arccos(np.maximum(-1,np.minimum(1,dot)))
			sumSquareDistance+=np.square(angle*radius[s])
		sumSquareDistance+=np.sum(np.square(p[sphereCount*3:,:]-nowPose[sphereCount*3:,:]),axis=0,keepdims=True)
	
		#hypothesis: with 1 nearest neighbor, it might be too sensive to outliers (average of 10 nearest neighbor might be better)
		#however, if the outliers is not too far, the dense denoising strategy nearby the collected data will do the job
		minIndex=np.argmin(sumSquareDistance)
		nearest[:,[i]]=nowPose[:,[minIndex]]
		distance[0,i]=np.sqrt(sumSquareDistance[0,minIndex])
	
	return nearest,distance

def randomOuterPoseAndExpectedGradientWithSphere(nowPose,sphereCount,radius,scalarRange,count,kNearest=25):
	#scalarRange does not contain sphere-related dimension

	outerPose=np.zeros([nowPose.shape[0],count])

	#random possible pose
	#spherical data need Gaussian random and normalize to sphere
	for s in range(sphereCount):
		r=np.random.randn(3,count)
		r=r/np.linalg.norm(r,axis=0,keepdims=True)*radius[s]
		outerPose[s*3:s*3+3,:]=r
	
	#scalar dimension need uniform random
	outerPose[sphereCount*3:,:]=np.random.rand(scalarRange.shape[0],count)*(scalarRange[:,[1]]-scalarRange[:,[0]])+scalarRange[:,[0]]

	#search for the nearest neighbor
	target=np.zeros([nowPose.shape[0],count])
	for i in range(count):
		p=outerPose[:,[i]]
		sumSquareDistance=np.zeros([1,nowPose.shape[1]])	#to compare
		for s in range(sphereCount):
			dot=np.sum(p[s*3:s*3+3,:]*nowPose[s*3:s*3+3,:],axis=0,keepdims=True)/(radius[s]*radius[s])
			angle=np.arccos(np.maximum(-1,np.minimum(1,dot)))
			sumSquareDistance+=np.square(angle*radius[s])
		sumSquareDistance+=np.sum(np.square(p[sphereCount*3:,:]-nowPose[sphereCount*3:,:]),axis=0,keepdims=True)
	
		#hypothesis: with 1 nearest neighbor, it might be too sensive to outliers (average of 20 nearest neighbor might be better)
		#however, if the outliers is not too far, the dense denoising strategy nearby the collected data will do the job
		if kNearest<=1:
			target[:,[i]]=nowPose[:,[np.argmin(sumSquareDistance)]]	#one nearest neighbor is not good enough
		else:
			kNearestIndex=np.argpartition(sumSquareDistance.flatten(),kNearest)[0:kNearest]
			kNearestPose=nowPose[:,kNearestIndex]
			#get average from kNearestPose (and must stay inside the possible space)
			targetTemp=np.mean(kNearestPose,axis=1,keepdims=True)
			for s in range(sphereCount):
				targetTemp[s*3:s*3+3,:]=radius[s]*targetTemp[s*3:s*3+3,:]/np.linalg.norm(targetTemp[s*3:s*3+3,:])

			target[:,[i]]=targetTemp

	#calculate expected gradient (point away from the nearest neighbor)
	# outerPose -> target
	surfaceDistance=np.zeros([sphereCount,count])
	for s in range(sphereCount):
		dot=np.sum(outerPose[s*3:s*3+3,:]*target[s*3:s*3+3,:],axis=0,keepdims=True)/(radius[s]*radius[s])
		angle=np.arccos(np.maximum(-1,np.minimum(1,dot)))
		surfaceDistance[s,:]=angle*radius[s]

	otherDistance=outerPose[sphereCount*3:,:]-target[sphereCount*3:,:]		#critical bug fixed
	
	mixedDistance = np.vstack([surfaceDistance,otherDistance])
	unitMixedDistance = mixedDistance/np.linalg.norm(mixedDistance,axis=0,keepdims=True)

	#noiseDirection must be tangent to the sphere, at the noisy spot, and point away from the original point, and has unit norm
	roughDirection=outerPose-target
	expectedGradient=np.zeros(outerPose.shape)
	for s in range(sphereCount):
		tangentDirection=projectToPlane(outerPose[s*3:s*3+3,:],roughDirection[s*3:s*3+3,:],keepSize=False)	#just want direction
		tangentDirection/=np.linalg.norm(tangentDirection,axis=0,keepdims=True)
		expectedGradient[s*3:s*3+3,:]=tangentDirection*unitMixedDistance[[s],:]
	expectedGradient[sphereCount*3:,:]=unitMixedDistance[sphereCount:,:]

	return outerPose,expectedGradient

def randomOuterPoseAndExpectedGradientWithSphere_farAndNear(nowPose,sphereCount,radius,scalarRange,count,kNearest=25,includeNearSide=True,nearRatio=1.0):
	#nearRatio=1.0 is adding point right at the target
	#scalarRange does not contain sphere-related dimension
	print('preparing outerPose data...')
	outerPose=np.zeros([nowPose.shape[0],count])

	#random possible pose
	#spherical data need Gaussian random and normalize to sphere
	print('random points...')
	for s in range(sphereCount):
		r=np.random.randn(3,count)
		r=r/np.linalg.norm(r,axis=0,keepdims=True)*radius[s]
		outerPose[s*3:s*3+3,:]=r
	
	#scalar dimension need uniform random
	outerPose[sphereCount*3:,:]=np.random.rand(scalarRange.shape[0],count)*(scalarRange[:,[1]]-scalarRange[:,[0]])+scalarRange[:,[0]]

	#search for the nearest neighbor
	print('nearest neighbor search...')
	target=np.zeros([nowPose.shape[0],count])
	beforeTarget=np.zeros(target.shape)
	for i in range(count):
		print(str(i)+'/'+str(count),end='\r')
		p=outerPose[:,[i]]
		sumSquareDistance=np.zeros([1,nowPose.shape[1]])	#to compare
		for s in range(sphereCount):
			dot=np.sum(p[s*3:s*3+3,:]*nowPose[s*3:s*3+3,:],axis=0,keepdims=True)/(radius[s]*radius[s])
			angle=np.arccos(np.maximum(-1,np.minimum(1,dot)))
			sumSquareDistance+=np.square(angle*radius[s])
		sumSquareDistance+=np.sum(np.square(p[sphereCount*3:,:]-nowPose[sphereCount*3:,:]),axis=0,keepdims=True)
	
		#hypothesis: with 1 nearest neighbor, it might be too sensive to outliers (average of 20 nearest neighbor might be better)
		#however, if the outliers is not too far, the dense denoising strategy nearby the collected data will do the job
		if kNearest<=1:
			target[:,[i]]=nowPose[:,[np.argmin(sumSquareDistance)]]	#one nearest neighbor is not good enough
		else:
			kNearestIndex=np.argpartition(sumSquareDistance.flatten(),kNearest)[0:kNearest]
			kNearestPose=nowPose[:,kNearestIndex]
			#get average from kNearestPose (and must stay inside the possible space)
			targetTemp=np.mean(kNearestPose,axis=1,keepdims=True)
			for s in range(sphereCount):
				targetTemp[s*3:s*3+3,:]=radius[s]*targetTemp[s*3:s*3+3,:]/np.linalg.norm(targetTemp[s*3:s*3+3,:])

			target[:,[i]]=targetTemp

	#calculate expected gradient (point away from the nearest neighbor)
	print('calculate expected gradient...')
	# outerPose -> target
	surfaceDistance=np.zeros([sphereCount,count])
	for s in range(sphereCount):
		dot=np.sum(outerPose[s*3:s*3+3,:]*target[s*3:s*3+3,:],axis=0,keepdims=True)/(radius[s]*radius[s])
		angle=np.arccos(np.maximum(-1,np.minimum(1,dot)))
		surfaceDistance[s,:]=angle*radius[s]

	otherDistance=outerPose[sphereCount*3:,:]-target[sphereCount*3:,:]		#critical bug fixed
	
	mixedDistance = np.vstack([surfaceDistance,otherDistance])
	unitMixedDistance = mixedDistance/np.linalg.norm(mixedDistance,axis=0,keepdims=True)

	#noiseDirection must be tangent to the sphere, at the noisy spot, and point away from the original point, and has unit norm
	roughDirection=outerPose-target

	#on the remote side
	expectedGradient=np.zeros(outerPose.shape)
	for s in range(sphereCount):
		tangentDirection=projectToPlane(outerPose[s*3:s*3+3,:],roughDirection[s*3:s*3+3,:],keepSize=False)	#just want direction
		tangentDirection/=np.linalg.norm(tangentDirection,axis=0,keepdims=True)
		expectedGradient[s*3:s*3+3,:]=tangentDirection*unitMixedDistance[[s],:]
	expectedGradient[sphereCount*3:,:]=unitMixedDistance[sphereCount:,:]

	if includeNearSide:
		'''
		#can reuse roughDirection but project differently
		expectedGradientNear=np.zeros(outerPose.shape)
		for s in range(sphereCount):
			tangentDirection=projectToPlane(target[s*3:s*3+3,:],roughDirection[s*3:s*3+3,:],keepSize=False)
			tangentDirection/=np.linalg.norm(tangentDirection,axis=0,keepdims=True)
			expectedGradientNear[s*3:s*3+3,:]=tangentDirection*unitMixedDistance[[s],:]
		expectedGradientNear[sphereCount*3:,:]=unitMixedDistance[sphereCount:,:]

		return np.hstack([outerPose,target]),np.hstack([expectedGradient,expectedGradientNear])
		'''

		#use spherical interpolation between outerPose and target to get "beforeTarget point" 		
		for i in range(count):
			for s in range(sphereCount):
				#need rotation axis
				fullAngle=surfaceDistance[s,i]/radius[s]
				axisOfRotation=np.cross(outerPose[s*3:s*3+3,i],target[s*3:s*3+3,i])
				axisOfRotation/=np.linalg.norm(axisOfRotation)	
				#rotate currentPoint with axis and angle
				beforeTarget[s*3:s*3+3,i]=rotateAxisAngle(outerPose[s*3:s*3+3,i],axisOfRotation,fullAngle*nearRatio)
				#adjust it back to the correct radius (remove collective error)
				beforeTarget[s*3:s*3+3,i]=radius[s]*beforeTarget[s*3:s*3+3,i]/np.linalg.norm(beforeTarget[s*3:s*3+3,i])

		beforeTarget[sphereCount*3:,:]=nearRatio*target[sphereCount*3:,:]+(1-nearRatio)*outerPose[sphereCount*3:,:]	#linear interpolation for scalar value

		#they are equal
		#print(target)
		#print("\n==========\n")
		#print(beforeTarget)

		expectedGradientNear=np.zeros(outerPose.shape)
		for s in range(sphereCount):
			tangentDirection=projectToPlane(beforeTarget[s*3:s*3+3,:],roughDirection[s*3:s*3+3,:],keepSize=False)
			tangentDirection/=np.linalg.norm(tangentDirection,axis=0,keepdims=True)
			expectedGradientNear[s*3:s*3+3,:]=tangentDirection*unitMixedDistance[[s],:]

		expectedGradientNear[sphereCount*3:,:]=unitMixedDistance[sphereCount:,:]

		return np.hstack([outerPose,beforeTarget]),np.hstack([expectedGradient,expectedGradientNear])

	else:
		return outerPose,expectedGradient

def rotation_matrix(axis, angle):
	"""
	Return the rotation matrix associated with counterclockwise rotation about
	the given axis by theta radians.
	https://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
	"""
	axis = np.asarray(axis)
	axis = axis/math.sqrt(np.dot(axis, axis))
	a = math.cos(angle/2.0)
	b, c, d = -axis*math.sin(angle/2.0)
	aa, bb, cc, dd = a*a, b*b, c*c, d*d
	bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
	return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
					 [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
					 [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

#v = [3, 5, 0]
#axis = [4, 4, 1]
#theta = 1.2 

def rotateAxisAngle(vector, axis, angle):
	return np.dot(rotation_matrix(axis,angle), vector)

def easyMoveOnSphere(currentPoint,moveVector,sphereCount,scale,scalarBound=None):	#for one point
	#simply add and normalize
	currentPoint+=moveVector
	for s in range(sphereCount):
		currentPoint[s*3:s*3+3,:]=currentPoint[s*3:s*3+3,:]*scale[s]/np.linalg.norm(currentPoint[s*3:s*3+3,:])

	if scalarBound is not None:
		currentPoint[sphereCount*3:,:]=np.maximum(scalarBound[:,[0]],np.minimum(scalarBound[:,[1]],currentPoint[sphereCount*3:,:]))

	return currentPoint

def preciseMoveOnSphere(currentPoint,moveVector,sphereCount,radius,scalarBound=None):	#for one point
	#move on sphere
	for s in range(sphereCount):
		surfaceDistance=np.linalg.norm(moveVector[s*3:s*3+3,:])
		angle=surfaceDistance/radius[s]
		axisOfRotation=np.cross(currentPoint[s*3:s*3+3,0],moveVector[s*3:s*3+3,0])
		axisOfRotation/=np.linalg.norm(axisOfRotation)	
		#rotate currentPoint with axis and angle
		currentPoint[s*3:s*3+3,0]=rotateAxisAngle(currentPoint[s*3:s*3+3,0],axisOfRotation,angle)
		#adjust it back to the correct radius (remove collective error)
		currentPoint[s*3:s*3+3,:]=radius[s]*currentPoint[s*3:s*3+3,:]/np.linalg.norm(currentPoint[s*3:s*3+3,:])

	#move scalar normally
	currentPoint[sphereCount*3:,:]+=moveVector[sphereCount*3:,:]
	if scalarBound is not None:
		currentPoint[sphereCount*3:,:]=np.maximum(scalarBound[:,[0]],np.minimum(scalarBound[:,[1]],currentPoint[sphereCount*3:,:]))

	return currentPoint

def isInside(currentScalar,scalarRange):
	for i in range(currentScalar.shape[0]):
		if currentScalar[i,0]<scalarRange[i,0] or currentScalar[i,0]>scalarRange[i,1]:
			return False
	
	return True

def correctPose(currentPose, sphereCount, radius, scalarBound=None):
	for s in range(sphereCount):
		#adjust it back to the correct radius (remove collective error)
		currentPose[s*3:s*3+3,:]=radius[s]*currentPose[s*3:s*3+3,:]/np.linalg.norm(currentPose[s*3:s*3+3,:])
	
	if scalarBound is not None:
		currentPose[sphereCount*3:,:]=np.maximum(scalarBound[:,[0]],np.minimum(scalarBound[:,[1]],currentPose[sphereCount*3:,:]))

	return currentPose

def correctHead(currentHead, currentPose, sphereCount ):
	#fix heading direction of spherical dimension (must be tangent to sphere surface)
	for s in range(sphereCount):
		currentHead[s*3:s*3+3,:]=projectToPlane(currentPose[s*3:s*3+3,:],currentHead[s*3:s*3+3,:],keepSize=True)
	
	currentHead = currentHead/np.linalg.norm(currentHead,axis=0,keepdims=True)	#projected back to the unit hypersphere

	return currentHead

def plotSequence(yList):
	fig=plt.figure()
	for y in yList:
		plt.plot(np.arange(y.shape[0]),y)
	plt.show()

def saveScatterPlot(XY, filename, fig=None, bound=True):
	if(fig==None):
		fig=plt.figure(figsize=(8, 8))	#in inch

	plt.scatter(XY[0,:],XY[1,:],color='r',s=1)
	plt.axis('equal')	#allow true square scaling

	if bound:
		plt.axis([-1.1,1.1,-1.1,1.1])

	plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
	fig.clf()

def saveScatterPlotWithColorLabel(XY_list, filename, fig=None, bound=True):
	if(fig==None):
		fig=plt.figure(figsize=(8, 8))	#in inch

	for XY in XY_list:
		plt.scatter(XY[0,:],XY[1,:],s=1)
	plt.axis('equal')	#allow true square scaling

	if bound:
		plt.axis([-1.1,1.1,-1.1,1.1])

	plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
	fig.clf()

def savePrePhaseAndPhasePlot(prePhaseXY, phaseXY, filename, fig=None, srcColorList=None):
	if(fig==None):
		fig=plt.figure(figsize=(8, 8))	#in inch

	if(srcColorList==None):
		plt.scatter(prePhaseXY[0,:],prePhaseXY[1,:],color='r',s=1)
	else:
		plt.scatter(prePhaseXY[0,:],prePhaseXY[1,:],c=srcColorList,s=1)

	if(phaseXY is not None):
		plt.scatter(phaseXY[0,:],phaseXY[1,:],color='g',s=1)
	
	#center blue dot
	plt.scatter([0],[0],color='b',s=3)

	plt.axis('equal')	#allow true square scaling

	#plt.axis([-1.1,1.1,-1.1,1.1])

	plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
	fig.clf()
#LP add from previous util
def saveRatPhaseRainbowPlot(ratPose,phaseRat, filename, fig=None):
	if(fig==None):
		fig=plt.figure(figsize=(8, 8))	#in inch

	plt.figure(fig.number)

	plt.rcParams['image.cmap'] = 'gist_rainbow'
	plt.scatter(ratPose[0,:],ratPose[1,:],c=phaseRat/np.pi,s=5,vmin=0,vmax=2,zorder=1)
	cb=plt.colorbar(format=matplotlib.ticker.FormatStrFormatter('%g $\pi$'))
	
	plt.axis('equal')	#allow true square scaling
	plt.grid()

	if filename!='':
		plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
		fig.clf()
	else:
		plt.show()	

def saveRatPhaseRainbowPlot2(ratPose,phaseRat, filename, fig=None):
	if(fig==None):
		fig=plt.figure(figsize=(8, 8))	#in inch

	plt.figure(fig.number)

	plt.rcParams['image.cmap'] = 'gist_rainbow'
	plt.scatter(ratPose[2,:],ratPose[3,:],c=phaseRat/np.pi,s=5,vmin=0,vmax=3,zorder=1)####20231113
	cb=plt.colorbar(format=matplotlib.ticker.FormatStrFormatter('%g $\pi$'))
	
	plt.axis('equal')	#allow true square scaling
	plt.grid()

	if filename!='':
		plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
		fig.clf()
	else:
		plt.show()	

def saveRatPhaseRainbowPlot3(ratPose,phaseRat, filename, fig=None):
	if(fig==None):
		fig=plt.figure(figsize=(8, 8))	#in inch

	plt.figure(fig.number)

	plt.rcParams['image.cmap'] = 'gist_rainbow'
	plt.scatter(ratPose[4,:],ratPose[5,:],c=phaseRat/np.pi,s=5,vmin=0,vmax=2,zorder=1)
	cb=plt.colorbar(format=matplotlib.ticker.FormatStrFormatter('%g $\pi$'))
	
	plt.axis('equal')	#allow true square scaling
	plt.grid()

	if filename!='':
		plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
		fig.clf()
	else:
		plt.show()	

def saveRatPhaseRainbowPlot4(ratPose,phaseRat, filename, fig=None):
	if(fig==None):
		fig=plt.figure(figsize=(8, 8))	#in inch

	plt.figure(fig.number)

	plt.rcParams['image.cmap'] = 'gist_rainbow'
	plt.scatter(ratPose[6,:],ratPose[7,:],c=phaseRat/np.pi,s=5,vmin=0,vmax=2,zorder=1)
	cb=plt.colorbar(format=matplotlib.ticker.FormatStrFormatter('%g $\pi$'))
	
	plt.axis('equal')	#allow true square scaling
	plt.grid()

	if filename!='':
		plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
		fig.clf()
	else:
		plt.show()	


def savePhaseRainbowPlot(centerPose,phase, filename, fig=None):
    if(fig==None):
        fig=plt.figure(figsize=(8, 8))    #in inch

    plt.figure(fig.number)

    plt.rcParams['image.cmap'] = 'gist_rainbow'
    plt.scatter(centerPose[0,:],centerPose[1,:],c=phase/np.pi,s=5,vmin=0,vmax=2,zorder=1)
    cb=plt.colorbar(format=matplotlib.ticker.FormatStrFormatter('%g $\pi$'))

    plt.axis('equal')    #allow true square scaling
    plt.grid()

    if filename!='':
        plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
        fig.clf()
    else:
        plt.show()
        


def savePhaseRainbowPlot2(centerPose,phase, filename, fig=None):
	if(fig==None):
		fig=plt.figure(figsize=(8, 8))	#in inch

	plt.figure(fig.number)

	plt.rcParams['image.cmap'] = 'gist_rainbow'
	plt.scatter(centerPose[2,:],centerPose[3,:],c=phase/np.pi,s=5,vmin=0,vmax=2,zorder=1)
	cb=plt.colorbar(format=matplotlib.ticker.FormatStrFormatter('%g $\pi$'))
	
	plt.axis('equal')	#allow true square scaling
	plt.grid()

	if filename!='':
		plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
		fig.clf()
	else:
		plt.show()	

def savePhaseRainbowPlot3(centerPose,phase, filename, fig=None):
	if(fig==None):
		fig=plt.figure(figsize=(8, 8))	#in inch

	plt.figure(fig.number)

	plt.rcParams['image.cmap'] = 'gist_rainbow'
	plt.scatter(centerPose[4,:],centerPose[5,:],c=phase/np.pi,s=5,vmin=0,vmax=2,zorder=1)
	cb=plt.colorbar(format=matplotlib.ticker.FormatStrFormatter('%g $\pi$'))
	
	plt.axis('equal')	#allow true square scaling
	plt.grid()

	if filename!='':
		plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
		fig.clf()
	else:
		plt.show()	

def savePhaseRainbowPlot4(centerPose,phase, filename, fig=None):
	if(fig==None):
		fig=plt.figure(figsize=(8, 8))	#in inch

	plt.figure(fig.number)

	plt.rcParams['image.cmap'] = 'gist_rainbow'
	plt.scatter(centerPose[6,:],centerPose[7,:],c=phase/np.pi,s=5,vmin=0,vmax=2,zorder=1)
	cb=plt.colorbar(format=matplotlib.ticker.FormatStrFormatter('%g $\pi$'))
	
	plt.axis('equal')	#allow true square scaling
	plt.grid()

	if filename!='':
		plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
		fig.clf()
	else:
		plt.show()	

def saveMarginPlot(prePhaseXY, phaseXY, prePhaseMargin, filename, fig=None, srcColorList=None):
	if(fig==None):
		fig=plt.figure(figsize=(8, 8))	#in inch

	plt.figure(fig.number)
	
	if(prePhaseMargin is not None):
		plt.scatter(prePhaseMargin[0,:],prePhaseMargin[1,:],color='lightgrey',s=1)	#draw first to be behind

	if(srcColorList==None):
		plt.scatter(prePhaseXY[0,:],prePhaseXY[1,:],color='r',s=1)
	else:
		plt.scatter(prePhaseXY[0,:],prePhaseXY[1,:],c=srcColorList,s=1)

	if(phaseXY is not None):
		plt.scatter(phaseXY[0,:],phaseXY[1,:],color='g',s=1)
	
	

	#center blue dot
	plt.scatter([0],[0],color='b',s=3)

	plt.axis('equal')	#allow true square scaling

	#plt.axis([-1.1,1.1,-1.1,1.1])

	plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
	fig.clf()

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector,axis=0,keepdims=True)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.sum(v1_u*v2_u,axis=0,keepdims=True), -1.0, 1.0))

def interpolatePose(poseA,poseB,aRatio,sphereCount):
	ans=aRatio*poseA+(1-aRatio)*poseB
	for s in range(sphereCount):
		ans[s*3:s*3+3,:]=ans[s*3:s*3+3,:]/np.linalg.norm(ans[s*3:s*3+3,:])

	return ans

def interpolatePose1D(poseA,poseB,aRatio,sphereCount):
	ans=aRatio*poseA+(1-aRatio)*poseB
	for s in range(sphereCount):
		ans[s*3:s*3+3]=ans[s*3:s*3+3]/np.linalg.norm(ans[s*3:s*3+3])

	return ans

def averageConsecutivePoses(poses,sphereCount):	#poses=(n,?)
	ans=np.mean(poses,axis=1,keepdims=False)
	for s in range(sphereCount):
		ans[s*3:s*3+3]=ans[s*3:s*3+3]/np.linalg.norm(ans[s*3:s*3+3])

	return ans	#(n,)

def smoothPhase(phase,wing=3):
	ans=np.zeros(phase.shape)
	phaseX=np.cos(phase)
	phaseY=np.sin(phase)
	wing=3
	sequenceLength,=phase.shape
	for i in range(sequenceLength):
		if(i<wing):
			begin=0
			end=i+wing+1
		elif(i>=sequenceLength-wing):
			begin=i-wing
			end=sequenceLength
		else:
			begin=i-wing
			end=i+wing+1
		
		#count=end-begin
		meanX=np.mean(phaseX[begin:end])
		meanY=np.mean(phaseY[begin:end])
		ans[i]=np.arctan2(meanY,meanX)%(2*np.pi)
	
	return ans

def calculatePhaseProgress(phase):
	sequenceLength,=phase.shape

	app=np.zeros(phase.shape)
	app[0]=0	
	pro=np.zeros(phase.shape)
	pro[0]=0
	for i in range(1,sequenceLength):
		progress=(phase[i]-phase[i-1])%(2*np.pi)	#always give positive value here
		if(progress>=np.pi):
			progress-=(2*np.pi)	#make it negative

		app[i]=app[i-1]+progress	#can be both positive and negative
		pro[i]=progress
	
	return pro,app

def phaseDiffAbs(a,b):
	p=(a-b)%(2*np.pi)
	if(p>=np.pi):
		p-=(2*np.pi)
	
	return np.abs(p)

def phaseDiffAbsArray(a,b):
	p=(a-b)%(2*np.pi)
	for i in range(p.shape[0]):
		if(p[i]>=np.pi):
			p[i]-=(2*np.pi)
	
	return np.abs(p)

def circularDTW(phase,periodLength,trim=True,unhealthy=False):	
	#this phase is unsegmented. it is just a long sequence of extracted phase
	#starting phase doesn't need to be zero, ending phase doesn't need to be 2*pi	
	perfect=np.arange(periodLength)*(2*np.pi/periodLength)
	l,=phase.shape
	table=np.zeros([periodLength,l])

	#pre-calculate the cost
	cost=np.zeros([periodLength,l])
	for i in range(periodLength):
		for j in range(l):
			cost[i,j]=phaseDiffAbs(perfect[i],phase[j])

	#fill the first column with original cost, 
	#starting phase will be determined when tracked back from the ending point
	for i in range(0,periodLength):
		table[i,0]=cost[i,0]

	for j in range(1,l):	#work one column at a time
		#there are 3 sources

		#first and second source are from previous column
		for i in range(periodLength):
			table[i,j]=cost[i,j]+min([table[i,j-1],table[(i-1)%periodLength,j-1]])
		
		#third source is from the same column (must start from minimum)
		min_i=np.argmin(table[:,j])
		for dummy_i in range(min_i+1,min_i+periodLength):
			i=dummy_i%periodLength
			pre_i=(i-1)%periodLength
			table[i,j]=min([table[i,j],cost[i,j]+table[pre_i,j]])

	#search the ending phase
	finalPhaseIndex=np.argmin(table[:,-1])

	#trackback to find solution
	#compact=np.zeros([periodLength,2],dtype=int)-1	#compact map doesn't work in this case
	compactListReverse=[]
	linkCountListReverse=[]

	end_ij_listReverse=[]	#list of tuple
	begin_ij_listReverse=[]	#list of tuple

	expand=np.zeros([l,2],dtype=int)-1				#2 for start and end

	i=finalPhaseIndex
	j=l-1

	end_ij_listReverse.append((i,j))

	compact=np.zeros([periodLength,2],dtype=int)-1	#for the first and the last segment, a lot of -1 will stay to tell that those phase are not reached

	compact[i,:]=[j,j]
	expand[j,:]=[i,i]

	periodLinkCounter=1
	while j>0:	#track to the first column, i can loop around
		minIndex=np.argmin([ table[(i-1)%periodLength,j-1], table[(i-1)%periodLength,j], table[i,j-1] ])
		
		i_before=i
		j_before=j
		if(minIndex==0):
			i=(i-1)%periodLength
			j=j-1
			#new block for i and j
			#compact[i,:]=[j,j]
			expand[j,:]=[i,i]

		elif(minIndex==1):
			i=(i-1)%periodLength
			#new block for i only
			#compact[i,:]=[j,j]
			expand[j,0]=i

		elif(minIndex==2):
			j=j-1
			#new block for j only
			#compact[i,0]=j
			expand[j,:]=[i,i]


		if(minIndex==0 or minIndex==1):

			if(i_before==0 and i==periodLength-1):	#enter new period
				compactListReverse.append(compact)
				linkCountListReverse.append(periodLinkCounter)

				begin_ij_listReverse.append((i_before,j_before))
				end_ij_listReverse.append((i,j))

				#reset
				compact=np.zeros([periodLength,2],dtype=int)-1
				periodLinkCounter=0

			compact[i,:]=[j,j]
		else:
			compact[i,0]=j
		
		periodLinkCounter+=1

	#for the head period
	compactListReverse.append(compact)
	linkCountListReverse.append(periodLinkCounter)
	begin_ij_listReverse.append((i,j))

	#DTW calculation is done, the rest is extra calculation

	#reverse everthing back to the correct order
	compactMap = np.stack(compactListReverse[::-1], axis=0)	#(m,periodLength,2)	#be careful, this include incomplete period at the head and the tail !
	linkCount = linkCountListReverse[::-1]
	end_ij = end_ij_listReverse[::-1]
	begin_ij = begin_ij_listReverse[::-1]	#this is not used, just for validation.

	#calculate dtw distance in each period
	m=compactMap.shape[0]
	print(compactMap.shape)
	dtwCostPerLink = np.zeros(m)
	previousPeriodEndCost=0
	for k in range(m):
		ei,ej = end_ij[k]
		totalCost=table[ei,ej]-previousPeriodEndCost
		dtwCostPerLink[k]=totalCost/linkCount[k]
		previousPeriodEndCost=table[ei,ej]

	if unhealthy:
		if m>10:	#trim only when a lot of data is there
			#remove the first and the last section
			dtwCostPerLink=dtwCostPerLink[1:-1]
			compactMap=compactMap[1:-1,:,:]
			m-=2

	else:
		if trim==True:
			#remove the first and the last section
			dtwCostPerLink=dtwCostPerLink[1:-1]
			compactMap=compactMap[1:-1,:,:]
			m-=2	#BUG: m becomes negative here

			if(m<=5):
				print('m<=5 in healthy case')
				exit()

	#calculate periodMap
	periodMap=np.zeros(l,dtype=int)-1
	sectionBeginIndex=np.zeros(m,dtype=int)	#this is not visualIndex
	sectionEndIndex=np.zeros(m,dtype=int)		#this is not visualIndex
	for k in range(m):
		periodMap[compactMap[k,0,0]:compactMap[k,periodLength-1,1]+1]=k		#this section belong to period k (if overlap, the period on the right side will take)
		
		sectionBeginIndex[k]=compactMap[k,0,0]
		sectionEndIndex[k]=compactMap[k,periodLength-1,1]

	return expand, compactMap, periodMap, sectionBeginIndex, sectionEndIndex, dtwCostPerLink


def matchPhaseDTW(phase,periodLength):	#this phase is just a short phase segment that is not sorted, (but may pass some smooth filter)
	perfect=np.arange(periodLength)*(2*np.pi/periodLength)
	l,=phase.shape
	table=np.zeros([periodLength,l])
	
	table[0,0]=phaseDiffAbs(perfect[0],phase[0])
	for i in range(1,periodLength):
		table[i,0]=table[i-1,0]+phaseDiffAbs(perfect[i],phase[0])
	
	for j in range(1,l):
		table[0,j]=table[0,j-1]+phaseDiffAbs(perfect[0],phase[j])
	
	for i in range(1,periodLength):
		for j in range(1,l):
			table[i,j]=phaseDiffAbs(perfect[i],phase[j])+min([table[i-1,j-1],table[i-1,j],table[i,j-1]])
	
	#trackback to find solution
	compact=np.zeros([periodLength,2],dtype=int)-1
	expand=np.zeros([l,2],dtype=int)-1

	i=periodLength-1
	j=l-1

	compact[i,:]=[j,j]
	expand[j,:]=[i,i]

	walkCounter=0
	while i>0 or j>0:
		
		if(i>0 and j>0):
			minIndex=np.argmin([table[i-1,j-1],table[i-1,j],table[i,j-1]])
		elif(i>0 and j==0):
			minIndex=1
		elif(i==0 and j>0):
			minIndex=2	

		if(minIndex==0):
			i-=1
			j-=1
			#new block for i and j
			compact[i,:]=[j,j]
			expand[j,:]=[i,i]
		elif(minIndex==1):
			i-=1
			#new block for i only
			compact[i,:]=[j,j]
			expand[j,0]=i
		elif(minIndex==2):
			j-=1
			#new block for j only
			compact[i,0]=j
			expand[j,:]=[i,i]
		
		walkCounter+=1
		
	return compact, expand, table[periodLength-1,l-1]/walkCounter

def rotateAndAzimuthalEquidistantProjectTo2D(rotationMatrix, sphericalData):	#sphericalData=(3,?)
	#rotate data
	rotatedData=np.dot(rotationMatrix,sphericalData)
	#azimuthal equidistant projection (2D)
	azeq=rotatedData[0:2,:]/np.linalg.norm(rotatedData[0:2,:],axis=0,keepdims=True)*np.arccos(np.clip(rotatedData[2:3,:],-1.0,1.0))

	return azeq	#(2,?)

def convertOriginalPoseByAzeqProjection(rotationMatrixList,pose):
	sphereCount=len(rotationMatrixList)
	n,m=pose.shape
	converted=np.zeros([n-sphereCount,m])
	for s in range(sphereCount):
		sData=pose[s*3:s*3+3,:]	#(3,?)
		azData=rotateAndAzimuthalEquidistantProjectTo2D(rotationMatrixList[s],sData)	#(2,?)
		converted[s*2:s*2+2,:]=azData
	converted[sphereCount*2:,:]=pose[sphereCount*3:,:]

	return converted

def projectAzeqTo3DandRotateBack(rotationMatrix, azeqData):	#azeqData=(2,?)
	radialDistance=np.linalg.norm(azeqData,axis=0,keepdims=True) #(1,?)
	xy=azeqData/radialDistance*np.sin(radialDistance)	#(2,?)
	z=np.cos(radialDistance)	#(1,?)
	sData=np.concatenate([xy,z],axis=0)
	ans=np.dot(np.transpose(rotationMatrix),sData)
	return ans

def convertAzeqToOriginalPose(rotationMatrixList,azeqPose):
	sphereCount=len(rotationMatrixList)
	n,m=azeqPose.shape
	ans=np.zeros([n+sphereCount,m])
	for s in range(sphereCount):
		azData=azeqPose[s*2:s*2+2,:]	#(2,?)
		sData=projectAzeqTo3DandRotateBack(rotationMatrixList[s],azData)	#(3,?)
		ans[s*3:s*3+3,:]=sData
	ans[sphereCount*3:,:]=azeqPose[sphereCount*2:,:]
	return ans

def angleDifferenceInRad(a,b):	#a=(3,) b=(3,)	#a and b must be normalized from outside this function
	d=np.dot(a,b)
	d=max(-1,d)
	d=min(1,d)
	return np.arccos(d)

def savePhase3dPlot(centerPose, phase, filename, fig=None):
    # Create figure and 3D axis
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw={'projection': '3d'})  # in inches
    # plt.figure(fig.number)
    # Set color map
    plt.rcParams['image.cmap'] = 'gist_rainbow'
    
    # Scatter plot
	# 0: X
	# 1: Z
	# 2: Frame
    scatter = ax.scatter(centerPose[2,:], centerPose[0,:], centerPose[1,:], 
                         c=phase/np.pi, s=4, vmin=0, vmax=2, zorder=1)
		  
    
    # Set axis limits properly
    ax.set_xlim(0, 75)   # X-axis range
    ax.set_ylim(-4, 4) # Y-axis range
    ax.set_zlim(-4, 4)  # Correct way to set Z-axis range
	 
    # ax.set_ylim(ax.get_ylim()[::-1]) # Y-axis switch
    
	
    # Set box aspect to make axes proportional
    ax.set_box_aspect([1,1,1])  # Ensures equal axis scaling in 3D
    
    # Add colorbar safely
    cb = plt.colorbar(scatter, format=matplotlib.ticker.FormatStrFormatter('%g π'))

    # Enable grid
    plt.grid()

    # Save or show the figure
    if filename:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig)  # Proper cleanup
    else:
        plt.show()



# def savePhasePerCycleSpiral(centerPose, phase, filename, fig=None):
#     # Create figure and 3D axis
#     fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw={'projection': '3d'})  # in inches
#     plt.figure(fig.number)
#     # Set color map
#     plt.rcParams['image.cmap'] = 'gist_rainbow'
    
#     # Scatter plot
# 	# 0: X
#     x_centerPose = centerPose[0, :]
# 	# 1: Z
#     z_centerPose = centerPose[1, :]
# 	# 2: Frame
#     frame_centerPose = centerPose[2, :]

# 	## STEP1: Loop Extraction
# 	## otsu thresholding peak detection
# 	## peak from X-axis
#     peak_centerPose = 


# 	## save between peaks

# 	## interpolate

# 	## Mean of cycle



#     # scatter = ax.scatter(centerPose[2,:], centerPose[0,:], centerPose[1,:], 
#     #                      c=phase/np.pi, s=4, vmin=0, vmax=2, zorder=1)
	
# 	scatter = ax.scatter(cycle_centerPose, x_mean_centerPose, z_mean_centerPose, 
# 						c=phase/np.pi, s=4, vmin=0, vmax=2, zorder=1)
    
#     # Set axis limits properly
#     ax.set_xlim(0, 75)   # X-axis range
#     ax.set_ylim(-4, 4)   # Y-axis range
#     ax.set_zlim(-4, 4)  # Correct way to set Z-axis range
	
#     # Set box aspect to make axes proportional
#     ax.set_box_aspect([1,1,1])  # Ensures equal axis scaling in 3D
    
#     # Add colorbar safely
#     cb = plt.colorbar(scatter, format=matplotlib.ticker.FormatStrFormatter('%g π'))

#     # Enable grid
#     plt.grid()

#     # Save or show the figure
#     if filename:
#         plt.savefig(filename, bbox_inches='tight', pad_inches=0)
#         plt.close(fig)  # Proper cleanup
#     else:
#         plt.show()

from tkinter import Tk

class Clipboard:

	def __init__(self):
		self.r = Tk()
		self.r.withdraw()
	
	def copy(self,message):
		self.r.clipboard_clear()
		self.r.clipboard_append(message)
		self.r.update() # now it stays on the clipboard after the window is closed

	def __exit__(self, exc_type, exc_value, traceback):
		self.r.destroy()

class PCA:

	def __init__(self, data):	#data=(D,m)
		D,m=data.shape
		cov=np.dot(data,np.transpose(data))/m

		eigVals,eigVecs=np.linalg.eigh(cov)	#not in ordered
		idx = eigVals.argsort()[::-1]   #sort from max to min
		
		eigVals = eigVals[idx]
		eigVecs = eigVecs[:,idx]	#each column is one eigen vector

		self.eigVals=eigVals
		self.eigVecs=eigVecs



if __name__ == '__main__':
	c=Clipboard()
	c.copy("hey")
	print("done")
#ghhghghg
def choose_threshold(data, method="mean_std", percentile=90):
    """
    Dynamically choose a threshold based on the data and the method provided.

    Parameters:
    data (numpy array): The input data to determine the threshold.
    method (str): The method to choose the threshold, options are:
                  - "mean_std" (mean + standard deviation)
                  - "mad" (Median Absolute Deviation)
                  - "percentile" (based on a specified percentile)
                  - "otsu" (Otsu's method for bimodal distributions)
    percentile (int): Used if method is "percentile". Default is 90.
    
    Returns:
    threshold (float): The chosen threshold.
    """
    if method == "mean_std":
        threshold = np.mean(data) + np.std(data)
    elif method == "mad":
        median = np.median(data)
        mad = np.median(np.abs(data - median))  # Median Absolute Deviation
        threshold = median + 3 * mad  # Adjust the multiplier as needed
    elif method == "percentile":
        threshold = np.percentile(data, percentile)
    elif method == "otsu":
        hist, bin_edges = np.histogram(data, bins=256)
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]
        mean1 = np.cumsum(hist * bin_mids) / weight1
        mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
        inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        index_of_max_variance = np.argmax(inter_class_variance)
        threshold = bin_mids[:-1][index_of_max_variance]
    else:
        raise ValueError("Invalid method. Choose from 'mean_std', 'mad', 'percentile', 'otsu'.")
    
    return threshold


def detect_peaks(data, threshold=None, distance=5, threshold_method="mean_std", percentile=90):
    """
    Detect peaks in the data using scipy's find_peaks and dynamically choose threshold if not provided.
    
    Parameters:
    data (numpy array): The input data to detect peaks from.
    threshold (float): The minimum height required for a peak to be detected.
    distance (int): Minimum horizontal distance between neighboring peaks.
    threshold_method (str): The method to choose the threshold ('mean_std', 'mad', 'percentile', 'otsu').
    percentile (int): Percentile used if threshold_method is 'percentile'.
    
    Returns:
    peaks (numpy array): The indices of the peaks in the data.
    properties (dict): Properties of the detected peaks, like heights.
    """
    # If no threshold is provided, dynamically select it based on the chosen method
    if threshold is None:
        threshold = choose_threshold(data, method=threshold_method, percentile=percentile)
    
    # Detect peaks
    peaks, properties = signal.find_peaks(data, height=threshold, distance=distance)
    
    return peaks, properties

def get_between_peak(x: np.array, peaks, return_n_peak=False):

    cycle_list = []


    for count_peak, peak in enumerate(peaks):

        if count_peak == 0:

            cycle = x[0: peaks[count_peak]]

        elif count_peak == len(peaks) - 1:

            cycle = x[peaks[count_peak]: ]

        else:
            cycle = x[peaks[count_peak - 1]: peaks[count_peak]]


        # appending the signal to list of cycles

        cycle_list.append(cycle)

    if return_n_peak:

        return cycle_list, len(peaks)
    
    else:

        return cycle_list


def interpolate_signal_window(input_signal, target_length):
    """
    Interpolates a signal window to a specified length.

    Parameters:
    input_signal (numpy.ndarray): The input signal window to be interpolated.
    target_length (int): The desired length of the interpolated signal.

    Returns:
    numpy.ndarray: The interpolated signal with the specified length.
    """
    original_length = len(input_signal)
    if original_length == target_length:
        return input_signal

    # Original indices
    original_indices = np.arange(original_length)

    # Target indices
    target_indices = np.linspace(0, original_length - 1, target_length)

    # Interpolate using numpy.interp
    interpolated_signal = np.interp(target_indices, original_indices, input_signal)

    return interpolated_signal

#funct working

def savePhasePerCycleSpiral(centerPose, phase, filename, fig=None):

	## Create figue and 3D axis

	fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw={"projection": "3d"})

	plt.figure(fig.number)

	plt.rcParams["image.cmap"] = "gist_rainbow"

	## Scatter plots

	x_centerPose = centerPose[0, :]
	z_centerPose = centerPose[1, :]
	frame_centerPose = centerPose[2, :]


	## STEP 1: Extract the cycles
	peak_centerPose, _ = detect_peaks(x_centerPose, threshold_method="otsu")
	
	segment_x_centerPose = get_between_peak(x_centerPose, peak_centerPose)
	segment_z_centerPose = get_between_peak(z_centerPose, peak_centerPose)

	segment_phase = get_between_peak(phase, peak_centerPose)


	## STEP 2: getting max length of all the signal for interpolation

	max_len = np.max([
		np.max([len(cycle) for cycle in segment_x_centerPose]),
		np.max([len(cycle) for cycle in segment_z_centerPose]),
		np.max([len(cycle) for cycle in segment_phase])
	])
	
	print('max len=',max_len)

	min_len = np.min([
		np.min([len(cycle) for cycle in segment_x_centerPose]),
		np.min([len(cycle) for cycle in segment_z_centerPose]),
		np.min([len(cycle) for cycle in segment_phase])
	])

	print('min len=',min_len)

	mean_len = np.min([
		np.mean([len(cycle) for cycle in segment_x_centerPose]),
		np.mean([len(cycle) for cycle in segment_z_centerPose]),
		np.mean([len(cycle) for cycle in segment_phase])
	])

	print('mean len=',mean_len)
	
	interpolated_segment_x_centerPose = np.array([interpolate_signal_window(input_signal, target_length=max_len) for input_signal in segment_x_centerPose])
	interpolated_segment_z_centerPose = np.array([interpolate_signal_window(input_signal, target_length=max_len) for input_signal in segment_z_centerPose])
	interpolated_segment_phase = np.array([interpolate_signal_window(input_signal, target_length=max_len) for input_signal in segment_phase])



	## STEP 3: Finding representatives

	mean_segment_x_centerPose = np.mean(interpolated_segment_x_centerPose, axis=0)
	mean_segment_z_centerPose = np.mean(interpolated_segment_z_centerPose, axis=0)
	mean_segment_phase = stats.circmean(interpolated_segment_phase, axis=0)


	sd_segment_x_centerPose = np.std(interpolated_segment_x_centerPose)
	sd_segment_z_centerPose =np.std(interpolated_segment_z_centerPose)
	sd_segment_phase = np.std(interpolated_segment_phase)

	## visualization

	# scatter = ax.scatter(
	# 	np.arange(max_len),
	# 	mean_segment_x_centerPose,
	# 	mean_segment_z_centerPose,
	# 	c = mean_segment_phase
	# )

	scatter = ax.scatter(
		np.arange(max_len),
		mean_segment_x_centerPose,
		mean_segment_z_centerPose,
		c = mean_segment_phase
	)

	# ax.set_xlim(0, 75)
	# ax.set_ylim(-4, 4)
	# ax.set_zlim(-4, 4)

	ax.set_box_aspect([1, 1, 1])

	cb = plt.colorbar(scatter, format=matplotlib.ticker.FormatStrFormatter("%g π"),shrink = 0.5)

	plt.grid()

	if filename:
		
		plt.savefig(filename, bbox_inches='tight', pad_inches=0)
		pickle.dump(fig, open(filename.replace("png", "pkl"), 'wb'))
		# plt.show() #try to 3d
		plt.close(fig)
		

	else:

		plt.show()



def save1000PhaseSpiral(centerPose, phase, filename, fig=None):


	## Create figue and 3D axis

	fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw={"projection": "3d"})

	plt.figure(fig.number)

	plt.rcParams["image.cmap"] = "gist_rainbow"

	## Scatter plots geyyyy

	x_centerPose = centerPose[0, :]
	z_centerPose = centerPose[1, :]
	frame_centerPose = centerPose[2, :]

	#pick first 100

	x_centerPose_1000 = x_centerPose[0:1001]
	z_centerPose_1000 = z_centerPose[0:1001]
	frame_centerPose_1000 = frame_centerPose[0:1001]
	phase_1000 = phase[0:1001]

	scatter = ax.scatter(
		frame_centerPose_1000,
		x_centerPose_1000,
		z_centerPose_1000,
		c = phase_1000
	)

	# ax.set_xlim(0, 75)
	# ax.set_ylim(-4, 4)
	# ax.set_zlim(-4, 4)

	ax.set_box_aspect([1, 1, 1])

	cb = plt.colorbar(scatter, format=matplotlib.ticker.FormatStrFormatter("%g π"),shrink = 0.5)

	plt.grid()

	if filename:
		
		plt.savefig(filename, bbox_inches='tight', pad_inches=0)
		pickle.dump(fig, open(filename.replace("png", "pkl"), 'wb'))
		# plt.show() #try to 3d
		plt.close(fig)
		

	else:

		plt.show()


def saveFirstHalfPhaseSpiral(centerPose, phase, filename, fig=None):


	## Create figue and 3D axis

	fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw={"projection": "3d"})

	plt.figure(fig.number)

	plt.rcParams["image.cmap"] = "gist_rainbow"

	## Scatter plots geyyyy

	x_centerPose = centerPose[0, :]
	z_centerPose = centerPose[1, :]
	frame_centerPose = centerPose[2, :]

	#pick first 100
	x_centerPose_firsthalf = x_centerPose[0:6001]
	z_centerPose_firsthalf = z_centerPose[0:6001]
	frame_centerPose_firsthalf = frame_centerPose[0:6001]
	phase_firsthalf = phase[0:6001]

	# #นอน
	# scatter = ax.scatter(
	# 	frame_centerPose_firsthalf,
	# 	x_centerPose_firsthalf,
	# 	z_centerPose_firsthalf,
	# 	c = phase_firsthalf
	# )
	#ตั้ง
	scatter = ax.scatter(
		z_centerPose_firsthalf,
		frame_centerPose_firsthalf,
		x_centerPose_firsthalf,
		c = phase_firsthalf
	)

	# ax.set_xlim(0, 75)
	# ax.set_ylim(-4, 4)
	# ax.set_zlim(-4, 4)

	ax.set_box_aspect([1, 1, 1])

	cb = plt.colorbar(scatter, format=matplotlib.ticker.FormatStrFormatter("%g π"),shrink = 0.5)

	plt.grid()

	if filename:
		
		plt.savefig(filename, bbox_inches='tight', pad_inches=0)
		pickle.dump(fig, open(filename.replace("png", "pkl"), 'wb'))
		# plt.show() #try to 3d
		plt.close(fig)
		

	else:

		plt.show()


def saveLateHalfPhaseSpiral(centerPose, phase, filename, fig=None):


	## Create figue and 3D axis

	fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw={"projection": "3d"})

	plt.figure(fig.number)

	plt.rcParams["image.cmap"] = "gist_rainbow"

	## Scatter plots geyyyy

	x_centerPose = centerPose[0, :]
	z_centerPose = centerPose[1, :]
	frame_centerPose = centerPose[2, :]

	#pick first 100
	x_centerPose_latehalf = x_centerPose[6001:12002]
	z_centerPose_latehalf = z_centerPose[6001:12002]
	frame_centerPose_latehalf = frame_centerPose[6001:12002]
	phase_latehalf = phase[6001:12002]

	# #แนวนอน
	# scatter = ax.scatter(
	# 	frame_centerPose_latehalf,
	# 	x_centerPose_latehalf,
	# 	z_centerPose_latehalf,
	# 	c = phase_latehalf
	# )
	#แนวตั้ง
	scatter = ax.scatter(
		z_centerPose_latehalf,
		frame_centerPose_latehalf,
		x_centerPose_latehalf,
		c = phase_latehalf
	)
	

	ax.set_ylim(6000, 12000)
	# ax.set_ylim(-4, 4)
	# ax.set_zlim(-4, 4)

	ax.set_box_aspect([1, 1, 1])

	cb = plt.colorbar(scatter, format=matplotlib.ticker.FormatStrFormatter("%g π"),shrink = 0.5)

	plt.grid()

	if filename:
		
		plt.savefig(filename, bbox_inches='tight', pad_inches=0)
		pickle.dump(fig, open(filename.replace("png", "pkl"), 'wb'))
		# plt.show() #try to 3d
		plt.close(fig)
		

	else:

		plt.show()

def evaluateplot(centerPose,centerPose_2, phase, filename, fig=None):
	x_centerPose = centerPose[0, :]
	z_centerPose = centerPose[1, :]
	frame_centerPose = centerPose[2, :]
	x_centerPose_2 = centerPose_2[0, :]
	z_centerPose_2 = centerPose_2[1, :]
	# frame_centerPose_2 = centerPose_2[2, :]

	x_centerPose_1000 = x_centerPose[15:801]
	z_centerPose_1000 = z_centerPose[15:801]
	frame_centerPose_1000 = frame_centerPose[15:801]
	phase_1000 = phase[15:801]

	x_centerPose_2_1000 = x_centerPose_2[15:801]
	z_centerPose_2_1000 = z_centerPose_2[15:801]
	# frame_centerPose_2_1000 = frame_centerPose_2[0:1001]

	# peaks, _ = find_peaks(frame_centerPose_1000, height=0)
	plt.figure(figsize=(8, 20))

	#velocity

	# plt.subplot(5, 1, 1)
	# plt.plot(frame_centerPose_1000,x_centerPose_1000)
	# plt.title('x_velocity')
	# plt.xlabel('frame')
	# plt.ylabel('x_velocity')
	# peaks, _ = find_peaks(x_centerPose_1000, height=0)
	# peak_pos = frame_centerPose_1000[peaks]
	# plt.plot(peak_pos, x_centerPose_1000[peaks], "x")
	# peak_frame = frame_centerPose_1000[peaks]
	# peaks_as_string = f"Top Peaks: {peak_frame[:]}"
	# plt.text(0.5, -0.2, peaks_as_string, transform=plt.gca().transAxes, 
    #      ha='center', fontsize=8, color='red')





	plt.subplot(5,1,2)
	plt.plot(frame_centerPose_1000,z_centerPose_1000)
	plt.title('z_velocity')
	plt.xlabel('frame')
	plt.ylabel('z_velocity')
	TOpeaks, _ = find_peaks(z_centerPose_1000, height=0.4, distance=50)
	peak_TO = frame_centerPose_1000[TOpeaks]
	plt.plot(peak_TO, z_centerPose_1000[TOpeaks], "x")
	TO_frame = frame_centerPose_1000[TOpeaks]
	TO_as_string = f"Toe off: {TO_frame[:]}"
	plt.text(0.5, -0.2, TO_as_string, transform=plt.gca().transAxes, 
         ha='center', fontsize=8, color='red')
	
	HSpeaks, _ = find_peaks(-z_centerPose_1000, height=(None, 0.3), distance=50)
	peak_HS = frame_centerPose_1000[HSpeaks]
	plt.plot(peak_HS, z_centerPose_1000[HSpeaks], "x")	
	HS_frame = frame_centerPose_1000[HSpeaks]
	HS_as_string = f"Heel strike Peaks: {HS_frame[:]}"
	plt.text(0.5, -0.3, HS_as_string, transform=plt.gca().transAxes, 
         ha='center', fontsize=8, color='blue')	
	

	plt.subplot(5, 1, 1)
	plt.plot(frame_centerPose_1000,x_centerPose_1000)
	plt.title('x_velocity')
	plt.xlabel('frame')
	plt.ylabel('x_velocity')
	peaks, _ = find_peaks(x_centerPose_1000, height=0)
	peak_pos = frame_centerPose_1000[peaks]
	plt.plot(peak_pos, x_centerPose_1000[peaks], "x",color = 'black')
	peak_frame = frame_centerPose_1000[peaks]
	peaks_as_string = f"Top Peaks: {peak_frame[:]}"
	plt.text(0.5, -0.1, peaks_as_string, transform=plt.gca().transAxes, 
         ha='center', fontsize=8, color='green')
	plt.plot(peak_TO, z_centerPose_1000[TOpeaks], "x")
	TO_frame = frame_centerPose_1000[TOpeaks]
	plt.text(0.5, -0.2, TO_as_string, transform=plt.gca().transAxes, 
         ha='center', fontsize=8, color='red')
	plt.plot(peak_HS, z_centerPose_1000[HSpeaks], "x")		
	HS_frame = frame_centerPose_1000[HSpeaks]
	plt.text(0.5, -0.3, HS_as_string, transform=plt.gca().transAxes, 
         ha='center', fontsize=8, color='blue')	
	
	


	plt.subplot(5, 1, 3)
	plt.plot(frame_centerPose_1000,x_centerPose_2_1000)
	plt.title('x_position')
	plt.xlabel('frame')
	plt.ylabel('x_position')
	peaks, _ = find_peaks(x_centerPose_2_1000, height=0)
	peak_pos = frame_centerPose_1000[peaks]
	plt.plot(peak_pos, x_centerPose_2_1000[peaks], "x",color = 'black')
	peak_frame = frame_centerPose_1000[peaks]
	peaks_as_string = f"Top Peaks: {peak_frame[:]}"
	plt.text(0.5, -0.1, peaks_as_string, transform=plt.gca().transAxes, 
         ha='center', fontsize=8, color='green')
	plt.plot(peak_TO, z_centerPose_1000[TOpeaks], "x")
	TO_frame = frame_centerPose_1000[TOpeaks]
	plt.text(0.5, -0.2, TO_as_string, transform=plt.gca().transAxes, 
         ha='center', fontsize=8, color='red')
	plt.plot(peak_HS, z_centerPose_1000[HSpeaks], "x")		
	HS_frame = frame_centerPose_1000[HSpeaks]
	plt.text(0.5, -0.3, HS_as_string, transform=plt.gca().transAxes, 
         ha='center', fontsize=8, color='blue')	

	plt.subplot(5,1,4)
	plt.plot(frame_centerPose_1000,z_centerPose_2_1000)
	plt.title('z_position')
	plt.xlabel('frame')
	plt.ylabel('z_position')
	peaks, _ = find_peaks(z_centerPose_2_1000, height=0.5)
	peak_pos = frame_centerPose_1000[peaks]
	plt.plot(peak_pos, x_centerPose_2_1000[peaks], "x",color = 'black')
	peak_frame = frame_centerPose_1000[peaks]
	peaks_as_string = f"Top Peaks: {peak_frame[:]}"
	plt.text(0.5, -0.1, peaks_as_string, transform=plt.gca().transAxes, 
         ha='center', fontsize=8, color='green')
	plt.plot(peak_TO, z_centerPose_1000[TOpeaks], "x")
	TO_frame = frame_centerPose_1000[TOpeaks]
	plt.text(0.5, -0.2, TO_as_string, transform=plt.gca().transAxes, 
         ha='center', fontsize=8, color='red')
	plt.plot(peak_HS, z_centerPose_1000[HSpeaks], "x")		
	HS_frame = frame_centerPose_1000[HSpeaks]
	plt.text(0.5, -0.3, HS_as_string, transform=plt.gca().transAxes, 
         ha='center', fontsize=8, color='blue')	
	

	plt.subplot(5, 1, 5)
	plt.plot(frame_centerPose_1000,phase_1000)
	plt.title('phase')
	plt.xlabel('frame')
	plt.ylabel('phase')
	peaks, _ = find_peaks(phase_1000, height=5.9)
	peak_pos = frame_centerPose_1000[peaks]
	plt.plot(peak_pos, x_centerPose_2_1000[peaks], "x",color = 'black')
	peak_frame = frame_centerPose_1000[peaks]
	peaks_as_string = f"Top Peaks: {peak_frame[:]}"	
	plt.text(0.5, -0.1, peaks_as_string, transform=plt.gca().transAxes, 
         ha='center', fontsize=8, color='green')
	plt.plot(peak_TO, z_centerPose_1000[TOpeaks], "x")
	TO_frame = frame_centerPose_1000[TOpeaks]
	plt.text(0.5, -0.2, TO_as_string, transform=plt.gca().transAxes, 
         ha='center', fontsize=8, color='red')
	plt.plot(peak_HS, z_centerPose_1000[HSpeaks], "x")		
	HS_frame = frame_centerPose_1000[HSpeaks]
	plt.text(0.5, -0.3, HS_as_string, transform=plt.gca().transAxes, 
         ha='center', fontsize=8, color='blue')	
	
	plt.tight_layout()

	if filename:
		
		plt.savefig(filename, bbox_inches='tight', pad_inches=0)
		pickle.dump(fig, open(filename.replace("png", "pkl"), 'wb'))
		# plt.show() #try to 3d
		plt.close(fig)
		

	else:

		plt.show()

	return TOpeaks , HSpeaks


def pca_hilbert_transform_with_phase_color(analytic_signal):
	instantaneous_phase = np.unwrap(analytic_signal)

	instantaneous_phase_mod_2pi = np.mod(instantaneous_phase, 2* np.pi)

	colorized = instantaneous_phase_mod_2pi / np.pi

	return colorized

def clo():
	return None
	


def XZPhase(centerPose,centerPose_2, phase, filename, HSpeaks, TOpeaks, fig=None):
	x_centerPose = centerPose[0, :]
	z_centerPose = centerPose[1, :]
	frame_centerPose = centerPose[2, :]
	x_centerPose_2 = centerPose_2[0, :]
	z_centerPose_2 = centerPose_2[1, :]


	x_centerPose_1000 = x_centerPose
	z_centerPose_1000 = z_centerPose
	frame_centerPose_1000 = frame_centerPose
	phase_1000 = phase
	phase_c = pca_hilbert_transform_with_phase_color(phase_1000)

	x_centerPose_2_1000 = x_centerPose_2
	z_centerPose_2_1000 = z_centerPose_2

	## Dealing with toe-off and heal-strike peaks plots
	peak_HS = frame_centerPose_1000[HSpeaks]
	peak_TO = frame_centerPose_1000[TOpeaks]

	x_HS = x_centerPose_2_1000[HSpeaks]
	z_HS = z_centerPose_2_1000[HSpeaks]
	
	x_TO = x_centerPose_2_1000[TOpeaks]
	z_TO = z_centerPose_2_1000[TOpeaks]

	phase_HS = phase_1000[HSpeaks]
	phase_TO = phase_1000[TOpeaks]

	fig, ax = plt.subplots(1, 1, figsize=(30 ,20), sharex=True, sharey=True)
	ax.scatter(x_centerPose_2_1000, z_centerPose_2_1000 , label=f"Trajectory", c=phase_c)
	# ax.scatter(x_HS, z)
	# ax[1].scatter(z_centerPose_1000[..., 0], z_centerPose_1000[..., 1] , label=f"z_position", c=phase_c)
	# ax[2].scatter(pred[..., 0], pred[..., 1], label="Prediction", c=pred_c)
	ax.set_title(f"phase and position")
	# ax[1].set_title(f"Groud Truth: Target {target_name} of No. {random_sample}")
	# ax[2].set_title(f"Model Prediction: {target_name} MSE:{mse}, gPSI:{g_psi}")
	ax.scatter(x_HS, z_HS, label=f"Heal Strike", c="#54534f", marker="D", s=50)
	ax.scatter(x_TO, z_TO, label=f"Toe-Off", c="#292723", marker="*", s=50)
	plt.figtext(0.7, 0.25, 'Mean HS phase: ' + str(np.mean(phase_HS)/math.pi) + 'PI', fontsize=20, color='blue')
	plt.figtext(0.7, 0.35, 'Mean TO phase: ' + str(np.mean(phase_TO)/math.pi) + 'PI', fontsize=20, color='red')
	# cb = fig.colorbar(ax.collections, format=matplotlib.ticker.FormatStrFormatter('%g $\pi$'), shrink=1)
	cb = fig.colorbar(ax.collections[0], format=matplotlib.ticker.FormatStrFormatter('%g $\pi$'), shrink=1)
	# cb = fig.colorbar(ax[1].collections[0], format=matplotlib.ticker.FormatStrFormatter('%g $\pi$'), shrink=1)
	# cb = fig.colorbar(ax[2].collections[0], format=matplotlib.ticker.FormatStrFormatter('%g $\pi$'), shrink=1)
	ax.legend()
	ax.grid()
	# ax[1].grid()
	# ax[2].grid()

	ax.set_xlabel("X Position")
	# ax[1].set_xlabel("X Position")
	# ax[2].set_xlabel("X Position")

	ax.set_ylabel("Y Position")
	# ax[1].set_ylabel("Y Position")
	# ax[2].set_ylabel("Y Position")
	plt.grid()

	if filename:
		
		plt.savefig(filename, bbox_inches='tight', pad_inches=0)
		pickle.dump(fig, open(filename.replace("png", "pkl"), 'wb'))
		# plt.show() #try to 3d
		plt.close(fig)
		

	else:

		plt.show()

	

# def savePhasePerCycleSpiral(centerPose, phase, filename, fig=None):

# 	## Create figue and 3D axis
# 	phase = np.unwrap(phase)

# 	fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw={"projection": "3d"})

# 	plt.figure(fig.number)

# 	plt.rcParams["image.cmap"] = "gist_rainbow"

# 	## Scatter plots

# 	x_centerPose = centerPose[0, :]
# 	z_centerPose = centerPose[1, :]
# 	frame_centerPose = centerPose[2, :]


# 	## STEP 1: Extract the cycles
# 	peak_centerPose, _ = detect_peaks(x_centerPose, threshold_method="otsu")
# 	# peak_centerPose, _ = detect_peaks(phase, threshold_method="otsu")

# 	segment_x_centerPose = get_between_peak(x_centerPose, peak_centerPose)
# 	segment_z_centerPose = get_between_peak(z_centerPose, peak_centerPose)

# 	segment_phase = get_between_peak(phase, peak_centerPose)


# 	## STEP 2: getting max length of all the signal for interpolation

# 	max_len = np.max([
# 		np.max([len(cycle) for cycle in segment_x_centerPose]),
# 		np.max([len(cycle) for cycle in segment_z_centerPose]),
# 		np.max([len(cycle) for cycle in segment_phase])
# 	])

	
# 	interpolated_segment_x_centerPose = np.array([interpolate_signal_window(input_signal, target_length=max_len) for input_signal in segment_x_centerPose])
# 	interpolated_segment_z_centerPose = np.array([interpolate_signal_window(input_signal, target_length=max_len) for input_signal in segment_z_centerPose])
# 	interpolated_segment_phase = np.array([interpolate_signal_window(input_signal, target_length=max_len) for input_signal in segment_phase])



# 	## STEP 3: Finding representatives

# 	mean_segment_x_centerPose = np.mean(interpolated_segment_x_centerPose, axis=0)
# 	mean_segment_z_centerPose = np.mean(interpolated_segment_z_centerPose, axis=0)
# 	mean_segment_phase = np.mean(interpolated_segment_phase, axis=0)

# 	sd_segment_x_centerPose = np.std(interpolated_segment_x_centerPose)
# 	sd_segment_z_centerPose =np.std(interpolated_segment_z_centerPose)
# 	sd_segment_phase = np.std(interpolated_segment_phase)

# 	## visualization

# 	scatter = ax.scatter(
# 		np.arange(max_len),
# 		mean_segment_x_centerPose,
# 		mean_segment_z_centerPose,
# 		c = mean_segment_phase
# 	)

# 	# ax.set_xlim(0, 75)
# 	# ax.set_ylim(-4, 4)
# 	# ax.set_zlim(-4, 4)

# 	ax.set_box_aspect([1, 1, 1])

# 	cb = plt.colorbar(scatter, format=matplotlib.ticker.FormatStrFormatter("%g π"),shrink = 0.5)

# 	plt.grid()

# 	if filename:

# 		plt.savefig(filename, bbox_inches='tight', pad_inches=0)
# 		plt.close(fig)

# 	else:

# 		plt.show()

