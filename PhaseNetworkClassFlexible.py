import numpy as np
import tensorflow.compat.v1 as tf
#import tensorflow.compat.v1 as tf#try
#tf.disable_v2_behavior()#try
#import setting

class EmptyContext():
	def __init__(self):
		pass
	def __enter__(self):
		pass
	def __exit__(self, type, value, traceback):
		pass

# https://stackoverflow.com/questions/41695893/tensorflow-conditionally-add-variable-scope?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
def selectedDevice(gpuID=-1):
	if gpuID==-1:
		return EmptyContext()	#just use default device which might be CPU or GPU:0
	else:
		return tf.device('/device:GPU:'+str(gpuID))

class PhaseNetwork:

	def nnFlow(self, L, W, b, input, outputName=None):
		## Tensorflow input layer -> Reshaping as Flattening (shape[self.inputShape, a, b, c]->shape[self.inputSize, a*b*c]
		flatInput=tf.reshape(input,[self.inputSize,-1])
		## List of input
		A=[flatInput]

		for i in range(1,len(L)-1):

			## Forward passing -> x_hat = tanh(Wx + b) ***(EXCEPT LAST LAYER)***
			Ai=tf.nn.tanh(tf.matmul(W[i],A[i-1])+b[i]) #,name="A"+str(i))
			### storage any outputs	
			A.append(Ai)

		i=len(L)-1 ## Pinning to the last layer
		#A.append(tf.nn.sigmoid(tf.matmul(W[i],A[i-1]+b[i])))	#last layer
		A.append(tf.matmul(W[i],A[i-1])+b[i])	#linear for last layer -> No Activation 

		prePhase=A[i]  ## defining prePhase 

		if(outputName==None):
			phaseXY=tf.nn.l2_normalize(prePhase,axis=0)	#this is a unit circle
			phaseRad=tf.atan2(phaseXY[1,:],phaseXY[0,:]) #,name="phaseRad")	#use for output only
		else:
			phaseXY=tf.nn.l2_normalize(prePhase,axis=0,name=outputName+"-phaseXY")	#this is a unit circle
			phaseRad=tf.atan2(phaseXY[1,:],phaseXY[0,:],name=outputName+"-phaseRad") #,name="phaseRad")	#use for output only

		return prePhase, phaseXY, phaseRad 	#pre-phase layer

	def __init__(self, D, windowWing=1, gpuID=-1):	#D=8 for kinectAlone, D=10 for kinect+ and mocap

		#set all parameters
		h=25#30#forelimb25#hindlimb25
		hiddenLayers=[h,h,h,h,h]#[h,h,h,h,h]#forelimb#[h,h,h,h,h,h,h]#hindlimb[h,h,h,h,h]

		#Ap=2*np.pi/180		#2deg for 30 fps
		#Ap=-2*np.pi/180		#constant negative value works well but not stable, the result can flip and move phase in negative way
		Ap_firstHalf =-2.3*np.pi/180#-2*np.pi/180#forelimb-1.8*np.pi/180#hindlimb-2.3*np.pi/180
		Ap_secondHalf=-2.3*np.pi/180#-2*np.pi/180#forelimb-1.8*np.pi/180#hindlimb-2.3*np.pi/180
		

		Bp=4*np.pi/180#15.0*np.pi/180#forelimb3.8*np.pi/180#hindlimb4*np.pi/180

		Cp = (-180 + 45)*np.pi/180	#on the left #(-180 + 45)*np.pi/180#forelimb(-180 + 45)*np.pi/180#hindlimb(-180 + 45)*np.pi/180

		centerSigma=1#1#forelimb1#hindlimb1
		distWeight=0.45#for maxDist#0.02#forelimb0.45#hindlimb0.1
		singWeight=0.55#0.02#forelimb0.55#hindlimb0.4
		marginWeight=0.55#0.02#forelimb0.55#hindlimb0.4


		##########################################
		###########define calculation graph#######
		##########################################
		windowSize=1+windowWing*2
		
		#D=setting.sphereCount*3+setting.scalarCount

		#sess = tf.InteractiveSession()
		self.inputSize=D*2*windowSize
		outputSize=2	#2 for x and y (before normalized)

		phaseGraph=tf.Graph()

		with phaseGraph.as_default():
			print(gpuID)
			with selectedDevice(gpuID):	#defalut device if gpuID==-1

				#placeholder
				inputPlace  = tf.placeholder(tf.float32, shape=[2,D,windowSize, None], name="input")	#2 is for POSE and HEAD ### Current Phase
				inputPlace2 = tf.placeholder(tf.float32, shape=[2,D,windowSize, None], name="input2") ### Next phase
				#firstLayerPlace3 = tf.placeholder(tf.float32, shape=[inputSize, None], name="input3")
				sessionSegmentMatrix = tf.placeholder(tf.float32, shape=[None, None])	#(pairCount, sessionCount)

				global_step_tensor = tf.Variable(0, trainable=False, name='global_step') ### Logging traning steps

				Ap = tf.Variable(Ap_firstHalf, dtype=tf.float32, trainable=False)	#Lower limit of the  phase progression loss, this variable will be changed at 5000th iteration

				L=[self.inputSize]+hiddenLayers+[outputSize]
				# print(L)

				#name is important if you want to save the session
				W=[0]	#keep variable, W[1] should be W1 (no W0)
				b=[0]	#keep variable, b[1] should be b1 (no b0)
				

				## Constuction of the network
				for i in range(1,len(L)):
					Wi=tf.Variable(tf.random_normal([L[i],L[i-1]], 0, np.sqrt(1.0/float(L[i-1]))),name="W"+str(i))
					bi = tf.Variable(tf.zeros([L[i],1]),name="b"+str(i))
					W.append(Wi)
					b.append(bi)
				
				#add noise to the current input
				inputPose=inputPlace[0,:,:,:]
				inputHead=inputPlace[1,:,:,:]

				# inputPose = tf.Print(inputPose, [tf.shape(inputPose)], "InputPose:")
				
				
				randomShape=tf.shape(inputPose) ### shape of the data point 
				noisyPose=inputPose + tf.random_normal(randomShape,mean=0,stddev=0.5)
				noisyHead=tf.nn.l2_normalize(inputHead + tf.random_normal(randomShape,mean=0,stddev=0.5),axis=1) ## normalized to make headding +/- equally
				## concatenation of 2 channels 
				noisyInput=tf.stack([noisyPose,noisyHead], axis=0)

				## forward passing of the noisy input
				prePhase3,phaseXY3,phaseRad3 = self.nnFlow(L,W,b,noisyInput)

				prePhaseMargin=prePhase3
				
				## Forward passing of @t(n)
				prePhase ,phaseXY ,phaseRad = self.nnFlow(L,W,b,inputPlace,"output")
				# prePhase = tf.Print(prePhase, [tf.shape(prePhase)], "prePhase:")
				## Forward passion of @t(n+1)
				prePhase2,phaseXY2,phaseRad2= self.nnFlow(L,W,b,inputPlace2)		
				
				#calculate penalty
				currentPhaseXY=phaseXY
				nextPhaseXY   =phaseXY2
				
				## Calculate the progress direction of phase difference
				crossProductZ = tf.multiply(currentPhaseXY[0,:], nextPhaseXY[1,:]) - tf.multiply(currentPhaseXY[1,:], nextPhaseXY[0,:])	#positive=counter-clockwise
				dotProduct = tf.reduce_sum(tf.multiply(currentPhaseXY, nextPhaseXY), axis=0)
				# dotProduct = tf.Print(dotProduct, [tf.shape(dotProduct)], "dotProduct:")
				phaseProgress = tf.atan2(crossProductZ, dotProduct)	#(-pi to pi)
				
				
				'''
				#this is for 2 parameters (Ap and Bp)
				below1=tf.less(phaseProgress,Ap)
				y1=0
				y2=np.pi/2
				x1=-np.pi
				x2=Ap
				below1P=tf.cos((phaseProgress*(y2-y1)+y1*x2-y2*x1)/(x2-x1))

				below2=tf.less_equal(phaseProgress,Bp)
				below2P=tf.fill(tf.shape(phaseProgress), 0.0)

				y1=-np.pi/2
				y2=0
				x1=Bp
				x2=np.pi
				elseP=tf.cos((phaseProgress*(y2-y1)+y1*x2-y2*x1)/(x2-x1))

				speedPenalty=tf.reduce_mean(tf.where(below1,below1P,tf.where(below2,below2P,elseP)))
				'''

				#this is for 3 parameters (Cp, Ap, Bp)
				isInCA=tf.logical_and(tf.greater_equal(phaseProgress,Cp),tf.less(phaseProgress,Ap))	#range [Cp,Ap)
				y1=0
				y2=0.5*np.pi
				x1=Cp
				x2=Ap
				CA_penalty=tf.cos((phaseProgress*(y2-y1)+y1*x2-y2*x1)/(x2-x1))

				isInAB=tf.logical_and(tf.greater_equal(phaseProgress,Ap),tf.less_equal(phaseProgress,Bp))	#range [Ap,Bp]
				AB_penalty=tf.fill(tf.shape(phaseProgress), 0.0)

				#from B to C 
				y1=-0.5*np.pi  #1.5*np.pi
				y2=0 #2*np.pi
				x1=Bp
				x2=2*np.pi+Cp
				aboveB_penalty=tf.cos((phaseProgress*(y2-y1)+y1*x2-y2*x1)/(x2-x1))
				belowC_penalty=tf.cos(((phaseProgress+2*np.pi)*(y2-y1)+y1*x2-y2*x1)/(x2-x1))  #below C is shifted to be above 180

				### Ask for initiative of the function

				isAboveB=tf.greater(phaseProgress,Bp)
				## if-else application of the condition -> finding mean of the batch
				speedPenalty=tf.reduce_mean(tf.where(isInCA,CA_penalty,tf.where(isInAB,AB_penalty,tf.where(isAboveB,aboveB_penalty, belowC_penalty))))

				#the original penalty
				#backwardPenalty=tf.reduce_mean(tf.nn.relu(-(crossProductZ-minimumPhaseProgressRad)))

				### centroid loss
				avgCentroidOfEachSession = tf.matmul(phaseXY,sessionSegmentMatrix)/tf.reduce_sum(sessionSegmentMatrix,axis=0,keepdims=True)
				# avgCentroidOfEachSession = tf.Print(avgCentroidOfEachSession, [tf.shape(avgCentroidOfEachSession)], "avgCentroidOfEachSession:")
				radiusSquareOfEachSession=tf.reduce_sum(tf.square(avgCentroidOfEachSession),axis=0)
				maxDistributionPenalty=distWeight*tf.reduce_max(radiusSquareOfEachSession)

				badDistributionPenalty=distWeight*tf.reduce_sum(tf.square(tf.reduce_mean(phaseXY,axis=1)))	#1.0 is the worst

				singularityPenalty=	singWeight*tf.reduce_mean((1/(centerSigma*np.sqrt(2*np.pi)))*tf.exp(-0.5*tf.square(tf.norm(prePhase,axis=0)/centerSigma))) ### Why norm 
				marginPenalty=	  marginWeight*tf.reduce_mean((1/(centerSigma*np.sqrt(2*np.pi)))*tf.exp(-0.5*tf.square(tf.norm(prePhaseMargin,axis=0)/centerSigma)))

				#cost=tf.add_n([speedPenalty,badDistributionPenalty,singularityPenalty,marginPenalty],name='cost')
				#cost=tf.add_n([speedPenalty,maxDistributionPenalty,singularityPenalty,marginPenalty],name='cost')#LP
				cost=tf.add_n([speedPenalty,badDistributionPenalty,singularityPenalty,marginPenalty],name='cost')#LP

				optimizer = tf.train.AdamOptimizer()

				train = optimizer.minimize(cost,global_step=global_step_tensor, var_list=W[1:]+b[1:])

				self.initializer=tf.global_variables_initializer()
				self.global_step_tensor=global_step_tensor
				#self.firstLayerPlace=firstLayerPlace
				self.inputPlace=inputPlace
				self.inputPlace2=inputPlace2
				self.sessionSegmentMatrix=sessionSegmentMatrix
				self.cost=cost
				self.speedPenalty=speedPenalty
				self.badDistributionPenalty=badDistributionPenalty
				self.maxDistributionPenalty=maxDistributionPenalty
				self.singularityPenalty=singularityPenalty
				self.marginPenalty=marginPenalty

				self.train=train
				self.prePhase=prePhase
				self.phaseXY=phaseXY
				self.phaseRad=phaseRad
				self.prePhaseMargin=prePhaseMargin

				self.assignSecondAp = tf.assign(Ap, Ap_secondHalf)	#to be run at 5000th iteration

		#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90)	
		#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth=True
		self.sess = tf.Session(graph=phaseGraph,config=config)

