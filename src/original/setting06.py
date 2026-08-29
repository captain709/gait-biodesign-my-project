import numpy as np

sphereCount=0
scalarCount=2

pcaGroup =[  2,  2,  1,  1,  1,  1,  1,  1, 1]	#target at azeq form

windowWing=3
regressWing=3

predictionGap=1	 #this should be important
periodLength=50

cropHead=0
cropTail=0

'''
scalarActualRange=np.zeros([scalarCount,2])
scalarActualRange[0,:]=[-90,90]
scalarActualRange[1,:]=[-90,90]
scalarActualRange[2,:]=[-90,90]
scalarActualRange[3,:]=[-90,90]
scalarActualRange[4,:]=[-90,90]
scalarActualRange[5,:]=[-100,100]

#this is done manually (calculate from the full range)
equalScaling=np.ones([8,1])
equalScaling[0,0]=2/np.pi	#one sphere (upper arm)
equalScaling[1,0]=2/np.pi 	#one sphere (forearm)
equalScaling[2,0]=3/180		#trunk x
equalScaling[3,0]=3/180		#trunk y
equalScaling[4,0]=3/180		#trunk z
equalScaling[5,0]=3/180		#preshoulderFront
equalScaling[6,0]=3/180		#preshoulderUp
equalScaling[7,0]=3/200		#forearmPronation
#equalScaling.dump(intermediate+"equalRandomDistance.dat")
'''
traceColors=[	
			'#1f77b4',
			'#ff7f0e',
			'#2ca02c',
			'#d62728',
			'#9467bd',
			'#8c564b',
			'#e377c2',
			'#7f7f7f',
			'#bcbd22',
			#'#17becf',
			#'#030303',	#
			'#419388',	#
			
			'#080808',	#wristX
			'#080808',	#wristY
			'#080808',	#wristZ
		]

traceName=[
	'upper arm (X)',
	'upper arm (Y)',
	'upper arm (Z)',
	'forearm (X)',
	'forearm (Y)',
	'forearm (Z)',
	'trunk frontal flexion (X)',
	'trunk rotation (Y)',
	'trunk lateral flexion (z)',
	#'scapular protraction',
	#'scapular elevation',
	'forearm pronation',

	'wrist (X)',
	'wrist (Y)',
	'wrist (Z)'
]

zTraceName=[
	'upper arm',
	'forearm',
	'trunk frontal flexion (X)',
	'trunk rotation (Y)',
	'trunk lateral flexion (z)',
	#'scapular protraction',
	#'scapular elevation',
	'forearm pronation',

	'wrist (X)',
	'wrist (Y)',
	'wrist (Z)'
]

legendNameList=[
	'UA Pnt.',#'Upper arm point.',
	'FA Pnt.',#'Forearm point.',
	'TS Ffx.',#'Torso front. flex.',
	'TS Turn',#'Torso L/R turn',
	'TS Lfx.',#'Torso lat. flex.',
	#'SCP Prt.',#'Scapula protract.',
	#'SCP Elv.',#'Scapula elevation',
	'FA Prn.',#'Forarm pronation'
	'WR X',	#wrist
	'WR Y',
	'WR Z',
]