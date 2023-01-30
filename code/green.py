#### This class contains the interaction rules for phototaxis and learning in a swarm of force-aligning active agents
#### It defines the function and a wrapper

### Updated 20230130 by Matan Yah Ben Zion Cleaning up and WELL DOCUMENTED
### Updated 20221017 by Matan Yah Ben Zion Implement Learning Robots Walk In light
### Updated 20220611 by Matan Yah Ben Zion General Clean-up
### Updated 20220527 by Matan Yah Ben Zion implementing periodic boundary condition

################# Written Originally by Naomi Oppenheimer and Matan Yah Ben Zion #################
#################################### Tel Aviv University #########################################


import numpy as np
	
def l2n(x):
	return np.array(x);
################################################# Phototaxiing Swarm in a Periodic Box ##############################################
def _grLearningPhototaxis(sw,sp,sP,boxSize):
### This is the main function to define the dynamics to propagate a swarm with force-alignnment and learning to phototaxi
### Here x,y, a are the configurational degrees of freedom of each robot
### th, and re are the internal threshold (policy) and reward (re)
### slowDown parameter sets the relative speed of robot when light threshold is triggered
### Light intensity has 3 regions [0,0.5-thresholdWidth/2, 0.5+threhosldWidth/2, 1]
### Robots with a light threshold in the mid range stop in the light
### The global threshold is 0.612
	
	
	rTol = 1E-8 # threshold for computing particles overlap
	r2Tol = np.power(rTol,2) 

	# Assign shorter name for various parameters (for improved readability)
	NN = sw.N
	
	wS = sw.wS
	v0 = sw.v0
	wA = sw.wA
	rSteric = sw.rSteric
	slowDown = sw.slowDown
	
	# Assign shorter names for positions and orienations arrays:
	
	sx = sp[0,:]
	sy = sp[1,:]
	snx = sp[2,:]
	sny = sp[3,:]
	sthreshold = sp[4,:]
	sreward = sp[5,:]
	
	#lightSpot radius
	rLightSpot = 0.16*boxSize # Illuminated radius ratio.	
	
	## Calculate the squares for later reference
	rLightSpot2 = np.power(rLightSpot,2) 
	
	thresholdWidth = 0.25
	ambientLight   = 0.5 - thresholdWidth/2
	spotLight      = 0.5 + thresholdWidth/2
	
	for i in range(NN):
		#Copy the individual parameters of each bot to update along the simulation step
		rewardi = sreward[i]
		thresholdi = sthreshold[i]
		
		for j in range(NN):
			if i != j:
				
				dx = sx[i] - sx[j]
				dy = sy[i] - sy[j]
				
				#find vectorial minimal separation on a torus
				dxA = np.abs(dx)
				dyA = np.abs(dy)
				
				if dxA > boxSize/2:
					dx = np.sign(dx)*(dxA-boxSize)
						
				if dyA > boxSize/2:
					dy = np.sign(dy)*(dyA-boxSize)
					
				r2 = np.power(dx,2) + np.power(dy,2)
				
				if r2 > r2Tol: 
				
					fSteric = 0
					rSteric2 = np.power(2*rSteric,2)

					if r2<rSteric2:

						r = np.sqrt(r2)
						irSteric = 2*rSteric/r
						fSteric = -wS*(1-irSteric)

						sP[0,i] += dx*fSteric
						sP[1,i] += dy*fSteric
						
						#Compare rewards upon collision
						#If the other bot's reward is greater
						#adopt its threshold
						rewardj = sreward[j]
						if rewardi < rewardj:
							thresholdi = sthreshold[j]
				
		# Calc distance from center of "light spot"
		rCentral2 = np.power(sx[i]-boxSize/2,2)+np.power(sy[i]-boxSize/2,2) 
		
		# Set robot's "observed light" by its location
		light = ambientLight		
		
		#If the robot is in the light spot increase its reward.
		#If its outside of the light spot decrease the reward to a minimum of 0
		if rCentral2 < rLightSpot2:
			light = spotLight			
			rewardi += 1

		else:
			#decrease reward outside the light
			rewardi -= 1
			if rewardi < 0:
				rewardi = 0

		sP[4,i] = thresholdi
		sP[5,i] = rewardi
		
		#set robots speed given its threshold and measured light
		v = v0

		#if local light is above threshold, change speed
		if thresholdi < light: 
			v *= slowDown
		 
		fAlign = wA*(snx[i]*sP[1,i]-sny[i]*sP[0,i]) #n cross force
		sP[2,i] +=  - sny[i]*fAlign
		sP[3,i] +=  + snx[i]*fAlign#(sny[i] - snx[i]*fAllign)*normFAllign - sny[i]
			
		sP[0,i] += v*snx[i] #currently spin is not normalized
		sP[1,i] += v*sny[i]
		


##############################################################################################################################
################################################# Container Class ############################################################
##############################################################################################################################

class green():
	'''
	A container class for the green function
	'''
#######################	
	def grLearningPhototaxis(sw,sp,boxSize):
		'''
		Propagates a swarm.
		Particles are in a periodic box of size boxSize.
		Particles interact through a soft core repulsion.
		Particles align with force according to Ben Zion et al, arXiv 2022.
		Each particle has a light threshold and its self-propulsion is adjusted given its current light measure.
		This leads to slow-in-light when the particle is indeed in the light with the correct threshold.
		This leads to slowdown also when a particle is not in the light but with a low threshold.
		'''
		NN = sw.N;
		
		sP = np.zeros(np.shape(sp)); 
		_grLearningPhototaxis(sw,sp,sP,boxSize);

		return sP	
	