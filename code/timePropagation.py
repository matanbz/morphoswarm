####################################################
################ Time Propagation ##################
####################################################

### Updated by Matan Yah Ben Zion 20230130 - cleaned and WELL COMMENTED
### Updated by Matan Yah Ben Zion 20220527 - to allow periodic boundary condition

################# Written Originally by Naomi Oppenheimer and Matan Yah Ben Zion #################
#################################### Tel Aviv University #########################################

#### This class is the core of the simulation engine
#### It then propagates in time the next position of each particle in the swarm
#### Time propagation of simulation using 5th order Runge Kutta #######


import numpy as np
from scipy.stats import norm
import pickle

class timePropagation():
	'''
	Time propagation wrapper. Propagates a swarm system of particles.
	Particle parameters (position, orientation, reward and threshold) are defined in ps matrix
	once propagates, runs a 5th order runge kuta algorithm using the selected function.
	During the simulation tSave equispaced snapshots are saved to tArray files on the hard drive (usually 10 files)
	'''
    
	def __init__(self, dt,sw,fileBaseName):
		self.dt = dt
		self.sw = sw
		self.fileBaseName = fileBaseName
		
		self.Z = np.zeros((6,self.sw.N))
			
			
	def timeProp5RungePeriodic(self,T,tSave,tArray,greenFunc,kT,boxSize):
		'''
		T: number of time steps
		tSave: number of snap shots to save
		tArray: number of files to which save the snapshots
		greenFunc: interaction rules propogator
		kT: temperature for random noise
		'''
		
		self.T = T           # total number of number of timesteps
		self.tSave = tSave   # How many snap shots to take from the simulation
		self.tArray = tArray # How many files to save those snapshots equily distributed along simulation
		self.kT = kT         # Temperature in arbitrary units
		

		sw = self.sw;		
		sp = np.stack((sw.x,sw.y,sw.nx,sw.ny,sw.th,sw.re))
		
		self.SP = np.zeros( np.concatenate( ([np.int_(self.tSave/self.tArray)],np.shape(sp)) ))

		self.scaleNoise = np.sqrt(2.0*self.dt*self.kT) 
		
		# Numericaly solve the differential equation using incremental time propagation
		"""
		Using fifth order Runge-Kutta with constant step size 				
		""" 
		b1 = 35./384; b1s = 5179./57600; b2 = 0.; b2s = 0.; b3 = 500./1113; b3s = 7571./16695; 
		b4 = 125./192; b4s = 393./640; b5 = -2187./6784; b5s = -92097./339200; b6 = 11./84; b6s = 187./2100; 
		b7s = 1./40; 
		a21 = 1./5; 
		a31 = 3./40; a32 = 9./40; 
		a41 = 44./45; a42 = -56./15; a43 = 32./9;
		a51 = 19372./6561; a52 = -25360./2187; a53 = 64448./6561; a54 = -212./729;
		a61 = 9017./3168; a62 = -355./33; a63 = 46732./5247; a64 = 49./176; a65 = -5103./18656;

		for ttt in range(self.tArray):
			for tt in range(np.int(self.tSave/self.tArray)):
				self.SP[tt] = sp

				for t in range(np.int_(self.T/self.tSave)):
					########################## Numerical Solver #######################
					
					SP0 = sp[:4,:].copy() 

					self.Z = greenFunc(sw,sp,boxSize) 
					
					k1 = self.dt*self.Z[:4,:] 
					sp[:4,:] = SP0 + a21*k1;   

					self.Z = greenFunc(sw,sp,boxSize)
					k2 = self.dt*self.Z[:4,:]
					sp[:4,:] = SP0 + a31*k1 + a32*k2; 

					self.Z = greenFunc(sw,sp,boxSize)
					k3 = self.dt*self.Z[:4,:]
					sp[:4,:] = SP0 + a41*k1 + a42*k2 + a43*k3;

					self.Z = greenFunc(sw,sp,boxSize)
					k4 = self.dt*self.Z[:4,:]
					sp[:4,:] = SP0 + a51*k1 + a52*k2 + a53*k3 + a54*k4;

					self.Z = greenFunc(sw,sp,boxSize)
					k5 = self.dt*self.Z[:4,:]
					sp[:4,:] = SP0 + a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5;

					self.Z = greenFunc(sw,sp,boxSize)
					k6 = self.dt*self.Z[:4,:]

					advance = (b1*k1+b2*k2+b3*k3+b4*k4+b5*k5+b6*k6) #the value of a step according to fifth order Runge Kuta

					################################# Advance #################################

					#Advance positions:
					sp[:4,:] = SP0[:4,:] + advance[:4,:]
					
					#Add noise to orientation:
					noise = norm.rvs(size=np.shape(sp[2:4,:]), scale=self.scaleNoise);
					sp[2:4,:] += noise
					
					################################# Parameters passing #################################
					self.Z = greenFunc(sw,sp,boxSize) #calculate once more to find the change in parameters following the green function
					#Update the swarm system parameters
					sp[4:,:] = self.Z[4:,:]

					
					################################# Periodic boundary condition ##################################
					sp[0,:] %= boxSize; 
					sp[1,:] %= boxSize;
					
					################################# Normalize the orientation vector #################################
					invN = 1/np.sqrt(np.power(sp[2,:],2)+np.power(sp[3,:],2))
					sp[2,:] *=invN;
					sp[3,:] *=invN;
										
#save the file
			fileName = self.fileBaseName+'_'+str(ttt).zfill(3)+'_of_'+str(self.tArray).zfill(3)+'.pkl'
			fHand = open(fileName, 'wb')
			pickle.dump(self.SP[:,:,:],fHand)
			fHand.close()