####################################################
########### Swarm class ##############
####################################################

### Updated by Matan Yah Ben Zion 20230130 - cleaned and WELL DOCUMENTED
### Updated by Matan Yah Ben Zion 20221014 - added reinforcement learning 
### Updated by Matan Yah Ben Zion 20220524 - added phototaxis

################# Written Originally by Naomi Oppenheimer and Matan Yah Ben Zion #################
#################################### Tel Aviv University #########################################

#### This class defines the variables of the swarm
#### The swarm consists of N agents with the following characteristics:
#### Location (x,y), orientation (nx,,ny), threshold(th), and reward (re)
#### th is th threshold for phototaxis
#### re is the embedded reward

import numpy as np

def l2n(x):
	return np.array(x);


class swarm():
	def __init__(self, N,v0=10,rSteric = 0.32,wS=1E2,wA=-2,slowDown=1/15):
		#Declare swarm class internal variables
		
		#Assign constants:
		self.N = N;
		self.v0 = v0;
		self.wA = wA;
		self.wS = wS;
		self.rSteric = rSteric;
		self.slowDown = slowDown
		
		#Allocate variables:
		self.x = np.zeros((1*self.N),dtype=np.float32);
		self.y = np.zeros((1*self.N),dtype=np.float32);
		self.nx = np.zeros((1*self.N),dtype=np.float32);
		self.ny = np.zeros((1*self.N),dtype=np.float32);
		self.re = np.zeros((1*self.N),dtype=np.float32);
		self.th = np.zeros((1*self.N),dtype=np.float32);


	def initializeSwarm(self,boxSize=1):
        #Set intial positions and orienations, thresholds, and rewards
		self.initializeRandomPositions(boxSize=boxSize);
		self.initializeRandomOrientations();
		self.initializeRandomThresholds();
		self.initializeRewards();

	def initializeRandomPositions(self,boxSize=1):
		#initial particles potision randomly
		self.x = np.random.rand(1*self.N)*boxSize;
		self.y = np.random.rand(1*self.N)*boxSize;

	def initializeRandomOrientations(self):
		#Initial particles' orientations randomly
		a = np.random.rand(1*self.N)*np.pi; #Angle between 0-2pi
		self.nx = np.cos(a)
		self.ny = np.sin(a)
				
	def initializeRandomThresholds(self):
		#Initial particles' orientations randomly
		self.th = np.random.rand(1*self.N); #threshold between 0..1
		
	def initializeRewards(self):
		self.re = np.zeros(1*self.N);