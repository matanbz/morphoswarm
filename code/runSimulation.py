#### This file initializes and runs a simulation of a phototaxis learning swarm of force-aligning active agents

### Updated 20230130 by Matan Yah Ben Zion. Clean up and WELL DOCUMENTED
### Updated 20221017 by Matan Yah Ben Zion. Added learning and walk in light
### Updated 20220611 by Matan Yah Ben Zion. Clean up
### Updated 20220527 by Matan Yah Ben Zion. Added period boundary condition

################# Written Originally by Naomi Oppenheimer and Matan Yah Ben Zion #################
#################################### Tel Aviv University #########################################


import numpy as np
import time
import pickle
import sys
from datetime import datetime

from os import mkdir

def l2n(x): return np.array(x);
def n2l(x): return list(x)

import swarm as swarm
import green as gr
import timePropagation as tp

import importlib  #importing the two other files to be used

importlib.reload(swarm)
importlib.reload(gr)
importlib.reload(tp)

def generateFileBaseName():
    dirName = datetime.today().strftime('%Y%m%d%H%M%S')+'results'+\
			'_dt'+'{:.0e}'.format(dt)+\
			'_N'+'{:.0e}'.format(N)+\
			'_T'+'{:.0e}'.format(T)+\
			'_kT'+'{:.0e}'.format(kT)+\
			'_rSteric'+'{:.1e}'.format(rSteric)+\
			'_wS'+'{:.0e}'.format(wS)+\
			'_V'+'{:.0e}'.format(v0)+\
			'_wA'+'{:.0e}'.format(wA)+\
            '_box'+'{:.0e}'.format(boxSize);

    mkdir(dirName)

    fileBaseName = dirName+'/file_'
    return fileBaseName

####### Set up simulation parameters

######### Timestep size
dt = 1E-4#5E-5 

######### Number of time steps
T = 1E4#1E4

######### Number of particles
N = 16

###### File saving paramters
#Total number of swarm snapshots to save
tSave = 1000 

# Number of files to produce during the run (each file will contain tSave/tArray snapshots)
tArray = 10 


#### Temperature 
kT = 2 
#kT = 100

#### Radius for steric interaction
rSteric = 20/150*4.8/2

#Core stiffness
wS = 1E2

#### Nominal active speed
v0 = 30

#### Aligment Strength
wA  = -3#E3

#### Slowdown factor when robot measures above threshold light
slowDown = 1/15

greenFunc = gr.green.grLearningPhototaxis 

################# RUN ###############

# The following define the parameters of the swarm

N = int(N)


######### Size of periodic box 
# rescale box by swarm size to keep constant area fraction

boxSize = 20*(N/128)**0.5 

#Initialize the eco system

sw = swarm.swarm(N=N,v0=v0,rSteric = rSteric,wS=wS,wA=wA,slowDown=slowDown)

# Random initial conditions:
sw.initializeSwarm(boxSize = boxSize) 


# Special settings:

# Start with "correct" threshold (comment out for randomly initialized swarm)
#sw.th = l2n([0.501]*N)


fileBaseName = generateFileBaseName() +'learningSwarm_'

######### Begin the simulation

# Initialize the propagator
prop = tp.timePropagation(dt,sw,fileBaseName)
print('Results will be saved to the following base name:')
print(fileBaseName)

ti = time.time()

# Start the simiulation
prop.timeProp5RungePeriodic(T,tSave,tArray,greenFunc,kT,boxSize)

tf = time.time()

print('run time ' +str((tf-ti)) +' seconds' );
