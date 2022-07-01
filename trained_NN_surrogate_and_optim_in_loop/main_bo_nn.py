import argparse
import os
import numpy as np
from utils import *
import pandas as pd
import shutil
import glob 
import subprocess
import time
import sys
import pymoo 
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.factory import get_termination
from pymoo.algorithms.soo.nonconvex.nelder_mead import NelderMead
sys.dont_write_bytecode = True
import GPyOpt
import torch
import torch.nn as nn
from student_model import SNet
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from utils import *
#from trainer import model_trainer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from lp import load_N_predict
import shutil
import random

r1l=50; r1h=600;r2l=1;r2h=1850;r3l=50;r3h=600; r4l=50;r4h=200;r5l=1;r5h=5;r6l=1;r6h=50; 

input_size=6                             # input size may change if integer/ordinal type variable and represented by one-hot encoding
num_variable = 6                        # number of variables  both real & int type 
output_size=1                            # number of output 
ranges=[r1l,r1h,r2l,r2h,r3l,r3h,r4l,r4h,r5l,r5h,r6l,r6h]                # ranges in form of [low1,high1,low2,high2,...]

mask=['real','real','real','real','real','real']                     # datatype ['dtype1','dtype2']
categories=[[None],[None],[None],[None],[None],[None]]  




path='./models/nn_acc_5percent.pt'
sim_file_name= 'exp3_D165_tl 2000_for_foam'      # Need to change for each experiment
D=165; total_len=1200               #define problem requirement
dim=4;n_gen=20;pop_size=5           #GA settings
####Dont change below it #####
file_name= sim_file_name+'.csv'
max_file= sim_file_name+'_max.csv'
min_file= sim_file_name+'_min.csv'
opt_file=sim_file_name+'_optimal.csv'


dia=180;tl=1750

def _evaluate(x):
        #print('self tl is:',self.tl ,'x is:',x)
        b= tl-x[0][0]-x[0][1]
        #print('c is:',b ,'x is:',x) 
        X= np.array([x[0][0],b,x[0][1],dia,x[0][2]*0.1,x[0][3]])
        X= X.reshape(1,-1)
        #print('X is:',X[0])
        copied_test_data=np.copy(X)
        fitted_text_X= data_preperation(copied_test_data,mask,np.array(ranges),categories)
        #fitted_text_X = SimDataset(fitted_test_data)
        
        #print('fitted X:',fitted_text_X)
        #print('Model is:',path)
        neuralNet= SNet(input_size,output_size)
        
        try: 
          neuralNet.load_state_dict(torch.load(path))       
          #print("Loaded earlier trained model successfully")
        except: 
          print('Couldnot find weights of NN')  
           
        with torch.no_grad(): 
            output = neuralNet(torch.from_numpy(fitted_text_X).float())
              
        output=output.cpu().detach().numpy()
        return output[0][0]




if __name__=='__main__':
	
	random.seed(10)

	bounds = [{'name': 'myring_a', 'type': 'continuous', 'domain': (10,573)},
	        {'name': 'myring_c', 'type': 'continuous', 'domain': (10,573)},
            {'name': 'n', 'type': 'continuous', 'domain': (10,50)},
            {'name': 'theta', 'type': 'continuous', 'domain': (1,50)}]
	"""        
            {'name': 'tail1_y', 'type': 'continuous', 'domain': (1,95.5)},
            {'name': 'tail2_y', 'type': 'continuous', 'domain': (1,95.5)},
            {'name': 'tail3_y', 'type': 'continuous', 'domain': (1,95.5)},
            {'name': 'tail4_y', 'type': 'continuous', 'domain': (1,95.5)}]  
	
	"""

    
	################################################


	max_time  = None 
	max_iter  = 100
	num_iter=20
	batch= int(max_iter/num_iter)
	tolerance = 1e-8     # distance between two consecutive observations 
	data_file_name='bo_lcb_nn'   
	#################################################
	already_run = len(glob.glob(data_file_name))
	print('file exist?:',already_run)

	print('Batch is:',batch)
	for i in range(num_iter): 

	 if already_run==1:
	   evals = pd.read_csv(data_file_name, index_col=0, delimiter="\t")
	   Y = np.array([[x] for x in evals["Y"]])
	   X = np.array(evals.filter(regex="var*"))
	   myBopt2D = GPyOpt.methods.BayesianOptimization(_evaluate, bounds,model_type = 'GP',X=X, Y=Y,
                                              acquisition_type='LCB',  
                                              exact_feval = True) 

	   print('In other runs run')
	 else: 
	   myBopt2D = GPyOpt.methods.BayesianOptimization(f=_evaluate,
                                              domain=bounds,
                                              model_type = 'GP',
                                              acquisition_type='LCB',  
                                              exact_feval = True) 
	   already_run=1
	   print('In 1st run')
	 print('------Running batch is:',i) 
   
 # --- Run the optimization 
	 try:
	  myBopt2D.run_optimization(batch,eps = tolerance,verbosity=True)  
	 except KeyboardInterrupt:
	  pass
 
	 sim_data_x= myBopt2D.X;
	 myBopt2D.save_evaluations(data_file_name)


	myBopt2D.plot_acquisition()  
	myBopt2D.plot_convergence()
	





