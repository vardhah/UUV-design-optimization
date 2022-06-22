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

r1l=50; r1h=600;r2l=1;r2h=1850;r3l=50;r3h=600; r4l=50;r4h=200;r5l=1;r5h=5;r6l=1;r6h=50; 

input_size=6                             # input size may change if integer/ordinal type variable and represented by one-hot encoding
num_variable = 6                        # number of variables  both real & int type 
output_size=1                            # number of output 
ranges=[r1l,r1h,r2l,r2h,r3l,r3h,r4l,r4h,r5l,r5h,r6l,r6h]                # ranges in form of [low1,high1,low2,high2,...]

mask=['real','real','real','real','real','real']                     # datatype ['dtype1','dtype2']
categories=[[None],[None],[None],[None],[None],[None]]  




path='./models/nn_acc_5percent.pt'


def run(test_data): 
    copied_test_data=np.copy(test_data)
    fitted_test_data= data_preperation(copied_test_data,mask,np.array(ranges),categories)
    fitted_text_X = SimDataset(fitted_test_data)
    
    print('fitted X:',fitted_text_X)
    print('Model is:',path)
    neuralNet= SNet(input_size,output_size)
        
    try: 
        neuralNet.load_state_dict(torch.load(path))       
        print("Loaded earlier trained model successfully")
    except: 
        print('Couldnot find weights of NN')  
           
    with torch.no_grad(): 
            output = neuralNet(torch.from_numpy(fitted_text_X).float())
              
    output=output.cpu().detach().numpy()
    return output



from pymoo.core.problem import ElementwiseProblem

class MyProblem(ElementwiseProblem):

    def __init__(self,d,tl):
        self.dia=d ; self.tl=tl
        super().__init__(n_var=4,
                         n_obj=1,
                         n_constr=0,
                         xl=np.array([10,10,10,1]),
                         xu=np.array([573,573,50,50]))

    def _evaluate(self, x, out, *args, **kwargs):
        #print('self tl is:',self.tl ,'x is:',x)
        b= self.tl-x[0]-x[1]
        #print('c is:',b ,'x is:',x) 
        X= np.array([x[0],b,x[1],self.dia,x[2]*0.1,x[3]])
        X= X.reshape(1,-1)
        print('X is:',X[0])
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
        print('Output is:',output[0][0])
        out["F"] = [output[0][0]]
        #out["G"] = [np.array([1])]




if __name__=='__main__':
	
	############################
	data_file_name='pymoo_nn.csv'   
	D=191; total_len=1330
	dim=4;n=100;

	pop_size=5
	n_gen=100
	#################################################
	#given total length & D => need to find a,c,n,theta
	problem = MyProblem(D,total_len) 
    
	already_run = len(glob.glob(data_file_name))
	print('file exist?:',already_run)
	#############
	
	###########Genetic algorithm ######################
	algorithm = GA(pop_size=pop_size,eliminate_duplicates=True)
	termination = get_termination("n_eval", n)
	res = minimize(problem,algorithm,termination,verbose=True, save_history=True)
	print('X is:', res.X)
	print('F is:', res.F)
	##################################
	"""
	######### Nealder Mead ###########################
	algorithm = NelderMead()
	termination = get_termination("n_eval", 100)
	res = minimize(problem,algorithm, termination,seed=1,verbose=False,save_history=True)
	###################################################
	"""

