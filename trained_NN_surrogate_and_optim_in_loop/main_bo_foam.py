import argparse
import os
import numpy as np
from utils import *
import pandas as pd
import shutil
import glob 
import subprocess
import time
#from run_dexof import *
import sys 
from cfd_sim.run_dexof import *
from cfd_sim.dexof_reader_class import parse_dex_file
import GPyOpt
from subprocess import PIPE, run
import random
sys.dont_write_bytecode = True
from numpy.random import seed


input_size=2                             # input size may change if integer/ordinal type variable and represented by one-hot encoding
cad_storage_name= './cad_sim/design_points.csv'
cfd_storage_name= './cfd_sim/design_points.csv'

num_iteration=20                        # Number of iteration of sampling
init_samples=50 
budget_samples=50                        # Number of samples-our budget
ranges=[-10,0,-6.5,0]                    # ranges in form of [low1,high1,low2,high2,...]

src= './cad_sim/stl_repo'
dst='./cfd_sim/stl_cfd'

#d=191; tl=1330    #Exp1
d=190; tl=1330    #Exp2
csv_filename_bo_foam= 'bo_lcb_foam_simtime_D'+str(d)+'_L'+str(tl)+'.csv'

def delete_dir(loc):
    print('*Deleted directory:',loc)
    shutil.rmtree(loc)

def copy_dir(src,dst):
	print('*Copied directory from',src,'to destination:',dst)
	shutil.copytree(src, dst)

def deletefiles(loc):
	print('Deleted files from location:',loc)
	file_loc= loc+'/*'
	files = glob.glob(file_loc)
	for f in files:
		os.remove(f)

def copy_file(src,dst):
	print('*Copied file from',src,'to destination:',dst)
	shutil.copy(src, dst)

def save_design_points(x):
    np.savetxt(cad_storage_name,x,  delimiter=',')
    np.savetxt(cfd_storage_name,x,  delimiter=',')


flag=0
def run_cad_cfd(x):
	global flag,_data
	start=time.time()
	print('shape of x:',x.shape)
	save_design_points(np.array([x[0][0],x[0][1],x[0][2],x[0][3],d,tl]))
	b= tl-x[0][0]-x[0][1]
	delete_dir(dst)
	subprocess.call('./cad_sim/run_cad.sh')
	copy_dir(src,dst)
	deletefiles(src)
	prev = os.path.abspath(os.getcwd()) # Save the real cwd
	print('prev is',prev)
	cfd_sim_path= prev+'/cfd_sim'
	print('func path is:',cfd_sim_path)
	os.chdir(cfd_sim_path)
	result = main_run()
	os.chdir(prev)
	end=time.time(); sim_time=end-start
	sim_data= np.array([x[0][0],b,x[0][1],d,x[0][2],x[0][3],result,sim_time])
	if flag==0:
        	_data= sim_data.reshape(1,-1) ; flag=1
	else: 
		_data= np.concatenate((_data,sim_data.reshape(1,-1)),axis=0)
		np.savetxt(csv_filename_bo_foam,_data,  delimiter=',')
	return result


if __name__=='__main__':
	
	seeds=101
	bounds = [{'name': 'myring_a', 'type': 'continuous', 'domain': (50,50)},
	        {'name': 'myring_c', 'type': 'continuous', 'domain': (50,600)},
            {'name': 'n', 'type': 'continuous', 'domain': (10,50)},
            {'name': 'theta', 'type': 'continuous', 'domain': (1,50)}]

    
	################################################

	seed(seeds)
	max_time  = None 
	max_iter  = 50
	num_iter=9
	batch= int(max_iter/num_iter)
	tolerance = 1e-8     # distance between two consecutive observations 
	data_file_name='bo_lcb_foam'   
	#################################################
	already_run = len(glob.glob(data_file_name))
	print('file exist?:',already_run)

	print('Batch is:',batch)
	for i in range(num_iter): 

	 if already_run==1:
	   evals = pd.read_csv(data_file_name, index_col=0, delimiter="\t")
	   Y = np.array([[x] for x in evals["Y"]])
	   X = np.array(evals.filter(regex="var*"))
	   myBopt2D = GPyOpt.methods.BayesianOptimization(run_cad_cfd, bounds,model_type = 'GP',X=X, Y=Y,
                                              acquisition_type='LCB',  
                                              exact_feval = True) 

	   print('In other runs run')
	 else: 
	   myBopt2D = GPyOpt.methods.BayesianOptimization(f=run_cad_cfd,
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
	
