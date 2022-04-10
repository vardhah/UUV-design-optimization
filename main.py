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
from cfd_sim.run_dexof import run_dex
from cfd_sim.dexof_reader_class import parse_dex_file
import GPyOpt


sys.dont_write_bytecode = True



input_size=2                             # input size may change if integer/ordinal type variable and represented by one-hot encoding
cad_storage_name= './cad_sim/design_points.csv'
cfd_storage_name= './cfd_sim/design_points.csv'

num_iteration=20                        # Number of iteration of sampling
init_samples=50 
budget_samples=50                        # Number of samples-our budget
ranges=[-10,0,-6.5,0]                    # ranges in form of [low1,high1,low2,high2,...]

src= './cad_sim/stl_repo'
dst='./cfd_sim/stl_cfd'


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

def run_cad_cfd(x):
	save_design_points(x)
	delete_dir(dst)
	subprocess.call('./cad_sim/run_cad.sh')
	copy_dir(src,dst)
	deletefiles(src)
	prev = os.path.abspath(os.getcwd()) # Save the real cwd
	print('prev is',prev)
	cfd_sim_path= prev+'/cfd_sim'
	print('func path is:',cfd_sim_path)
	os.chdir(cfd_sim_path)
	result = run_dex()
	os.chdir(prev)
	return result


if __name__=='__main__':
	
	#run_cad_cfd(np.array([1,2]))
	
	bounds = [{'name': 'first_y', 'type': 'continuous', 'domain': (0,50)},
            {'name': 'second_y', 'type': 'continuous', 'domain': (0,50)},
            {'name': 'third_y', 'type': 'continuous', 'domain': (0,50)},
            {'name': 'fourth_y', 'type': 'continuous', 'domain': (0,50)},
            {'name': 'fifth_y', 'type': 'continuous', 'domain': (0,50)}]

	################################################


	max_time  = None 
	max_iter  = 100
	num_iter=10
	batch= int(max_iter/num_iter)
	tolerance = 1e-8     # distance between two consecutive observations 

	#################################################
	first_run=1 
	print('Batch is:',batch)
	for i in range(num_iter): 

	 if first_run==0:
	   evals = pd.read_csv("sea_parrot", index_col=0, delimiter="\t")
	   Y = np.array([[x] for x in evals["Y"]])
	   X = np.array(evals.filter(regex="var*"))
	   myBopt2D = GPyOpt.methods.BayesianOptimization(run_cad_cfd, bounds,model_type = 'GP',X=X, Y=Y,
                                              acquisition_type='EI',  
                                              normalize_Y = True,
                                              acquisition_weight = 2) 

	   print('In other runs run')
	 else: 
	   myBopt2D = GPyOpt.methods.BayesianOptimization(f=run_cad_cfd,
                                              domain=bounds,
                                              model_type = 'GP',
                                              acquisition_type='EI',  
                                              normalize_Y = True,
                                              acquisition_weight = 2) 
	   first_run=0
	   print('In 1st run')
	 print('------Running batch is:',i) 
   
 # --- Run the optimization 
	 try:
	  myBopt2D.run_optimization(batch,verbosity=True)  
	 except KeyboardInterrupt:
	  pass
 
	 sim_data_x= myBopt2D.X;
	 myBopt2D.save_evaluations("sea_parrot")


	myBopt2D.plot_acquisition()  
	myBopt2D.plot_convergence()
	
