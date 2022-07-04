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


sys.dont_write_bytecode = True

d=191;tl=1330

cad_storage_name= './cad_sim/design_points.csv'
cfd_storage_name= './cfd_sim/design_points.csv'

src= './cad_sim/stl_repo'
dst='./cfd_sim/stl_cfd'

def generate_intial_seed(n=5):
    ds= random_sampling(dim,n,ranges)
    np.savetxt(initial_seed.csv,x,  delimiter=',')


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
    #np.savetxt(cfd_storage_name,x,  delimiter=',')

def run_cad_cfd(x):
	save_design_points(np.array([x[0],x[1],x[2],x[3],d,tl]))
	delete_dir(dst)
	subprocess.call('./cad_sim/run_cad.sh')
	copy_file(cad_storage_name,cfd_storage_name)
	copy_dir(src,dst)
	deletefiles(src)
	prev = os.path.abspath(os.getcwd()) # Save the real cwd
	print('prev is',prev)
	cfd_sim_path= prev+'/cfd_sim'
	print('func path is:',cfd_sim_path)
	os.chdir(cfd_sim_path)
	result = main_run()
	os.chdir(prev)
	return result



def doe(runid,doe_strategy,seeds):
	############################
	data_file_name='./data/doe_'+doe_strategy+str(runid)+'.csv'   
	dim=4;n=50;D=191
	#################################################
	#given total_len & D => need to find a,c,n,theta
	ranges=[10,3*D,10,3*D,10,50,1,50]    
    
	already_run = len(glob.glob(data_file_name))
	print('file exist?:',already_run)
	if already_run==1:
	    multi_runresults=np.loadtxt(data_file_name, delimiter=",",skiprows=0, dtype=np.float32)
	    multi_runresults= np.atleast_2d(multi_runresults)
	    #print('shape of multi_runresults:',multi_runresults.shape)
	max_iter  = 1
	#################################

	for i in range(max_iter):
		if doe_strategy=='vmc':
			ds= random_sampling(dim,n,ranges,seeds)
		elif doe_strategy=='lhc':	
			ds= lhc_samples_maximin(n,dim,ranges,seeds)  #maximin LHC
		else: 
			print('Unknown sampling strategy')
		print('ds is:',ds.shape[0])
		for i in range(ds.shape[0]):
		 already_run = len(glob.glob(data_file_name))	
		 design_point= ds[i]	
		 print('design point is:',design_point)
		 fd=run_cad_cfd(design_point)
		 #fd=10
		 #print('fd is:',fd)
		 sim_data=np.append(ds[i],fd).reshape(1,-1); print('Shape of sim_data:',sim_data.shape)
		 if already_run==0:
		   multi_runresults= sim_data
		 else:
		   multi_runresults= np.concatenate((multi_runresults,sim_data),axis=0)
		 #print('multirun result:',multi_runresults)
		 np.savetxt(data_file_name,multi_runresults,  delimiter=',')
         
	

if __name__=='__main__':
	aqu1='vmc'; aqu2='lhc'
	run=[1,2,3,4,5]; seeds=[11,13,17,19,23]
	for i in range(len(run)):
		doe(run[i],aqu1,seeds[i])
		doe(run[i],aqu2,seeds[i])  	
	
