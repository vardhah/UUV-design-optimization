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


sys.dont_write_bytecode = True


cad_storage_name= './cad_sim/design_points.csv'
cfd_storage_name= './cfd_sim/design_points.csv'

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
    #np.savetxt(cfd_storage_name,x,  delimiter=',')

def run_cad_cfd(x):
	save_design_points(x)

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


if __name__=='__main__':
	
	############################
	data_file_name='vmc.csv'   
	#D=np.random.rand()*150+50;
	dim=5;n=1
	#################################################
	#Range[total_length,Diameter,nose,tail,n,theta]
	#ranges=[1000,2000,D,3*D,D,3*D,10,50,1,50]    
    
	max_iter  = 1000
	#################################

	for i in range(max_iter):
		D=np.random.rand()*150+50;
		ranges=[1000,2000,D,3*D,D,3*D,10,50,1,50]  
		ds= random_sampling(dim,n,ranges)
		ds= np.append(D,ds)
		print('ds is:',ds, 'its shaape is',ds.shape[0])
		design_point= ds
		print('design point is:',design_point)
		fd=run_cad_cfd(design_point)
		 
