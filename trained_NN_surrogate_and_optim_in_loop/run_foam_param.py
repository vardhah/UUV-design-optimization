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

sys.dont_write_bytecode = True


                             # input size may change if integer/ordinal type variable and represented by one-hot encoding
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
	result = run_dex()
	os.chdir(prev)
	return result



def _evaluate(x):
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
        result = run_dex()
        os.chdir(prev)
        return result




if __name__=='__main__':
        # DS : 
	#### total_len  = 1000-2000;                     % units in mm  
	#### Diameter   = 50-200 ;                       % units in mm 
	#### nose(a)       = 50-600                         % units in mm 
	#### tail(c)       = 50-600                         % units in mm 
	#### n          = 10-50                          % dimensionless parameter 
	#### theta      = 1-50 ;	                      % Semi-angle of tail 

	a=572.71;c=441.91;d=65; n=10.0; theta=3.07;tl=1250
	 
	#Do not delete beyond it.. 
	x= np.array([a,c,n,theta,d,tl])
	drag=_evaluate(x)
	print('x is:',x,'drag is:',drag)



