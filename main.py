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

sys.dont_write_bytecode = True



input_size=2                             # input size may change if integer/ordinal type variable and represented by one-hot encoding



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


def run_cad_cfd():
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
	print(result)


if __name__=='__main__':
	
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
	print(result)



