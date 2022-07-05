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
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.factory import get_termination
from pymoo.algorithms.soo.nonconvex.nelder_mead import NelderMead
sys.dont_write_bytecode = True
import random
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



from pymoo.core.problem import ElementwiseProblem

class CFDProblem(ElementwiseProblem):

    def __init__(self,d,tl,filename):
        self.d=d;self.tl=tl;self.flag=0; self.data=None; self.filename=filename
        
        super().__init__(n_var=4,
                         n_obj=1,
                         n_constr=0,
                         xl=np.array([10,10,10,1]),
                         xu=np.array([573,573,50,50]))

    def _evaluate(self, x, out, *args, **kwargs):
        save_design_points(np.array([x[0],x[1],x[2],x[3],self.d,self.tl]))
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
        out["F"] = [result]
     
        
        if self.flag==0:
        	self._data= np.append(x,result).reshape(1,-1) ; self.flag=1
        else: 
        	self._data= np.concatenate((self._data,np.append(x,result).reshape(1,-1)),axis=0)
        	np.savetxt(self.filename,self._data,  delimiter=',')




class DummyProblem(ElementwiseProblem):

    def __init__(self,d,tl,filename):
        self.d=d;self.tl=tl; self.flag=0; self.data=None; self.filename=filename
        
        super().__init__(n_var=4,
                         n_obj=1,
                         n_constr=0,
                         xl=np.array([10,10,10,1]),
                         xu=np.array([573,573,50,50]))

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = [x[0]+x[1]+x[2]+x[3]+self.d+self.tl]
        #out["G"] = [g1]
        if self.flag==0:
        	self._data= np.append(x,out["F"][0]).reshape(1,-1) ; self.flag=1
        else: 
        	self._data= np.concatenate((self._data,np.append(x,out["F"][0]).reshape(1,-1)),axis=0)
        	np.savetxt(self.filename,self._data,  delimiter=',')





def run_pymoo(run_id=0,optimiser='GA',seeds=0):
	############################
	data_file_name='./data/pymoo_'+optimiser+str(run_id)+'.csv'   
	D=191;tl=1330
	dim=4;n=50;
	pop_size=5
	#################################################
	#given total length & D => need to find a,c,n,theta
	problem = CFDProblem(D,tl,data_file_name) 
	#problem = DummyProblem(D,tl,data_file_name)     
	already_run = len(glob.glob(data_file_name))
	print('file exist?:',already_run)

	

	###########Genetic algorithm ######################
	if optimiser=='GA':
		algorithm = GA(pop_size=pop_size,eliminate_duplicates=True)
		termination = get_termination("n_eval", n)
		res = minimize(problem,algorithm,termination,seed=seeds,verbose=True, save_history=True)
	##################################

	elif optimiser=='NM':
		algorithm = NelderMead()
		termination = get_termination("n_eval", n)
		res = minimize(problem,algorithm, termination,seed=seeds,verbose=False,save_history=True)
	else:
		print('****No valid optimiser****')
		

if __name__=='__main__':
	aqu1='GA'; aqu2='NM'
	run=[1,2,3,4,5]; seeds=[11,13,17,19,21]
	for i in range(len(run)):
		run_pymoo(run[i],aqu1,seeds[i])
		run_pymoo(run[i],aqu2,seeds[i])  
	
