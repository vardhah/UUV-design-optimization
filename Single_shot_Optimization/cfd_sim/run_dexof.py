# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 17:02:38 2022

@author:Vardhan Harsh (Vanderbilt University)
"""

import numpy as np
import os 
import subprocess
import glob 
import time
import re
import shutil

data_file_name='vmc_cfd3.csv' 

#######
path_stl = './stl_cfd/'
base_dexfile='rough_mesh_8cores.dex'
result_folder='./result_logs'
stl_name = 'swordfish' 
from .dexof_reader_class import parse_dex_file
default_max=1100
######

def run_dex(mesh=0.2):

 files= os.listdir(path_stl)
 aoa= 0
 result_array=[]

 dexof_editor= parse_dex_file(base_dexfile)
 print('Loaded files')
 itr=0
 for file_name in files:
    #vel=np.random.rand(1)*9+1
    dexof_editor.setall_mesh(mesh)
    print('--------------------------------------')
    print('---Running CFD iteration:', itr,'----------')
    
    arg_=['./run_dexof.sh']
    input_stl_file= path_stl
    print('stl file name:',file_name)
    input_stl_file+=file_name
    print('full path of stl file:',input_stl_file)
    
    dexof_editor.set_input_file(input_stl_file)
    split_dexfilename=base_dexfile.split('.')
    stlfile_name= file_name.split('.stl')
    _dex_file_name=split_dexfilename[0]+'_'+stlfile_name[0]
    new_dex_file_name=_dex_file_name+'.dex'
    print('New dex file name is:',new_dex_file_name)
    
    with open(new_dex_file_name, 'w') as f:
     f.write(dexof_editor.contents)
     
    exp_index=int(stlfile_name[0].split(stl_name)[1])
  
    arg_.append(new_dex_file_name)
    arg_.append(input_stl_file)
    #_aoa_str=str(round(aoa[aoa_index],1))
    _aoa_str=str(aoa)
    arg_.append(_aoa_str) 
    print('arg is:',arg_)
    start_time= time.time()
    subprocess.call(arg_)
    try:
     #reading results from folder
     folder='dir_'+_dex_file_name+'_aoa_'+_aoa_str
     src_resultfile='./'+folder+'/results.log'
     with open(src_resultfile, 'r') as f:
      result_output=f.read()
    
     Fd_out=re.search(r"Total\s+:\s*\(\s*\d*\.\d*",result_output)
     if Fd_out: 
        Fd_found = Fd_out.group(0).split(': (')[1]
        Fd_found= float(Fd_found)
     else: 
        Fd_found= default_max
     if Fd_found==0: 
        Fd_found=default_max

     print("######################################") 
     print("######################################")   
     print('Drag force is:', Fd_found)
     print("######################################")
     end_time=time.time()
     sim_time=end_time-start_time
     print('sim time is:',sim_time)
     os.remove(new_dex_file_name)
     os.remove('seaglider_out.stl')
     os.remove('temp.stl')
     shutil.rmtree(folder)
     del arg_
     return Fd_found
    except:
     return np.atleast_2d(np.array(default_max*2))  


def main_run():
    start=time.time()
    alt_mesh=[0.15,0.1,0.08,0.06,0.05]
    already_run = len(glob.glob(data_file_name))
    print('file exist?:',already_run)
    if already_run==1:
        multi_runresults=np.loadtxt(data_file_name, delimiter=",",skiprows=0, dtype=np.float32)
        multi_runresults= np.atleast_2d(multi_runresults)
        #print('shape of multi_runresults:',multi_runresults.shape)

    design_set_load= np.loadtxt('./design_points.csv', delimiter = ",")
    mesh=0.2
    Fd_found= run_dex(mesh)
    
    #print('Fd found is:',Fd_found,type(Fd_found))
    if (Fd_found>=1000):
      for i in range(len(alt_mesh)):
         mesh=alt_mesh[i]
         Fd_found= run_dex(mesh)
         if (Fd_found<1000):    
            break

    end=time.time(); sim_time=end-start
    drag_data=np.array([Fd_found,mesh,sim_time])
    data= np.atleast_2d(np.append(design_set_load,drag_data))
    print('--------->Data is:',data)
   
    #if already_run==0:
    #       multi_runresults= data
    # else:
    #       multi_runresults= np.concatenate((multi_runresults,data),axis=0)
    #     #print('multirun result:',multi_runresults)
    #np.savetxt(data_file_name,multi_runresults,  delimiter=',')
    return Fd_found
    

     
