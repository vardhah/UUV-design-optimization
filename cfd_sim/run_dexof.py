# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 17:02:38 2022

@author:Vardhan Harsh (Vanderbilt University)
"""

import numpy as np
import os 
import subprocess
from dexof_reader_class import parse_dex_file
import time
import re
import shutil

path_stl = './stl_cfd/'
base_dexfile='rough_mesh_8cores.dex'
design_set_load= np.loadtxt('./rudder_design_points.csv', delimiter = ",")
result_folder='result_logs'


files= os.listdir(path_stl)
aoa= 0
result_array=[]

dexof_editor= parse_dex_file(base_dexfile)
print('Loaded files')
itr=0
for file_name in files:
    print('--------------------------------------')
    print('---Running iteration:', itr,'----------')
    
    arg_=['./run_dexof.sh']
    input_stl_file= './stl_cfd/'
    #print('stl file name:',file_name)
    input_stl_file+=file_name
    #print('full path of stl file:',input_stl_file)
    
    dexof_editor.set_input_file(input_stl_file)
    split_dexfilename=base_dexfile.split('.')
    stlfile_name= file_name.split('.stl')
    _dex_file_name=split_dexfilename[0]+'_'+stlfile_name[0]
    new_dex_file_name=_dex_file_name+'.dex'
    print('New dex file name is:',new_dex_file_name)
    
    with open(new_dex_file_name, 'w') as f:
     f.write(dexof_editor.contents)
     
    exp_index=int(stlfile_name[0].split('rudder_uuv')[1])
  
    arg_.append(new_dex_file_name)
    arg_.append(input_stl_file)
    #_aoa_str=str(round(aoa[aoa_index],1))
    _aoa_str=str(aoa)
    arg_.append(_aoa_str) 
    print('arg is:',arg_)
    start_time= time.time()
    
    subprocess.call(arg_)
    
    #reading results from folder
    folder='dir_'+_dex_file_name+'_aoa_'+_aoa_str
    src_resultfile='./'+folder+'/results.log'
    dst_result_file='./'+result_folder+'/result_'+str(exp_index)+'.log'
    shutil.copyfile(src_resultfile, dst_result_file)
    
    
    with open(src_resultfile, 'r') as f:
     result_output=f.read()
    #print('folder is:',folder,'result output is:',result_output )
    Fd_out=re.search(r"Total\s+:\s*\(\s*\d*\.\d*",result_output)
    #print(Fd_out)
    if Fd_out: 
        Fd_found = Fd_out.group(0).split(': (')[1]
        Fd_found= float(Fd_found)
    else: 
        Fd_found= 2
    
    print('Drag force is:', Fd_found)
    end_time=time.time()
    sim_time=end_time-start_time
    del arg_
    print('sim time is:',sim_time)
    #print(design_set_load[aoa_index,:].shape)
    
    result=np.concatenate((design_set_load[(exp_index-1),:],np.array([Fd_found,sim_time]))).reshape(1,-1)
    if itr ==0:
        result_array=result
        itr+=1
    else: 
        result_array= np.concatenate((result_array,result),axis=0)
        
    print('-----------------------------------')
    print(result_array)
    np.savetxt('sim_out_rudder.csv',result_array,delimiter=',')
   
    os.remove(new_dex_file_name)
    shutil.rmtree(folder)
    
