# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 09:51:57 2021

@author: HPP
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from student_model import SNet
import matplotlib.pyplot as plt

# On a given 'test_data' , 'input and output size of network' and 'model location',  it loads the file and make prediction
# return: test prediction   
#to run:  lnp=load_N_predict() ; lnp.run() 

class load_N_predict():
    def __init__(self,test_data,input_size=1,output_size=1,model_loc="./models/test/",net_type='S'):
         self.test_data = test_data
         self.input_size=input_size
         self.output_size=output_size
         self.model_loc=model_loc
         self.net_type=net_type

    def to_tensor(self, numpy_array):
        return torch.from_numpy(numpy_array).float()
    
    def to_array(self, torch_tensor):
        return torch_tensor.cpu().float().numpy()
    
    # to do the prediction function 
    def getPrediction(self, state,model):
        with torch.no_grad():
            action_original = model(self.to_tensor(state))
        return self.to_array(action_original)

    def run(self):
      try:
        #model = os.listdir(self.model_loc)
        #print('model is:',self.model_loc,'net type is:',self.net_type)
        if self.net_type=='S':
          neuralN=SNet(self.input_size,self.output_size)        
          #model_name=self.model_loc+model[0]
          model_name=self.model_loc
          print('predicting from model:',model_name)
          neuralN.load_state_dict(torch.load(model_name))
        test_prediction=self.getPrediction(self.test_data,neuralN)
        
      except:
        print("Cannot find model weights in this directory")
      return test_prediction
