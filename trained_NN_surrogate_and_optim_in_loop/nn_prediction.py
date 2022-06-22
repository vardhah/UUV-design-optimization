import os
import glob
import pathlib
import numpy as np
import torch
import torch.nn as nn
from student_model import SNet
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from utils import *
#from trainer import model_trainer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from lp import load_N_predict
import shutil




r1l=50; r1h=600;r2l=1;r2h=1850;r3l=50;r3h=600; r4l=50;r4h=200;r5l=1;r5h=5;r6l=1;r6h=50; 

input_size=6                             # input size may change if integer/ordinal type variable and represented by one-hot encoding
num_variable = 6                        # number of variables  both real & int type 
output_size=1                            # number of output 
ranges=[r1l,r1h,r2l,r2h,r3l,r3h,r4l,r4h,r5l,r5h,r6l,r6h]                # ranges in form of [low1,high1,low2,high2,...]

mask=['real','real','real','real','real','real']                     # datatype ['dtype1','dtype2']
categories=[[None],[None],[None],[None],[None],[None]]  




# hyperparameters ()
batch_size = 32
device='cuda'
loss_fn=nn.L1Loss()
learning_rate=0.001
num_co=[]



first_run=1
device='cuda'

nnstorage=glob.glob("./models/*.pt")
print('nns are:',nnstorage) 

test_data= np.loadtxt("./data/dataware/test_data.txt", delimiter=" ",skiprows=0, dtype=np.float32)
result_file_name= './data/prediction_result.csv'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used

def estimate_accuracy(test_data,prediction):
    print('test_data:',test_data.shape,'prediction:',prediction.shape)
    copied_test_data=np.copy(test_data)
    ground_truth=label_data(copied_test_data,prediction)
    
    index_f = np.where(ground_truth[:,-1]==1)
    index_p = np.where(ground_truth[:,-1]==0)
      
    failed_gt= ground_truth[index_f[0]]
    passed_gt=ground_truth[index_p[0]]

    print('outlier is:',failed_gt.shape[0])
    accuracy=( passed_gt.shape[0]/(passed_gt.shape[0]+failed_gt.shape[0]))
    print('Accuracy is:', accuracy) 
    return accuracy


def calc_residual(truth,pred):
    delz= np.subtract(truth,pred)
    residual=np.divide(np.abs(delz),np.abs(truth))
    sum_residual= np.sum(residual)/truth.shape[0]
    print('Shape of residual vector is:',residual.shape,'avg residual is:',sum_residual)

    
def calc_deviation(truth,pred):
    delz= np.subtract(truth,pred)
    var= np.sum(np.square(delz))/truth.shape[0]
    print('variance is:',var)
    std= np.power(var,0.5)
    print('std is:',std)
    return std

def run(): 
    first_run=1
    copied_test_data=np.copy(test_data)
    fitted_test_data= data_preperation(copied_test_data,mask,np.array(ranges),categories)
        
    testing_data = SimDataset(fitted_test_data)
    fitted_text_X= fitted_test_data[:,:-1]; fitted_test_y=fitted_test_data[:,-1]
    print('fitted X:',fitted_text_X,'fitted test Y:',fitted_test_y)
        
    for nn in nnstorage:
      path=nn
      print('Model is:',path)
      neuralNet= SNet(input_size,output_size)
        
      try: 
        neuralNet.load_state_dict(torch.load(path))       
        print("Loaded earlier trained model successfully")
      except: 
        print('Couldnot find weights of NN')  
           
      with torch.no_grad(): 
            output = neuralNet(torch.from_numpy(fitted_text_X).float())
              
      output=output.cpu().detach().numpy()
      estimate_accuracy(test_data,output)
      if first_run==1:
           print('in 1st runid')
           multi_runresults= np.array(output).reshape(-1,1)
           first_run=0
      else: 
           multi_runresults= np.concatenate((multi_runresults,np.array(output).reshape(-1,1)),axis=1)
      np.savetxt(result_file_name,multi_runresults,  delimiter=',')  


def load_and_average(): 
     predicted_data= np.loadtxt(result_file_name, delimiter=",",skiprows=0, dtype=np.float32)
     print('shape of predicted matrix:',predicted_data.shape) 
     mean_prediction= np.average(predicted_data, axis=1)
     print('test prediction:',mean_prediction)
     estimate_accuracy(test_data,mean_prediction)
     return mean_prediction     

if __name__ == "__main__":
        run()  
        #pred=load_and_average()
        pred= np.loadtxt(result_file_name, delimiter=",",skiprows=0, dtype=np.float32)
        test= test_data[:,-1]
        print('pred shape:',pred.shape,'test shape is:', test.shape)
        calc_residual(test,pred)
        calc_deviation(test,pred)
        """
       
        ###Load evaluation data ,derived ground_truth and created storage for result  
        test_data= np.loadtxt("./data/dataware/test_data.txt", delimiter=" ",skiprows=0, dtype=np.float32)
        
        copied_test_data=np.copy(test_data)
        fitted_test_data= data_preperation(copied_test_data,mask,np.array(ranges),categories)
        
        testing_data = SimDataset(fitted_test_data)
        fitted_text_X= fitted_test_data[:,:-1]; fitted_test_y=fitted_test_data[:,-1]
        print('fitted X:',fitted_text_X,'fitted test Y:',fitted_test_y)
        
        for nn in nnstorage:
         path=nn
         neuralNet= SNet(input_size,output_size)
        
         try: 
           neuralNet.load_state_dict(torch.load(path))       
           print("Loaded earlier trained model successfully")
         except: 
           print('Couldnot find weights of NN')  
         
           
         with torch.no_grad(): 
            output = neuralNet(torch.from_numpy(fitted_text_X).float())
              

         copied_test_data=np.copy(test_data)
         output=output.cpu().detach().numpy()
         ground_truth=label_data(copied_test_data,output)
    
         index_f = np.where(ground_truth[:,-1]==1)
         index_p = np.where(ground_truth[:,-1]==0)
      
         failed_gt= ground_truth[index_f[0]]
         passed_gt=ground_truth[index_p[0]]

         result=( passed_gt.shape[0]/(passed_gt.shape[0]+failed_gt.shape[0]))
         print('Accuracy is:', result) 
         if first_run==1:
           print('in 1st runid')
           multi_runresults= np.array(output).reshape(-1,1)
           first_run=0
         else: 
           multi_runresults= np.concatenate((multi_runresults,np.array(output).reshape(-1,1)),axis=1)
         print('result is:',multi_runresults)
         np.savetxt(result_file_name,multi_runresults,  delimiter=',')
         """
      
	



