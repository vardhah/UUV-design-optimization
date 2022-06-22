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
from pytorch_model_summary import summary


######change for each run_id
#run=[1,2,3,4,5]
run=[2]
####################
r1l=50; r1h=600;r2l=1;r2h=1850;r3l=50;r3h=600; r4l=50;r4h=200;r5l=1;r5h=5;r6l=1;r6h=50; 


search_itr=5 
input_size=6                             # input size may change if integer/ordinal type variable and represented by one-hot encoding
num_variable = 6                        # number of variables  both real & int type 
output_size=1                            # number of output 
ranges=[r1l,r1h,r2l,r2h,r3l,r3h,r4l,r4h,r5l,r5h,r6l,r6h]                # ranges in form of [low1,high1,low2,high2,...]

mask=['real','real','real','real','real','real']                     # datatype ['dtype1','dtype2']
categories=[[None],[None],[None],[None],[None],[None]]  




#Training hyperparameters ()
max_epoch = 400
at_least_epoch=25
batch_size = 32
device='cuda'
loss_fn=nn.L1Loss()
learning_rate=0.001
num_co=[]

#select the training regime:(one shot vs piecewise sqeezing)
N = []; minim=100; maxim=2400; sample=0
while sample<maxim:
   sample+=minim; N.append(sample)
print('Samples to evaluate is:',N)
result_file_name= './data/sim_result_uni_random.txt'
        
####docs#####
#Dataset :  eval_data: Container of leftover train data; traini_data:current itr training data; test_data: test data

       
device='cpu'
print('run is:',run)

if __name__ == "__main__":
       for runid in run: 
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used
        ###Load evaluation data ,derived ground_truth and created storage for result  
        earlier_passed_gt=0
        result=[]
        for n in N:
         flag_first=0
         #give the reference to test data
         test_data= np.loadtxt("./data/dataware/test_data.txt", delimiter=" ",skiprows=0, dtype=np.float32)
         if n==minim:
           print('****n is:',n)
           #Loading train data  bucket ( then divide it as train and validation set)
           pool_mesh= np.loadtxt("./data/dataware/train_data.txt", delimiter=" ",skiprows=0, dtype=np.float32)
           sim_data, eval_data=data_split_size(pool_mesh,minim)
           traini_data= sim_data

         else:
           print('****n is:',n)  
           t_data, eval_data=data_split_size(eval_data,minim)
           e_traini_data= t_data
           traini_data=np.concatenate((traini_data,e_traini_data),axis=0)
           
         
         print('train size:',traini_data.shape,'test_data:',test_data.shape,'eval size:',eval_data.shape) 

         ###################################################################################
         ################# Dont change anything beyond it ##################################
         ###################################################################################


         ########## prepare data for testing ####
         eval_mesh= test_data[:,0:6]
         #print('size of eval_mesh',eval_mesh.shape)
         f_ground_eval_mesh= test_data[:,6]
         #print('size of f_ground_eval',f_ground_eval_mesh.shape)
         copied_eval_mesh=np.copy(eval_mesh)
         fitted_eval_mesh= data_preperation(copied_eval_mesh,mask,np.array(ranges),categories)



         ############# create data holder for train and test
         train_data,validation_data= data_split(traini_data,proportion=0.1)
         copied_train_data=np.copy(train_data)
         copied_validation_data=np.copy(validation_data)
         #print('copied_train_data', copied_train_data[0])
         fitted_train_data= data_preperation(copied_train_data,mask,np.array(ranges),categories)
         fitted_validation_data= data_preperation(copied_validation_data,mask,np.array(ranges),categories)
         #########################################################
         #print('fitted_train_data', fitted_train_data[0])

         train_data = SimDataset(fitted_train_data)
         validate_data = SimDataset(fitted_validation_data)
         print('length of train data:',len(train_data))
	 
         path='./models/nn_rand_uni_'+str(runid)+'.pt'
         name= 'nn_rand_uni_'+str(runid)+'.pt'
         directory= './models/'
         last_itr_path='./models/nn_rand_uni_'+str(runid)+'_last_itr.pt'
         
         if n!= minim:
           shutil.copy(path,last_itr_path)
           got_better_model=False
         for _ in range(search_itr): 
          print('-------------<<<<<<<<Training SNET>>>>>>>>>>>>>-----------')
          neuralNet= SNet(input_size,output_size)
          try: 
           neuralNet.load_state_dict(torch.load(path))       
           print("Loaded earlier trained model successfully")
          except: 
           neuralNet= neuralNet.apply(initialize_weights)
           print('Randomly initialising weights')  
          #neuralNet= SNet(input_size,output_size).apply(initialize_weights)
          model = neuralNet.to(device) 
          optimizer = optim.Adam(model.parameters(), lr=learning_rate)
          epoch=0; loss_train=[];loss_validate=[]     
          
          #print(summary(model, torch.zeros((batch_size,6)), show_input=True))
          while True: 
            #print('training epoch:',epoch)   
            if epoch > max_epoch:
                break    
            try:
                dataloader = DataLoader(train_data, batch_size, True)
                correct = 0
                for x, y in dataloader:
                	y=y.view(-1,1)
                	x, y = x.to(device), y.to(device)
                	output = model(x)
                	loss = loss_fn(output, y)
                	optimizer.zero_grad()
                	loss.backward() 
                	optimizer.step()
                	correct+= loss.item()
                train_loss=correct/len(train_data); loss_train.append(train_loss)
                
                with torch.no_grad(): 
                  dataloader = DataLoader(validate_data, batch_size, True)
                  correct = 0
                  for x, y in dataloader:
                    y=y.view(-1,1)
                    x, y = x.to(device), y.to(device)
                    output = model(x)
                    loss = loss_fn(output, y)
                    correct += loss.item()
                validate_loss= correct/len(dataloader); loss_validate.append(validate_loss) 
                
                if epoch <= at_least_epoch:
                  whichmodel=epoch  
                  torch.save(model.state_dict(), path)
                #if epoch%20==0:
                   #print('epoch is:',epoch)
                if epoch> at_least_epoch:
                 diff_loss=np.absolute(train_loss-validate_loss)
                 if flag_first==0: 
                   torch.save(model.state_dict(), path)
                   whichmodel=epoch 
                   flag_first=1
                   last_diff_loss=diff_loss

                 elif flag_first==1:
                  if last_diff_loss>diff_loss:
                   torch.save(model.state_dict(), path); whichmodel=epoch ;
                   last_diff_loss=diff_loss

            except KeyboardInterrupt:
                break
           
            epoch+=1

          #fig=plt.figure(figsize=(9,6))
          #plt.plot(loss_train,label='training')
          #plt.plot(loss_validate,label='validate')
          #plt.legend()
          #plt.show()

         
          print('--> Saved model is from', whichmodel , ' epoch')
         

         
          #prediction on eval data on student model
          lnp_e_m=load_N_predict(fitted_eval_mesh,input_size,output_size,path,'S')
          eval_mesh_pred=lnp_e_m.run()

          copied_dense_sample_data=np.copy(test_data)
          ground_truth=label_data(copied_dense_sample_data,eval_mesh_pred)
    
          index_f = np.where(ground_truth[:,-1]==1)
          index_p = np.where(ground_truth[:,-1]==0)
      
          failed_gt= ground_truth[index_f[0]]
          passed_gt=ground_truth[index_p[0]]
          #print('passed gt shape:',passed_gt.shape[0])
          if (passed_gt.shape[0]/(passed_gt.shape[0]+failed_gt.shape[0]))>earlier_passed_gt:
             earlier_passed_gt=(passed_gt.shape[0]/(passed_gt.shape[0]+failed_gt.shape[0]))
             current_passed_gt=earlier_passed_gt
             got_better_model=True
             print('***************Hip Hip Hurray****<got better model>****************')
             break
          else: 
             shutil.copy(last_itr_path,path) 
        
         print('=================================================================')
         if got_better_model==False:
            lnp_e_m=load_N_predict(fitted_eval_mesh,input_size,output_size,path,'S')
            eval_mesh_pred=lnp_e_m.run()
            copied_dense_sample_data=np.copy(test_data)
            ground_truth=label_data(copied_dense_sample_data,eval_mesh_pred)
            index_f = np.where(ground_truth[:,-1]==1)
            index_p = np.where(ground_truth[:,-1]==0)
            failed_gt= ground_truth[index_f[0]]
            passed_gt=ground_truth[index_p[0]]
            result.append( passed_gt.shape[0]/(passed_gt.shape[0]+failed_gt.shape[0]))
            current_passed_gt= passed_gt.shape[0]/(passed_gt.shape[0]+failed_gt.shape[0])
         else:
           result.append( passed_gt.shape[0]/(passed_gt.shape[0]+failed_gt.shape[0]))
         print('***n is:',n,'Number of failed data:',failed_gt.shape[0],'Number of passed data:',passed_gt.shape[0],'percentage is:',passed_gt.shape[0]/(passed_gt.shape[0]+failed_gt.shape[0]))
         
        
        if runid==1:
           print('in 1st runid')
           multi_runresults= np.array(result).reshape(1,-1)
        else: 
           multi_runresults= np.concatenate((multi_runresults,np.array(result).reshape(1,-1)),axis=0)
        print('result is:',multi_runresults)
        np.savetxt(result_file_name,multi_runresults,  delimiter=',')


      
	



