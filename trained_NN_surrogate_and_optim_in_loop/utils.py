# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 16:11:45 2021

@author: HPP
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
device = torch.device("cpu")
from sklearn.preprocessing import OneHotEncoder
import copy 
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans


class utilities():
    def __init__(self):
        self.lr=0

    def set_lr(self, lr):
    	 self.lr = lr
    def set_lr_auto(self):
       self.lr = np.random.choice(np.logspace(-3, 0, base=10))

    def get_optimizer(self,model):
     optimizer_class = optim.Adam  
     #print('***learning rate is:',self.lr)
     return optimizer_class(model.parameters(), lr=0.001)

    def get_lossfunc(self,net_type):
        if net_type=='S':
            return nn.L1Loss()
        elif net_type=='T':
            return nn.BCELoss()

def data_partioning(self,data,population):
      #based on population 
     a_list=np.arange(data.shape[0])
     np.random.shuffle(a_list)
     a_split=np.array_split(a_list,population)
     data_split=[]
     for i in range(len(a_split)):
        data_split.append(data[a_split[i]])
     return data_split

def append_data_splitmode(data,new_data,population):
    snew=np.array_split(new_data, population)
    for i in range(population):
      data[i]=np.append(data[i],snew[i],axis=0)
    return data

def append_data_randomly_splitmode(data,new_data,population):
    sel_pop=np.random.randint(population,size=1)
    #print('selected population:',sel_pop,"data size is", len(data) )
    data[sel_pop[0]]=np.append(data[sel_pop[0]],new_data,axis=0)
    return data


def scale_data(data,ranges):
     minimum=[] ; total_range=[]
     for i in range(data.shape[1]):
       minimum.append(ranges[2*i])
       total_range.append((ranges[2*i+1]-ranges[2*i]))
     #print('min is:',minimum,'Range is:',total_range,'data b/f scaling is:',data)
     minimum=np.array(minimum).reshape(1,-1)
     total_range=np.array(total_range).reshape(1,-1)
     data= np.divide((data-minimum),total_range)
     #print('min is:',minimum,'Range is:',total_range,'data a/f scaling is:',data)
     return data

def rescale_data(data,ranges,mask):
    #print('data b/f rescaling:',data)
    for i in range(len(mask)):
      #print('i is:',i,'mask is:',mask[i])
     
      if mask[i]=='real':
         data[:,i]= (data[:,i]*(ranges[2*i+1]- ranges[2*i]))+ranges[2*i]
      elif mask[i]=='int':
         data[:,i]= (data[:,i]*(ranges[2*i+1]- ranges[2*i]))+ranges[2*i]
         data[:,i]=np.array(data[:,i], dtype=np.int16)
    #print('data after rescaling:',data)
    return data

#on a given input prepare data for training
def data_preperation(data,mask,ranges,cat):    
    for i in range(len(mask)):
      #print('i is:',i,'mask is:',mask[i])
      if mask[i]=='real':
         data[:,i]= (data[:,i]-ranges[2*i])/(ranges[2*i+1]- ranges[2*i])
      elif mask[i]=='int':
         enc = OneHotEncoder(categories=cat[i])
         data[:,i]=enc.transform(data[:,i]).toarray()
    return data
      
def update_lbtm(data,size):
    return data[0:size,:] 


def label_data(data,stest_pred):
      # create label for data(if predicted vale is >/< 10% of error then it labels it '1' or else it is '0')
      ones=np.ones(stest_pred.shape[0])
      zeros= np.zeros(stest_pred.shape[0])
      #print('test shape:',test_data[:,-1].shape,'zeros shape:',zeros.shape,'ones shape:',ones.shape,'stest shape',stest_pred.flatten().shape)
      result = np.where(np.absolute((data[:,-1]-stest_pred.flatten())) > (0.05*np.absolute(data[:,-1])),ones,zeros)
      data[:,-1]=result
      return data

def choose_samples_epsilon(pool_data,pool_pred,selection_prob,num_of_samples,epsilon):
    a_list=np.arange(pool_data.shape[0])
    #print('pool prediction is:',pool_pred)
    index = np.where(pool_pred> selection_prob)     # find indices in pool_data which have high probability to fail
    #print('index of high prob is:',index) 
    passed_samples_size=len(index[0]) 
    #print('total high prob samples size are:',passed_samples_size)
    passed_pool_data=pool_data[index[0]]
    leftover=np.delete(a_list,index[0])
    #print('left over samples in pool data are:',leftover)
    failed_pool_data=pool_data[leftover] 
    print('size of passed data:',passed_pool_data.shape,'failed pool data:',failed_pool_data.shape)
    
    num_of_samples_from_greedy= int(num_of_samples*epsilon)      #find number of samples from being greedy
    #print('number of greedy samples:',num_of_samples_from_greedy)
    #if passed sample bucket is greater than number of sample required from greedy approach
    if passed_samples_size>num_of_samples_from_greedy:
       #print('++++++', 'In if loop') 
       selected_samples,pass_pool_leftover=data_split_size(passed_pool_data,num_of_samples_from_greedy)
       #print('passed pool leftover:',pass_pool_leftover.shape,'selected samples:',selected_samples.shape)
       pool_data=np.concatenate((pass_pool_leftover,failed_pool_data),axis=0)
       num_of_random_samples= num_of_samples-selected_samples.shape[0]
       t_data, rest_pool_data=data_split_size(pool_data,num_of_random_samples)
       selected_samples= np.concatenate((selected_samples,t_data),axis=0)
       
    else: 
      #print('++++++', 'In else loop') 
      selected_samples= passed_pool_data
      num_of_random_samples= num_of_samples-selected_samples.shape[0]
      #print('number of random_samples:',num_of_random_samples,'no of passed samples:',selected_samples.shape)
      t_data, rest_pool_data=data_split_size(failed_pool_data,num_of_random_samples)
      selected_samples= np.concatenate((selected_samples,t_data),axis=0)
    #print("---->Total selected samples :",selected_samples.shape)
    return selected_samples,rest_pool_data


def choose_topb_samples(pool_data,pool_pred,selection_prob,num_of_samples):
    #print('pool_data shape:',pool_data.shape,'pool_pred shape',pool_pred.shape)
    pool_data_sorted = pool_data[np.argsort(-1*pool_pred[:, 0])]
    pool_pred_sorted = pool_pred[np.argsort(-1*pool_pred[:, 0])]
    a_list=np.arange(pool_data.shape[0])
    index = np.where(pool_pred> selection_prob)     # find indices in pool_data which have high probability to fail
    passed_samples_size=len(index[0]) 
    print('size of passed data:',passed_samples_size)
    selected_samples= pool_data_sorted[:num_of_samples,:]
    rest_pool_data = pool_data_sorted[num_of_samples:,:]
    selected_probability= pool_pred_sorted[:num_of_samples,:].flatten()
    print('shape of selected-samples',selected_samples.shape,'rest pool data:',rest_pool_data.shape,'pool_data:',pool_data.shape) 
    return selected_samples,rest_pool_data,selected_probability


def choose_samples_weighted_diversity(data,pred,selection_prob,num_of_beta_samples,num_of_sel_samples):
    print('beta samples are',num_of_beta_samples,'selcted samples are:',num_of_sel_samples)
    selected,rejected,weights=choose_topb_samples(data,pred,selection_prob,num_of_beta_samples)
    print('*******shape of weights:', weights.shape)
    closest_samples,leftoversamples=kmeancluster_weighted(selected,num_of_sel_samples,weights)
    total_leftover=np.concatenate((rejected,leftoversamples),axis=0)
    print('size of in data:',data.shape,'closeset:',closest_samples.shape,'leftover:',total_leftover.shape)
    return closest_samples,total_leftover

def kmeancluster(X,num_cluster):
    km = KMeans(n_clusters=num_cluster,init='k-means++').fit(X)
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)
    return closest

def kmeancluster_weighted(X,num_cluster,weights):
    total_list=np.arange(X.shape[0])
    km = KMeans(n_clusters=num_cluster,init='k-means++').fit(X[:,:-1],sample_weight = weights)
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X[:,:-1])   
    leftover=np.delete(total_list,closest)
    closest_data= X[closest]
    left_over_data= X[leftover]
    return closest_data,left_over_data

def data_split_size(data,size):
      #on a given dataset return the splitted data=> train_data(based on size),validate_data(leftover)
      a_list=np.arange(data.shape[0])
      np.random.shuffle(a_list)
      alist=a_list[0:size]
      train_data=data[alist]
      d=np.arange(data.shape[0])
      leftover=np.delete(d,alist)
      validate_data=data[leftover]
      return train_data,validate_data


def data_split(data,proportion=0.2):
      a_list=np.arange(data.shape[0])
      np.random.shuffle(a_list)
      alist=a_list[0:int(data.shape[0]*(1-proportion))]
      train_data=data[alist]
      d=np.arange(data.shape[0])
      leftover=np.delete(d,alist)
      validate_data=data[leftover]
      return train_data,validate_data

def initialize_weights(m):
     if isinstance(m, nn.Linear):
      #nn.init.kaiming_uniform_(m.weight.data)
      torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
      torch.nn.init.constant_(m.bias.data, 0)

def create_datafiles(data,test_fraction=0.1):
     a_list=np.arange(data.shape[0])
     np.random.shuffle(a_list)
     alist=a_list[0:int(data.shape[0]*(1-test_fraction))]
     train_data=data[alist]
     d=np.arange(data.shape[0])
     leftover=np.delete(d,alist)
     test_data=data[leftover]
     #print('train_data to create file:',train_data,'test data in create file:',test_data)
     np.savetxt('./data/train_data.txt', train_data,  delimiter=' ')
     np.savetxt('./data/test_data.txt', test_data, delimiter=' ')
     return 0

def create_files(data,file_name):  
     name_file= './data/'+file_name+'.txt'
     np.savetxt(name_file, data,  delimiter=' ')
     return 0

class SimDataset(torch.utils.data.Dataset):
  def __init__(self, dataset):
    x_tmp = dataset[:,0:-1]
    y_tmp = dataset[:,-1]
    #print('X_tmp is:',x_tmp,'Y_tmp is:',y_tmp)

    self.x = torch.tensor(x_tmp,
      dtype=torch.float32).to(device)
    self.y = torch.tensor(y_tmp,
      dtype=torch.float32).to(device)

  def __len__(self):
    return len(self.x)  # required

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    preds = self.x[idx,:]
    pol = self.y[idx]
    sample = \
      { preds, pol }
    return preds,pol


