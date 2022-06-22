import torch.nn as nn
import torch.nn.functional as F
import torch 

HIDDEN1_UNITS = 128
HIDDEN2_UNITS = 64
HIDDEN3_UNITS = 64
HIDDEN4_UNITS = 32
HIDDEN5_UNITS = 32
HIDDEN6_UNITS = 64
HIDDEN7_UNITS = 64
HIDDEN8_UNITS = 8
class SNet(nn.Module):
    """
    def __init__(self, input_size, output_size):
      super().__init__()
      self.layers = nn.Sequential(
        nn.Linear(input_size, HIDDEN1_UNITS),
        #nn.LeakyReLU(0.2, True),
        nn.PReLU(HIDDEN1_UNITS),
        nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS),
        #nn.Tanh(),
        nn.PReLU(HIDDEN2_UNITS),
        nn.Linear(HIDDEN2_UNITS, HIDDEN3_UNITS),
        nn.PReLU(HIDDEN3_UNITS),
        nn.Linear(HIDDEN3_UNITS, HIDDEN4_UNITS),
        nn.PReLU(HIDDEN4_UNITS),
        nn.Linear(HIDDEN4_UNITS, HIDDEN5_UNITS),
        nn.PReLU(HIDDEN5_UNITS),
        nn.Linear(HIDDEN5_UNITS, output_size),
      )
    
    def forward(self, x):
        return self.layers(x)    


    """
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, HIDDEN1_UNITS)
        self.prelu= nn.PReLU()
        self.fc2 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.fc3 = nn.Linear(HIDDEN2_UNITS, HIDDEN3_UNITS)
        self.fc4 = nn.Linear(HIDDEN3_UNITS, HIDDEN4_UNITS)
        self.fc5 = nn.Linear(HIDDEN4_UNITS, HIDDEN5_UNITS)
        self.fc6 = nn.Linear(HIDDEN5_UNITS, HIDDEN6_UNITS)
        self.fc7 = nn.Linear(HIDDEN6_UNITS, HIDDEN7_UNITS)
        self.fc8 = nn.Linear(HIDDEN7_UNITS, HIDDEN8_UNITS)
        self.fc9 = nn.Linear(HIDDEN8_UNITS, output_size)
        #self.tanh= nn.Tanh()
     
    def forward(self, x):
        x=self.fc1(x)
        x1=F.relu(x) 
        x1 = self.fc2(x1)
        x2=F.relu(x1)
        x2 = self.fc3(x2)
        x3=F.relu(x2)
        x3=torch.add(x2, x3)
        nn.Dropout(p=0.2)
        x3 = self.fc4(x3)
        x4=F.relu(x3)
        x4 = self.fc5(x4)
        x5=F.relu(x4)
        x5 = self.fc6(x5)
        x5=F.relu(x5)
        nn.Dropout(p=0.2)
        x5 = self.fc7(x5)
        x5=F.relu(x5)
        x5 = self.fc8(x5)
        x5=F.relu(x5)
        x5 = self.fc9(x5)
        return x5
    """ 
    
    def forward(self, x):
        x=self.fc1(x)
        x1=torch.cat((F.relu(x),self.prelu(x)),1)   
        x2 = self.fc2(x1)
        x2=torch.cat((F.relu(x2),self.prelu(x2)),1)
        x3 = self.fc3(x2)
        x3=torch.cat((F.relu(x3),self.prelu(x3)),1)
        x4 = self.fc4(x3)
        x4=torch.cat((F.relu(x4),self.prelu(x4)),1)
        x5 = self.fc5(x4)
        x5=torch.cat((F.relu(x5),self.prelu(x5)),1)
        x6 = self.fc6(x5)
        return x6
   """  
