import numpy as np
from vessel_class import GliderVessel


fin_h=np.array([0.25,0.5,1])
fin_l=np.array([0.2, 0.4, 0.8])
fin_t=np.array([5, 10, 20])

ds1,ds2,ds3= np.meshgrid(fin_h,fin_l,fin_t)
design_space=np.vstack((ds1.ravel(),ds2.ravel(),ds3.ravel())).T
#print(design_space)
index= np.arange(1,28).reshape(-1,1)
glider_design_with_index= np.hstack((index,design_space))
print(glider_design_with_index)
np.savetxt('./data/rudder_study_design_points.csv', glider_design_with_index, delimiter=',')


#Importing vessel seed design
vessel = GliderVessel('./cad/VUnderwater.FCStd') 


for i in range(glider_design_with_index.shape[0]):
    experiment_id=glider_design_with_index[i][0]
    print('----------->Experiment id is:',experiment_id)
    r_height= glider_design_with_index[i][1]
    r_len = glider_design_with_index[i][2]
    r_thickness = glider_design_with_index[i][3]
    vessel.set_fins(r_height,r_len,r_thickness)
    vessel.get_rudder_details()
    vessel.create_stl(experiment_id)
