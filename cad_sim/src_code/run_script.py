import numpy as np
from vessel_class import GliderVessel


dp= np.loadtxt('design_points.csv', delimiter=',')
print('design_points are:',dp,'shape:',dp.shape)

#Importing vessel seed design
vessel = GliderVessel('./seed_cad/swordfish_cfd.FCStd') 

#tail_Rad=dp[0] ; tail_Len=dp[1] 
F_y=dp[0]; S_y=dp[1]; T_y=dp[2]; Four_y=dp[3]; Five_y=dp[4]
print('dp1:',dp[0],'dp2:',dp[1],'dp3:',dp[2],'dp4:',dp[3],'dp5:',dp[4])



#vessel.set_tail(tail_Len,tail_Rad)

vessel.set_y_loc(F_y,S_y,T_y,Four_y,Five_y)
vessel.print_info()
vessel.create_stl(1)

"""
for i in range(glider_design_with_index.shape[0]):

    print('----------->Experiment id is:',experiment_id)
    r_height= glider_design_with_index[i][0]
    r_len = glider_design_with_index[i][1]
    r_thickness = glider_design_with_index[i][2]
    vessel.set_fins(r_height,r_len,r_thickness)
    vessel.get_rudder_details()
    vessel.create_stl(experiment_id)
"""