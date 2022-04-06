import numpy as np
from vessel_class import GliderVessel


dp= np.loadtxt('design_points.csv', delimiter=',')
print('design_points are:',dp)

#Importing vessel seed design
vessel = GliderVessel('./seed_cad/swordfish_cfd.FCStd') 
vessel.print_info()

#vessel.set_tail(dp[])
#vessel.set_y_loc()


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