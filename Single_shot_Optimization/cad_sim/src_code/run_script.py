import numpy as np
from vessel_class import GliderVessel
import math 
import matplotlib.pyplot as plt

remus_volume=27818.58916  # in cubic cm 

#optimal_nt=np.array([29.79,67.30,81.90,95,89.52,66.74,38.26,24.93])
dp= np.loadtxt('design_points.csv', delimiter=',')
#print('design_points are:',dp,'shape:',dp.shape)

#Importing vessel seed design
vessel = GliderVessel('./seed_cad/Remus_Myring_hull.FCStd') 
#dp= np.append(np.array([b,D]),ds[i])
#print('******design point is:******',dp,'dp shape  is:',dp.shape)

######Setting vehicle details###


#d= dp[1]

a= dp[0]
c= dp[1]
n=dp[2]/10
theta=dp[3]
d=dp[4]
tl=[5]
b= tl-a-c

r= d*0.5






vessel.set_fairing_rad(r)
vessel.set_nose_len(a)
vessel.set_fairing_len(b)
vessel.set_tail_len(c)

##########
body=vessel.get_fairing_details()
print('----> body is:',body)


nose_loc= vessel.get_nose_x_loc()
tail_loc= vessel.get_tail_x_loc()+a+b
print('=> nose_x:',nose_loc)
print('=> tail_x:',tail_loc)
################


###Get volume and apped to design point
volume=vessel.get_outer_volume()
myring= np.array([a,b,c,d,n,theta])
dp=np.append(myring,volume)
print('******design point is:******',dp,'dp shape  is:',dp.shape)

np.savetxt('design_points.csv',dp,delimiter=',')
#########
def estimate_nose(a,d,x,n): 
    return 0.5*d*np.power((1-np.power(((x-a)/a),2)),(1/n))

def estimate_tail(a,b,c,d,x,theta):  
    theta=theta*math.pi/180
    y1=0.5*d 
    y2= np.power((x-a-b),2)*((3*d)/(2*c*c) - math.tan(theta)/c)
    y3= ((d/(c*c*c)) - math.tan(theta)/(c*c))*np.power((x-a-b),3)
    return (y1-y2+y3)    




####if only nose tail
nose_y=estimate_nose(a,d,nose_loc,n)
tail_y= estimate_tail(a,b,c,d,tail_loc,theta)
nose_tail_y= np.append(nose_y,tail_y)
print('=> nose_tail_y:',nose_tail_y)

vessel.set_nose_tail_y(nose_tail_y)









#########################################
#######Donot edit beyond this ###########
########################################

#################Save fig of hull 2D
x_n=np.array([0,a/5,2*a/5,3*a/5,4*a/5,a]);
x_t=np.array([a+b,a+b+c/5,a+b+2*c/5,a+b+3*c/5,a+b+4*c/5,a+b+c]);
x_b= np.array([a,a+b*1/6,a+b*2/6,a+b*3/6,a+b*4/6,a+b*5/6,a+b])
y_b=np.array([r,r,r,r,r,r,r])
y_b=y_b.astype('float64')

y= estimate_nose(a,d,x_n,n)
z= estimate_tail(a,b,c,d,x_t,theta)



figname= './fig_hull/GA_a'+str(round(a, 2))+'c_'+str(round(c, 2))+'n_'+str(round(n, 2))+'_theta_'+str(round(theta, 2))+'.png'
plt.figure(figsize=(10,3))
plt.plot(x_n,y)
plt.plot(x_t,z)
plt.plot(x_b,y_b)
plt.plot(x_n,-1*y)
plt.plot(x_t,-1*z)
plt.plot(x_b,-1*y_b)

plt.savefig(figname)
plt.close() 


vessel.print_info()
vessel.create_stl(1)

