# try:
#     import mujoco
#     import mujoco.viewer
#     import time
#     import numpy as np
#     from matplotlib import pyplot as plt
#     import math
#     import random
#     from filterpy.kalman import ExtendedKalmanFilter
#     from filterpy.common import Q_discrete_white_noise
# except Exception as e:
#     print (e)

import mujoco
import mujoco.viewer
import time
import numpy as np
from matplotlib import pyplot as plt
import math
from math import cos, sin
import random
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise


    
#print ("import complete")
model = mujoco.MjModel.from_xml_path ("models/case_2_pend.xml")

#print ("compilation complete")
#model.opt.timestep = 0.002
#model = mujoco.MjModel.from_xml_path ("tutorial_models/3D_pendulum_actuator.xml")
#model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
data = mujoco.MjData(model)

class Modif_EKF(ExtendedKalmanFilter):
    def __init__(self, dim_x, dim_z, dim_u = 0):
        super().__init__(dim_x, dim_z, dim_u)
    def predict_x(self, u = 0):
        
        g = - model.opt.gravity[2]
        # self.x = x + dt * np.array([x[1], l_1 / I * (math.sin(x[0]) * (g * (m1 / 2 + m2) - u[1]) + u[0] * math.cos(x[0]))]).reshape((2, 1))
        # x = self.x
        # dx = np.zeros(4).reshape((4, 1))
        # dx1 = dt * x[1][0]
        # dx3 = dt * x[3][0]
        # # dx2 = dt / I * (l_1 * sin(x[0]) * g * (m1 / 2 + m2) + u[0] * (l_1 * sin(x[0] + x[2]) - l_2 / 2 * cos(2 * x[2])) + u[1] * (l_1 * sin(x[0] + x[2]) + l_2 / 2 * cos(2 * x[2])))
        # dx2 = dt / I  * (l_1 * sin(x[0]) * g * (m1 / 2 + m2))
        # dx4 = dt * l_2 / I2 / 2 * (u[0] - u[1])
        # # print (dx1)
        # # print (dx2)
        # # print (dx3)
        # # print (dx4)
        # dx = np.array([dx1, dx2, dx3, dx4]).reshape((4, 1))
        
        # self.x += dx
        
        #dx = np.zeros(4).reshape((4, 1))
        x = self.x
        # dx2 = dt / I  * (l_1 * sin(x[0, 0]) * g * (m1 / 2 + m2))
        # dx2 = dt / I  * (l_1 * sin(x[0]) * g * (m1 / 2 + m2) + u[0] * (l_1 * sin(x[0] + x[2]) - l_2 / 2 * cos(2 * x[2])) + u[1] * (l_1 * sin(x[0] + x[2]) + l_2 / 2 * cos(2 * x[2])))
        dx2 = dt / I  * (l_1 * sin(x[0]) * g * (m1 / 2 + m2) - u[0] * (l_1 * sin(x[0] - x[2]) - l_2 / 2) - u[1] * (l_1 * sin(x[0] - x[2]) + l_2 / 2))
        # dx2 = dt / I  * (l_1 * sin(x[0]) * g * (m1 / 2 + m2) - u[0] * (l_1 * sin(x[0] + x[2]) - l_2 / 2) - u[1] * (l_1 * sin(x[0] + x[2]) + l_2 / 2))
        dx4 = dt * l_2 / I2 / 2 * (u[0] - u[1])
        
        dxa = np.array([0, dx2, 0, dx4]).reshape((4, 1))
        self.x += dxa
        
        x = self.x
        dx1 = dt * x[1][0]
        dx3 = dt * x[3][0]
        
        dxb = np.array([dx1, 0, dx3, 0]).reshape((4, 1))
        self.x += dxb
        # print (dx1)
        # print (dx2)
        # print (dx3)
        # print (dx4)
        




start_angle = 0.3

#start_angle = math.pi - start_angle

data.qpos[0] = start_angle

data.qpos[1] = -start_angle

#data.qvel[1] = 0.05

mujoco.mj_forward(model, data)

mujoco.mju_zero(data.sensordata)

#l_2 = mujoco.mju_dist3(data.joint("prop_base_joint").xanchor, data.site("leg_end").xpos)
l_1 = 0.5
b_1 = 0.01

l_2 = 0.2

m1 = data.body("pendulum").cinert[9]
m2 = data.body("prop_bar").cinert[9]

# I = l_1**2 * (m1 / 3 + m2) + 2 / 5 * m2 * r_2**2
I2 = m2 * (l_2**2 / 12)
I1 = l_1**2 * m1 / 3 #+ l_1**2 * m2 + # + I2
I = I1 + l_1**2 * m2 #+ I2
#I = 
#print (data.sensor("IMU_acc").data)

main_bodies_names = ["propeller_base", "leg_1"]

blue_color = np.array ([0, 0, 1, 1])
red_color = np.array ([1, 0, 0, 1])
green_color = np.array([0, 1, 0, 1])
yellow_color = np.array([1, 1, 0, 1])
cyan_color = np.array([0, 1, 1, 1])

exit = False
pause = True
step = False

sim_time = []
sim_theta = []
est_theta = []
sim_theta2 = []
est_theta2 = []
sim_vel = []
est_vel = []
sim_vel2 = []
sim_vel_sum = []
est_vel_sum = []
sim_acc = []
est_acc = []
meas_diff = np.zeros((3, 0))
sim_meas = np.zeros((3, 0))
est_meas = np.zeros((3, 0))


ext_force = np.zeros((model.nbody, 6))
ext_force [1][2] = 1 #vertical force



id_mat = np.eye(3)

blue_color = np.array ([0, 0, 1, 1])
red_color = np.array ([1, 0, 0, 1])
green_color = np.array([0, 1, 0, 1])

arrow_shape = np.array ([0.012, 0.012, 1])
arrow_dim = np.zeros(3)
arrow_quat = np.zeros(4)
arrow_mat = np.zeros(9)
loc_force = np.zeros(3)


k_red = 0.1



k_theta = 0.7 #rotation visualization amplifier
q_err = np.zeros(4)
perp_quat = np.zeros(4)
perp_axangle = np.zeros(4)
res_quat = np.zeros(4)
temp_force = np.zeros(3)

angle_err = []
force_norm = []
prop_force = np.zeros((3, 0))
curr_force = np.zeros(3)

ekf_count = 0
count_max = 1

dt = model.opt.timestep

#ekf_theta = ExtendedKalmanFilter (2, 3) #1 dim for angular velocity, 2 for acceleration
#ekf_theta = ExtendedKalmanFilter (2, 3, 2) #1 dim for angular velocity, 2 for acceleration
# ekf_theta = Modif_EKF(2, 3, 2)
ekf_theta = Modif_EKF(4, 3, 2)

ekf_theta.x [0] = data.qpos[0].copy()
ekf_theta.x [2] = data.qpos[1].copy() + data.qpos[0].copy()

ekf_theta.x[1] = data.qvel[0]
ekf_theta.x[3] = data.qvel[1] 

# ekf_theta.x [0] += random.random() * 0.1

# ekf_theta.x [2] += random.random() * 0.1


# ekf_theta.x[3] += random.random()*0.02

# ekf_theta.x [0] += 0.1

# ekf_theta.x [2] += -0.1

#print (ekf_theta.x[0], data.qpos[0])

""" ekf_theta.R [0, 0] = model.sensor("IMU_gyro").noise
ekf_theta.R [1, 1] = model.sensor("IMU_acc").noise
ekf_theta.R [2, 2] = model.sensor("IMU_acc").noise """

# ekf_theta.R [0, 0] = 0.011
# ekf_theta.R [1, 1] = 0.009
# ekf_theta.R [2, 2] = 0.01


#print ([model.sensor("IMU_gyro").noise, model.sensor("IMU_acc").noise, model.sensor("IMU_acc").noise])
#print (ekf_theta.R)
#ekf_theta.Q = Q_discrete_white_noise (3, dt = dt, var = 0.0001)
ekf_theta.Q *= 0.001
ekf_theta.R *= 0.001
#ekf_theta.Q *= 0
""" ekf_theta.R *= 0.0001
ekf_theta.Q *= 0.0001 """

#ekf_theta.P *= 10

appl_force = np.zeros(3)

f1 = 0
f2 = 0

kp = 0.5
kd = 0.1

kp_2 = 2
kd_2 =  1.2

k_f = 0.2 #force visualization amplifier

des_pos = 0

P_threshold = 0.01

def draw_vector(viewer, idx, arrow_pos, arrow_color, arrow_dir, arrow_norm):
    
    mujoco.mju_copy(loc_force, arrow_dir)
    mujoco.mju_normalize (loc_force)
    mujoco.mju_quatZ2Vec(arrow_quat, loc_force)
    mujoco.mju_quat2Mat(arrow_mat, arrow_quat)
    mujoco.mju_copy(arrow_dim, arrow_shape)
    arrow_dim [2] = arrow_shape [2] * arrow_norm
    mujoco.mjv_initGeom(viewer.user_scn.geoms[idx],\
            type = mujoco.mjtGeom.mjGEOM_ARROW, size = arrow_dim,\
            pos = arrow_pos, mat = arrow_mat.flatten(), rgba = arrow_color)

def draw_vector_euler(viewer, idx, arrow_pos, arrow_color, arrow_norm, euler_ang):
    
    # mujoco.mju_copy(loc_force, arrow_dir)
    # mujoco.mju_normalize (loc_force)
    # mujoco.mju_quatZ2Vec(arrow_quat, loc_force)
    # mujoco.mju_quat2Mat(arrow_mat, arrow_quat)
    
    # arrow_quat = np.zeros(4)
    # arrow_mat =
    
    global arrow_mat, arrow_quat
    
    mujoco.mju_euler2Quat(arrow_quat, euler_ang, "XYZ")
    mujoco.mju_quat2Mat(arrow_mat, arrow_quat)
    mujoco.mju_copy(arrow_dim, arrow_shape)
    arrow_dim [2] = arrow_shape [2] * arrow_norm
    mujoco.mjv_initGeom(viewer.user_scn.geoms[idx],\
            type = mujoco.mjtGeom.mjGEOM_ARROW, size = arrow_dim,\
            pos = arrow_pos, mat = arrow_mat.flatten(), rgba = arrow_color)

        
def control_callback (model, data):
    
    global appl_force, f1, f2
    
    g = - model.opt.gravity[2]
    
    x = ekf_theta.x.copy()
    
    P = ekf_theta.P.copy()
    
    x1 = x[0, 0]
    x2 = x[1, 0]
    x3 = x[2, 0]
    x4 = x[3, 0]
    
    x1 = x1 % (2 * math.pi)
    x3 = x3 % (2 * math.pi)
                
    if x1 > math.pi:
        x1 -= 2 * math.pi
    if x3 > math.pi:
        x3 -= 2 * math.pi
    
    f1 = 0
    f2 = 0
    
    bal_force = np.array([0, 0, - l_1 * g * (m1 / 2 + m2) * sin(x1)])
    # bal_force = np.array([0, 0, l_1 * g * (m1 / 2 + m2) * sin(data.qpos[0])])
    
    #to finish
    
    cont_force = np.zeros(3)
    
    #print (P [0, 0], P [1, 1])
    
    #if P [0, 0] < P_threshold and P [1, 1] < P_threshold:
    
    
    if np.trace(P) < 4 * P_threshold:           
            
        des_pos_2 = 0
        des_M_2 = kp * (des_pos_2 - x3) - kd * x4
        
        des_M_2 *= I2
        
        des_M_2 = 0
        
        f1 += 1/l_2 * des_M_2
        f2 -= 1/l_2 * des_M_2
        
        des_M1 = kp * (des_pos - data.qpos[0]) - kd * data.qvel[0]
        # des_M1 = kp * (des_pos - x1) - kd * x3
        
        des_M1 = I *  des_M1
        # des_M1 = - I *  des_M1
        
        
        
        
        s1 = math.sin(x1)
        c1 = math.cos(x1)
        # if abs(c1) < 0.5:
        #     #cont_force_x =  des_M1 / (l_1 * math.cos(data.qpos[0]))
        #     cont_force_x =  des_M1 / (l_1 * c1)
        #     #cont_force_y = -  des_M1 / (l_1 * math.sin(data.qpos[0]))
        # elif abs(s1) < 0.5:
        #     cont_force_z = -  des_M1 / (l_1 * s1)
        # else:
        #     cont_force_x =  des_M1 / (l_1 * c1) / 2
        #     cont_force_z = -  des_M1 / (l_1 * s1) / 2
        
        # cont_f_den = 2 * l_1 * sin(x1 + x3)
        cont_f_den = 2 * l_1 * sin(2 * data.qpos[0] + data.qpos[1])
        if abs(cont_f_den) > 0.001: 
            
            cont_int = (des_M1 + des_M_2 * cos(2*x3)) / cont_f_den 
            
            f1 += cont_int
            f2 += cont_int
        
        
        
        #print (data.ctrl)
        
        # f1 = cont_f_1
        # f2 = cont_f_2
    else:
        bal_force = 0.8 * bal_force
        #cont_f_den = 0
    
    # den_bal_force = 2 * l_1 * sin(x1 + x3)
    den_bal_force = 2 * l_1 * sin(2 * data.qpos[0] + data.qpos[1])
    if abs(den_bal_force) > 0.001: 
        bal_int = bal_force [2] / den_bal_force
        
        f1 += bal_int
        f2 += bal_int
    
    
    print (f1, f2)
    data.actuator("propeller1").ctrl = f1
    data.actuator("propeller2").ctrl = f2
    
        
    # if apply_force:
    #     data.actuator("propeller1").ctrl = 5
    # else:
    #     data.actuator("propeller1").ctrl = 0
    
    

#mujoco.set_mjcb_control(control_callback)




apply_force = False
def kb_callback(keycode):
    global f1, f2
    #print(chr(keycode))
    if chr(keycode) == ' ':
        global exit
        exit = not exit
    if chr(keycode) == 'P':
        global pause
        pause = not pause
    if chr(keycode) == 'S':
        global step
        step = not step
    if chr(keycode) == 'F':
        # global apply_force
        # apply_force = not apply_force
        f1 = 1 - f1
        data.actuator("propeller1").ctrl = f1
    
    if chr(keycode) == 'G':
        f2 = 1 - f2
        data.actuator("propeller2").ctrl = f2
        
    if chr(keycode) == 'H':
        
        f1 = 1 - f1
        data.actuator("propeller1").ctrl = f1
        f2 = 1 - f2
        data.actuator("propeller2").ctrl = f2
        
#x = np.zeros(6)
def h_x (x):
    global appl_force
    
    c1 = math.cos(x[0])
    s1 = math.sin(x[0])
    g = - model.opt.gravity[2]
    
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    
    
    H = np.zeros (3)
    # H[0] = x [2] + x[3]
    
    # H [1] = - l_1 * x[2]**2 * sin(x[0])
    # H [1]+= l_1 / I * cos(x[0]) * (l_1 * sin(x[0]) * g * (m1 / 2 + m2) + f1 * (l_1 * sin(x[1]) + l_2 * sin(2 * (x[0] + x[1]))) + f2 * l_1 * sin(2 * x[0] + x[1]))
    
    # H [2] = - l_1 * x[2]**2 * cos(x[0])
    # H [2]-= l_1 / I * sin(x[0]) * (l_1 * math.sin(x[0]) * g * (m1 / 2 + m2) + f1 * (l_1 * math.sin(x[1]) + l_2 * math.sin(2 * (x[0] + x[1]))) + f2 * l_1 * math.sin(2 * x[0] + x[1]))
    
    
    H[0] = x4
    
    # H [1] = l_1 * sin(x1) * g * (m1 / 2 + m2)
    # H [1]+= f1 * (l_1 * sin(x1 + x3) - l_2 / 2 * cos(2 * x3))
    # H [1]+= f2 * (l_1 * sin(x1 + x3) + l_2 / 2 * cos(2 * x3))
    # H [1]*= l_1 / I  * cos(x1)
    # H [1]-= l_1 * x2**2 * sin(x1)
    
    # H [2] = l_1 * sin(x1) * g * (m1 / 2 + m2)
    # H [2]+= f1 * (l_1 * sin(x1 + x3) - l_2 / 2 * cos(2 * x3))
    # H [2]+= f2 * (l_1 * sin(x1 + x3) + l_2 / 2 * cos(2 * x3))
    # H [2]*= l_1 / I  * sin(x1)
    # H [2]-= l_1 * x2**2 * cos(x1)
    
    # H [1] = - l_1 * x2**2 * sin(x1)
    # H [1]+= l_1 * cos(x1) / I  * (l_1 * sin(x1) * g * (m1 / 2 + m2) + f1 * (l_1 * sin(x1 + x3) - l_2 / 2 * cos(2 * x3)) + f2 * (l_1 * sin(x1 + x3) + l_2 / 2 * cos(2 * x3)))
    
    # H [2] = -l_1 * x2**2 * cos(x1)
    # H [2]-= l_1 * sin(x1) / I  * (l_1 * sin(x1) * g * (m1 / 2 + m2) + f1 * (l_1 * sin(x1 + x3) - l_2 / 2 * cos(2 * x3)) + f2 * (l_1 * sin(x1 + x3) + l_2 / 2 * cos(2 * x3)))
    
    H [1] = l_1 * s1 * g * (m1 / 2 + m2)
    # H [1]-= f1 * (l_1 * sin(x1 - x3) - l_2 / 2)
    # H [1]-= f2 * (l_1 * sin(x1 - x3) + l_2 / 2)
    H [1]+= f1 * (l_1 * sin(x3 - x1) + l_2 / 2)
    H [1]+= f2 * (l_1 * sin(x3 - x1) - l_2 / 2)
    H [1]*= l_1 / I * c1
    H [1]-= l_1 * x2**2 * s1
    
    H [2] = l_1 * s1 * g * (m1 / 2 + m2)
    H [2]-= f1 * (l_1 * sin(x1 - x3) - l_2 / 2)
    H [2]-= f2 * (l_1 * sin(x1 - x3) + l_2 / 2)
    H [2]*= - l_1 / I * s1
    H [2]-= l_1 * x2**2 * c1
    
    return H.reshape((3, 1)) 

def H_jac (x):
    g = - model.opt.gravity[2]
    #global appl_force
    
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]

    c1 = math.cos(x1)
    s1 = math.sin(x1)
    H = np.zeros((3, 4))
    H_test = np.zeros((3, 4))
    
    """ H [0, 2] = 1
    H [0, 3] = 1
    
    
    
    H [1, 0] = - l_1 * x3**2 * c1
    H [1, 0]+= l_1 / I  * l_1 * cos(2 * x1) * g * (m1 / 2 + m2)
    H [1, 0]+= l_1 / I  * f1 * (l_2 * (2 * c1 * cos(2 * (x1 + x2)) - s1 * sin(2 * (x1 + x2))) - l_1 * s1 * sin(x2))
    H [1, 0]+= l_1 / I  * f2 * l_1 * (2 * c1 * cos(2 * x1 + x2) - s1 * sin(2 * x1 + x2))
    
    H [1, 1] = f1 * (cos(x2) * l_1 + l_2 * 2 * cos(2 * (x1 + x2)))
    H [1, 1]+= f2 * l_1 * cos(2 * x1 + x2)
    H [1, 1]*= l_1 / I  * c1
    
    H [1, 2] = -2 * l_1 * x3 * s1
    
    H [2, 2] = -2 * l_1 * x3 * c1
    
    H [2, 1] = f1 * (l_1 * cos(x2) + 2 * l_2 * cos(2 * (x1 + x2)))
    H [2, 1]+= f2 * l_1 * cos(2 * x1 + x2)
    H [2, 1]*= - l_1 / I
    
    #potential index error?
    H [2, 2] = l_1 * sin(2 * x1) * g * (m1 / 2 + m2)
    H [2, 2]+= f1 * (l_2 * (c1 * sin(2 * (x1 + x2)) + 2 * s1 * cos(2 * (x1 + x2))) + l_1 * c1 * sin(x2))
    H [2, 2]+= f2 * l_1 * (c1 * sin(2 * x1 + x2) + 2 * s1 * cos(2 * x1 + x2))
    H [2, 2]*= - l_1 / I
    H [2, 2]+= l_1 * x3**2 * s1 """

    
    
    
    
    """ H [1, 0] = l_1 * cos(2 * x1) * g * (m1 / 2 + m2)
    H [1, 0]+= f1 * (l_1 * cos(2 * x1 + x3) + l_2 / 2 * s1 * cos(2 * x3))
    H [1, 0]+= f2 * (l_1 * cos(2 * x1 + x3) - l_2 / 2 * s1 * cos(2 * x3))
    H [1, 0]*= l_1 / I 
    H [1, 0]-= l_1 * x2**2 * c1
    
    H [1, 1] = -2 * l_1 * x2 * s1
    
    H [1, 2] = f1 * (l_1 * cos(x1 + x3) + l_2 * sin(2 * x3))
    H [1, 2]+= f2 * (l_1 * cos(x1 + x3) - l_2 * sin(2 * x3))
    H [1, 2]*= l_1 / I  * c1
    
    
    
    H [2, 1] = -2 * l_1 * x2 * c1
    
    H [2, 2] = f1 * (l_1 * cos(x1 + x3) + l_2 * sin(2 * x3))
    H [2, 2]+= f2 * (l_1 * cos(x1 + x3) - l_2 * sin(2 * x3))
    H [2, 2]*= - l_1 / I  * s1
    
    H [2, 0] = l_1 * sin(2 * x1) * g * (m1 / 2 + m2)
    H [2, 0]+= f1 * (l_1 * sin(2 * x1 + x3) - l_2 / 2 * c1 * cos(2 * x3))
    H [2, 0]+= f2 * (l_1 * sin(2 * x1 + x3) + l_2 / 2 * c1 * cos(2 * x3))
    H [2, 0]*= - l_1 / I 
    H [2, 0]+= l_1 * x2**2 * s1 """

    """ H_test [1, 0] = l_1 * cos(2 * x1) * g * (m1 / 2 + m2)
    H_test [1, 0]+= f1 * (l_1 * cos(2 * x1 + x3) + l_2 / 2 * s1 * cos(2 * x3))
    H_test [1, 0]+= f2 * (l_1 * cos(2 * x1 + x3) - l_2 / 2 * s1 * cos(2 * x3))
    H_test [1, 0]*= l_1 / I 
    H_test [1, 0]-= l_1 * x2**2 * c1
    
    H_test [1, 1] = -2 * l_1 * x2 * s1
    
    H_test [1, 2] = f1 * (l_1 * cos(x1 + x3) + l_2 * sin(2 * x3))
    H_test [1, 2]+= f2 * (l_1 * cos(x1 + x3) - l_2 * sin(2 * x3))
    H_test [1, 2]*= l_1 / I  * c1
    
    
    
    H_test [2, 1] = -2 * l_1 * x2 * c1
    
    H_test [2, 2] = f1 * (l_1 * cos(x1 + x3) + l_2 * sin(2 * x3))
    H_test [2, 2]+= f2 * (l_1 * cos(x1 + x3) - l_2 * sin(2 * x3))
    H_test [2, 2]*= - l_1 / I  * s1
    
    H_test [2, 0] = l_1 * sin(2 * x1) * g * (m1 / 2 + m2)
    H_test [2, 0]+= f1 * (l_1 * sin(2 * x1 + x3) - l_2 / 2 * c1 * cos(2 * x3))
    H_test [2, 0]+= f2 * (l_1 * sin(2 * x1 + x3) + l_2 / 2 * c1 * cos(2 * x3))
    H_test [2, 0]*= - l_1 / I 
    H_test [2, 0]+= l_1 * x2**2 * s1 """
    
    H [0, 3] = 1
    
    
    H [1, 0] = l_1 * cos(2 * x1) * g * (m1 / 2 + m2)
    H [1, 0]-= f1 * (l_1 * cos(2 * x1 - x3) + l_2 / 2 * s1)
    H [1, 0]-= f2 * (l_1 * cos(2 * x1 - x3) - l_2 / 2 * s1)
    H [1, 0]*= l_1/I 
    H [1, 0]-= l_1*x2**2*c1
    
    H [1, 1] = - 2 * l_1 * x2 * s1
    
    H [1, 2] = f1 * l_1 * cos(x1 - x3)
    H [1, 2]+= f2 * l_1 * cos(x1 - x3)
    H [1, 2]*= l_1 / I * c1 
    
    
    H [2, 0] = l_1 * sin(2 * x1) * g * (m1 / 2 + m2)
    H [2, 0]-= f1 * (l_1 * sin(2 * x1 - x3) - l_2 / 2 * c1)
    H [2, 0]-= f2 * (l_1 * sin(2 * x1 - x3) + l_2 / 2 * c1)
    H [2, 0]*= - l_1 / I 
    H [2, 0]+= l_1 * x2**2 * s1
    
    H [2, 1] = - 2 * l_1 * x2 * c1
    
    H [2, 2] = f1 * l_1 * cos(x1 - x3)
    H [2, 2]+= f2 * l_1 * cos(x1 - x3)
    H [2, 2]*= - l_1 / I * s1 
    
    H_test [0, 3] = 1
    
    
    H_test [1, 0] = l_1 * cos(2 * x1) * g * (m1 / 2 + m2)
    H_test [1, 0]-= f1 * (c1 * l_1 * cos(x1 - x3) - s1 * (l_1 * sin(x1 - x3) - l_2 / 2))
    H_test [1, 0]-= f2 * (c1 * l_1 * cos(x1 - x3) - s1 * (l_1 * sin(x1 - x3) + l_2 / 2))
    H_test [1, 0]*= l_1/I 
    H_test [1, 0]-= l_1*x2**2*c1
    
    H_test [1, 1] = - 2 * l_1 * x2 * s1
    
    H_test [1, 2] = f1 * l_1 * cos(x1 - x3)
    H_test [1, 2]+= f2 * l_1 * cos(x1 - x3)
    H_test [1, 2]*= l_1 / I * c1 
    
    
    H_test [2, 0] = l_1 * sin(2 * x1) * g * (m1 / 2 + m2)
    H_test [2, 0]-= f1 * (c1 * (l_1 * sin(x1 - x3) - l_2 / 2) + s1 * l_1 * cos(x1 - x3))
    H_test [2, 0]-= f2 * (c1 * (l_1 * sin(x1 - x3) + l_2 / 2) + s1 * l_1 * cos(x1 - x3))
    H_test [2, 0]*= - l_1 / I 
    H_test [2, 0]+= l_1 * x2**2 * s1
    
    H_test [2, 1] = - 2 * l_1 * x2 * c1
    
    H_test [2, 2] = f1 * l_1 * cos(x1 - x3)
    H_test [2, 2]+= f2 * l_1 * cos(x1 - x3)
    H_test [2, 2]*= - l_1 / I * s1 
    
    print (H - H_test)
    
    return H

def compute_z (data, x):
    ang_data = data.sensor ("IMU_gyro").data[1].copy() 
                
    acc_data = data.sensor ("IMU_acc").data.copy()
    
    # print (acc_data, data.cacc[1])
    
    
    
    
    # ang_data += random.gauss(0, model.sensor("IMU_gyro").noise[0])
    # for i in range (0, 3):
    #     acc_data [i] += random.gauss(0, model.sensor("IMU_acc").noise[0])
    
    
    
    #q_est = x[0]
    #q_est = - x[0]
    #q_est = data.qpos[0]
    q_est = x[2]
    #q_est = - x[2]
    
    R = np.array([[math.cos(q_est), 0, math.sin(q_est)], [0, 1, 0], [-math.sin(q_est), 0, math.cos(q_est)]])
    
    mujoco.mju_mulMatVec (acc_data, R, acc_data.copy())
    
    
    #print (acc_data, data.body("prop_bar").cacc[3:])
    
    
    acc_data = acc_data + model.opt.gravity#.reshape((3, 1))
    
    
    #print (acc_data, data.body("end_mass").cacc[3:] + model.opt.gravity)
    """ sim_acc = l_1 * (data.qacc[0] * np.array([math.cos(data.qpos[0]), 0, - math.sin(data.qpos[0])]) - data.qvel[0]**2 * np.array([math.sin(data.qpos[0]), 0, math.cos(data.qpos[0])])) 
    print (acc_data, sim_acc) """
    
    
    
    #z = np.array([ang_data, acc_data [0], acc_data [1]]).reshape ((3, 1)) 
    #z = np.array([ang_data, acc_data [0], acc_data [1] - g]).reshape ((3, 1)) 
    return np.array([ang_data, acc_data [0], acc_data [2]]).reshape ((3, 1)) 
    #z = np.array([ang_data, - acc_data [0], - acc_data [2]]).reshape ((3, 1)) 
    
z = np.zeros(3)

#print ("setup sim")


try:        
    with mujoco.viewer.launch_passive(model, data, key_callback= kb_callback) as viewer:
        #print ("visualizer setup")
        try:
            viewer.lock()
            #print ("lock")
            #viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            
            #print (int(mujoco.mjtFrame.mjFRAME_BODY), int(mujoco.mjtFrame.mjFRAME_GEOM), int(mujoco.mjtFrame.mjFRAME_SITE))
            
            """ for i in range (model.nsite + model.njnt):
                mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn) """
            
            mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
            mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
            
            mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
            mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
            
            mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
            mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
            #print ("geoms added")
            #print ('simulation setup completed')
            viewer.sync()
        except Exception as e:
            print (e)
        
        #print ("sim_start")
        while viewer.is_running() and (not exit) :
            step_start = time.time()
            #mujoco.mj_forward(model, data)
            #print ('running iteration')
            
            
            if not pause or step:
                step = False
                with viewer.lock():
                    
                    
                    #print (data.qvel[1] + data.qvel[0])
                    sim_time.append (data.time)
                    sim_theta.append(data.qpos[0])
                    
                    sim_theta2.append(data.qpos[1] + data.qpos[0])
                    """ est_q = ekf_theta.x[0] % (2 * math.pi)
                    
                    if est_q > math.pi:
                        est_q -= 2 * math.pi """
                    est_q = ekf_theta.x[0].copy()
                    est_theta.append(est_q)
                    
                    est_q2 = ekf_theta.x[2].copy()
                    est_theta2.append(est_q2)
                    """ angle.append(data.qpos[0])
                    torque.append(data.ctrl[0]) """
                    
                    c_t_est = np.zeros(3)
                    
                    # sim_meas = np.append(sim_meas, z.reshape((3, -1)), axis = 1)
                    # est_meas = np.append(est_meas, h_x(ekf_theta.x).reshape((3, -1)), axis = 1)

                    sim_meas = np.concatenate((sim_meas.copy(), z.reshape((3, -1))), axis = 1)
                    #est_meas = np.concatenate((est_meas.copy(), h_x(ekf_theta.x).reshape((3, -1))), axis = 1)
                    # meas_diff = np.concatenate((meas_diff.copy(), z.reshape((3, -1)) - h_x(ekf_theta.x).reshape((3, -1))), axis = 1)
                    # x_sim = np.array([data.qpos[0], data.qvel[0], data.qpos[1] + data.qpos[0], data.qvel[1] + data.qvel[0]]).reshape((4, 1))
                    # est_meas = np.concatenate((est_meas.copy(), h_x(x_sim).reshape((3, -1))), axis = 1)
                    # meas_diff = np.concatenate((meas_diff.copy(), z.reshape((3, -1)) - h_x(x_sim).reshape((3, -1))), axis = 1)
                    sim_h = np.array([data.qvel[0] + data.qvel[1], - l_1 * data.qvel[0]**2 * sin(data.qpos[0]) + l_1 * data.qacc[0] * cos(data.qpos[0]), - l_1 * data.qvel[0]**2 * cos(data.qpos[0]) - l_1 * data.qacc[0] * sin(data.qpos[0])]).reshape((3, 1))
                    est_meas = np.concatenate((est_meas, sim_h), axis = 1)
                    meas_diff = np.concatenate((meas_diff.copy(), z.reshape((3, -1)) - sim_h), axis = 1)
                    # est_meas = np.append(est_meas, h_x(x_sim).reshape((-1, 3)), axis = 0)
                    # sim_meas = np.append(sim_meas, h_x(ekf_theta.x).reshape((-1, 3)), axis = 0)
                    #meas_diff = np.append(meas_diff.reshape((3, -1)), z.reshape((3, -1)) - h_x(ekf_theta.x).reshape((3, -1))).reshape((3, -1))
                    # meas_diff = np.append(meas_diff.reshape((3, -1)), z.reshape((3, -1)) - h_x(x_sim).reshape((3, -1))).reshape((3, -1))
                    
                    sim_vel.append(data.qvel[0])
                    est_vel.append(ekf_theta.x[1].copy())
                    sim_vel2.append(data.qvel[1])
                    sim_vel_sum.append(2 * data.qvel[0] + data.qvel[1])
                    est_vel_sum.append(data.sensor ("IMU_gyro").data[1].copy())
                    
                    sim_acc.append(data.qacc[0])
                    x = ekf_theta.x.copy()
                    g = -model.opt.gravity[2]
                    est_acc_1 = 1 / I  * (l_1 * sin(x[0]) * g * (m1 / 2 + m2) - f1 * (l_1 * sin(x[0] - x[2]) - l_2 / 2) - f2 * (l_1 * sin(x[0] - x[2]) + l_2 / 2))
                    est_acc.append(est_acc_1)
                    
                    q_est = ekf_theta.x[0]
                    # #q_est = - ekf_theta.x[0]
                    
                    c_t_est = l_1 * np.array([math.sin(q_est), 0, math.cos(q_est)])  
                    # #print (q_est)  
                    # c_t_est = l_1 / 2 * np.array([math.sin(q_est), 0, math.cos(q_est)])    
                    # #c_t_est = - l_1 / 2 * np.array([math.sin(q_est), 0, math.cos(q_est)]) 
                    
                    #print (dir(data.site('stick_end')))
                    
                    
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[0],\
                        type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .06 * np.ones(3),\
                        pos = data.subtree_com[1], mat = np.eye(3).flatten(), rgba = blue_color)
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[1],\
                        type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .035 * np.ones(3),\
                        pos = c_t_est, mat = np.eye(3).flatten(), rgba = green_color)
                    # mujoco.mjv_initGeom(viewer.user_scn.geoms[0],\
                    #     type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .06 * np.ones(3),\
                    #     pos = data.body("prop_bar").xpos, mat = np.eye(3).flatten(), rgba = blue_color)
                    # mujoco.mjv_initGeom(viewer.user_scn.geoms[1],\
                    #     type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .035 * np.ones(3),\
                    #     pos = data.site("IMU_loc").xpos, mat = np.eye(3).flatten(), rgba = green_color)
                    
                    q1_est = ekf_theta.x[0]
                    q2_est = ekf_theta.x[2]
                    est_prop1_pos = l_1 * np.array([sin(q1_est), 0, cos(q1_est)]) + l_2/2 * np.array([-cos(q2_est), 0, sin(q2_est)])
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[2],\
                        type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .06 * np.ones(3),\
                        pos = data.site("prop_1").xpos, mat = np.eye(3).flatten(), rgba = yellow_color)
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[3],\
                        type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .035 * np.ones(3),\
                        pos = est_prop1_pos, mat = np.eye(3).flatten(), rgba = cyan_color)
                
                    

                    #draw_vector(viewer, 2, data.site("IMU_loc").xpos, red_color, appl_force, 0.05 * mujoco.mju_norm(appl_force))
                    w_bar_pitch =  data.qpos[1] + data.qpos[0]
                    draw_vector_euler(viewer, 4, data.site("prop_1").xpos, red_color, k_f * f1, np.array([0, w_bar_pitch, 0]))
                    draw_vector_euler(viewer, 5, data.site("prop_2").xpos, red_color, k_f * f2, np.array([0, w_bar_pitch, 0]))
                    viewer.user_scn.ngeom =  6
                    
                    #debug 
                    
                    x1_acc = 1 / I  * (l_1 * sin(x[0]) * g * (m1 / 2 + m2) - f1 * (l_1 * sin(x[0] - x[2]) - l_2 / 2) - f2 * (l_1 * sin(x[0] - x[2]) + l_2 / 2))
                    
                    sim_acc_1 =  1 / I  * (l_1 * sin(data.qpos[0]) * g * (m1 / 2 + m2) - f1 * (l_1 * sin(data.qpos[0] + data.qpos[1]) + l_2 / 2) - f2 * (l_1 * sin(data.qpos[0] + data.qpos[1]) - l_2 / 2))
                    # print (x1_acc, data.qacc[0])
                    print (sim_acc_1 - data.qacc[0])
                    
                    
                    x = ekf_theta.x.copy()
                    g = - model.opt.gravity[2]
                    
                    # f1 = 0
                    # f2 = 0
                    
                    
                    F = np.eye(4) + dt * np.eye(4, 4, 1)
                    F [1, 2] = 0
                    
                    #print (F)
                    
                    # F [2, 0] = dt / I * (l_1 * math.cos(x[0]) * g * (m1 / 2 + m2) + f1 * 2 * l_2 * math.cos(2 * (x[0] + x[1])) + f2 * 2 * l_1 * math.cos(2 * x[0] + x[1]))
                    
                    # F [2, 1] = dt / I * (f1 * (l_1 * math.cos(x[1]) + 2 * l_2 * math.cos(2 * (x[0] + x[1]))) + f2 * l_1 * math.cos(2 * x[0] + x[1]))
                    
                    F [1, 0] = l_1 * cos(x[0]) * g * (m1 / 2 + m2)
                    F [1, 0]-= f1 * l_1 * cos(x[0] - x[2])
                    F [1, 0]-= f2 * l_1 * cos(x[0] - x[2])
                    F [1, 0]*= dt / I 
                    
                    F [1, 2] = f1 * l_1 * cos(x[0] - x[2])
                    F [1, 2]+= f2 * l_1 * cos(x[0] - x[2])
                    F [1, 2]*= dt / I 
                    
                    
                    
                    ekf_theta.F = F.copy()
                    
                    B = np.zeros((4, 2))
                    
                    # B [2, 0] = dt / I * (l_1 * math.sin(x[1]) + l_2 * math.sin(2 * (x[0] + x[1])))
                    
                    # B [2, 1] = dt / I * l_1 * math.sin(2 * x[0] + x[1])
                    
                    # B [3, 0] = dt / I2 * l_2 / 2
                    
                    # B [3, 1] = - dt / I2 * l_2 / 2
                    
                    B [3, 0] = dt / I2 * l_2 / 2 
                    B [3, 1] = - dt / I2 * l_2 / 2
                    
                    B [1, 0] = - dt / I * (l_1 * sin(x[0] - x[2]) - l_2 / 2)
                    B [1, 1] = - dt / I * (l_1 * sin(x[0] - x[2]) + l_2 / 2)
                    
                    ekf_theta.B = B.copy()
                    
                    #f_u = np.array([appl_force[0], appl_force[2]]).reshape((2, 1))
                    f_u = np.array([f1, f2]).copy()
                    
                    ekf_theta.predict(f_u)
                    # ekf_theta.predict(np.array([0, 0]))
                    
                    #if (ekf_count == 0) and False:
                    if (ekf_count == 0): 
                                                                
                        
                        #mujoco.mj_forward(model, data)
                        z = compute_z(data, ekf_theta.x)
                        
                        #ekf_theta.update(z = z, HJacobian = H_jac, Hx = h_x)
                        #H_jac(ekf_theta.x)
                        
                        # print (h_x(ekf_theta.x) - np.array([data.qvel[0] + data.qvel[1], l_1 * data.qvel[0]**2 * - sin(data.qpos[0]) + l_1 * data.qacc[0] * cos(data.qpos[0]), l_1 * data.qvel[0]**2 * - cos(data.qpos[0]) - l_1 * data.qacc[0] * sin(data.qpos[0])]).reshape((3, 1)))
                        
                    
                        
                    ekf_count = (ekf_count + 1) % count_max
                    
                    
                    
                    # try:
                    #     mujoco.mj_step(model, data)
                    # except Exception as e:
                    #     print (e)
                    #     exit()
                    mujoco.mj_step(model, data)

                    
                    #mujoco.mjv_updateScene(model, data, viewer.opt, viewer.perturb, viewer.cam, mujoco.mjtCatBit.mjCAT_ALL, viewer.user_scn)
                        
                
                #mujoco.mj_kinematics(model, data)    
                viewer.sync() 
                
                
                
                
            time_to_step = model.opt.timestep - (time.time() - step_start)
            if (time_to_step > 0):
                time.sleep(time_to_step)
except Exception as e:
    print (e)            

#print (angle_err)
print (sim_meas.size)
sim_meas = np.asarray(sim_meas).reshape((3, -1))
est_meas = np.asarray(est_meas).reshape((3, -1))
print (sim_meas.size)
fig1, ax = plt.subplots(3, 2, sharex = True)

#ax[0].plot(sim_time, np.array([angle_err, force_norm]).reshape((-1, 2)))
#ax[0].legend(['theta', 'total force'])
""" ax.plot(sim_time, sim_theta)
ax.plot(sim_time, est_theta)
#ax.plot(sim_time, [s - e for s,e in zip(sim_theta, est_theta)])
ax.legend(['sim', 'est', 'diff']) """
ax[0][0].plot(sim_time, sim_theta)
ax[0][0].plot(sim_time, est_theta)
ax[0][1].plot(sim_time, [s - e for s,e in zip(sim_theta, est_theta)])
ax[0][0].legend(['sim', 'est'])
ax[0][1].legend(['sim', 'est'])

ax[1][0].plot(sim_time, sim_vel)
ax[1][0].plot(sim_time, est_vel)
ax[1][1].plot(sim_time, [s - e for s,e in zip(sim_vel, est_vel)])
ax[0][0].legend(['sim', 'est'])
ax[0][1].legend(['sim', 'est'])

ax[2][0].plot(sim_time, sim_acc)
ax[2][0].plot(sim_time, est_acc)
ax[2][1].plot(sim_time, [s - e for s,e in zip(sim_acc, est_acc)])
ax[0][0].legend(['sim', 'est'])
ax[0][1].legend(['sim', 'est'])
fig1.waitforbuttonpress()

fig1, ax = plt.subplots(1, 2, sharex = True)

ax[0].plot(sim_time, sim_theta2)
ax[0].plot(sim_time, est_theta2)
ax[1].plot(sim_time, [s - e for s, e in zip(sim_theta2, est_theta2)])
ax[0].legend(['sim', 'est'])

fig1.waitforbuttonpress()


# #fig1.clear()
# """ fig1, ax = plt.subplots(3, 1, sharex = True)
# for i in range (3):
#     ax[i].plot(sim_time, sim_meas[i, :].reshape((-1, 1)))
#     ax[i].plot(sim_time, est_meas[i].reshape((-1, 1)) - 10)
#     ax[i].legend(['sim', 'est'])
# #plt.show()
# fig1.waitforbuttonpress() """

# #fig1.clear()
# #print (meas_diff.shape)
# fig1, ax = plt.subplots(3, 1, sharex = True)
# for i in range (3):
#     ax[i].plot(sim_time, sim_meas[i].reshape((-1, 1)))
#     ax[i].plot(sim_time, est_meas[i].reshape((-1, 1)))
#     ax[i].plot(sim_time, meas_diff[i, :])
#     #ax[i].legend(['diff'])
#     ax[i].legend(['sim', 'est', 'diff'])
#     #ax[i].legend(['sim', 'est'])
# #plt.show()
# fig1.waitforbuttonpress()
fig1, ax = plt.subplots(3, 2, sharex = True)
for i in range (3):
    ax[i][0].plot(sim_time, sim_meas[i, :])
    ax[i][0].plot(sim_time, est_meas[i, :])
    ax[i][1].plot(sim_time, meas_diff[i, :])
    ax[i][1].legend(['diff'])
    #ax[i].legend(['sim', 'est', 'diff'])
    ax[i][0].legend(['sim', 'est'])
#plt.show()
fig1.waitforbuttonpress()

# fig1, ax = plt.subplots(4, 1, sharex = True)

# #ax[0].plot(sim_time, np.array([angle_err, force_norm]).reshape((-1, 2)))
# #ax[0].legend(['theta', 'total force'])
# ax[0].plot(sim_time, sim_vel)
# ax[1].plot(sim_time, sim_vel2)
# ax[2].plot(sim_time, sim_vel_sum)
# ax[3].plot(sim_time, est_vel_sum)
# #ax.legend(['sim', 'est'])
# plt.waitforbuttonpress()