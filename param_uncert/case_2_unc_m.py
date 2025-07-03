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
        s1 = sin(x[0])
        c1 = cos(x[0])
        s2 = sin(x[1])
        c2 = cos(x[1])
        
        I = compute_I1(x[4, 0], x[5, 0])
        I2 = compute_I2(x[5, 0])
        
        # if abs(I) > 0.001 and abs(I2) > 0.001:
            
        # dx3 = + dt / I * l_1 * sin(x[0]) * g * (m1 / 2 + m2)
        # dx4 = - dt / I * l_1 * sin(x[0]) * g * (m1 / 2 + m2)
        dx3 = dt * (+ 1 / I * (l_1 * s1 * g * (x[4, 0] / 2 + x[5, 0]) + l_1 * s2 * (u[0] + u[1]) + l_2 / 2 * (u[0] - u[1])))
        
        dx4 = - 1 / I * (l_1 * s1 * g * (x[4, 0] / 2 + x[5, 0]) + l_1 * s2 * (u[0] + u[1])) + (1 / I2 - 1 / I) * l_2 / 2 * (u[0] - u[1])
        dx4*= dt
        dxa = np.array([0, 0, dx3, dx4, 0, 0]).reshape((-1, 1))
        self.x += dxa
        x = self.x
        dx1 = dt * x[2, 0]
        dx2 = dt * x[3, 0]
        
        dxb = np.array([dx1, dx2, 0, 0, 0, 0]).reshape((-1, 1))
        self.x += dxb
        
        # dx = np.array([dx1, dx2, dx3, dx4]).reshape((4, 1))
        # self.x += dx
        # print (dx1)
        # print (dx2)
        # print (dx3)
        # print (dx4)
        




start_angle = 0.5

#start_angle = math.pi - start_angle

data.qpos[0] = start_angle

data.qpos[1] = -start_angle
# data.qpos[1] = 0

#data.qvel[1] = 0.05

mujoco.mj_forward(model, data)

mujoco.mju_zero(data.sensordata)

#l_2 = mujoco.mju_dist3(data.joint("prop_base_joint").xanchor, data.site("leg_end").xpos)
l_1 = 0.5
b_1 = 0.01

l_2 = 0.2

# m1 = data.body("pendulum").cinert[9]
# m2 = data.body("prop_bar").cinert[9]

m1 = model.body("pendulum").mass[0]
m2 = model.body("prop_bar").mass[0]



# I = l_1**2 * (m1 / 3 + m2) + 2 / 5 * m2 * r_2**2
# I2 = model.body("prop_bar").inertia[1]
# I1 = model.body("pendulum").inertia[1]
# I = I1 + l_1**2 * (m1 / 4 + m2)

# I2 = l_2**2 * m2 / 12
# I1 = l_1**2 * m1 / 12
# I = I1 + l_1**2 * (m1 / 4 + m2)

def compute_I1(m_1, m_2):
    return l_1**2 * (m_1 / 3 + m_2)
def compute_I2(m_2):
    return l_2**2 * m_2 / 12



# print (I1, model.body("pendulum").inertia[1])

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

est_l_1 = []
est_l_2 = []

acc_err = []
dim_z = 6
meas_diff = np.zeros((dim_z, 0))
sim_meas = np.zeros((dim_z, 0))
est_meas = np.zeros((dim_z, 0))


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
# ekf_theta = Modif_EKF(4, 3, 2)
ekf_theta = Modif_EKF(6, 6, 2)

ekf_theta.x [0] = data.qpos[0].copy()
ekf_theta.x [1] = data.qpos[1].copy()

ekf_theta.x [2] = data.qvel[0]
ekf_theta.x [3] = data.qvel[1] 

ekf_theta.x[4] = model.body("pendulum").mass
ekf_theta.x[5] = model.body("prop_bar").mass


# ekf_theta.x [0] += random.random() * 0.1

# ekf_theta.x [1] += random.random() * 0.1

# ekf_theta.x[2] += random.random() * 0.02
# ekf_theta.x[3] += random.random() * 0.02

# ekf_theta.x[4] += random.random() * 0.03
# ekf_theta.x[5] += random.random() * 0.03

# ekf_theta.x [0] += 0.1

# ekf_theta.x [2] += -0.1

#print (ekf_theta.x[0], data.qpos[0])

""" ekf_theta.R [0, 0] = model.sensor("IMU_gyro").noise
ekf_theta.R [1, 1] = model.sensor("IMU_acc").noise
ekf_theta.R [2, 2] = model.sensor("IMU_acc").noise """


ekf_theta.R [0, 0] = 0.011
ekf_theta.R [1, 1] = 0.09
ekf_theta.R [2, 2] = 0.01

ekf_theta.R [3, 3] = 0.011
ekf_theta.R [4, 4] = 0.09
ekf_theta.R [5, 5] = 0.01


#print ([model.sensor("IMU_gyro").noise, model.sensor("IMU_acc").noise, model.sensor("IMU_acc").noise])
#print (ekf_theta.R)
#ekf_theta.Q = Q_discrete_white_noise (3, dt = dt, var = 0.0001)
ekf_theta.Q *= 0.001
# ekf_theta.R *= 0.001
#ekf_theta.Q *= 0
""" ekf_theta.R *= 0.0001
ekf_theta.Q *= 0.0001 """

#ekf_theta.P *= 10

P_val = np.zeros((6, 0))

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
    
    # bal_force = np.array([0, 0, - l_1 * g * (m1 / 2 + m2) * sin(x1)])
    # bal_force = np.array([0, 0, l_1 * g * (m1 / 2 + m2) * sin(data.qpos[0])])
    bal_force = - g * (m1 / 2 + m2) * sin(x1)
    
    cont_force_1 = 0
    cont_force_2 = 0
    
    #print (P [0, 0], P [1, 1])
    
    #if P [0, 0] < P_threshold and P [1, 1] < P_threshold:
    
    
    if np.trace(P) < 4 * P_threshold or True:           
            
        des_pos_2 = 0
        des_acc_2 = kp * (des_pos_2 - x2) - kd * x4
        # des_acc_2 *= -1
        
        # des_M_2 = I2 * des_acc_2
        
        # des_M_2 = 0
        
        # des_acc_1 = kp * (des_pos - data.qpos[0]) - kd * data.qvel[0]
        des_acc_1 = kp * (des_pos - x1) - kd * x3
        # des_acc_1 *= -1
        
        # des_M_1 = I *  des_acc_1
        # des_M_1 = - I *  des_M_1
        
        f_cont_2 = I2 * (des_acc_2 + des_acc_1) / l_2
        f1 += f_cont_2
        f2 -= f_cont_2

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
        cont_f_den = 2 * l_1 * sin(x2)
        if abs(cont_f_den) > 0.001: 
            
            cont_int = ((I - I2) * des_acc_1 - I2 * des_acc_2) / cont_f_den
            
            f1 += cont_int
            f2 += cont_int
        
        
        
        #print (data.ctrl)
        
        # f1 = cont_f_1
        # f2 = cont_f_2
    else:
        bal_force = 0.8 * bal_force
        #cont_f_den = 0
    
    # den_bal_force = 2 * l_1 * sin(x1 + x3)
    bal_int = 0
    bal_den = 2 * sin(x2)
    if bal_den > 0.001:
        bal_int = bal_force / bal_den
    f1 += bal_int
    f2 += bal_int  
       
    
    
    #print (f1, f2)
    data.actuator("propeller1").ctrl = f1
    data.actuator("propeller2").ctrl = f2
    
        
    # if apply_force:
    #     data.actuator("propeller1").ctrl = 5
    # else:
    #     data.actuator("propeller1").ctrl = 0
    
    

# mujoco.set_mjcb_control(control_callback)




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
    
    
    
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    
    c1 = math.cos(x1)
    s1 = math.sin(x1)
    g = - model.opt.gravity[2]
    
    
    H = np.zeros ((6, 1))
    
    I = compute_I1(x5,x6)
    I2 = compute_I2(x6)
    
    H[0] = x3
    
    H [1] = l_1 * s1 * g * (x5 / 2 + x6)
    H [1]+= l_1 * sin(x2) * (f1 + f2)
    H [1]+= l_2 / 2 * (f1 - f2)
    H [1]*= l_1 / I * c1
    H [1]-= l_1 * x3**2 * s1
    
    H [2] = l_1 * s1 * g * (x5 / 2 + x6)
    H [2]+= l_1 * sin(x2) * (f1 + f2)
    H [2]+= l_2 / 2 * (f1 - f2)
    H [2]*= - l_1 / I * s1
    H [2]-= l_1 * x3**2 * c1
    
    H [3] = x3 + x4
    
    H [4] = - l_2 / 2 * (x3 + x4)**2 * cos(x1 + x2) - l_2**2 / (4 * I2) * sin(x1 + x2) * (f1 - f2) + H [1]
    
    H [5] = l_2 / 2 * (x3 + x4)**2 * sin(x1 + x2) - l_2**2 / (4 * I2) * cos(x1 + x2) * (f1 - f2) + H [2]
    
    
    return H.reshape((-1, 1)) 

def H_jac (x):
    g = - model.opt.gravity[2]
    #global appl_force
    
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    # est_l1 = x5
    # est_l2 = x6

    c1 = math.cos(x1)
    s1 = math.sin(x1)
    # H = np.zeros((6, 4))
    H = np.zeros((6, 6))
    # H_test = np.zeros((3, 4))
    
    I = compute_I1(x5, x6)
    I2 = compute_I2(x6)
    
    
    H [0, 2] = 1
    
    
    H [1, 0] = l_1 * cos(2 * x1) * g * (x5 / 2 + x6)
    H [1, 0]-= l_1 * s1 * sin(x2) * (f1 + f2)
    H [1, 0]-= l_2 / 2 * s1 * (f1 - f2)
    H [1, 0]*= l_1 / I 
    H [1, 0]-= l_1 * x3**2 * c1
    
    H [1, 1] = l_1**2 / I * c1 * cos(x2) * (f1 + f2)
    
    H [1, 2] = - 2 * l_1 * x3 * s1
    
    H [1, 4] = + c1 / ((x5 + 3 * x6)**2) * (s1 * g * 3 / 2 * x6 - 3 * sin(x2) * (f1 + f2) - 3 / 2 * l_2 / l_1 * (f1 - f2))
    
    H [1, 5] = - c1 / ((x5 + 3 * x6)**2) * (s1 * g * 3 / 2 * x5 + 9 * sin(x2) * (f1 + f2) + 9 / 2 * l_2 / l_1 * (f1 - f2))
    
    
    H [2, 0] = l_1 * sin(2 * x1) * g * (x5 / 2 + x6)
    H [2, 0]+= l_1 * c1 * sin(x2) * (f1 + f2)
    H [2, 0]+= l_2 / 2 * c1 * (f1 - f2)
    H [2, 0]*= - l_1 / I 
    H [2, 0]+= l_1 * x3**2 * s1
    
    H [2, 1] = - l_1**2 / I * s1 * cos(x2) * (f1 + f2)
    
    H [2, 2] = - 2 * l_1 * x3 * c1
    
    H [2, 4] = - s1 / ((x5 + 3 * x6)**2) * (s1 * g * 3 / 2 * x6 - 3 * sin(x2) * (f1 + f2) - 3 / 2 * l_2 / l_1 * (f1 - f2))
    
    H [2, 5] = + s1 / ((x5 + 3 * x6)**2) * (s1 * g * 3 / 2 * x5 + 9 * sin(x2) * (f1 + f2) + 9 / 2 * l_2 / l_1 * (f1 - f2))
    
    
    H [3, 2] = 1
    H [3, 3] = 1
    
    
    H [4, 0] = l_2 / 2 * (x3 + x4)**2 * sin(x1 + x2) - l_2**2 / (4 * I2) * cos(x1 + x2) * (f1 - f2)
    
    H [4, 1] = H [4, 0]
    
    # H [4, 0]+= H [1, 0]
    
    # H [4, 1]+= H [1, 1]
    
    H [4, 2] = - l_2 * (x3 + x4) * cos(x1 + x2)
    
    H [4, 3] = H [4, 2]
    
    # H [4, 2]+= H [1, 2]
    
    # H [4, 3]+= H [1, 3]
    
    H [4, 5] = 3 / (x6**2) * sin(x1 + x2) * (f1 - f2)
    
    # H [4, 5]+= H [1, 5]
    
    # H [4, 4]+= H [1, 4]

    
    H [5, 0] = l_2 / 2 * (x3 + x4)**2 * cos(x1 + x2) + l_2 **2 / (4 * I2) * sin(x1 + x2) * (f1 - f2)
    
    H [5, 1] = H [5, 0]
    
    # H [5, 0]+= H [2, 0]
    
    # H [5, 1]+= H [2, 1]
    
    H [5, 2] = l_2 * (x3 + x4) * sin(x1 + x2)
    
    H [5, 3] = H [5, 2]
    
    # H [5, 2]+= H [2, 2]
    
    # H [5, 3]+= H [2, 3]
    
    H [5, 5] = - 3 / (x6**2) * cos(x1 + x2) * (f1 - f2)
    
    # H [5, 5]+= H [2, 5]
    
    # H [5, 4]+= H [2, 4]
    
    """ H [4, 0] = l_2 / 2 * (x3**2 + x4**2) * sin(x1 + x2) - l_2**2 / (4 * I2) * cos(x1 + x2) * (f1 - f2)
    
    H [4, 1] = H [4, 0]
    
    H [4, 2] = - l_2 * x3 * cos(x1 + x2)
    
    H [4, 3] = - l_2 * x4 * cos(x1 + x2)
    
    H [5, 0] = l_2 / 2 * (x3**2 + x4**2) * cos(x1 + x2) + l_2**2 / (4 * I2) * sin(x1 + x2) * (f1 - f2)
    
    H [5, 1] = H [5, 0]
    
    H [5, 2] = l_2 * x3 * sin(x1 + x2)
    
    H [5, 3] = l_2 * x4 * sin(x1 + x2) """
    return H

def compute_z (data, x):
    ang_data_1 = data.sensor ("IMU_1_gyro").data[1].copy() 
                
    acc_data_1 = data.sensor ("IMU_1_acc").data.copy()
    
    ang_data_2 = data.sensor("IMU_2_gyro").data[1].copy()
    
    acc_data_2 = data.sensor("IMU_2_acc").data.copy()
    
    # print (acc_data, data.cacc[1])
    
    
    
    
    # ang_data_1 += random.gauss(0, model.sensor("IMU_1_gyro").noise[0])
    # for i in range (0, 3):
    #     acc_data_1 [i] += random.gauss(0, model.sensor("IMU_1_acc").noise[0])
    
    # ang_data_2 += random.gauss(0, model.sensor("IMU_1_gyro").noise[0])
    # for i in range (0, 3):
    #     acc_data_2 [i] += random.gauss(0, model.sensor("IMU_1_acc").noise[0])
    
    q_est_1 = x[0]
    #q_est = - x[0]
    #q_est = data.qpos[0]
    q_est_2 = x[0] + x[1]
    #q_est = - x[2]
    
    R1 = np.array([[math.cos(q_est_1), 0, math.sin(q_est_1)], [0, 1, 0], [-math.sin(q_est_1), 0, math.cos(q_est_1)]])
    
    R2 = np.array([[math.cos(q_est_2), 0, math.sin(q_est_2)], [0, 1, 0], [-math.sin(q_est_2), 0, math.cos(q_est_2)]])
    
    
    mujoco.mju_mulMatVec (acc_data_1, R1, acc_data_1.copy())
    
    mujoco.mju_mulMatVec (acc_data_2, R2, acc_data_2.copy())
    
    
    #print (acc_data, data.body("prop_bar").cacc[3:])
    #print (acc_data - data.body("prop_bar").cacc[3:])
    
    
    
    acc_data_1 += model.opt.gravity#.reshape((3, 1))
    acc_data_2 += model.opt.gravity
    
    # acc_data_2 -= acc_data_1
    #print (acc_data, data.body("end_mass").cacc[3:] + model.opt.gravity)
    """ sim_acc = l_1 * (data.qacc[0] * np.array([math.cos(data.qpos[0]), 0, - math.sin(data.qpos[0])]) - data.qvel[0]**2 * np.array([math.sin(data.qpos[0]), 0, math.cos(data.qpos[0])])) 
    print (acc_data, sim_acc) """
    
    # z = np.array([ang_data_1, acc_data_1[0], acc_data_1[2], ang_data_2, acc_data_2[0] - acc_data_1[0], acc_data_2[2] - acc_data_1[2]])
    z = np.array([ang_data_1, acc_data_1[0], acc_data_1[2], ang_data_2, acc_data_2[0], acc_data_2[2]])
    
    #z = np.array([ang_data, acc_data [0], acc_data [1]]).reshape ((3, 1)) 
    #z = np.array([ang_data, acc_data [0], acc_data [1] - g]).reshape ((3, 1)) 
    #return np.array([ang_data, acc_data [0], acc_data [2]]).reshape ((-1, 1)) 
    #z = np.array([ang_data, - acc_data [0], - acc_data [2]]).reshape ((3, 1)) 
    
    return z.reshape((-1, 1))
    
z = np.zeros((ekf_theta.dim_z, 1))
h_est = np.zeros((ekf_theta.dim_z, 1))

#print ("setup sim")


     
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
                
                # print (ekf_theta.x, np.array([data.qpos, data.qvel]).reshape((4, 1)))
                
                #print (data.qvel[1] + data.qvel[0])
                sim_time.append (data.time)
                sim_theta.append(data.qpos[0])
                
                sim_theta2.append(data.qpos[1])
                """ est_q = ekf_theta.x[0] % (2 * math.pi)
                
                if est_q > math.pi:
                    est_q -= 2 * math.pi """
                est_q = ekf_theta.x[0].copy()
                est_theta.append(est_q)
                
                est_q2 = ekf_theta.x[1].copy()
                est_theta2.append(est_q2)
                """ angle.append(data.qpos[0])
                torque.append(data.ctrl[0]) """
                
                c_t_est = np.zeros(3)
                
                """ # sim_meas = np.append(sim_meas, z.reshape((3, -1)), axis = 1)
                # est_meas = np.append(est_meas, h_x(ekf_theta.x).reshape((3, -1)), axis = 1)

                # sim_meas = np.concatenate((sim_meas.copy(), z.reshape((3, -1))), axis = 1)
                # est_meas = np.concatenate((est_meas.copy(), h_x(ekf_theta.x).reshape((3, -1))), axis = 1)
                # meas_diff = np.concatenate((meas_diff.copy(), z.reshape((3, -1)) - h_x(ekf_theta.x).reshape((3, -1))), axis = 1)
                # x_sim = np.array([data.qpos[0], data.qpos[1], data.qvel[0], data.qvel[1]]).reshape((4, 1))
                # est_meas = np.concatenate((est_meas.copy(), h_x(x_sim).reshape((3, -1))), axis = 1)
                # meas_diff = np.concatenate((meas_diff.copy(), z.reshape((3, -1)) - h_x(x_sim).reshape((3, -1))), axis = 1)
                # sim_h = np.array([data.qvel[0] + data.qvel[1], - l_1 * data.qvel[0]**2 * sin(data.qpos[0]) + l_1 * data.qacc[0] * cos(data.qpos[0]), - l_1 * data.qvel[0]**2 * cos(data.qpos[0]) - l_1 * data.qacc[0] * sin(data.qpos[0])]).reshape((3, 1))
                # est_meas = np.concatenate((est_meas, sim_h), axis = 1)
                # meas_diff = np.concatenate((meas_diff.copy(), z.reshape((3, -1)) - sim_h), axis = 1)
                # est_meas = np.append(est_meas, h_x(x_sim).reshape((-1, 3)), axis = 0)
                # sim_meas = np.append(sim_meas, h_x(ekf_theta.x).reshape((-1, 3)), axis = 0)
                #meas_diff = np.append(meas_diff.reshape((3, -1)), z.reshape((3, -1)) - h_x(ekf_theta.x).reshape((3, -1))).reshape((3, -1))
                # meas_diff = np.append(meas_diff.reshape((3, -1)), z.reshape((3, -1)) - h_x(x_sim).reshape((3, -1))).reshape((3, -1)) """
                
                sim_meas = np.concatenate((sim_meas.copy(), z), axis = 1)
                est_meas = np.concatenate((est_meas.copy(), h_est), axis = 1)
                meas_diff = np.concatenate((meas_diff.copy(), z - h_est), axis = 1)
                
                # print (I, model.body("pendulum").inertia + l_1**2 / 4 * m1 + l_1 **2 * m2)
                sim_vel.append(data.qvel[0])
                est_vel.append(ekf_theta.x[2].copy())
                sim_vel2.append(data.qvel[1])
                # sim_vel_sum.append(2 * data.qvel[0] + data.qvel[1])
                # est_vel_sum.append(data.sensor ("IMU_gyro").data[1].copy())
                
                est_l_1.append(ekf_theta.x[4].copy())
                est_l_2.append(ekf_theta.x[5].copy())
                
                sim_acc.append(data.qacc[0])
                x = ekf_theta.x.copy()
                g = -model.opt.gravity[2]
                I = compute_I1(x[4], x[5])
                
                # est_acc_1 = 1 / I  * (l_1 * sin(x[0]) * g * (m1 / 2 + m2))
                # est_acc_1 = (+ 1 / I * (l_1 * sin(x[0]) * g * (m1 / 2 + m2) + l_1 * sin(x[1]) * (f1 + f2) + l_2 / 2 * (f1 - f2)))
                est_acc_1 = (+ 1 / I * (l_1 * sin(x[0]) * g * (x[4] / 2 + x[5]) + l_1 * sin(x[1]) * (f1 + f2) + l_2 / 2 * (f1 - f2)))
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
                q2_est = ekf_theta.x[1]
                
                q_rot_est = q1_est + q2_est
                est_prop1_pos = c_t_est + l_2 / 2 * np.array([-cos(q_rot_est), 0, sin(q_rot_est)])
                 
                mujoco.mjv_initGeom(viewer.user_scn.geoms[2],\
                    type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .06 * np.ones(3),\
                    pos = data.site("prop_1").xpos, mat = np.eye(3).flatten(), rgba = yellow_color)
                mujoco.mjv_initGeom(viewer.user_scn.geoms[3],\
                    type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .035 * np.ones(3),\
                    pos = est_prop1_pos, mat = np.eye(3).flatten(), rgba = cyan_color)
                
                # mujoco.mjv_initGeom(viewer.user_scn.geoms[2],\
                #     type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .06 * np.ones(3),\
                #     pos = data.site("IMU_loc").xpos, mat = np.eye(3).flatten(), rgba = yellow_color)
                # mujoco.mjv_initGeom(viewer.user_scn.geoms[3],\
                #     type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .05 * np.ones(3),\
                #     pos = data.site("test_pos").xpos, mat = np.eye(3).flatten(), rgba = cyan_color)
            
                x_sim = np.array([data.qpos, data.qvel]).reshape((4, 1))
                # sim_h = h_x(x_sim)
                # acc_err_curr = math.sqrt(mujoco.mju_norm(np.array([data.sensor("IMU_acc").data[0], 0, data.sensor("IMU_acc").data[2]]) + model.opt.gravity - np.array([sim_h[1, 0], 0, sim_h[2, 0]])))
                # sim_h = np.array([0, - l_1 * data.qvel[0]**2 * sin(data.qpos[0]) + l_1 * data.qacc[0] * cos(data.qpos[0]), - l_1 * data.qvel[0]**2 * cos(data.qpos[0] - l_1 * data.qacc[0] * sin(data.qpos[0]))])
                # meas_acc_norm = mujoco.mju_norm(np.array([data.sensor("IMU_acc").data[0], 0, data.sensor("IMU_acc").data[2]]).reshape((3, -1)) + model.opt.gravity.reshape((3, -1)))
                # sim_acc_norm = mujoco.mju_norm(np.array([sim_h[1], 0, sim_h[2]]).reshape((3, -1)))
                # acc_err_curr =  meas_acc_norm - sim_acc_norm
                # # print (acc_err_curr)
                # acc_err.append(acc_err_curr)
                
                # print (mujoco.mju_dist3(data.site("IMU_loc").xpos, data.joint("prop_orientation_joint").xanchor))

                #draw_vector(viewer, 2, data.site("IMU_loc").xpos, red_color, appl_force, 0.05 * mujoco.mju_norm(appl_force))
                w_bar_pitch =  data.qpos[1] + data.qpos[0]
                draw_vector_euler(viewer, 4, data.site("prop_1").xpos, red_color, k_f * f1, np.array([0, w_bar_pitch, 0]))
                draw_vector_euler(viewer, 5, data.site("prop_2").xpos, red_color, k_f * f2, np.array([0, w_bar_pitch, 0]))
                viewer.user_scn.ngeom =  6
                
                #debug 
                
                # x1_acc = 1 / I  * (l_1 * sin(x[0]) * g * (m1 / 2 + m2) - f1 * (l_1 * sin(x[0] - x[2]) - l_2 / 2) - f2 * (l_1 * sin(x[0] - x[2]) + l_2 / 2))
                
                # sim_acc_1 =  1 / I  * (l_1 * sin(data.qpos[0]) * g * (m1 / 2 + m2) - f1 * (l_1 * sin(data.qpos[0] + data.qpos[1]) + l_2 / 2) - f2 * (l_1 * sin(data.qpos[0] + data.qpos[1]) - l_2 / 2))
                # # print (x1_acc, data.qacc[0])
                # print (sim_acc_1 - data.qacc[0])
                
                # print (mujoco.mju_dist3(data.joint("pend_joint").xanchor, data.site("IMU_loc").xpos))
                
                mujoco.mj_step(model, data)
                
                x = ekf_theta.x.copy()
                g = - model.opt.gravity[2]
                I = compute_I1(ekf_theta.x[4], ekf_theta.x[5])
                I2 = compute_I2(ekf_theta.x[5])
                
                # f1 = 0
                # f2 = 0
                
                
                F = np.eye(6) + dt * np.diag(np.array([1, 1, 0, 0]), 2)
                
                
                # print (F)
                
                # F [2, 0] = dt / I * (l_1 * math.cos(x[0]) * g * (m1 / 2 + m2) + f1 * 2 * l_2 * math.cos(2 * (x[0] + x[1])) + f2 * 2 * l_1 * math.cos(2 * x[0] + x[1]))
                
                # F [2, 1] = dt / I * (f1 * (l_1 * math.cos(x[1]) + 2 * l_2 * math.cos(2 * (x[0] + x[1]))) + f2 * l_1 * math.cos(2 * x[0] + x[1]))
                
                F [2, 0] = l_1 * cos(x[0]) * g * (x[4] / 2 + x[5])
                F [2, 0]*= dt / I 
                
                F [2, 1] = l_1 * cos(x[1]) * (f1 + f2)
                F [2, 1]*= dt / I 
                
                F [2, 4] = 3 / 2 * x[5] / l_1 * sin(x[0]) * g - 3 / l_1 * sin(x[1]) * (f1 + f2) - 3 / 2 * l_2 / (l_1**2) * (f1 - f2)
                F [2, 4]*= dt / ((x[4] + 3 * x[5])**2)                
                
                F [2, 5] = 3 / 2 * x[4] / l_1 * sin(x[0]) * g + 9 / l_1 * sin(x[1]) * (f1 + f2) + 9 / 2 * l_2 / (l_1**2) * (f1 - f2)
                F [2, 5]*= - dt / ((x[4] + 3 * x[5])**2)
                
                
                F [3, 0] = l_1 * cos(x[0]) * g * (x[4] / 2 + x[5])
                F [3, 0]*= - dt / I
                
                F [3, 1] = l_1 * cos(x[1]) * (f1 + f2)
                F [3, 1]*= - dt / I
                
                F [3, 4] = 3 / 2 * x[5] / l_1 * sin(x[0]) * g - 3 / l_1 * sin(x[1]) * (f1 + f2) - 3 / 2 * l_2 / (l_1**2) * (f1 - f2)
                F [3, 4]*= - dt / ((x[4] + 3 * x[5])**2)
                
                F [3, 5] = 3 / 2 * x[4] / l_1 * sin(x[0]) * g + 9 / l_1 * sin(x[1]) * (f1 + f2) + 9 / 2 * l_2 / (l_1**2) * (f1 - f2)
                F [3, 5]*= 1 / ((x[4] + 3 * x[5])**2)
                F [3, 5]-= 6 * l_2 / (l_1**2) / (x[5]**2) * (f1 - f2)
                F [3, 5]*= dt
                
                
                
                
                ekf_theta.F = F.copy()
                
                B = np.zeros((4, 2))
                
                B [2, 0] = l_1 * sin(x[1]) + l_2 / 2
                B [2, 0]*= dt / I
                
                B [2, 1] = l_1 * sin(x[1]) - l_2 / 2
                B [2, 1]*= dt / I
                
                
                B [3, 0] = - l_1 / I * sin(x[1]) + (1 / I2 - 1 / I) * l_2 / 2
                B [3, 0]*= dt
                
                B [3, 1] = - l_1 / I * sin(x[1]) - (1 / I2 - 1 / I) * l_2 / 2
                B [3, 1]*= dt
                
                ekf_theta.B = B.copy()
                
                #f_u = np.array([appl_force[0], appl_force[2]]).reshape((2, 1))
                f_u = np.array([f1, f2]).copy()
                
                ekf_theta.predict(f_u)
                # ekf_theta.predict(np.array([0, 0]))
                
                #if (ekf_count == 0) and False:
                if (ekf_count == 0): 
                                                            
                    
                    mujoco.mj_forward(model, data)
                    z = compute_z(data, ekf_theta.x)
                    h_est = h_x(ekf_theta.x)
                    
                    ekf_theta.update(z = z, HJacobian = H_jac, Hx = h_x)
                    
                    #H_jac(ekf_theta.x)
                    # H = H_jac(ekf_theta.x)
                    # P = ekf_theta.P.copy()
                    # R = ekf_theta.R
                    
                    # K = P@H.T@np.linalg.inv(H@P@H.T + R)
                    
                    # print (K @ (z - h_est))
                    
                    # print (h_x(ekf_theta.x) - np.array([data.qvel[0] + data.qvel[1], l_1 * data.qvel[0]**2 * - sin(data.qpos[0]) + l_1 * data.qacc[0] * cos(data.qpos[0]), l_1 * data.qvel[0]**2 * - cos(data.qpos[0]) - l_1 * data.qacc[0] * sin(data.qpos[0])]).reshape((3, 1)))
                    
                
                # print (np.diag(ekf_theta.P))
                P_val = np.concatenate((P_val, np.diag(ekf_theta.P).reshape((-1, 1))), axis = 1)
                ekf_count = (ekf_count + 1) % count_max
                
                
                
                # try:
                #     mujoco.mj_step(model, data)
                # except Exception as e:
                #     print (e)
                #     exit()
                

                
                #mujoco.mjv_updateScene(model, data, viewer.opt, viewer.perturb, viewer.cam, mujoco.mjtCatBit.mjCAT_ALL, viewer.user_scn)
                    
            
            #mujoco.mj_kinematics(model, data)    
            viewer.sync() 
            
            
            
            
        time_to_step = model.opt.timestep - (time.time() - step_start)
        if (time_to_step > 0):
            time.sleep(time_to_step)
           


print (ekf_theta.x)
#print (angle_err)
# print (sim_meas.size)
# sim_meas = np.asarray(sim_meas).reshape((3, -1))
# est_meas = np.asarray(est_meas).reshape((3, -1))
# print (sim_meas.size)
fig1, ax = plt.subplots(4, 2, sharex = True)

#ax[0].plot(sim_time, np.array([angle_err, force_norm]).reshape((-1, 2)))
#ax[0].legend(['theta', 'total force'])
""" ax.plot(sim_time, sim_theta)
ax.plot(sim_time, est_theta)
#ax.plot(sim_time, [s - e for s,e in zip(sim_theta, est_theta)])
ax.legend(['sim', 'est', 'diff']) """
ax[0][0].plot(sim_time, sim_theta)
ax[0][0].plot(sim_time, est_theta)
ax[0][1].plot(sim_time, [s - e for s,e in zip(sim_theta, est_theta)])
# ax[0][1].plot(sim_time, [(1 if s > e else -1) for s,e in zip(sim_theta, est_theta)])
ax[0][0].legend(['sim', 'est'])
ax[0][1].legend(['sim', 'est'])

ax[1][0].plot(sim_time, sim_vel)
ax[1][0].plot(sim_time, est_vel)
ax[1][1].plot(sim_time, [s - e for s,e in zip(sim_vel, est_vel)])
# ax[1][1].plot(sim_time, [(1 if s > e else -1) for s,e in zip(sim_vel, est_vel)])
ax[0][0].legend(['sim', 'est'])
ax[0][1].legend(['sim', 'est'])

ax[2][0].plot(sim_time, sim_acc)
ax[2][0].plot(sim_time, est_acc)
ax[2][1].plot(sim_time, [s - e for s,e in zip(sim_acc, est_acc)])
# ax[2][1].plot(sim_time, [(1 if s > e else -1) for s,e in zip(sim_acc, est_acc)])
ax[0][0].legend(['sim', 'est'])
ax[0][1].legend(['sim', 'est'])

ax[3][0].plot(sim_time, est_l_1)
ax[3, 0].plot(sim_time, model.body("pendulum").mass * np.ones(len(sim_time)))
ax[3, 1].plot(sim_time, est_l_1 - model.body("pendulum").mass)
fig1.waitforbuttonpress()

fig1, ax = plt.subplots(2, 2, sharex = True)

# ax[0].plot(sim_time, sim_theta2)
# ax[0].plot(sim_time, est_theta2)
# ax[1].plot(sim_time, [s - e for s, e in zip(sim_theta2, est_theta2)])
# ax[0].legend(['sim', 'est'])
ax[0][0].plot(sim_time, sim_theta2)
ax[0][0].plot(sim_time, est_theta2)
ax[0][1].plot(sim_time, [s - e for s, e in zip(sim_theta2, est_theta2)])
ax[0, 0].legend(['sim', 'est'])

ax[1, 0].plot(sim_time, est_l_2)

close = False
while(not close):
    close = fig1.waitforbuttonpress()




fig1, ax = plt.subplots(dim_z, 2, sharex = True)
for i in range (dim_z):
    # ax[i][0].plot(sim_time[:-1], sim_meas[i, 1:])
    # ax[i][0].plot(sim_time[:-1], est_meas[i, :-1])
    ax[i][0].plot(sim_time[:], sim_meas[i, :])
    ax[i][0].plot(sim_time[:], est_meas[i, :])
    ax[i][1].plot(sim_time, meas_diff[i, :])
    ax[i][1].legend(['diff'])
    #ax[i].legend(['sim', 'est', 'diff'])
    ax[i][0].legend(['sim', 'est'])
#plt.show()
close = False
while(not close):
    close = fig1.waitforbuttonpress()

# fig1, ax = plt.subplots(1, 1, sharex = True)

# ax.plot(sim_time, acc_err)

# # ax.legend(['sim', 'est'])

# close = False
# while(not close):
#     close = fig1.waitforbuttonpress()

fig1, ax = plt.subplots(1, 1, sharex = True)

for k_P_val in P_val:
    ax.plot(sim_time, k_P_val)

ax.legend(['x1', 'x2', 'x3', 'x4'])

close = False
while(not close):
    close = fig1.waitforbuttonpress()

