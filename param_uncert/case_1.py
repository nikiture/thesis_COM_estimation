import mujoco
#import mediapy
#from IPython import display
#from IPython.display import clear_output
import mujoco.viewer
import time
import numpy as np
from matplotlib import pyplot as plt
#import itertools
import math
import random
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise


    

model = mujoco.MjModel.from_xml_path ("models/2d_pend.xml")
#model.opt.timestep = 0.002
#model = mujoco.MjModel.from_xml_path ("tutorial_models/3D_pendulum_actuator.xml")
#model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
data = mujoco.MjData(model)

class Modif_EKF(ExtendedKalmanFilter):
    def __init__(self, dim_x, dim_z, dim_u = 0):
        super().__init__(dim_x, dim_z, dim_u)
    def predict_x(self, u = 0):
        g = - model.opt.gravity[2]
        self.x = x + dt * np.array([x[1], l_1 / I * (math.sin(x[0]) * (g * (m1 / 2 + m2) - u[1]) + u[0] * math.cos(x[0]))]).reshape((2, 1))




start_angle = 0.1

#start_angle = math.pi - start_angle

data.qpos = start_angle

mujoco.mj_forward(model, data)

mujoco.mju_zero(data.sensordata)

#l_2 = mujoco.mju_dist3(data.joint("prop_base_joint").xanchor, data.site("leg_end").xpos)
l_1 = 0.5
b_1 = 0.01

r_2 = 0.05

m1 = data.body("pendulum").cinert[9]
m2 = data.body("end_mass").cinert[9]

I = l_1**2 * (m1 / 3 + m2) + 2 / 5 * m2 * r_2**2
#I = 
#print (data.sensor("IMU_acc").data)

main_bodies_names = ["propeller_base", "leg_1"]

blue_color = np.array ([0, 0, 1, 1])
red_color = np.array ([1, 0, 0, 1])
green_color = np.array([0, 1, 0, 1])
yellow_color = np.array([1, 0, 1, 1])

exit = False
pause = True
step = False

sim_time = []
sim_theta = []
est_theta = []
sim_vel = []
est_vel = []
meas_diff = np.zeros((3, 0))
sim_meas = np.zeros((0, 3))
est_meas = np.zeros((0, 3))


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


k_f = 0.18 #force visualization amplifier
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

dt = model.opt.timestep * count_max

#ekf_theta = ExtendedKalmanFilter (2, 3) #1 dim for angular velocity, 2 for acceleration
#ekf_theta = ExtendedKalmanFilter (2, 3, 2) #1 dim for angular velocity, 2 for acceleration
ekf_theta = Modif_EKF(2, 3, 2)

ekf_theta.x [0] = data.qpos[0]

ekf_theta.x [0] += random.random() * 0.05

#print (ekf_theta.x[0], data.qpos[0])

""" ekf_theta.R [0, 0] = model.sensor("IMU_gyro").noise
ekf_theta.R [1, 1] = model.sensor("IMU_acc").noise
ekf_theta.R [2, 2] = model.sensor("IMU_acc").noise """

ekf_theta.R [0, 0] = 0.011
ekf_theta.R [1, 1] = 0.009
ekf_theta.R [2, 2] = 0.01


#print ([model.sensor("IMU_gyro").noise, model.sensor("IMU_acc").noise, model.sensor("IMU_acc").noise])
#print (ekf_theta.R)
#ekf_theta.Q = Q_discrete_white_noise (3, dt = dt, var = 0.0001)
ekf_theta.Q *= 0.001
#ekf_theta.R *= 0.001
#ekf_theta.Q *= 0
""" ekf_theta.R *= 0.0001
ekf_theta.Q *= 0.0001 """

#ekf_theta.P *= 10

appl_force = np.zeros(3)

kp = 0.5
kd = 0.1

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



        
def control_callback (model, data):
    
    global appl_force
    
    g = - model.opt.gravity[2]
    
    x = ekf_theta.x.copy()
    
    P = ekf_theta.P.copy()
    
    x1 = x[0, 0]
    x2 = x[1, 0]
    
    x1 = x1 % (2 * math.pi)
                
    if x1 > math.pi:
        x1 -= 2 * math.pi
    
    """ x1 = data.qpos[0]
    x2 = data.qvel[0] """
    
    """ x1 = - x[0, 0]
    x2 = - x[1, 0] """
    
    bal_force = np.array([0, 0, g * (m1 / 2 + m2)])
    
    cont_force = np.zeros(3)
    
    #print (P [0, 0], P [1, 1])
    
    if P [0, 0] < P_threshold and P [1, 1] < P_threshold:
        #des_moment = kp * (des_pos - data.qpos[0]) - kd * data.qvel[0]
        des_moment = kp * (des_pos - x1) - kd * x2
        
        des_moment = I * des_moment
        #des_moment = - I * des_moment
        
        s1 = math.sin(x1)
        c1 = math.cos(x1)
        cont_force_x = 0
        cont_force_z = 0
        if abs(c1) < 0.5:
            #cont_force_x = des_moment / (l_1 * math.cos(data.qpos[0]))
            cont_force_x = des_moment / (l_1 * c1)
            #cont_force_y = - des_moment / (l_1 * math.sin(data.qpos[0]))
        elif abs(s1) < 0.5:
            cont_force_z = - des_moment / (l_1 * s1)
        else:
            cont_force_x = des_moment / (l_1 * c1) / 2
            cont_force_z = - des_moment / (l_1 * s1) / 2
            
        
        cont_force[0] = cont_force_x
        cont_force [2] = cont_force_z
    else:
        bal_force = 0.8 * bal_force
    
    #cont_force = np.zeros(3)
    #bal_force = np.array([0, 0, g * (m1 / 2 + m2)])
    #appl_force = np.array([cont_force_x, 0, bal_force])
    mujoco.mju_add (appl_force, cont_force, bal_force)
    
    #print (cont_force_x)
    #print (bal_force)
    #print (appl_force)
    
    mujoco.mju_copy(data.body("end_mass").xfrc_applied[:3], appl_force)
    
    #print (data.xfrc_applied)
    
    
    
mujoco.set_mjcb_control(control_callback)







def kb_callback(keycode):
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
    """ if chr(keycode) == 'F':
        global appl_force
        appl_force = not appl_force """
        
#x = np.zeros(6)
def h_x (x):
    global appl_force
    Fx = appl_force[0]
    Fy = appl_force[2]
    """ Fx = 0
    Fy = 0 """
    c1 = math.cos(x[0])
    s1 = math.sin(x[0])
    g = - model.opt.gravity[2]
    #ang_acc = l_1 / I * (-(m1 / 2 + m2) * g * s1)
    
    H = np.zeros (3)
    H[0] = x [1]
    
    #H [1] = - l_1 * x[1]**2 * s1 + l_1**2 / I * c1 * (- g * (m1 / 2 + m2) * s1 + Fx * c1 + Fy * s1)
    
    H [1] = - l_1 * x[1]**2 * s1 + l_1**2 / I * c1 * (g * (m1 / 2 + m2) * s1 + Fx * c1 - Fy * s1)
    
    H [2] = - l_1 * x[1]**2 * c1 - l_1**2 / I * s1 * (g * (m1 / 2 + m2) * s1 + Fx * c1 - Fy * s1)
    
    """ H [1] = - l_1 * x[1]**2 * s1 - l_1**2 / I * c1 * (g * (m1 / 2 + m2) * s1 + Fx * c1 - Fy * s1)
    
    H [2] = - l_1 * x[1]**2 * c1 + l_1**2 / I * s1 * (g * (m1 / 2 + m2) * s1 + Fx * c1 - Fy * s1) """
    
    return H.reshape((3, 1)) 

def H_jac (x):
    g = - model.opt.gravity[2]
    global appl_force
    Fx = appl_force[0]
    Fy = appl_force[2]
    
    x1 = x[0]
    x2 = x[1]
    #x1 = - x[0]
    #x2 = - x[1]
    """ Fx = 0
    Fy = 0 """
    c1 = math.cos(x1)
    s1 = math.sin(x1)
    H = np.zeros((3, 2))
    
    H [0, 1] = 1
    
    """ H [1, 1] = - 2 * x[1] * l_1 * s1
    
    H [1, 0] = - l_1 * x[1]**2 * c1 - l_1**2 / I * ((2 * c1**2 - 1) * (g * (m1 / 2 + m2) - Fy) + 2 * s1 * c1 * Fx) """
    
    H [1, 1] = - 2 * x2 * l_1 * s1
    
    H [1, 0] = - l_1 * x2**2 * c1 + l_1**2 / I * ((c1**2 - s1**2) * (g * (m1 / 2 + m2) - Fy) - 2 * s1 * c1 * Fx)
    
    H [2, 1] = - 2 * x2 * l_1 * c1
    
    H [2, 0] = l_1 * x2**2 * s1 - l_1**2 / I * (2 * s1 * c1 * (g * (m1 / 2 + m2) - Fy) + (c1**2 - s1**2) * Fx) 
    
    """ H [1, 1] = - 2 * x2 * l_1 * s1
    
    H [1, 0] = - l_1 * x2**2 * c1 - l_1**2 / I * ((c1**2 - s1**2) * (g * (m1 / 2 + m2) - Fy) - 2 * s1 * c1 * Fx)
    
    H [2, 1] = - 2 * x2 * l_1 * c1
    
    H [2, 0] = l_1 * x2**2 * s1 + l_1**2 / I * (2 * s1 * c1 * (g * (m1 / 2 + m2) - Fy) + (c1**2 - s1**2) * Fx) """ 


    return H
    
z = np.zeros(3)

        
        
with mujoco.viewer.launch_passive(model, data, key_callback= kb_callback) as viewer:
    
    viewer.lock()
    
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
    
    #print (int(mujoco.mjtFrame.mjFRAME_BODY), int(mujoco.mjtFrame.mjFRAME_GEOM), int(mujoco.mjtFrame.mjFRAME_SITE))
    
    """ for i in range (model.nsite + model.njnt):
        mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn) """
    
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    #print ('simulation setup completed')
    viewer.sync()
    
    #print (mujoco.mjtCatBit.mjCAT_ALL)
    while viewer.is_running() and (not exit) :
        step_start = time.time()
        #mujoco.mj_forward(model, data)
        #print ('running iteration')
        
        
        if not pause or step:
            step = False
            with viewer.lock():
                
                sim_time.append (data.time)
                sim_theta.append(data.qpos[0])
                """ est_q = ekf_theta.x[0] % (2 * math.pi)
                
                if est_q > math.pi:
                    est_q -= 2 * math.pi """
                est_q = ekf_theta.x[0]
                est_theta.append(est_q)
                """ angle.append(data.qpos[0])
                torque.append(data.ctrl[0]) """
                sim_meas = np.append(sim_meas, z.reshape((-1, 3)), axis = 0)
                est_meas = np.append(est_meas, h_x(ekf_theta.x).reshape((-1, 3)), axis = 0)
                
                sim_vel.append(data.qvel[0])
                est_vel.append(ekf_theta.x[1])
                
                q_est = ekf_theta.x[0]
                #q_est = - ekf_theta.x[0]
                
                #c_t_est = l_1 * np.array([- math.sin(q_est), 0, math.cos(q_est)])  
                #print (q_est)  
                c_t_est = l_1 / 2 * np.array([math.sin(q_est), 0, math.cos(q_est)])    
                #c_t_est = - l_1 / 2 * np.array([math.sin(q_est), 0, math.cos(q_est)]) 
                
                #print (dir(data.site('stick_end')))
                
                """ draw_vector(viewer, 0, np.array(data.geom("propeller_body").xpos), blue_color, data.xfrc_applied[2][0:3], k_f * mujoco.mju_norm(data.xfrc_applied[2][0:3]))
                draw_vector(viewer, 1, np.array(data.joint('pend_joint').xanchor), red_color, data.xfrc_applied[2][3:], k_f * mujoco.mju_norm(data.xfrc_applied[2][3:]))
                draw_vector(viewer, 2, np.array(data.joint('pend_joint').xanchor), green_color, q_err[1:4]/math.sin(p_err/2), k_theta * p_err) """
                #viewer.user_scn.ngeom = 3
                mujoco.mjv_initGeom(viewer.user_scn.geoms[0],\
                type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .06 * np.ones(3),\
                pos = data.subtree_com[1], mat = np.eye(3).flatten(), rgba = blue_color)
                mujoco.mjv_initGeom(viewer.user_scn.geoms[1],\
                    type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .035 * np.ones(3),\
                    pos = c_t_est, mat = np.eye(3).flatten(), rgba = green_color)
                """ mujoco.mjv_initGeom(viewer.user_scn.geoms[0],\
                    type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .06 * np.ones(3),\
                    pos = data.body("end_mass").xpos, mat = np.eye(3).flatten(), rgba = blue_color)
                mujoco.mjv_initGeom(viewer.user_scn.geoms[1],\
                    type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .035 * np.ones(3),\
                    pos = data.site("IMU_loc").xpos, mat = np.eye(3).flatten(), rgba = green_color) """
            
                

                draw_vector(viewer, 2, data.site("IMU_loc").xpos, red_color, appl_force, 0.05 * mujoco.mju_norm(appl_force))
                
                viewer.user_scn.ngeom =  3
                
                """ print (I)
                print (data.body("pendulum").cinert) """
                
                #mujoco.mj_step(model, data)
                
                
                
                g = - model.opt.gravity[2]
                
                #F = np.eye(2) + dt * np.array([[0, 1], [(3 / 2) / l_2 * model.opt.gravity[2] * math.cos(ekf_theta.x[0]), 0]])
                #print (ekf_theta.x)
                x = ekf_theta.x.copy()
                Fy = appl_force[2]
                Fx = appl_force[0]
                """ Fy = 0
                Fx = 0 """
                
                F = np.eye(2) + dt * np.array([[0, 1], [0, 0]])
                
                #print (F)
                
                F [1, 0] = dt * l_1 / I * (math.cos(x[0]) * (g * (m1 / 2 + m2) - Fy) - math.sin(x[0]) * Fx)
                #F [1, 0] = - dt * l_1 / I * (math.cos(x[0]) * (g * (m1 / 2 + m2) - Fy) - math.sin(x[0]) * Fx)
                #F [1, 0] = dt * l_1 / I * (math.cos(x[0]) * (g * (m1 / 2 + m2)))
                
                
                
                ekf_theta.F = F.copy()
                
                B = np.zeros((2, 2))
                
                B [1, 0] = dt * l_1 / I * math.cos(x[0])
                
                B [1, 1] = - dt * l_1 / I * math.sin(x[0])
                
                """ B [1, 0] = - dt * l_1 / I * math.cos(x[0])
                
                B [1, 1] = dt * l_1 / I * math.sin(x[0]) """
                
                ekf_theta.B = B.copy()
                
                f_u = np.array([appl_force[0], appl_force[2]]).reshape((2, 1))
                
                ekf_theta.predict(f_u)
                
                #if (ekf_count == 0) and False:
                if (ekf_count == 0): 
                                                            
                    #ang_data = data.sensor ("IMU_gyro").data[2].copy() 
                    #print (data.sensor ("IMU_gyro").data)
                    mujoco.mj_forward(model, data)
                    ang_data = data.sensor ("IMU_gyro").data[1].copy() 
                    
                    #print (ang_data, data.qvel)
                    #print (data.sensor ("IMU_gyro").data, data.qvel)
                    
                    acc_data = data.sensor ("IMU_acc").data.copy()
                    
                    
                    #print (random.gauss(0, model.sensor("IMU_gyro").noise[0]))
                    
                    #print (random.gauss(0, model.sensor("IMU_gyro").noise[0]))
                    
                    ang_data += random.gauss(0, model.sensor("IMU_gyro").noise[0])
                    for i in range (0, 3):
                        acc_data [i] += random.gauss(0, model.sensor("IMU_acc").noise[0])
                    
                    
                    #print (acc_data)
                    #acc_data.shape = (3, 1)
                    #print (data.sensor ("IMU_acc").data)
                    q_est = x[0]
                    #q_est = - x[0]
                    #q_est = data.qpos[0]
                    
                    R = np.array([[math.cos(q_est), 0, math.sin(q_est)], [0, 1, 0], [-math.sin(q_est), 0, math.cos(q_est)]])
                    #print (R)
                    #print (math.cos(q_est))
                    #print (q_est[0], data.qpos[0])
                    #R = np.array([[math.cos(q_est), math.sin(q_est), 0], [-math.sin(q_est), math.cos(q_est), 0], [0, 0, 1], ])
                    
                    #print (R)
                    
                    #print (acc_data)
                    #acc_data = acc_data + model.opt.gravity
                    
                    #mujoco.mju_mulMatTVec (acc_data, R, acc_data.copy())
                    mujoco.mju_mulMatVec (acc_data, R, acc_data.copy())
                    
                    #acc_data.shape = (3, 1)
                    #print(acc_data)
                    
                    #acc_data = data.body("end_mass").cacc[3:]
                    
                    acc_data = acc_data + model.opt.gravity#.reshape((3, 1))
                    
                    
                    #print (acc_data, data.body("end_mass").cacc[3:] + model.opt.gravity)
                    """ sim_acc = l_1 * (data.qacc[0] * np.array([math.cos(data.qpos[0]), 0, - math.sin(data.qpos[0])]) - data.qvel[0]**2 * np.array([math.sin(data.qpos[0]), 0, math.cos(data.qpos[0])])) 
                    print (acc_data, sim_acc) """
                    
                    #print (l_1 / I * (math.sin(data.qpos[0]) * (g * (m1 / 2 + m2) - appl_force[2]) + math.cos(data.qpos[0]) * appl_force[0]), data.qacc[0])
                    
                    #z = np.array([ang_data, acc_data [0], acc_data [1]]).reshape ((3, 1)) 
                    #z = np.array([ang_data, acc_data [0], acc_data [1] - g]).reshape ((3, 1)) 
                    z = np.array([ang_data, acc_data [0], acc_data [2]]).reshape ((3, 1)) 
                    #z = np.array([ang_data, - acc_data [0], - acc_data [2]]).reshape ((3, 1)) 
                    
                    
                    #print(dir(data.body("leg_1")))
                    
                    """ meas_val.append(z.copy())
                    
                    est_meas.append(h_x(ekf_theta.x)) """
                    
                    
                    
                    """ print (data.qpos[0], ekf_theta.x[0])
                    print (data.qvel[0], ekf_theta.x[1])
                    print (data.qacc[0], l_1 / I * (math.sin(data.qpos[0]) * (g * (m1 / 2 + m2) - appl_force[2]) + math.cos(data.qpos[0]) * appl_force[0]))
                     """
                    #print (F[1, 0] * x[0] / dt, data.qacc[0])
                    
                    #print (z.reshape((1, 3)), h_x(ekf_theta.x).reshape((1, 3)))
                    resid = z - h_x(ekf_theta.x)
                    abs_resid = np.zeros ((3, 1))
                    np.absolute(resid.copy(), out = abs_resid)
                    #print (abs_resid)
                    #meas_diff = np.append(meas_diff.reshape(3, -1), abs_resid.reshape((3, 1))).reshape((3, -1))
                    
                    resid.shape = (1, 3)
                    #print (resid)
                    
                    
                    ekf_theta.update(z = z, HJacobian = H_jac, Hx = h_x)
                    #print ('post update: ', ekf_theta.x)
                    #ekf_theta.predict_update(z_sim, HJacobian = H_jac, Hx = h_x)
                    #ekf_theta.predict_update(data.sensor("IMU_gyro").data[1], HJacobian = H_jac, Hx = h_x)
                    #ekf_theta.predict_update(data.qvel[0], HJacobian = H_jac_ang, Hx = h_x_ang)
                    
                    #print (np.asarray(ekf_theta.x))
                
                    
                ekf_count = (ekf_count + 1) % count_max
                
                meas_diff = np.append(meas_diff.reshape(3, -1), resid.reshape((3, 1))).reshape((3, -1))
                
                mujoco.mj_step(model, data)
                
                

                
                #mujoco.mjv_updateScene(model, data, viewer.opt, viewer.perturb, viewer.cam, mujoco.mjtCatBit.mjCAT_ALL, viewer.user_scn)
                    
            
            #mujoco.mj_kinematics(model, data)    
            viewer.sync() 
            
            
            
            
        time_to_step = model.opt.timestep - (time.time() - step_start)
        if (time_to_step > 0):
            time.sleep(time_to_step)
            

#print (angle_err)
#  print (sim_meas)
sim_meas = np.asarray(sim_meas).reshape((3, -1))
est_meas = np.asarray(est_meas).reshape((3, -1))
fig1, ax = plt.subplots(2, 1, sharex = True)

#ax[0].plot(sim_time, np.array([angle_err, force_norm]).reshape((-1, 2)))
#ax[0].legend(['theta', 'total force'])
""" ax.plot(sim_time, sim_theta)
ax.plot(sim_time, est_theta)
#ax.plot(sim_time, [s - e for s,e in zip(sim_theta, est_theta)])
ax.legend(['sim', 'est', 'diff']) """
ax[0].plot(sim_time, sim_theta)
ax[0].plot(sim_time, est_theta)
#ax.plot(sim_time, [s - e for s,e in zip(sim_theta, est_theta)])
ax[0].legend(['sim', 'est'])

ax[1].plot(sim_time, sim_vel)
ax[1].plot(sim_time, est_vel)
#ax.plot(sim_time, [s - e for s,e in zip(sim_theta, est_theta)])
ax[1].legend(['sim', 'est'])


fig1.waitforbuttonpress()
#fig1.clear()
""" fig1, ax = plt.subplots(3, 1, sharex = True)
for i in range (3):
    ax[i].plot(sim_time, sim_meas[i, :].reshape((-1, 1)))
    ax[i].plot(sim_time, est_meas[i].reshape((-1, 1)) - 10)
    ax[i].legend(['sim', 'est'])
#plt.show()
fig1.waitforbuttonpress() """

#fig1.clear()
#print (meas_diff.shape)
fig1, ax = plt.subplots(3, 1, sharex = True)
for i in range (3):
    #ax[i].plot(sim_time, sim_meas[i].reshape((-1, 1)) - est_meas[i].reshape((-1, 1)))
    ax[i].plot(sim_time, meas_diff[i, :])
    ax[i].legend(['diff'])
#plt.show()
fig1.waitforbuttonpress()

""" _, ax = plt.subplots(1, 1, sharex = True)

#ax[0].plot(sim_time, np.array([angle_err, force_norm]).reshape((-1, 2)))
#ax[0].legend(['theta', 'total force'])
ax.plot(sim_time, sim_theta)
ax.plot(sim_time, est_theta)
ax.legend(['sim', 'est'])
plt.waitforbuttonpress() """