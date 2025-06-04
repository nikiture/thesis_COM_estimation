import mujoco
import mediapy
from IPython import display
from IPython.display import clear_output
import mujoco.viewer
import time
import numpy as np
from matplotlib import pyplot as plt
import itertools
import math
import random
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise


#random.seed(1)
model = mujoco.MjModel.from_xml_path("models/single_2D_pend.xml")
data = mujoco.MjData(model)

data.qpos[0] = 0.2

mujoco.mj_forward(model, data)
#mujoco.mj_forwardSkip(model, data, mujoco.mjtStage.)
#mujoco.mj_kinematics(model, data)

#mujoco.mju_zero(data.qvel)
#mujoco.mju_zero(data.qacc)
mujoco.mju_zero(data.sensordata)



l_2 = mujoco.mju_dist3(data.joint("prop_base_joint").xanchor, data.site("leg_end").xpos)

#print (data.sensor("IMU_acc").data)

main_bodies_names = ["propeller_base", "leg_1"]

blue_color = np.array ([0, 0, 1, 1])
red_color = np.array ([1, 0, 0, 1])
green_color = np.array([0, 1, 0, 1])
yellow_color = np.array([1, 0, 1, 1])

exit = False
pause = True
step = False

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
    c1 = math.cos(x[0])
    s1 = math.sin(x[0])
    H = np.zeros (3)
    H[0] = x [1]
    
    H[1] = l_2 / 2 * (x[2] * c1 - (x[1] ** 2) * s1)
    #H[1] *=  1/3
    
    H[2] = l_2 / 2 * (x[2] * s1 + (x[1] **2) * c1)
    
    """ H[0] = x [1]
    
    H[1] = l_2 / 2 * (- x[2] * c1 + (x[1] ** 2) * s1)
    
    H[2] = l_2 / 2 * (x[2] * s1 + (x[1] **2) * c1) """
    
    return H.reshape((3, 1)) 

def H_jac (x):
    c1 = math.cos(x[0])
    s1 = math.sin(x[0])
    H = np.zeros((3, 3))
    
    H [0, 1] = 1
    
    H [1, 0] = - l_2 / 2 * (x [2] * s1 + (x [1] ** 2) * c1)
    H [1, 1] = - l_2 * x [1] * s1
    H [1, 2] = l_2 / 2 * c1
    #H [1, :] *=  1/3
    
    H [2, 0] = l_2 / 2 * (x [2] * c1 - (x [1] ** 2) * s1)
    H [2, 1] = l_2 * x [1] * c1
    H [2, 2] = l_2 / 2 * s1
    
    """ H [0, 1] = 1
    
    H [1, 0] = l_2 / 2 * (x [2] * s1 + (x [1] ** 2) * c1)
    H [1, 1] = l_2 * x [1] * s1
    H [1, 2] = - l_2 / 2 * c1
    
    H [2, 0] = l_2 / 2 * (x [2] * c1 - (x [1] ** 2) * s1)
    H [2, 1] = l_2 * x [1] * c1
    H [2, 2] = l_2 / 2 * s1 """
    

    return H
    
ekf_count = 0
count_max = 1

dt = model.opt.timestep * count_max

ekf_theta = ExtendedKalmanFilter (3, 3) #1 dim for angular velocity, 2 for acceleration

ekf_theta.x [0] = data.qpos[0]

ekf_theta.x [0] += random.random() * 0.05

#print (ekf_theta.x[0], data.qpos[0])

""" ekf_theta.R [0, 0] = model.sensor("IMU_gyro").noise
ekf_theta.R [1, 1] = model.sensor("IMU_acc").noise
ekf_theta.R [2, 2] = model.sensor("IMU_acc").noise """

ekf_theta.R [0, 0] = 0.18
ekf_theta.R [1, 1] = 0.22
ekf_theta.R [2, 2] = 0.22


#print ([model.sensor("IMU_gyro").noise, model.sensor("IMU_acc").noise, model.sensor("IMU_acc").noise])
#print (ekf_theta.R)
#ekf_theta.Q = Q_discrete_white_noise (3, dt = dt, var = 0.0001)
ekf_theta.Q *= 0.0002
#ekf_theta.Q *= 0
""" ekf_theta.R *= 0.0001
ekf_theta.Q *= 0.0001 """

#ekf_theta.P *= 0.3




c_2_est_1 = []
c_2_est_2 = []
c_2_sim = []

ang_sim = []
ang_est = []
ang_err = []
sim_time = []
imu_acc = []

est_vel = []
sim_vel = []

ang_est = []
ang_vels = []
meas_diff = []

meas_val = []
est_meas = []

sim_x = []
est_x = []




#print (ekf_ang.x[:2])
with mujoco.viewer.launch_passive(model, data, key_callback = kb_callback) as viewer:
    viewer.lock()


    #viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
    
    #viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    
    #mujoco.mju_zero(data.sensordata)
    viewer.sync()
    #print (data.sensor ("IMU_acc").data)


    while viewer.is_running() and not exit:
        step_start = time.time()
        if (not pause) or step:
            step = False
            #print (ekf_ang.x)
            viewer.lock()
            mujoco.mj_step(model, data)
            #mujoco.mju_zero(data.qvel)
            
            #print (data.sensor ("IMU_acc").data)
            
            if (ekf_count == 0) or True:
                
                
                
                
                #F = np.eye(2) + dt * np.array([[0, 1], [(3 / 2) / l_2 * model.opt.gravity[2] * math.cos(ekf_theta.x[0]), 0]])
                
                F = np.eye(3) + dt * np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
                
                #F [1, 2] = dt * (3 / 2) / l_2 * model.opt.gravity[2] * math.cos(ekf_theta.x[0])
                
                F [2, 0] = (3 / 2) / l_2 * model.opt.gravity[2] * math.cos(ekf_theta.x[0])
                
                F [2, 2] = 0
                
                #attempt with x_prior instead of x
                #F [2, :] = np.array([(3 / 2) / l_2 * model.opt.gravity[2] * math.cos(ekf_theta.x_prior[0]), 0, 0])
                
                #print (F)
                                
                #print (ekf_theta.x[0], data.qpos[0])
                #print ('F: ', F)
                #print ('F_sim: ', F_sim)
                #print ('F_err: ', F - F_sim)
                #print (ekf_ang.P)
                
                ekf_theta.F = F.copy()
                
                ang_data = data.sensor ("IMU_gyro").data[1].copy() 
                
                acc_data = data.sensor ("IMU_acc").data.copy()
                
                #print (random.gauss(0, model.sensor("IMU_gyro").noise[0]))
                
                ang_data += random.gauss(0, model.sensor("IMU_gyro").noise[0])
                for i in range (0, 3):
                    acc_data [i] += random.gauss(0, model.sensor("IMU_acc").noise[0])
                
                
                #print (acc_data)
                #acc_data.shape = (3, 1)
                #print (data.sensor ("IMU_acc").data)
                q_est = ekf_theta.x[0].copy()
                #q_est = data.qpos[0]
                
                R = np.array([[math.cos(q_est), 0, math.sin(q_est)], [0, 1, 0], [-math.sin(q_est), 0, math.cos(q_est)]])
                
                #print (R)
                
                #print (acc_data)
                
                #mujoco.mju_mulMatTVec (acc_data, R, acc_data.copy())
                
                #print (acc_data, model.opt.gravity)
                #mujoco.mju_mulMatVec (acc_data, R, acc_data.copy())
                
                #acc_data.shape = (3, 1)
                #print(acc_data)
                
                
                acc_data = acc_data + model.opt.gravity#.reshape((3, 1))
                #print (acc_data)
                
                #print ('ang:', ang_data)
                
                #print (np.array([ang_data[0], acc_data [0], acc_data [2]]))
                
                z = np.array([ang_data, acc_data [0], acc_data [2]]).reshape ((3, 1)) 
                
                #print(dir(data.body("leg_1")))
                """ acc_sim = data.body("leg_1").cacc[3:].copy()
                
                #mujoco.
                
                mujoco.mju_mulMatTVec(acc_sim, R, acc_sim.copy())
                acc_sim += model.opt.gravity
                #print (acc_data - acc_sim)
                z_sim = np.array ([data.qvel[0], acc_sim[0], acc_sim[2]]).reshape ((3, 1))  """
                
                #print (data.sensor("IMU_acc").data - data.body("leg_1").cacc[3:])
                #print (acc_data - model.opt.gravity - data.body("leg_1").cacc[3:])
                
                #print (z - z_sim)
                
                
                #print (z)
                #print ('pre-update:', ekf_theta.x)
                #print ('measures: ', z)
                
                #print ('diff: ', z - h_x(ekf_theta.x.copy()))
                #meas_diff.append(z - h_x(ekf_theta.x.copy()))
                meas_val.append(z.copy())
                
                est_meas.append(h_x(ekf_theta.x))
                
                
                
                ekf_theta.predict_update(z, HJacobian = H_jac, Hx = h_x)
                #print ('post update: ', ekf_theta.x)
                #ekf_theta.predict_update(z_sim, HJacobian = H_jac, Hx = h_x)
                #ekf_theta.predict_update(data.sensor("IMU_gyro").data[1], HJacobian = H_jac, Hx = h_x)
                #ekf_theta.predict_update(data.qvel[0], HJacobian = H_jac_ang, Hx = h_x_ang)
               
            #mujoco.mj_step(model, data)
            curr_pos = data.body("leg_1").xpos.copy()
            c_2_sim.append([curr_pos[0], curr_pos[2]])
            sim_time.append(data.time)
            sim_vel.append(data.qvel[0])
            
            est_x.append(ekf_theta.x.copy())
            sim_x.append(np.array([data.qpos[0], data.qvel[0], data.qacc[0]]))
            
            
            
            #print (data.qpos[0], ekf_theta.x[0])
            #print (ekf_theta.x)
            
            ekf_count = (ekf_count + 1) % count_max
            
            #print (ekf_ang.x[-1, -1] - data.qvel[0])
            
            
            q_est = ekf_theta.x[0].copy()
            c_est = l_2 / 2 * np.array([-math.sin(q_est), -math.cos(q_est)])
            
            ang_err.append (data.qpos[0].copy() - q_est)
            ang_vels.append (data.qvel[0].copy() - ekf_theta.x[-1].copy())
            
            
            ang_est.append (ekf_theta.x[0])
            ang_sim.append (data.qpos[0])
            
            #print (ekf_ang.x[:2])
            c_2_est_1.append([c_est])
            
            
            c_t_est_1 = np.zeros(3)
            c_t_est_2 = np.zeros(3)
            
            #c_t_est = data.body(main_bodies_names[1]).cinert[9] * np.array([ekf_ang.x[0], 0, ekf_ang.x[1]])
            #print (ekf_ang.x)
            #print (ekf_ang.x[-2] - data.qpos[0])
            
            
            #print (ekf_acc.x[-1])
            c_t_est_1[0] += c_est[0]
            c_t_est_1[2] += c_est[1]
            
            c_t_est_1 /= model.body_subtreemass[1]
            #c_t_est_2 /= model.body_subtreemass[1]
            
            #print (data.body(main_bodies_names[1]).cinert[9])
            #print (c_t_est)
            
            
            
            c_t_est_1 += data.body(main_bodies_names[0]).xpos
            c_t_est_2 += data.body(main_bodies_names[0]).xpos

            
            #print (c_t_est_1, c_t_est_2)
            
            mujoco.mjv_initGeom(viewer.user_scn.geoms[0],\
                type = mujoco.mjtGeom.mjGEOM_SPHERE, size = .03 * np.ones(3),\
                pos = data.subtree_com[1], mat = np.eye(3).flatten(), rgba = red_color)
            
            mujoco.mjv_initGeom(viewer.user_scn.geoms[1],\
                type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .035 * np.ones(3),\
                pos = c_t_est_1, mat = np.eye(3).flatten(), rgba = green_color)
            """ mujoco.mjv_initGeom(viewer.user_scn.geoms[1],\
                type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .035 * np.ones(3),\
                pos = c_t_est_2, mat = np.eye(3).flatten(), rgba = blue_color) """
            
            viewer.user_scn.ngeom = 2
            
            
            #print (data.sensor("IMU_gyro").data)
            
            
            #print (C_0_t, C_1_t)
                
            """ mujoco.mjv_initGeom(viewer.user_scn.geoms[1],\
                type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .035 * np.ones(3),\
                pos = C_0_t [0:3], mat = np.eye(3).flatten(), rgba = green_color) """
            
            
            
            
            #viewer.user_scn.ngeom = 1
            
            
            
            
            viewer.sync()
            """ sim_time.append(data.time)
            angle_1.append(data.qpos[0])
            angle_2.append(data.qpos[1] + data.qpos[0]) """
            
        time_to_step = model.opt.timestep - (time.time() - step_start)
        if (time_to_step > 0):
            time.sleep(time_to_step)
            
#_, ax = plt.subplots(2, 1, sharex = True)


""" ax[0].plot(sim_time, np.array([sim_vel, est_vel]))
ax[0].legend(['angvel mj']) """

""" ax.plot(sim_time, np.asarray(ang_sim))
ax.plot(sim_time, np.asarray(ang_est))
ax.legend (['sim angle', 'est angle']) """
#ax.plot(sim_time, np.asarray(ang_err))
""" ax[0].plot(sim_time, np.asarray(ang_err))
ax[1].plot(sim_time, np.asarray(ang_vels)) """
""" ax[0].plot(sim_time, np.asarray(ang_sim))
ax[0].plot(sim_time, np.asarray(ang_est))
ax[0].legend (['sim angle', 'est angle'])
ax[1].plot(sim_time, np.asarray(meas_diff).reshape (-1, 3))
ax[1].legend(['ang diff', 'acc x diff', 'acc z diff']) """



""" ax.plot(sim_time, np.asarray(ang_est))
ax.legend(['angvel mj']) """
#ax[1].plot(sim_time, np.reshape(err_1, (-1, 1)))

""" ax[1].plot(sim_time, np.asarray(est_vel))
ax[1].legend(['angvel ekf']) """

""" imu_acc = np.asarray(imu_acc)
ax[2].plot(sim_time, np.reshape(imu_acc, (-1, 2)))
ax[2].legend(['meas acc']) """

meas_val = np.asarray(meas_val).reshape (-1, 3)

est_meas = np.asarray(est_meas).reshape(-1, 3)

sim_x = np.asarray(sim_x).reshape(-1, 3)

est_x = np.asarray(est_x).reshape(-1, 3)

_, ax = plt.subplots(3, 1, sharex = True)


for i in range (3):
    #ax[i].plot(sim_time, meas_val[:, i])
    #ax[i].plot(sim_time, est_meas[:, i])
    ax[i].plot(sim_time, sim_x[:, i])
    ax[i].plot(sim_time, est_x[:, i])
    ax[i].legend (['sim', 'est'])
    
plt.waitforbuttonpress()


_, ax2 = plt.subplots(3, 1, sharex = True)


for i in range (3):
    ax2[i].plot(sim_time, meas_val[:, i])
    ax2[i].plot(sim_time, est_meas[:, i])
    #ax2[i].plot(sim_time, meas_val[:, i] / est_meas[:, i])
    #ax[i].plot(sim_time, sim_x[:, i])
    #ax[i].plot(sim_time, est_x[:, i])
    ax2[i].legend (['sim', 'est', 'ratio'])

plt.waitforbuttonpress()