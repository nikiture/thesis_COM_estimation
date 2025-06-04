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



model = mujoco.MjModel.from_xml_path("models/single_2D_pend.xml")
data = mujoco.MjData(model)

data.qpos[0] = 0.3

mujoco.mj_forward(model, data)

l_2 = mujoco.mju_dist3(data.joint("prop_base_joint").xanchor, data.site("leg_end").xpos)

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
    #return np.array([[0, 0, 0, 0, 0, 1]]) @ np.array(x)
    #return x[-2:].copy()
    """ H = np.zeros((2, 6))
    H[:, -2:] = np.eye(2)
    h_res = H @ x
    print (h_res)
    return h_res.reshape(2, -1) """
    return np.array([x[-2], x[-1]])

def h_x_ang (x):
    #return np.array([[0, 0, 0, 0, 0, 1]]) @ np.array(x)
    return np.array([x [-1]])

def h_x_theta (x):
    #return np.array([[0, 0, 0, 0, 0, 1]]) @ np.array(x)
    return np.array(x [-1])
    

def H_jac (x):
    """ H = np.zeros(len(x))
    H[-1] = 1 """
    #print (H)
    #return np.concatenate([np.zeros(len(x) - 1), np.array([1])])
    #return np.array([[0, 0, 0, 0, 0, 1]])
    """ H = np.zeros((2, 6))
    H[:, -2:] = np.eye(2) """
    H = np.array([[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
    #print (H)
    return H
    #return H.T
    
def H_jac_ang (x):
    H = np.zeros((1, len(x)))
    H[-1] = 1
    #print (H)
    #return np.concatenate([np.zeros(len(x) - 1), np.array([1])])
    return H

def H_jac_thta (x):
    H = np.zeros((1, len(x)))
    H[-1] = 1
    #print (H)
    #return np.concatenate([np.zeros(len(x) - 1), np.array([1])])
    return H
    
    

#print (l_2)
""" ekf_ang = ExtendedKalmanFilter(6,  1)

ekf_ang.x[0] = - l_2 / 2 * math.sin(data.qpos[0])
ekf_ang.x[1] = - l_2 / 2 * math.cos(data.qpos[0])
ekf_ang.x[-2] = data.qpos[0]
z_dim = len(ekf_ang.z)
#ekf_ang.R = np.zeros((z_dim, z_dim))
ekf_ang.R *= 0.001
ekf_ang.Q *= 0.001 """

""" ekf_ang = ExtendedKalmanFilter(dim_x = 6,  dim_z = 1, dim_u = 1)

#print (dir(ekf_ang))
#print (ekf_ang.B)

ekf_ang.x[0] = - l_2 / 2 * math.sin(data.qpos[0])
ekf_ang.x[1] = - l_2 / 2 * math.cos(data.qpos[0])
ekf_ang.x[-2] = data.qpos[0]
z_dim = len(ekf_ang.z)
#ekf_ang.R = np.zeros((z_dim, z_dim))
ekf_ang.R *= 0.01
ekf_ang.Q *= 0.001
ekf_ang.B = np.zeros(6) """


ekf_theta = ExtendedKalmanFilter (2, 1)

ekf_theta.x [0] = data.qpos[0]

ekf_theta.R *= 0.01
ekf_theta.Q *= 0.01
""" ekf_theta.R *= 0.0001
ekf_theta.Q *= 0.0001 """

#ekf_theta.P *= 10

""" ekf_acc = ExtendedKalmanFilter(dim_x = 6, dim_z = 2)
ekf_acc.x[0] = - l_2 / 2 * math.sin(data.qpos[0])
ekf_acc.x[1] = - l_2 / 2 * math.cos(data.qpos[0])

#print (ekf_acc.x)
z_dim = len(ekf_acc.z)
#ekf_acc.R = np.zeros((z_dim, z_dim))
#ekf_acc.R = np.eye(z_dim) * 0.1
#ekf_acc.Q = np.eye(6) * 0.1
#ekf_acc.P = np.eye(6)
#print (ekf_acc.P)

ekf_acc.R *= 0.001
ekf_acc.Q *= 0.001
ekf_acc.P *= 0.1 """


c_2_est_1 = []
c_2_est_2 = []
c_2_sim = []
ang_err = []
sim_time = []
imu_acc = []

est_vel = []
sim_vel = []

ang_est = []
ang_vels = []



ekf_count = 0
count_max = 1

dt = model.opt.timestep * count_max

#print (ekf_ang.x[:2])
with mujoco.viewer.launch_passive(model, data, key_callback = kb_callback) as viewer:
    viewer.lock()


    #viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    
    #viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    viewer.sync()
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    
    


    while viewer.is_running() and not exit:
        step_start = time.time()
        if (not pause) or step:
            step = False
            #print (ekf_ang.x)
            viewer.lock()
            #mujoco.mju_zero(data.qvel)
            mujoco.mj_step(model, data)
            curr_pos = data.body("leg_1").xpos.copy()
            c_2_sim.append([curr_pos[0], curr_pos[2]])
            sim_time.append(data.time)
            sim_vel.append(data.qvel[0])
            
            
            if ekf_count == 0:
                """ F_sim = np.eye(6)
                F_sim[:2, 2:4] = dt * np.eye(2)
                q1 = data.qpos[0]
                #temp = l_2 / 2 * data.qvel[0] * dt
                #print (ekf_ang.x[-1])
                #print (temp)
                #print (np.diag([math.sin(q1), -math.cos(q1)]))
                F_sim[2:4, 2:4] = np.zeros((2, 2))
                F_sim[2:4, 4:] = - l_2 / 2 * dt * np.diag([math.cos(q1), math.sin(q1)])
                F_sim[4, -1] = dt """
                
                #print (F_sim)
                
                """ F = np.eye(6)
                F[:2, 2:4] = dt * np.eye(2)
                q1 = ekf_ang.x[-2, 0]
                temp = l_2 / 2 * ekf_ang.x[-1, -1] * dt
                #print (ekf_ang.x[-1])
                #print (temp)
                #print (np.diag([math.sin(q1), -math.cos(q1)]))
                F[2:4, 2:4] = np.zeros((2, 2))
                F[2:4, 4:] = - l_2 / 2 * dt * np.diag([math.cos(q1), math.sin(q1)])
                F[4, -1] = dt
                
                #print ('F: ', F)
                #print ('F_sim: ', F_sim)
                #print ('F_err: ', F - F_sim)
                #print (ekf_ang.P)
                
                ekf_ang.F = F.copy()
                
                B = np.zeros((6, 1))
                B[-1] = - dt / l_2 * math.sin (ekf_ang.x[-2, -1])
                #B[-1] = - dt / l_2 * math.sin (data.qpos[0])
                #B[-1] = dt / l_2 * math.sin (data.qpos[0])
                #print(mujoco.mju_norm(data.sensor("IMU_gyro").data), data.qvel[0])
                #print(data.sensor("IMU_gyro").data[1], data.qvel[0])
                #print (data.sensor("IMU_gyro").data)
                
                ekf_ang.predict_update(- data.sensor("IMU_gyro").data[1], HJacobian = H_jac_ang, Hx = h_x_ang, u = model.opt.gravity[2]) """
                
                #F = np.eye(2) + dt * np.array([[0, 1], [0, 0]]) + dt * np.diag([0, - (3 / 2) / l_2 * model.opt.gravity[2] * math.cos(ekf_theta.x[0])])
                
                F = np.eye(2) + dt * np.array([[0, 1], [(3 / 2) / l_2 * model.opt.gravity[2] * math.cos(ekf_theta.x[0]), 0]])
                
                #F = np.eye(2) + dt * np.array([[0, 1], [0, 0]]) + dt * np.diag([0, - (3 / 2) / l_2 * model.opt.gravity[2] * math.cos(data.qpos[0])])
                
                print (ekf_theta.x[0], data.qpos[0])
                print ('F: ', F)
                #print ('F_sim: ', F_sim)
                #print ('F_err: ', F - F_sim)
                #print (ekf_ang.P)
                
                ekf_theta.F = F.copy()
                
                B = np.zeros((2, 1))
                B[-1] = - dt / l_2 * math.sin (ekf_theta.x[0])
                #B[-1] = - dt / l_2 * math.sin (data.qpos[0])
                #B[-1] = dt / l_2 * math.sin (data.qpos[0])
                #print(mujoco.mju_norm(data.sensor("IMU_gyro").data), data.qvel[0])
                #print(data.sensor("IMU_gyro").data[1], data.qvel[0])
                #print (data.sensor("IMU_gyro").data)
                #ekf_theta.B = B.copy()
                
                ekf_theta.predict_update(data.sensor("IMU_gyro").data[1], HJacobian = H_jac_ang, Hx = h_x_ang)
                #ekf_theta.predict_update(data.qvel[0], HJacobian = H_jac_ang, Hx = h_x_ang)
                
                #ekf_ang.predict_update(data.qvel[0], HJacobian = H_jac_ang, Hx = h_x_ang, u = model.opt.gravity[2])
            
            """ est_vel.append(ekf_ang.x[-1, -1])
            
            ang_vels.append ([data.qvel[0].copy(), ekf_ang.x[-1, -1]])
            ang_err.append (data.qvel[0]- ekf_ang.x[-1, -1])
            ang_est.append(ekf_ang.x[-2, -1]) """
            
            ekf_count = (ekf_count + 1) % count_max
            
            #print (ekf_ang.x[-1, -1] - data.qvel[0])
            """ F_sim = np.eye(6)
            F_sim[:2, 2:4] = dt * np.eye(2)
            F_sim[2:4, 4:6] = dt * np.eye(2)
            
            ekf_acc.F = F_sim.copy() """
            
            """ acc_data = data.sensor("IMU_acc").data.copy()
            acc_data += model.opt.gravity
            acc_data = np.array([acc_data[0].copy(), acc_data[2].copy()])
            
            acc_data *= -1
            
            #acc_data = np.zeros(2)
            
            #print (acc_data)
            
            #ekf_acc.predict_update(acc_data, H_jac, h_x)
            #ekf_acc.update(acc_data, H_jac, h_x)
            imu_acc.append(acc_data) """
            """ ekf_acc.update(acc_data, HJacobian = H_jac, Hx = h_x)
            ekf_acc.predict() """
            
            #ang_err.append(ekf_ang.x[-2] - data.qpos[0])
            
            q_est = ekf_theta.x[0].copy()
            c_est = l_2 / 2 * np.array([-math.sin(q_est), -math.cos(q_est)])
            
            ang_err.append (data.qpos[0].copy() - q_est)
            ang_vels.append (data.qvel[0].copy() - ekf_theta.x[-1].copy())
            
            #print (ekf_ang.x[:2])
            c_2_est_1.append([c_est])
            #c_2_est_1.append([ekf_ang.x[:2, 0].copy()])
            #c_2_est_2.append(ekf_ang.x[:2, 0].copy())
            #c_2_est.append([ekf_acc.x[-1, :2].copy()])
            #print (data.qvel)
            #print (data.qpos)
            #mujoco.mju_zero(data.qvel)
            #data.qvel = np.zeros(len(data.qvel)).copy()
            #print (ekf_acc.x)
            
            c_t_est_1 = np.zeros(3)
            c_t_est_2 = np.zeros(3)
            
            #c_t_est = data.body(main_bodies_names[1]).cinert[9] * np.array([ekf_ang.x[0], 0, ekf_ang.x[1]])
            #print (ekf_ang.x)
            #print (ekf_ang.x[-2] - data.qpos[0])
            """ c_t_est_1[0] += data.body(main_bodies_names[1]).cinert[9] * ekf_ang.x[0, 0]
            c_t_est_1[2] += data.body(main_bodies_names[1]).cinert[9] * ekf_ang.x[1, 0]
            
            c_t_est_2[0] += data.body(main_bodies_names[1]).cinert[9] * ekf_ang.x[0, -1]
            c_t_est_2[2] += data.body(main_bodies_names[1]).cinert[9] * ekf_ang.x[1, -1 """
            
            #print (ekf_acc.x[:, -1])
            """ c_t_est[0] = data.body(main_bodies_names[1]).cinert[9] * ekf_acc.x[0, -1]
            c_t_est[2] = data.body(main_bodies_names[1]).cinert[9] * ekf_acc.x[1, -1] """
            
            """ c_t_est[0] += data.body(main_bodies_names[1]).cinert[9] * ekf_acc.x[0, 0]
            c_t_est[2] += data.body(main_bodies_names[1]).cinert[9] * ekf_acc.x[1, 0] """
            
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
            
_, ax = plt.subplots(1, 2, sharex = True)


""" ax[0].plot(sim_time, np.array([sim_vel, est_vel]))
ax[0].legend(['angvel mj']) """

#ax.plot(sim_time, np.asarray(ang_vels).reshape (-1, 2))
#ax.plot(sim_time, np.asarray(ang_err))
ax[0].plot(sim_time, np.asarray(ang_err))
ax[1].plot(sim_time, np.asarray(ang_vels))
""" ax.plot(sim_time, np.asarray(ang_est))
ax.legend(['angvel mj']) """
#ax[1].plot(sim_time, np.reshape(err_1, (-1, 1)))

""" ax[1].plot(sim_time, np.asarray(est_vel))
ax[1].legend(['angvel ekf']) """

""" imu_acc = np.asarray(imu_acc)
ax[2].plot(sim_time, np.reshape(imu_acc, (-1, 2)))
ax[2].legend(['meas acc']) """

plt.waitforbuttonpress()