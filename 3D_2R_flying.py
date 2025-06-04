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


model = mujoco.MjModel.from_xml_path("models/3D_2R_flying.xml")
data = mujoco.MjData(model)

#joint_names = ["ground_joint", "middle_joint", "prop_base_joint"]
joint_names = ["free_joint", "prop_base_joint_1", "prop_base_joint_2", "knee"]
main_body_names = ["propeller_base","leg_0", "leg_1", "leg_2"]
prop_body_names = ["propeller_1", "propeller_2"]



#print (dir(model.joint(joint_names[0])))

#print (model.opt.gravity)



""" mass_1 = data.cinert[1][9]
mass_2 = data.cinert[2][9] """

""" print (mass_2)
print (model.body_mass[2]) """

#start_angle = math.pi/3
#start_angle = 0


start_angle_1 = 0
start_axis_1 = np.zeros(3)
start_axis_1[1] = 1
start_axis_1 [0] = 1
mujoco.mju_normalize(start_axis_1) 

start_angle_2 = 0.3

start_angle_3 = 0.6

start_angle_4 = 0.4

mujoco.mju_axisAngle2Quat(data.qpos[3:7], start_axis_1, start_angle_1)

data.qpos[7] = start_angle_2
data.qpos[8] = start_angle_3
data.qpos[9] = start_angle_4

#print (data.qpos)

#print(len(data.qpos))
""" mujoco.mju_axisAngle2Quat(data.qpos[7:11], start_axis_2, start_angle_2)

data.qpos[11] = start_angle_3 """

#data.qpos[1] = -start_angle
#data.qpos[1] = -start_angle/2

#data.qpos[1] = math.atan(math.tan(start_angle) * mass_1 / ( 2 * (mass_1 + mass_2))) - start_angle

#des_angle_1 = 0


mujoco.mj_forward(model, data)


pause = True
exit = False

loc_force = np.zeros(3)
arrow_quat = np.zeros(4)
arrow_mat = np.zeros(9)
arrow_dim = np.zeros(3)
arrow_shape = np.array ([0.012, 0.012, 1])

blue_color = np.array ([0, 0, 1, 1])
red_color = np.array ([1, 0, 0, 1])
green_color = np.array([0, 1, 0, 1])
yellow_color = np.array([1, 0, 1, 1])


bal_force_1 = np.zeros(3)
bal_force_2 = np.zeros(3)
bal_int_1 = 0
bal_int_2 = 0
force_quat = np.zeros(4)
neg_force_quat = np.zeros(4)
vert_force = np.array([0, 0, 1])
force_dir = np.zeros(3)


k_f = 0.03

""" w_n = 15
dampening = 1
k_p = mass_2 * w_n * w_n
k_v = 2 * dampening * w_n * mass_2 """


appl_force = False
ext_force = np.zeros((model.nbody, 6))
ext_force [1][0] = 10
#ext_force [2][4] = 0.1

k_p = 1
k_v = 0.8
k_i = 0
k_w_1 = 1

err_1_prev = None
in_err = 0
i_err_1 = 0

err_2_prev = None
i_err_2 = 0

k_p_2 = 8
k_v_2 = 6
k_i_2 = 0.5
#k_red_2 = 0.5


l_1 = 0.5
l_2 = 0.2


sim_time = []
angle_1 = []
error_1 = []
i_error_1 = []
angle_2 = []
i_error_2 =[]

gravity_comp = True

inertia_flat = np.zeros(9)
inertia = np.zeros((3, 3))

inertia_offset = np.zeros((3, 3))


l = np.zeros(0)
#l_1 = 1
l = np.append(l, mujoco.mju_dist3(data.body("propeller_1").xpos, data.body("propeller_2").xpos))
l = np.append(l, mujoco.mju_dist3(data.joint(joint_names[1]).xanchor, data.joint(joint_names[2]).xanchor))
l = np.append(l, mujoco.mju_dist3(data.joint(joint_names[2]).xanchor, data.joint(joint_names[3]).xanchor))
l = np.append(l, mujoco.mju_dist3(data.joint(joint_names[3]).xanchor, data.site("foot").xpos))

#print (mujoco.mju_dist3(data.body("propeller_1").xpos, data.body("propeller_2").xpos))
#print (l)
#l = np.array ([l_1, l_2])

#l = [l_1]

T_0_i = np.eye(4)

#R_0_1 = np.eye(3)
R_0_1 = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
R_1_2 = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
R_2_3 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
R_3_4 = np.eye(3)

T_0_1 = np.eye(4)
T_0_1 [0:3, 0:3] = R_0_1.copy()
T_0_1 [0:3, 3] = data.qpos[:3].copy()

""" C_0_1 = np.zeros(3)
mujoco.mju_mulMatTVec (C_0_1, R_0_1, data.qpos[:3].copy())
T_0_1 [0:3, 3] = C_0_1.copy() """


T_1_2 = np.eye(4)
T_1_2 [:3, :3] = R_1_2.copy()
T_1_2 [0][3] = 0

T_2_3 = np.eye(4)
T_2_3 [:3, :3] = R_2_3.copy()
T_2_3 [0, 3] = 0

T_3_4 = np.eye(4)
T_3_4 [:3, :3] = R_3_4.copy()
T_3_4 [0, 3] = l[2]

T_i_i = np.array([T_0_1, T_1_2, T_2_3, T_3_4])

#print (T_i_i)

""" T_i_i = np.zeros((4, 4, 0))

T_i_i = np.append(T_i_i, np.eye(4).copy()) #world to first leg

T_i_i = np.append(T_i_i, np.eye(4).copy()) #first to second leg

T_i_i = np.append(T_i_i, np.eye(4).copy()) #second leg to propeller base

np.reshape(T_i_i, (4, 4, -1)) """

#print (T_i_i)


""" 
R_0_1 = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
T_i_i [0, 0:3, 0:3] = R_0_1.copy()

for i in range (0, len(T_i_i) - 1):
    T_i_i [i + 1, :3, 4] = np.array([l[i], 0, 0]) """

#T_i_i = np.array([T_0_1, T_1_2])

#print (T_i_i)

C_i_i = np.zeros((len(l), 4))

for k in range(0, len(l)):
    #C_i_i = np.append (C_i_i, np.transpose(np.array([l_i/2, 0, 0])))
    C_i_i [k][0] = l [k]/2
    C_i_i [k][3] = 1

C_i_i [0, 0] = 0
C_i_i [1, 0] = 0


#print (C_i_i)

C_0_i = np.ones((len(main_body_names), 4))
#print (C_i_i)
random.seed(3)





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
    
def draw_vector_quat(viewer, idx, arrow_pos, arrow_color, arrow_quat, arrow_norm):
    
    """ mujoco.mju_copy(loc_force, arrow_dir)
    mujoco.mju_normalize (loc_force)
    mujoco.mju_quatZ2Vec(arrow_quat, loc_force) """
    mujoco.mju_quat2Mat(arrow_mat, arrow_quat)
    mujoco.mju_copy(arrow_dim, arrow_shape)
    arrow_dim [2] = arrow_shape [2] * arrow_norm
    mujoco.mjv_initGeom(viewer.user_scn.geoms[idx],\
            type = mujoco.mjtGeom.mjGEOM_ARROW, size = arrow_dim,\
            pos = arrow_pos, mat = arrow_mat.flatten(), rgba = arrow_color)

def kb_callback(keycode):
    print(chr(keycode))
    if chr(keycode) == ' ':
        global exit
        exit = not exit
    if chr(keycode) == 'P':
        global pause
        pause = not pause
    """ if chr(keycode) == 'F':
        global appl_force
        appl_force = not appl_force """
        
def skew_mat(vec):
    """ res = np.array[[0, -vec[2], vec[1]], ] """
    res = np.cross(np.eye(3), vec)
    return res
        
        
def control_cb (model, data):
    #g = - model.opt.gravity[2]
    comp_force = - model.opt.gravity * model.body(main_body_names[0]).subtreemass
    data.body (prop_body_names [0]).xfrc_applied[:3] = comp_force.copy() / 2
    data.body (prop_body_names [1]).xfrc_applied[:3] = comp_force.copy() / 2
    
    
    

#mujoco.set_mjcb_control(control_cb)


#viewer = mujoco.viewer.launch_passive(model, data, key_callback = kb_callback)
with mujoco.viewer.launch_passive(model, data, key_callback = kb_callback) as viewer:
    viewer.lock()


    #viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    
    #viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    viewer.sync()
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    for _ in range (0, len(C_i_i)):
        mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    


    while viewer.is_running() and not exit:
        step_start = time.time()
        if not pause:
            viewer.lock()
            #mujoco.mju_zero(data.qvel)
            mujoco.mj_step(model, data)
            #print (data.qvel)
            #print (data.qpos)
            #mujoco.mju_zero(data.qvel)
            #data.qvel = np.zeros(len(data.qvel)).copy()

            
            
            
            mujoco.mjv_initGeom(viewer.user_scn.geoms[0],\
                type = mujoco.mjtGeom.mjGEOM_SPHERE, size = .03 * np.ones(3),\
                pos = data.subtree_com[1], mat = np.eye(3).flatten(), rgba = red_color)
            
            
            
            T_1_i = np.eye(4)
            C_1_t = np.zeros(4)
            R_i_i = np.zeros((model.njnt, 3, 3))
            T_i_i [0, 0:3, 3] = data.qpos[:3].copy()
            R_0_i = R_0_1.copy()
            C_1_i = np.zeros((len(l), 4))
            C_1_i [0, 3] = 1
            #R_0_i = R_0_i.flatten()
            #C_0_t [3] = 1
            #print (T_0_i)
            for i, jname in enumerate(joint_names):
                #print (i)
                jmod = model.joint (jname)
                jdata = data.joint (jname)
                #print (jmod.type)
                if jmod.type == mujoco.mjtJoint.mjJNT_HINGE:
                    alpha_i = jdata.qpos
                    R_i_i[i] = np.array([[math.cos(alpha_i), -math.sin(alpha_i), 0], [math.sin(alpha_i), math.cos(alpha_i), 0], [0, 0, 1]])
                if jmod.type == mujoco.mjtJoint.mjJNT_BALL:
                    #print (i)
                    q_curr = jdata.qpos.copy()
                    #print (q_curr)
                    #q_prev = data.
                    """ q_rot = data.body_quat[i + 1]
                    csi_cross = skew_mat(q_rot[1:])
                    R_i_i [i] = np.eye(3) + 2 * q_rot[0] * csi_cross + 2 * csi_cross @ csi_cross
                    #R_0_i = np.eye(3)
                    for k in range (0, i):
                        #R_0_i =  R_0_i @ R_i_i [k]
                        mujoco.mju_mulMatMat(R_0_i, R_0_i, R_i_i[k]) """
                    #q_curr[1:] = R_0_i * q_curr [1:]
                    #mujoco's link frames does not follow harvenberg convention, using R_0_1 to convert the quaternion axes to harvenberg convention axes
                    #(alternatively represents all user-defined lengths and transformations not anymore in harvenberg convention but on the one used by mujoco)
                    mujoco.mju_mulMatTVec(q_curr[1:], R_0_i, q_curr[1:].copy()) 
                    csi_cross = skew_mat(q_curr[1:])
                    csi_cross_sq = np.eye(3)
                    mujoco.mju_mulMatMat(csi_cross_sq, csi_cross, csi_cross)
                    R_i_i [i] = np.eye(3) + 2 * q_curr[0] * csi_cross + 2 * csi_cross_sq
                if jmod.type == mujoco.mjtJoint.mjJNT_FREE:
                    q_curr = jdata.qpos[3:].copy()
                    
                    #print (q_curr)
                    
                    mujoco.mju_mulMatVec(q_curr[1:], R_0_i, q_curr[1:].copy()) 
                    csi_cross = skew_mat(q_curr[1:])
                    csi_cross_sq = np.eye(3)
                    mujoco.mju_mulMatMat(csi_cross_sq, csi_cross, csi_cross)
                    R_i_i [i] = np.eye(3) + 2 * q_curr[0] * csi_cross + 2 * csi_cross_sq
                    
                #mujoco.mju_mulMatMat(R_0_i, R_0_i.copy(), R_i_i[i].copy())
                #R_0_i = R_0_i @ R_i_i [i]
                

                    
            for j, T_i in enumerate(T_i_i[1:]):
                #print (i)
                #T_0_i = T_0_i @ T_i_i [i, :, :]
                #alpha_i = data.qpos [i]
                i = j + 1
                #print (i)
                T_curr = T_i.copy()
                #print (T_curr, i)
                #R_curr = np.array([[math.cos(alpha_i), math.sin(alpha_i), 0], [-math.sin(alpha_i), math.cos(alpha_i), 0], [0, 0, 1]])
                #R_curr = np.array([[math.cos(alpha_i), -math.sin(alpha_i), 0], [math.sin(alpha_i), math.cos(alpha_i), 0], [0, 0, 1]])
                
                #mujoco.mju_mulMatMat (T_curr[0:3, 0:3], T_i_i [i, 0:3, 0:3], R_curr)
                #T_curr [:3, :3] = T_curr[:3, :3] @ R_i_i [i]
                #R_curr = np.eye(3)
                #R_curr = R_curr.flatten()
                R_curr = T_curr[0:3, 0:3].copy()
                mujoco.mju_mulMatMat(R_curr, T_i [0:3, 0:3], R_i_i [i])
                T_curr[0:3, 0:3] = R_curr.copy()
                #T_curr [0:3, 0:3] *= np.array([[math.cos(alpha_i), math.sin(alpha_i), 0], [-math.sin(alpha_i), math.cos(alpha_i), 0], [0, 0, 1]]) 
                #T_i_i [i, 0:3, 0:3] = np.array([[math.cos(alpha_i), math.sin(alpha_i), 0], [-math.sin(alpha_i), math.cos(alpha_i), 0], [0, 0, 1]])
                #T_i_i [i + 1, 0:3, 0:3] = np.array([[math.cos(alpha_i), -math.sin(alpha_i), 0], [math.sin(alpha_i), math.cos(alpha_i), 0], [0, 0, 1]])
                T_temp = T_1_i.copy()
                mujoco.mju_mulMatMat(T_1_i, T_temp, T_curr)
                #print (T_1_i, i)
                #print (i, T_i_i[i])
                #print (T_0_i)
                #print (C_i_i[i])
                
                #print (C_i_k)
                #C_0_i = data.cinert[i + 1][9] * T_0_i * C_i_k
                #C_0_i = np.dot(T_0_i, C_i_i [i])
                mujoco.mju_mulMatVec(C_1_i [i], T_1_i, C_i_i [i])
                

                
                #print (data.cinert [i + 1, 9])
                #C_0_t += data.cinert [i + 1, 9] * C_0_i [0:3]
                #C_1_t += data.body(main_body_names[i]).cinert[9] * C_1_i [i]
                mujoco.mju_addToScl(C_1_t, C_1_i[i], data.body(main_body_names[i]).cinert[9])
                #print (data.body(main_body_names[i]).cinert[9], i)
                
                
                #print (C_0_i[:3], i)
                #print (T_0_i, i)
                #print (data.cinert[i + 1, 9], i)
                #print (len(data.cinert))
                #print (dir(data.body(main_body_names[0])))
                #print (data.body(main_body_names[1]).cinert[9])
                #print (C_0_i)
                
            #print (model.body_subtreemass[1])
            #C_1_t += data.body(main_body_names[0]).cinert[9] * C_i_i [0, 0:3]
            mujoco.mju_addToScl(C_1_t, C_1_i[0], data.body(main_body_names[0]).cinert[9])
            C_1_t /= model.body_subtreemass[1]
            #print (data.cinert[:, 9])
                
            
            """ mujoco.mjv_initGeom(viewer.user_scn.geoms[1],\
                type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .04 * np.ones(3),\
                pos = C_0_t, mat = np.eye(3).flatten(), rgba = blue_color) """
            
            """ end_eff_1_1 = np.array([l[0], 0, 0])
            #end_eff_1_1 = np.array([0, 0, l[0]])
            end_eff_1_0 = np.zeros(3)
            mujoco.mju_rotVecQuat(end_eff_1_0, end_eff_1_1, data.qpos[0:4])
            mujoco.mju_mulMatVec(end_eff_1_0, R_0_1, end_eff_1_0.copy())
            q_curr = data.qpos[0:4]
            q_curr_0 = q_curr.copy()
            mujoco.mju_mulMatVec(q_curr_0 [1:], R_0_1.T, q_curr_0[1:].copy())
            csi_cross = skew_mat(q_curr_0[1:])
            csi_cross_sq = np.eye(3)
            mujoco.mju_mulMatMat(csi_cross_sq, csi_cross, csi_cross)
            R = np.eye(3) + 2 * q_curr_0[0] * csi_cross + 2 * csi_cross_sq
            
            mujoco.mju_mulMatMat (R, R_0_1, R.copy())
            #mujoco.mju_mulMatMat (R, R.copy(), R_0_1)
            
            mujoco.mju_mulMatVec(end_eff_1_0, R, end_eff_1_1)
            #mujoco.mju_mulMatVec(end_eff_1_0, R_0_1, end_eff_1_0.copy())
            print (end_eff_1_0)
            
            mujoco.mjv_initGeom(viewer.user_scn.geoms[1],\
                type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .04 * np.ones(3),\
                pos = end_eff_1_0, mat = np.eye(3).flatten(), rgba = blue_color) """
                
            T_0_1_curr = T_0_1.copy()
            T_0_1_curr [:3, 3] = data.qpos[:3].copy()
            #R_0_1_curr = T_0_1_curr[:3, :3].flatten
            R_0_1_curr = np.eye(3)
            mujoco.mju_mulMatMat(R_0_1_curr, T_0_1_curr[:3, :3].copy(), R_i_i[0].copy()) 
            T_0_1_curr [:3, :3] = R_0_1_curr.copy()
            #print (T_0_1_curr)
            
            C_1_t [3] = 1
            C_0_t = np.zeros(4)
            mujoco.mju_mulMatVec(C_0_t, T_0_1_curr, C_1_t)
            
            #print (C_0_t, C_1_t)
                
            mujoco.mjv_initGeom(viewer.user_scn.geoms[1],\
                type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .035 * np.ones(3),\
                pos = C_0_t [0:3], mat = np.eye(3).flatten(), rgba = green_color)
            """ mujoco.mjv_initGeom(viewer.user_scn.geoms[1],\
                type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .04 * np.ones(3),\
                pos = C_1_t [0:3], mat = np.eye(3).flatten(), rgba = green_color) """
            
            for i in range (0, len(C_1_i)):
                C_0_i = np.zeros(4)
                mujoco.mju_mulMatVec(C_0_i, T_0_1_curr, C_1_i[i])
                try:
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[i + 2],\
                        type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .03 * np.ones(3),\
                        pos = C_0_i [0:3], mat = np.eye(3).flatten(), rgba = blue_color)
                except Exception as e:
                    print(e)
                """ mujoco.mjv_initGeom(viewer.user_scn.geoms[i + 2],\
                    type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .04 * np.ones(3),\
                    pos = C_1_i [i, 0:3], mat = np.eye(3).flatten(), rgba = blue_color) """
                #print (C_0_i, i)
                #print (C_1_i [i], i)
                """ if i == 2:
                    print (C_1_i[2], C_0_i, T_0_1_curr) """
            
            viewer.user_scn.ngeom = 3 + len(C_1_i)
            
            #viewer.user_scn.ngeom = 1
            
            
            
            
            viewer.sync()
            """ sim_time.append(data.time)
            angle_1.append(data.qpos[0])
            angle_2.append(data.qpos[1] + data.qpos[0]) """
            
        time_to_step = model.opt.timestep - (time.time() - step_start)
        if (time_to_step > 0):
            time.sleep(time_to_step)
        
        
""" if viewer.is_running():
    viewer.close() """