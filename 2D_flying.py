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


model = mujoco.MjModel.from_xml_path("models/2D_flying.xml")
data = mujoco.MjData(model)

#joint_names = ["ground_joint", "middle_joint", "prop_base_joint"]
joint_names = ["free_joint", "prop_base_joint", "knee"]
main_body_names = ["propeller_base", "leg", "leg_2"]



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

start_angle_2 = math.pi/8

start_angle_3 = -2 * start_angle_2

mujoco.mju_axisAngle2Quat(data.qpos[3:7], start_axis_1, start_angle_1)

data.qpos[7] = start_angle_2
data.qpos[8] = start_angle_3

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
l = np.append(l, mujoco.mju_dist3(data.joint(joint_names[2]).xanchor, data.site("foot").xpos))

#print (mujoco.mju_dist3(data.body("propeller_1").xpos, data.body("propeller_2").xpos))
#print (l)
#l = np.array ([l_1, l_2])

#l = [l_1]

T_0_i = np.eye(4)

#R_0_1 = np.eye(3)
R_0_1 = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])

T_0_1 = np.eye(4)
T_0_1 [0:3, 0:3] = R_0_1
T_0_1 [0:3, 3] = data.qpos[:3].copy()

""" C_0_1 = np.zeros(3)
mujoco.mju_mulMatTVec (C_0_1, R_0_1, data.qpos[:3].copy())
T_0_1 [0:3, 3] = C_0_1.copy() """


R_1_2 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
#R_1_2 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

#R_0_2 = R_0_1 @ R_1_2

T_1_2 = np.eye(4)
T_1_2 [:3, :3] = R_1_2
T_1_2 [0][3] = 0

T_2_3 = np.eye(4)
T_2_3 [0, 3] = l [1]

T_i_i = np.array([T_0_1, T_1_2, T_2_3])

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

C_0_i = np.zeros((len(l), 4))
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
    #tot_mass = mujoco.mj_getTotalMass(model)
    
    
    #print (mass_1)
    
    """ global bal_force_1, bal_force_2
    stat_bal_force = - model.opt.gravity[2] * np.array([0, 0, mass_1/2 + mass_2])
    #bal_force = - model.opt.gravity[2] * np.array([mass_1/2 * math.tan(data.qpos[0]), 0, mass_1 + mass_2])
    bal_force_1 = stat_bal_force/2
    bal_force_2 = stat_bal_force/2
    #mujoco.mju_quatZ2Vec(force_quat, )
    
    
    des_angle_2 = -data.qpos[0]
    
    rot_force = k_p * (des_angle_2 - data.qpos[1]) - k_v * data.qvel[1]
    
    
    bal_force_2 += rot_force/2
    bal_force_1 -= rot_force/2
    
    data.body("propeller_1").xfrc_applied[0:3] = bal_force_1
    data.body("propeller_2").xfrc_applied[0:3] = bal_force_2 """
    
    """ if rot_force > 0:
        bal_force_2[2] += rot_force
    else:
        bal_force_1[2] -= rot_force """
    
    
    """ global bal_int_1, bal_int_2
    stal_bal_int = -model.opt.gravity[2] * (mass_1/2 + mass_2)
    
    baL_int_1 = stal_bal_int/2
    bal_int_2 = stal_bal_int/2
    
    des_angle_2 = -data.qpos[0]
    
    rot_force = k_p * (des_angle_2 - data.qpos[1]) - k_v * data.qvel[1]
    
    
    bal_int_2 += rot_force/2
    bal_int_1 -= rot_force/2
    
    data.actuator("propeller_1_thrust").ctrl = bal_int_1
    data.actuator("propeller_2_thrust").ctrl = bal_int_2 """
    
    
    global bal_int_1, bal_int_2, l_1, l_2, vert_force, force_dir, err_1_prev, i_err_1, in_err, error_1
    g = - model.opt.gravity[2]
    #stat_bal_force = - model.opt.gravity[2] * np.array([0, 0, mass_1/2 + mass_2])
    
    #bal_force = - model.opt.gravity[2] * np.array([mass_1/2 * math.tan(data.qpos[0]), 0, mass_1 + mass_2])
    
    if math.sin(2 * data.qpos[0] + data.qpos[1]) != 0:
        bal_int_1 = (math.sin(data.qpos[0])/math.sin(2 * data.qpos[0] + data.qpos[1])) * (mass_1/2 + mass_2) * g / 2
        bal_int_2 = bal_int_1 
    #mujoco.mju_quatZ2Vec(force_quat, )
    
    if not gravity_comp:
        bal_int_1 = 0
        bal_int_2 = 0
    
    
    
    err_1 =  - des_angle_1 + data.qpos[0]
    if err_1_prev is None:
        err_1_prev = err_1
    d_err_1 = (err_1 - err_1_prev) / model.opt.timestep
    i_err_1 += model.opt.timestep * err_1
    err_1_prev = err_1
    i_error_1.append(i_err_1)
    
    error_1.append(err_1)
    
    a_1_des = k_p * err_1 + d_err_1 * k_v + k_i * i_err_1
    
    body_mod = model.body("pendulum")
    body_data = data.body("pendulum")
    joint = data.joint("prop_base_joint")
    
    mujoco.mju_sqrMatTD (inertia, np.reshape(body_data.ximat, (3, 3)), body_mod.inertia)
    
    mujoco.mju_sqrMatTD(inertia_offset, skew_mat(body_data.xpos - joint.xanchor), None)
    
    inertia_flat = inertia.flatten()
    
    mujoco.mju_addTo(inertia_flat, inertia_offset.flatten())
    
    #a_1_des = k_w_1 * (data.qvel[1] - w_1_des)
    
    M_1_des = a_1_des * inertia_flat [4] #multiply by I 22
    #M_1_des = k_w_1 * a_1_des
    #print (data.cinert[1][3])
    #M_1_des = 0
    
    
    #M_1_des = k_p * (des_angle_1 - data.qpos[0]) - k_v * data.qvel[0]
    
    #print (M_1_des)
    
    """ bal_int_1 += M_1_des / (2 * l_1 * math.sin(2 * data.qpos[0] + data.qpos[1]))
    bal_int_2 += M_1_des / (2 * l_1 * math.sin(2 * data.qpos[0] + data.qpos[1])) """
    
    
    global err_2_prev, i_err_2, i_error_2
    des_angle_2 = -data.qpos[0]
    
    err_2 = des_angle_2 - data.qpos[1]
    if err_2_prev is None:
        err_2_prev = err_2
    
    d_err_2 = (err_2 - err_2_prev) / model.opt.timestep
    i_err_2 += model.opt.timestep * err_2
    i_error_2.append(i_err_2)
    err_2_prev = err_2
    
    
    
    M_l2 = k_p_2 * err_2 + d_err_2 * k_v_2 + k_i_2 * i_err_2
    
    body_mod = model.body("propeller_base")
    body_data = data.body("propeller_base")
    joint = data.joint("prop_base_joint")
    
    mujoco.mju_sqrMatTD (inertia, np.reshape(body_data.ximat, (3, 3)), body_mod.inertia)
    
    mujoco.mju_sqrMatTD(inertia_offset, skew_mat(body_data.xpos - joint.xanchor), None)
    
    inertia_flat = inertia.flatten()
    
    mujoco.mju_addTo(inertia_flat, inertia_offset.flatten())
    
    M_l2 = inertia_flat[4] * M_l2
    
    
    
    
    
    """ M_l2 = k_p_2 * (des_angle_2 - data.qpos[1]) - k_v_2 * data.qvel[1]
    
    #print(des_angle_2 - data.qpos[1])
    
    
    #M_l2 = - M_l2
    M_l2 = 0 """
    
    
    
    """ bal_int_1 -= M_l2/l_2
    bal_int_2 += M_l2/l_2 """
    
    bal_int_1 += (M_1_des + M_l2) / (2 * l_1 * math.sin(2 * data.qpos[0] + data.qpos[1]))
    bal_int_2 += (M_1_des + M_l2) / (2 * l_1 * math.sin(2 * data.qpos[0] + data.qpos[1]))
    
    bal_int_1 -= M_l2 / l_2
    bal_int_2 += M_l2 / l_2
    
    
    data.actuator("propeller_1_thrust").ctrl = bal_int_1
    data.actuator("propeller_2_thrust").ctrl = bal_int_2
    
    global appl_force, ext_force
    
    if appl_force:
        for i in range (0, model.nbody):
            mujoco.mju_addTo(data.xfrc_applied[i], ext_force [i])
    """ else:
        for i in range (0, model.nbody):
            mujoco.mju_zeros(data.xfrc_applied[i], ext_force [i]) """
    
    
    #print (data.qpos[0] - math.atan(bal_force[0]/bal_force[2]))
    
    
    
    #print (dir(data.geom("propeller_1")))
    
    #data.body("propeller_base").xfrc_applied[0:3] = bal_force
    
    
    
    
    
    """ curr_quat = data.body("propeller_1").xquat
    data.actuator("propeller_1_thrust").ctrl = np.dot(bal_force_1, curr_quat[1:4]/math.sin(math.acos(curr_quat[0]))) - rot_force
    curr_quat = data.body("propeller_2").xquat
    data.actuator("propeller_2_thrust").ctrl = np.dot(bal_force_2, curr_quat[1:4]/math.sin(math.acos(curr_quat[0]))) + rot_force """
    
    

#mujoco.set_mjcb_control(control_cb)


#viewer = mujoco.viewer.launch_passive(model, data, key_callback = kb_callback)
with mujoco.viewer.launch_passive(model, data, key_callback = kb_callback) as viewer:
    viewer.lock()


    #viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
    viewer.sync()
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
    mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)



    """ for i in range (0, model.nbody):
        mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn)
        mujoco.mjv_addGeoms(model, data, viewer.opt, viewer.perturb, mujoco.mjtCatBit.mjCAT_DECOR, viewer.user_scn) """

    while viewer.is_running() and not exit:
        step_start = time.time()
        if not pause:
            viewer.lock()
            mujoco.mj_step(model, data)
            
            """ for k in range (6, len(data.qvel)):
                data.qvel[k] = random.random()
            #print (data.qpos)
            data.qvel[:6] = np.zeros(6).copy() """
            
            
            
            mujoco.mjv_initGeom(viewer.user_scn.geoms[0],\
                type = mujoco.mjtGeom.mjGEOM_SPHERE, size = .03 * np.ones(3),\
                pos = data.subtree_com[1], mat = np.eye(3).flatten(), rgba = red_color)
            
            
            
            T_1_i = np.eye(4)
            C_1_t = np.zeros(4)
            R_i_i = np.zeros((model.njnt, 3, 3))
            T_i_i [0, 0:3, 3] = data.qpos[:3].copy()
            R_0_i = R_0_1.copy()
            C_1_i = np.zeros((len(l), 4))
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
                    R_i_i[i] = np.array([[math.cos(alpha_i), math.sin(alpha_i), 0], [-math.sin(alpha_i), math.cos(alpha_i), 0], [0, 0, 1]])
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
                """ R_0_curr = data.xmat[i].copy()
                R_0_prev = np.eye(3).flatten()
                
                R_i_prev = np.eye(3)
                R_i_prev.reshape ((1, 9))
                R_i_prev = np.eye(3)
                
                if i > 0:
                    mujoco.mju_copy(R_0_prev, data.xmat[i - 1])
                print (R_0_prev, R_i_prev)
                mujoco.mju_mulMatTMat(R_i_prev, R_0_prev.reshape(3, 3), R_0_curr.reshape(3, 3))
                R_i_i [i] = R_i_prev.reshape(3, 3) """
                """ #R_i_i [i] = R_0_1 @ data.xmat[i]
                R_0_i = np.eye(3)
                #mujoco.mju_mulMatMat(R_0_i, R_0_1.reshape(3, 3), data.xmat[i].reshape(3, 3))
                mujoco.mju_mulMatMat(R_0_i, np.eye(3), data.xmat[i].reshape(3, 3))
                R_i_i[i] = R_0_i.reshape(3, 3) """
                        
                    
            #print ()
                    
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
                type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .04 * np.ones(3),\
                pos = C_0_t [0:3], mat = np.eye(3).flatten(), rgba = green_color)
            """ mujoco.mjv_initGeom(viewer.user_scn.geoms[1],\
                type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .04 * np.ones(3),\
                pos = C_1_t [0:3], mat = np.eye(3).flatten(), rgba = green_color) """
            
            for i in range (0, 3):
                C_0_i = np.zeros(4)
                mujoco.mju_mulMatVec(C_0_i, T_0_1_curr, C_1_i[i])
                mujoco.mjv_initGeom(viewer.user_scn.geoms[i + 2],\
                    type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .02 * np.ones(3),\
                    pos = C_0_i [0:3], mat = np.eye(3).flatten(), rgba = blue_color)
                """ mujoco.mjv_initGeom(viewer.user_scn.geoms[i + 2],\
                    type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .04 * np.ones(3),\
                    pos = C_1_i [i, 0:3], mat = np.eye(3).flatten(), rgba = blue_color) """
            
            viewer.user_scn.ngeom = 5
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