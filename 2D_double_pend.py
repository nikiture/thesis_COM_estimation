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


model = mujoco.MjModel.from_xml_path("models/2D_double_pend.xml")
data = mujoco.MjData(model)

#print (model.opt.gravity)

mujoco.mj_forward(model, data)

""" mass_1 = data.cinert[1][9]
mass_2 = data.cinert[2][9] """

""" print (mass_2)
print (model.body_mass[2]) """

#start_angle = math.pi/3
#start_angle = 0


data.qpos[0] = math.pi/8
data.qpos[1] = - math.pi/3

#data.qpos[1] = -start_angle
#data.qpos[1] = -start_angle/2

#data.qpos[1] = math.atan(math.tan(start_angle) * mass_1 / ( 2 * (mass_1 + mass_2))) - start_angle

des_angle_1 = 0

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



#l_1 = 1
l_1 = mujoco.mju_dist3(data.joint("ground_joint").xanchor, data.joint("middle_joint").xanchor)
l_2 = mujoco.mju_dist3(data.joint("middle_joint").xanchor, data.site("stick_end").xpos)

l = np.array ([l_1, l_2])

#l = [l_1]

T_0_i = np.eye(4)

#R_0_1 = np.eye(3)
#R_0_1 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
R_0_1 = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])

T_0_1 = np.eye(4)
T_0_1 [0:3, 0:3] = R_0_1

T_1_2 = np.eye(4)
T_1_2 [0][3] = l_1

T_2_e = np.eye(4)
T_2_e [0, 3] = l_2

T_i_i = np.array([T_0_1, T_1_2, T_2_e])
#T_i_i = np.array([T_0_1, T_1_2])

#print (T_i_i)

C_i_i = np.zeros((len(l), 4))

for k in range(0, len(l)):
    #C_i_i = np.append (C_i_i, np.transpose(np.array([l_i/2, 0, 0])))
    C_i_i [k][0] = l [k]/2
    C_i_i [k][3] = 1


C_0_i = np.zeros(4)
print (C_i_i)
    




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


viewer = mujoco.viewer.launch_passive(model, data, key_callback = kb_callback)

viewer.lock()


#viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
viewer.sync()
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
        
        
        """ curr_body = data.body("propeller_1")
        draw_vector(viewer, 0, curr_body.xpos, green_color, bal_force_1, k_f *  mujoco.mju_norm(bal_force_1))
        curr_body = data.body("propeller_2")
        draw_vector(viewer, 1, curr_body.xpos, blue_color, bal_force_2, k_f *  mujoco.mju_norm(bal_force_2))
        viewer.user_scn.ngeom = 2 """
        
        """ curr_body = data.body("propeller_1")
        draw_vector_quat(viewer, 0, curr_body.xpos, green_color, curr_body.xquat, k_f *  bal_int_1)
        curr_body = data.body("propeller_2")
        draw_vector_quat(viewer, 1, curr_body.xpos, blue_color, curr_body.xquat, k_f *  bal_int_2)
        viewer.user_scn.ngeom = 2
        if appl_force:
            for i in range (0, model.nbody):
                curr_force = ext_force[i]
                draw_vector(viewer, 2 + 2 * i, data.xipos[i], yellow_color, curr_force[0:3], k_f * mujoco.mju_norm(curr_force[0:3]))
                draw_vector(viewer, 2 + 2 * i + 1, data.xpos[i], red_color, curr_force[3:], k_f * mujoco.mju_norm(curr_force[3:]))
            viewer.user_scn.ngeom += 2 * model.nbody """
        
        mujoco.mjv_initGeom(viewer.user_scn.geoms[0],\
            type = mujoco.mjtGeom.mjGEOM_SPHERE, size = .04 * np.ones(3),\
            pos = data.subtree_com[1], mat = np.eye(3).flatten(), rgba = red_color)
        
        T_0_i = np.eye(4)
        C_0_t = np.zeros(3)
        #C_0_t [3] = 1
        #print (T_0_i)
        for i in range (0, model.nbody - 1):
            #T_0_i = T_0_i @ T_i_i [i, :, :]
            alpha_i = data.qpos [i]
            T_curr = T_i_i [i].copy()
            R_curr = np.array([[math.cos(alpha_i), math.sin(alpha_i), 0], [-math.sin(alpha_i), math.cos(alpha_i), 0], [0, 0, 1]])
            #R_curr = np.array([[math.cos(alpha_i), -math.sin(alpha_i), 0], [math.sin(alpha_i), math.cos(alpha_i), 0], [0, 0, 1]])
            
            #mujoco.mju_mulMatMat (T_curr[0:3, 0:3], T_i_i [i, 0:3, 0:3], R_curr)
            T_curr [:3, :3] = T_curr[:3, :3] @ R_curr
            #T_curr [0:3, 0:3] *= np.array([[math.cos(alpha_i), math.sin(alpha_i), 0], [-math.sin(alpha_i), math.cos(alpha_i), 0], [0, 0, 1]]) 
            #T_i_i [i, 0:3, 0:3] = np.array([[math.cos(alpha_i), math.sin(alpha_i), 0], [-math.sin(alpha_i), math.cos(alpha_i), 0], [0, 0, 1]])
            #T_i_i [i + 1, 0:3, 0:3] = np.array([[math.cos(alpha_i), -math.sin(alpha_i), 0], [math.sin(alpha_i), math.cos(alpha_i), 0], [0, 0, 1]])
            T_temp = T_0_i.copy()
            mujoco.mju_mulMatMat(T_0_i, T_temp, T_curr)
            #print (i, T_i_i[i])
            #print (T_0_i)
            #print (C_i_i[i])
            
            #print (C_i_k)
            #C_0_i = data.cinert[i + 1][9] * T_0_i * C_i_k
            #C_0_i = np.dot(T_0_i, C_i_i [i])
            mujoco.mju_mulMatVec(C_0_i, T_0_i, C_i_i [i])
            
            print (C_0_i)
            
            print (data.cinert [i + 1, 9])
            C_0_t += data.cinert [i + 1, 9] * C_0_i [0:3]
            
            #print (C_0_i)
            
        #print (model.body_subtreemass[1])
        C_0_t /= model.body_subtreemass[1]
        
        print (C_0_t)
        #print (T_i_i [0])
            
        
        mujoco.mjv_initGeom(viewer.user_scn.geoms[1],\
            type = mujoco.mjtGeom.mjGEOM_LINEBOX, size = .04 * np.ones(3),\
            pos = C_0_t, mat = np.eye(3).flatten(), rgba = blue_color)
        
        viewer.user_scn.ngeom = 2
        
        """ mujoco.mjv_initGeom(viewer.user_scn.geoms[2],\
            type = mujoco.mjtGeom.mjGEOM_SPHERE, size = .05 * np.ones(3),\
            pos = data.xpos[2], mat = np.eye(3).flatten(), rgba = green_color)
        mujoco.mjv_initGeom(viewer.user_scn.geoms[3],\
            type = mujoco.mjtGeom.mjGEOM_SPHERE, size = .05 * np.ones(3),\
            pos = C_0_i[0:3], mat = np.eye(3).flatten(), rgba = yellow_color)
        viewer.user_scn.ngeom = 4 """
        
        #print (model.body_ipos)
        
        
        viewer.sync()
        """ sim_time.append(data.time)
        angle_1.append(data.qpos[0])
        angle_2.append(data.qpos[1] + data.qpos[0]) """
        
    time_to_step = model.opt.timestep - (time.time() - step_start)
    if (time_to_step > 0):
        time.sleep(time_to_step)
        
        
if viewer.is_running():
    viewer.close()
    
    
