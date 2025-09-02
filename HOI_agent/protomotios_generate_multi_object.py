import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch

import json 
import collections
from manip.lafan1.utils import quat_inv, quat_mul, quat_between, normalize, quat_normalize, quat_mul_vec
import pytorch3d.transforms as transforms 

from utils.visualize.simplify_loc2rot import joints2smpl
from human_body_prior.body_model.body_model import BodyModel
from utils.process_mesh_files import run_smplx_model, load_rest_pose_object_geometry, \
    load_object_geometry_w_rest_geo, create_ball_mesh_new, export_to_mesh, \
    save_verts_faces_to_mesh_file_w_object, sample_point_trajectory, interpolation_object_motion_new, \
    round_list, get_obb_vertices, interpolate_rotations
from manip.vis.blender_vis_mesh_motion import run_blender_rendering_and_save2video
import shutil
from scipy.interpolate import PchipInterpolator
from utils.a_star_planning import astar, GridMap, smooth_path, build_horizontal_polygon, \
    simplify_path_rdp, sample_path_by_distance, assign_frame_ids_by_distance, visualize_gridmap
import subprocess 

import random
seed=1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

joint_map = {
    'pelvis': 0,
    'left_hand': 22,
    'right_hand': 23,
    'left_foot': 10,
    'right_foot': 11
}

def specify_points(number_frames=120, points=[[50, 1, 1, 1]]):
    hint = np.zeros((number_frames, 3))
    for point in points:
        hint[point[0]] = point[1:]
    return hint

def zup_to_yup(joint_pos):
    trans_matrix = np.array([[1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0]])
    pose_seq_np_n = np.dot(joint_pos, trans_matrix)
    return pose_seq_np_n
def yup_to_zup(joint_pos):
    trans_matrix = np.array([[1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0]])
    pose_seq_np_n = np.dot(joint_pos, trans_matrix)
    return pose_seq_np_n

class protomotion_model:
    def __init__(self, device, save_motion_root, batch_size=1, vis_fps = 15, use_sample_contact_point=True, last_k_frame=1,
                      rest_object_geo_folder='processed_data/rest_object_geo',
                      scene_hight_file='processed_data/replica_processed/scene_floor_height.json', opt=None):
        self.device = device
        self.batch_size = batch_size
        
        self.model_path = opt.model_path

        self.last_k_frame = last_k_frame
        self.save_motion_root = save_motion_root
        if not os.path.exists(save_motion_root):
            os.makedirs(save_motion_root) 
        
        self.prepare_smpl_parameter()
        self.rest_object_geo_folder = rest_object_geo_folder
        self.all_scene_floor_height = json.load(open(scene_hight_file, 'r'))
        self.vis_fps = vis_fps
        self.use_sample_contact_point = use_sample_contact_point
        self.quick_vis=False
        
        self.y_up=False
        self.height_idx=1 if self.y_up else 2
        self.opt = opt

    def prepare_smpl_parameter(self):
        # Prepare SMPLX model 
        self.data_root_folder = 'data'
        soma_work_base_dir = os.path.join(self.data_root_folder, 'smpl_all_models')
        support_base_dir = soma_work_base_dir 
        surface_model_type = "smplx"
        surface_model_male_fname = os.path.join(support_base_dir, surface_model_type, "SMPLX_MALE.npz")
        surface_model_female_fname = os.path.join(support_base_dir, surface_model_type, "SMPLX_FEMALE.npz")
        dmpl_fname = None
        num_dmpls = None 
        num_expressions = None
        num_betas = 16 

        self.male_bm = BodyModel(bm_fname=surface_model_male_fname,
                        num_betas=num_betas,
                        num_expressions=num_expressions,
                        num_dmpls=num_dmpls,
                        dmpl_fname=dmpl_fname)
        self.female_bm = BodyModel(bm_fname=surface_model_female_fname,
                        num_betas=num_betas,
                        num_expressions=num_expressions,
                        num_dmpls=num_dmpls,
                        dmpl_fname=dmpl_fname)

        for p in self.male_bm.parameters():
            p.requires_grad = False
        for p in self.female_bm.parameters():
            p.requires_grad = False 

        self.male_bm = self.male_bm.cuda()
        self.female_bm = self.female_bm.cuda()
        
        self.bm_dict = {'male' : self.male_bm, 'female' : self.female_bm}

    def set_data_name(self, seq_name, p_idx, scene_name, gender='male', betas=None):
        self.seq_name = seq_name
        self.scene_name = scene_name
        self.p_idx = p_idx
        self.task_index = -1
        
        self.gender = gender
        self.object_name = seq_name.split('_')[1]
        self.betas = betas if betas is not None else np.array([[ 0.88815075,  0.06342907,  0.7364383 , -2.1567714 , -1.0417588 ,
                                                                -0.56648463,  4.172723  ,  1.4159995 ,  2.1836061 ,  2.5980496 ,
                                                                -2.3135738 , -0.69624144,  1.7862709 ,  0.01760743,  0.7098188 ,
                                                                1.5601856 ]])
        
        self.save_motion_folder = os.path.join(self.save_motion_root, f"{self.scene_name}_{self.seq_name}_{p_idx}")
        if os.path.exists(self.save_motion_folder):
            shutil.rmtree(self.save_motion_folder)  

        os.makedirs(self.save_motion_folder, exist_ok=True)  

        self.ball_mesh_save_folder = os.path.join(self.save_motion_folder, \
                                "ball_objs")
        self.mesh_save_folder = os.path.join(self.save_motion_folder, \
                        "objs_mesh")
        self.out_rendered_img_folder = os.path.join(self.save_motion_folder, \
                        "imgs_render")
        self.template_npz_folder = os.path.join(self.save_motion_folder, \
                        "template_npz_folder")
        self.temp_control_points_folder = os.path.join(self.save_motion_folder, \
                        "control_points")
        self.maskedmimic_res_folder = os.path.join(self.save_motion_folder, \
                        "maskedmimic_res")
        self.out_vid_file_path = os.path.join(self.save_motion_folder, \
                        "video.mp4")
        self.check_collision = self.opt.check_collision

        os.makedirs(self.ball_mesh_save_folder, exist_ok=True) 
        os.makedirs(self.mesh_save_folder, exist_ok=True) 
        os.makedirs(self.out_rendered_img_folder, exist_ok=True) 
        os.makedirs(self.temp_control_points_folder, exist_ok=True) 
        os.makedirs(self.maskedmimic_res_folder, exist_ok=True) 

        self.floor_blend_path = f'processed_data/blender_files/for_seq_in_{scene_name}.blend'
        self.target_lift_height = 1.2

    def get_human_object_mesh(self):
        # load last sample data
        init_position, init_rotation, init_humanml3d, init_object_pos_all, init_object_rot_all, prev_joint_point, prev_object_point_all, pose_aa = self.load_prev_npy_data(self.task_index+1)
        temp_seq_len = init_position.shape[1]
        
        if pose_aa is not None:
            rot_aa = torch.from_numpy(pose_aa).float().to(self.device).reshape(1, temp_seq_len, 22, 3)
        else:
            # do IK
            ik_path = os.path.join(self.save_motion_folder, 'ik.pt')
            if os.path.exists(ik_path):
                motion_tensor = torch.load(ik_path)
            else:
                j2s = joints2smpl(num_frames=temp_seq_len, device_id=0, cuda=True) 
                motion_tensor, opt_dict = j2s.joint2smpl(init_position[0].copy())
                torch.save(motion_tensor, ik_path)
            # run smplx
            rot_temp_local = motion_tensor[:,:22].permute(0, 3, 1, 2).float().to(self.device)
            rot_mat = transforms.rotation_6d_to_matrix(rot_temp_local)
            rot_aa = transforms.matrix_to_axis_angle(rot_mat)
        trans = torch.from_numpy(init_position[:,:,0]).float().to(self.device)
        betas = torch.from_numpy(self.betas).float().to(self.device)
        mesh_jnts, mesh_verts, mesh_faces = run_smplx_model(trans, rot_aa, betas, [self.gender], self.bm_dict)
        
        diff = trans[0,0:1] - mesh_jnts[0,0,0:1]
        mesh_jnts = mesh_jnts + diff
        mesh_verts = mesh_verts + diff

        mesh_jnts, mesh_verts, mesh_faces = mesh_jnts[0], mesh_verts[0], mesh_faces

        # recover to original scene height
        scene_height = self.all_scene_floor_height[self.scene_name]

        mesh_jnts[:,:,2] = mesh_jnts[:,:,2] + scene_height
        mesh_verts[:,:,2] = mesh_verts[:,:,2] + scene_height

        prev_joint_point[:,:,3] = prev_joint_point[:,:,3] + scene_height

        ball_mesh_path = os.path.join(self.ball_mesh_save_folder, "conditions.ply")
        # save joint condition
        for idx, joint_point in enumerate(prev_joint_point[0]):
            create_ball_mesh_new(joint_point[1:], ball_mesh_path, 'joint'+str(idx), int(joint_point[0]), ball_color=[22, 173, 100])

        transformed_obj_verts_all = []
        obj_mesh_faces_all = []
        object_name_all = []
        for name in init_object_pos_all:
            # get object mesh
            init_object_rot = torch.from_numpy(init_object_rot_all[name]).float().to(self.device)[0]
            init_object_pos = torch.from_numpy(init_object_pos_all[name]).float().to(self.device)[0]
            init_object_rot_mat = transforms.quaternion_to_matrix(init_object_rot)
            
            rest_verts, obj_mesh_faces = load_rest_pose_object_geometry(self.rest_object_geo_folder, self.object_name)
            obj_rest_verts = torch.from_numpy(rest_verts).float().to(self.device)
            transformed_obj_verts = load_object_geometry_w_rest_geo(init_object_rot_mat, init_object_pos, obj_rest_verts)

            transformed_obj_verts[:,:,2] = transformed_obj_verts[:,:,2] + scene_height
            
            prev_object_point_all[name][:,:,3] = prev_object_point_all[name][:,:,3] + scene_height

            # save condition data
            start_mesh_path = os.path.join(self.ball_mesh_save_folder, f"start_object_{name}.ply")
            export_to_mesh(transformed_obj_verts[0].cpu(), obj_mesh_faces, start_mesh_path)

            # save object condition
            for idx, object_point in enumerate(prev_object_point_all[name][0]):
                create_ball_mesh_new(object_point[1:], ball_mesh_path, f'object_{name}_'+str(idx), int(object_point[0]), ball_color=[255, 255, 0], radius=0.1)

            transformed_obj_verts_all.append(transformed_obj_verts.detach().cpu().numpy())
            obj_mesh_faces_all.append(obj_mesh_faces)
            object_name_all.append(name)

        # save human and object mesh
        save_verts_faces_to_mesh_file_w_object(mesh_verts.detach().cpu().numpy(), \
                    mesh_faces.detach().cpu().numpy(), \
                    transformed_obj_verts_all, obj_mesh_faces_all, self.mesh_save_folder, object_name_all)
        
        return mesh_jnts, mesh_verts, mesh_faces, transformed_obj_verts_all
    
    def render_video(self):
        run_blender_rendering_and_save2video(self.mesh_save_folder, self.out_rendered_img_folder, self.out_vid_file_path, 
                                             self.ball_mesh_save_folder, vis_object=True, vis_condition=True, \
                                            scene_blend_path=self.floor_blend_path, fps=self.vis_fps)

    def text_control_data(self, control_joints=None, control_points=None, text=None, number_frames=120):
        '''
        Here I just provide one simple example for combination contorlling (l_wrist, r_wrist)
        Need to optimize if want to add more...
        '''
        
        text = [text]

        number_frames = int(number_frames)
        control_dict = collections.defaultdict(list)

        for joint, waypoints in zip(control_joints, control_points):
            for frame, x, y, z in waypoints:    
                control_dict[joint_map[joint]].append([int(frame), x, y, z])
        
        control = [[]]
        joint_id = []
        for k, v in control_dict.items():
            control[-1].append(specify_points(number_frames, v))
            joint_id.append(k)

        control = np.stack(control)
        joint_id = np.array([joint_id,])


        control_full = np.zeros((len(control), number_frames, 22, 3)).astype(np.float32)
        for i in range(len(control)):
            mask = control[i].sum(-1) != 0
            control_ = (control[i] - self.raw_mean.reshape(22, 1, 3)[joint_id[i]]) / self.raw_std.reshape(22, 1, 3)[joint_id[i]]
            control_ = control_ * mask[..., np.newaxis]
            control_full[i, :, joint_id[i], :] = control_

        control_full = control_full.reshape((control.shape[0], number_frames, -1))
        return number_frames, text, control_full, joint_id
    
    def get_pred_object_mesh(self, object_pos, object_rot_mat, object_name, human_pos):
        bs = object_pos.shape[0]
        pred_obj_rot_mat = torch.from_numpy(object_rot_mat).reshape(bs, -1, 3, 3).float().to(self.device)
        pred_seq_com_pos = torch.from_numpy(object_pos.reshape(bs, -1, 3)).float().to(self.device)

        sampled_points_list = []
        obj_mesh_verts_list = []
        for idx in range(bs):
            curr_obj_rot_mat = pred_obj_rot_mat[idx]
            cur_pred_seq_com_pos = pred_seq_com_pos[idx]

            rest_verts, obj_mesh_faces = load_rest_pose_object_geometry(self.rest_object_geo_folder, object_name.replace('1', ''))
            obj_rest_verts = torch.from_numpy(rest_verts).float().to(self.device)            
            obj_mesh_verts = load_object_geometry_w_rest_geo(curr_obj_rot_mat, cur_pred_seq_com_pos, obj_rest_verts)

            obj_mesh_verts_list.append(obj_mesh_verts.unsqueeze(0))

            if human_pos is not None:
                max_height = obj_mesh_verts[0, :, self.height_idx].max()
                min_height = obj_mesh_verts[0, :, self.height_idx].min()
                height = max_height - min_height

                # set constrains for the object lift height
                z_min = min_height + 0.0 * height if min_height > 1 else min_height + 0.3 * height
                z_max = min_height + 0.99 * height if max_height < 1.2 else min_height + 0.5 * height

                z_range = [z_min.item(), z_max.item()]

                new_human_pos = human_pos.copy()
                new_human_pos[0,:2] += (human_pos[0, :2] - human_pos[1, :2])*3
                new_human_pos[1,:2] += (human_pos[1, :2] - human_pos[0, :2])*3

                sampled_points = sample_point_trajectory(obj_mesh_verts.clone(), obj_mesh_faces, z_range, torch.from_numpy(new_human_pos[0]).float().to(self.device), height_idx=self.height_idx, mode='sample')
                sampled_points2 = sample_point_trajectory(obj_mesh_verts.clone(), obj_mesh_faces, z_range, torch.from_numpy(new_human_pos[1]).float().to(self.device), height_idx=self.height_idx, mode='sample')
                sampled_points = torch.cat([sampled_points.unsqueeze(1),sampled_points2.unsqueeze(1)], dim=1)
                sampled_points_list.append(sampled_points.unsqueeze(0))
        
        # bs, T, n, 3
        return torch.concat(sampled_points_list, dim=0), pred_seq_com_pos, pred_obj_rot_mat, torch.concat(obj_mesh_verts_list, dim=0)
    
    def rotate_at_frame_w_obj(self, X, Q, control_points_list, n_past=1, y_up=False, object_pos_list=None, object_rot_list=None, last_k_frame=1, control_joint=None):
        """
        Re-orients the animation data according to the last frame of past context.

        :param X: tensor of positions of shape (Batchsize, T, Joints, 3)
        :param Q: tensor of quaternions (Batchsize, T, 1, 4)
        :obj_x: N X T X 3
        :obj_q: N X T X 4
        :trans2joint_list: N X 3 
        :param parents: list of parents' indices
        :param n_past: number of frames in the past context
        :return: The rotated positions X and quaternions Q
        """

        trans2joint_list = -X[:, n_past - 1, 0, :] # N X 3
        floor_height = X.min(axis=1).min(axis=1)[:,2]
        trans2joint_list[:, 2] = -floor_height

        # Get global quats and global poses (FK)
        global_q = Q 
        global_x = X + trans2joint_list[:, np.newaxis, np.newaxis, :] # N X T X 22 X 3

        key_glob_Q = global_q[:, n_past - 1 : n_past, 0:1, :]  # (B, 1, 1, 4)
        
        # The floor is on z = xxx. Project the forward direction to xy plane. 
        forward = np.array([1, 1, 0])[np.newaxis, np.newaxis, np.newaxis, :] * quat_mul_vec(
            key_glob_Q, np.array([1, 0, 0])[np.newaxis, np.newaxis, np.newaxis, :]
        ) # In rest pose, x direction is the body left direction, root joint point to left hip joint.  
        

        forward = normalize(forward)
        yrot = quat_normalize(quat_between(np.array([1, 0, 0]), forward))
        new_glob_Q = quat_mul(quat_inv(yrot), global_q)
        new_glob_X = quat_mul_vec(quat_inv(yrot), global_x)

        # Process control_point rotation and translation
        new_control_points_list = []
        for control_id, control_points in enumerate(control_points_list):
            if 0 not in np.array(control_points)[:,0]:
                control_joint_name = control_joint[control_id]
                control_points = [[0] + list(X[0,-1,joint_map[control_joint_name]])] + control_points
            control_points = np.array(control_points)
            obj_trans = control_points[:, np.newaxis, 1:] + trans2joint_list[:, np.newaxis, :] # N X T X 3  
            obj_trans = quat_mul_vec(quat_inv(yrot[:, 0, :, :]), obj_trans) # N X T X 3
            # obj_trans = obj_trans - trans2joint_list[:, np.newaxis, :] # N X T X 3 
            new_control_points = control_points.copy() 
            if y_up:
                obj_trans = zup_to_yup(obj_trans[:,:].copy())
            new_control_points[:,1:] = obj_trans[:,0,:]
            new_control_points[:,0] += last_k_frame-1

            new_control_points_list.append(new_control_points.tolist())
        
        Q = new_glob_Q
        X = new_glob_X 
        if y_up:
            X = zup_to_yup(X)

        # Process object rotation and translation 
        if object_pos_list is not None:
            for i, object_pos in enumerate(object_pos_list):
                object_rot = object_rot_list[i] # N X T X 4
                object_pos = object_pos + trans2joint_list[:, np.newaxis, :] # N X T X 3
                object_pos = quat_mul_vec(quat_inv(yrot[:, 0, :, :]), object_pos) # N X T X 3
                object_rot = quat_mul(quat_inv(yrot[:, 0, :, :]), object_rot)  
                object_rot = torch.from_numpy(object_rot).float()
                object_rot_mat = transforms.quaternion_to_matrix(object_rot)
                object_rot_mat = object_rot_mat.numpy()
                if y_up:
                    object_pos = zup_to_yup(object_pos)
                    object_rot = torch.from_numpy(object_rot).float()
                    object_rot_mat = transforms.quaternion_to_matrix(object_rot)
                    transform_matrix = torch.tensor([[1, 0, 0],
                                                    [0, 0, 1],
                                                    [0, -1, 0]], dtype=torch.float32).float()
                    object_rot_mat = torch.einsum('ij,bnjk->bnik', transform_matrix, object_rot_mat)
                    object_rot_mat = object_rot_mat.numpy()

                object_pos_list[i] = object_pos
                object_rot_list[i] = object_rot_mat
            return X, Q, new_control_points_list, yrot, trans2joint_list, object_pos_list, object_rot_list
        
        return X, Q, new_control_points_list, yrot, trans2joint_list
    

    def detect_object(self):
        ## load the environment state
        if self.task_index < 0:
            state = json.load(open(f'{self.opt.data_folder}/environment_state/{self.seq_name}_pidx_{self.p_idx}.json', 'r'))
        else:
            try:
                state = json.load(open(os.path.join(self.save_motion_folder, self.seq_name+"_pidx_"+str(self.p_idx)+"_task_"+str(self.task_index)+".json"), 'r'))
            except:
                state = json.load(open(f'{self.opt.data_folder}/environment_state/{self.seq_name}_pidx_{self.p_idx}.json', 'r'))
        
        CURRENT_STATE_ENVIRONMENT = ""
        ee_start_pelvis_position = state['human']['pelvis']
        ee_start_left_hand_position = state['human']['left_hand']
        ee_start_right_hand_position = state['human']['right_hand']

        ee_start_orientation = state['human']['orientation']
        CURRENT_STATE_ENVIRONMENT += f"***human***:\n"
        CURRENT_STATE_ENVIRONMENT += f"pelvis position: {ee_start_pelvis_position}\n"
        CURRENT_STATE_ENVIRONMENT += f"left hand position: {ee_start_left_hand_position}\n"
        CURRENT_STATE_ENVIRONMENT += f"right hand position: {ee_start_right_hand_position}\n"

        CURRENT_STATE_ENVIRONMENT += f"orientation: {ee_start_orientation}\n"
        for obj in state['objects']:
            CURRENT_STATE_ENVIRONMENT += f"***{obj['name']}***:\n" + f"position: {obj['position']}\n" + f"orientation: {obj['orientation']}\n" + f"sizes: {obj['sizes']}\n"
        
        print(CURRENT_STATE_ENVIRONMENT)
        return

    def load_prev_npy_data(self, task_index=None):
        # load last sample data
        init_humanml3d = None
        pose_aa = None
        init_object_xyz = {}
        init_object_rotation = {}
        prev_object_point = {}
        if task_index == 0:
            data_last = np.load(f'{self.opt.data_folder}/motion_data/{self.seq_name}_pidx_{self.p_idx}.npy',allow_pickle=True)[None][0]
            init_position = data_last['human']['joint_xyz'].reshape(1, -1, 24, 3) # z-up, 24 X 3
            init_rotation = data_last['human']['orientation'].reshape(1, -1, 1, 4) # z-up, 4

            pose_aa = data_last['human']['pose_aa'].reshape(1, -1, 22, 3)

            prev_joint_point = np.array([], dtype=np.float32).reshape(1, -1, 4) # z-up, 1 X N X 3
            
            for name in data_last['object']:
                init_object_xyz[name] = data_last['object'][name]['xyz'].reshape(1, -1, 3)
                init_object_rotation[name] = data_last['object'][name]['rotation'].reshape(1, -1, 4)
            
                prev_object_point[name] = np.array([], dtype=np.float32).reshape(1, -1, 4) # z-up, 1 X N X 3
        else:
            data_last = np.load(os.path.join(self.save_motion_folder, self.seq_name+"_pidx_"+str(self.p_idx)+"_task_"+str(task_index-1)+".npy"),allow_pickle=True)[None][0]
            init_position = data_last['human']['joint_xyz'] # z-up, 1 X T X 24 X 3
            init_rotation = data_last['human']['orientation'] # z-up, 1 X T X 1 X 4

            pose_aa = data_last['human']['pose_aa'].reshape(1, -1, 22, 3)

            prev_joint_point = data_last['human']['joint_point'].reshape(1, -1, 4) # z-up, 1 X N X 3
            for name in data_last['object']:
                init_object_xyz[name] = data_last['object'][name]['xyz'].reshape(1, -1, 3)
                init_object_rotation[name] = data_last['object'][name]['rotation'].reshape(1, -1, 4)

                prev_object_point[name] = data_last['object'][name]['object_point'].reshape(1, -1, 4) # z-up, 1 X N X 3

        return init_position, init_rotation, init_humanml3d, init_object_xyz, init_object_rotation, prev_joint_point, prev_object_point, pose_aa

    # use A* for the whole trajectory
    def quick_check_collision_with_astar_v3(self, subject_name, position, rotation, size, scene_state, detect_frames, object_points, diff_to_object=None, path_size=None):
        from shapely.geometry import Point

        xlim = [np.inf, -np.inf]
        ylim = [np.inf, -np.inf]
        resolution = 0.025

        seq_len = position.shape[0]
        all_polygons = []

        # Build horizontal collision polygons for all obstacles
        for obj in scene_state['objects']:
            if obj['name'] not in subject_name:
                obj_pos = np.array(obj['position'])
                obj_euler = torch.from_numpy(np.array(obj['orientation']))
                obj_rot = transforms.euler_angles_to_matrix(obj_euler, convention='XYZ').numpy()
                obj_size = np.array(obj['sizes'])
                obj_vertices = get_obb_vertices(obj_pos, obj_rot, obj_size / 2)

                # update xlim 和 ylim
                xs = obj_vertices[:, 0]
                ys = obj_vertices[:, 1]
                xlim[0] = min(xlim[0], xs.min())
                xlim[1] = max(xlim[1], xs.max())
                ylim[0] = min(ylim[0], ys.min())
                ylim[1] = max(ylim[1], ys.max())

                poly = build_horizontal_polygon(obj_vertices)
                poly_for_astar = build_horizontal_polygon(
                    get_obb_vertices(obj_pos, obj_rot, obj_size / 2 + 0.025)
                )
                all_polygons.append((poly, poly_for_astar, obj['name'], obj_pos, obj_size))

        # Check which frames collide
        collision_frames = []
        collision_objects = set()
        max_x_extent = 0.0
        max_y_extent = 0.0
        min_x_extent = np.inf
        min_y_extent = np.inf
        for idx in range(seq_len):

            vertices = get_obb_vertices(position[idx], rotation[idx], size / 2)
            # 
            xs = vertices[:, 0]
            ys = vertices[:, 1]
            x_extent = xs.max() - xs.min()
            y_extent = ys.max() - ys.min()

            max_x_extent = max(max_x_extent, x_extent)
            max_y_extent = max(max_y_extent, y_extent)
            min_x_extent = min(min_x_extent, x_extent)
            min_y_extent = min(min_y_extent, y_extent)
            flag = False
            for pt in vertices:
                pt_xy = Point(pt[0], pt[1])
                for poly, _, name, obj_pos, obj_size in all_polygons:
                    if poly.contains(pt_xy):
                        collision_frames.append(detect_frames[idx])
                        collision_objects.add((name, tuple(obj_pos.tolist()), tuple(obj_size.tolist())))
                        flag = True
                        break  # only log once per frame
                if flag:
                    break

        if not collision_frames:
            return None

        if 'human' in subject_name:
            segments = [(int(object_points[1,0]), int(object_points[-1,0]))]
        else:
            segments = [(int(object_points[2,0]), int(object_points[-1,0]))]

        # Build A* map
        grid = GridMap(xlim=xlim, ylim=ylim, resolution=resolution)
        labeled_objects = []
        for poly, poly_for_astar, name, pos, size in all_polygons:
            grid.set_obstacle_from_polygon(poly_for_astar)
            labeled_objects.append((name, pos))

        new_object_points = []
        last_end_idx = 0
        frame_ids = object_points[:, 0]

        # Report all collided objects
        object_report = "\n".join([
            f"- {name} at position {round_list(list(pos))} with size {round_list(list(size_))}"
            for name, pos, size_ in sorted(collision_objects)
        ])
        
        for seg_start, seg_end in segments:
            start_idx = np.searchsorted(frame_ids, seg_start, side='left') - 1
            end_idx = np.searchsorted(frame_ids, seg_end, side='right')
            start_idx = max(0, start_idx)
            end_idx = min(len(object_points) - 1, end_idx)

            start_point = object_points[start_idx, 1:].tolist()
            end_point = object_points[end_idx, 1:].tolist()
            fixed_z = object_points[0,3]

            obj_w=(max_x_extent+min_x_extent)/2
            obj_h=(max_y_extent+min_y_extent)/2
            
            start_obj_w=obj_w
            start_obj_h=obj_h
            goal_obj_w=obj_w
            goal_obj_h=obj_h
            if path_size is not None:
                path_obj_w = path_size[0]
                path_obj_h = path_size[1]
            else:
                path_obj_w=obj_w
                path_obj_h=obj_h
        
            path_2d = astar(grid, start_point[:2], end_point[:2], start_obj_w=start_obj_w, start_obj_h=start_obj_h,
                            path_obj_w=path_obj_w, path_obj_h=path_obj_h,
                            goal_obj_w=goal_obj_w, goal_obj_h=goal_obj_h,)
            
            if path_2d is None:
                visualize_gridmap(
                    grid,
                    start=list(object_points[0, 1:3]),
                    goal=list(object_points[-1, 1:3]),
                    labeled_objects=labeled_objects,
                    save_path=os.path.join(self.save_motion_folder, f'Astar_path_{self.task_index+1}.png')
                )

                return (f"{subject_name[0]} collides with an obstacle between frames {seg_start}-{seg_end}, and A* failed to find a path.",
                        f"Objects involved in collision:\n{object_report}\n\n"
                         "Please replan manually.")

            smoothed_path_2d = smooth_path(path_2d)
            smoothed_path_2d = simplify_path_rdp(smoothed_path_2d, epsilon=0.05)
            smoothed_path_2d = sample_path_by_distance(smoothed_path_2d, 0.1)
            if start_point[:2] not in smoothed_path_2d and (smoothed_path_2d[0], smoothed_path_2d[1]) not in smoothed_path_2d:
                smoothed_path_2d = [start_point[:2]] + smoothed_path_2d

            frame_id_list = assign_frame_ids_by_distance(smoothed_path_2d, frame_ids[start_idx], frame_ids[end_idx])

            new_segment = [
                [fid, pt[0], pt[1], fixed_z] for fid, pt in zip(frame_id_list, smoothed_path_2d)
            ]

            # Add valid previous waypoints
            for i in range(last_end_idx, start_idx):
                new_object_points.append(round_list(object_points[i].tolist()))

            new_object_points.extend([round_list(p) for p in new_segment])
            last_end_idx = end_idx

        # Add remaining waypoints
        if last_end_idx+1 == object_points.shape[0]:
            pass
        else:
            for i in range(last_end_idx, object_points.shape[0]):
                new_object_points.append(round_list(object_points[i].tolist()))

        new_object_points = [list(p) for p in set(tuple(p) for p in new_object_points)]
        new_object_points.sort(key=lambda x: x[0])

        visualize_gridmap(
            grid,
            start=list(object_points[0, 1:3]),
            goal=list(object_points[-1, 1:3]),
            path=[[x[1], x[2]] for x in new_object_points],
            labeled_objects=labeled_objects,
            save_path=os.path.join(self.save_motion_folder, f'Astar_path_{self.task_index+1}.png')
        )

        self.check_collision = False

        if diff_to_object is not None:
            new_object_points = np.array(new_object_points)
            
            # Calculate the length of diff_to_object (keep the distance between the two objects constant)
            offset_distance = min(np.linalg.norm(diff_to_object), np.linalg.norm(size[:2]) + 0.1)
            
            # Initialize a perpendicular vector to determine the initial lateral direction
            initial_perpendicular = None
            
            # Get the trajectories of the two objects
            object1_trajectory = []
            object2_trajectory = []
            
            for i in range(len(new_object_points)):
                center_point = new_object_points[i]
                
                if i > 0:
                    direction_vector = new_object_points[i, 1:] - new_object_points[i - 1, 1:]
                else:
                    direction_vector = np.array([0, 0, 0])
                
                # Calculate the unit vector perpendicular to the direction of motion
                if np.linalg.norm(direction_vector) > 1e-5:
                    # 2D case: (x, y) -> (-y, x) or (y, -x)
                    if len(direction_vector) == 2:
                        perpendicular = np.array([-direction_vector[1], direction_vector[0]])
                    # 3D case: need to choose a perpendicular direction
                    else:
                        # Use cross product to determine perpendicular direction
                        perpendicular = np.cross(direction_vector, [0, 0, 1])[:len(direction_vector)]
                    
                    # Normalize
                    if np.linalg.norm(perpendicular) > 1e-5:
                        perpendicular = perpendicular / np.linalg.norm(perpendicular)
                    else:
                        # If the perpendicular vector is zero, keep the previous direction
                        if initial_perpendicular is not None:
                            perpendicular = initial_perpendicular / np.linalg.norm(initial_perpendicular)
                        else:
                            # Use the original diff_to_object direction
                            perpendicular = diff_to_object / np.linalg.norm(diff_to_object)
                else:
                    # If the direction vector is zero, use the previous perpendicular vector or original diff_to_object
                    if initial_perpendicular is not None:
                        perpendicular = initial_perpendicular / np.linalg.norm(initial_perpendicular)
                    else:
                        perpendicular = diff_to_object / np.linalg.norm(diff_to_object)

                # Ensure consistent direction of the perpendicular vector
                if initial_perpendicular is not None:
                    # Check whether current perpendicular vector is aligned with the initial one
                    # If dot product is negative, the direction is opposite and needs to be flipped
                    if np.dot(perpendicular, initial_perpendicular) < 0:
                        perpendicular = -perpendicular
                else:
                    # Set the initial perpendicular vector
                    initial_perpendicular = perpendicular.copy()
                
                # Scale to the required distance
                perpendicular = perpendicular * offset_distance

                # Calculate positions of the two objects
                obj1_point = center_point.copy()
                obj1_point[1:] += perpendicular
                
                obj2_point = center_point.copy()
                obj2_point[1:] -= perpendicular
                
                object1_trajectory.append(obj1_point)
                object2_trajectory.append(obj2_point)
            
            # Convert to list format
            object1_points = [round_list(obj.tolist()) for obj in object1_trajectory]
            object2_points = [round_list(obj.tolist()) for obj in object2_trajectory]


            return (
                f"{subject_name[0]} and {subject_name[1]} were detected to collide in {len(segments)} time segment(s). "
                f"Collision was avoided using A* with smoothing.\n"
                f"Objects involved in collision:\n{object_report}\n\n"
                f"Here is the recommended collision-free trajectory for {subject_name[0]}:\n{object1_points}\n"
                f"Here is the recommended collision-free trajectory for {subject_name[1]}:\n{object2_points}\n"
                f"Please verify the new path to ensure no further collisions. If no collision, please let engineer generate codes for this trajectory."
            )
            
        return (
            f"{subject_name[0]} was detected to collide in {len(segments)} time segment(s). "
            f"Collision was avoided using A* with smoothing.\n"
            f"Objects involved in collision:\n{object_report}\n\n"
            f"Here is the recommended collision-free trajectory:\n{new_object_points}\n"
            f"Please verify the new path to ensure no further collisions. If no collision, please let engineer generate codes for this trajectory."
        )

    
    def quick_check_human_collision(self, control_joints, control_points, temp_frames, init_position, state):
        # quick check collision for human
        for idx, joint_name in enumerate(control_joints):
            if joint_name == 'pelvis':
                from scipy.interpolate import interp1d
                temp_control_points = np.array(control_points[idx])
                key_times = np.array(temp_control_points[:,0], np.int64)
                key_pos = temp_control_points[:,1:]
                if 0 not in key_times:
                    key_times = np.concatenate([np.array([0]), key_times], axis=0)
                    key_pos = np.concatenate([init_position[0,-1:,0], key_pos], axis=0)
                # temp_frames = number_frames - last_k_frame + 1
                temp_position = np.zeros((temp_frames, 3))
                for temp_idx in range(3):
                    interpolation_res = interp1d(key_times, key_pos[:, temp_idx], kind='linear', fill_value="extrapolate")
                    temp_position[:,temp_idx] = interpolation_res(range(temp_frames))
                temp_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).reshape(1,3,3)
                temp_rotation = np.repeat(temp_rotation, temp_frames, 0)

                num_frames = temp_position.shape[0]
                detect_frames = [i for i in range(num_frames) if i % 10 == 0]
                if num_frames-1 not in detect_frames:
                    detect_frames.append(num_frames-1)
                res = self.quick_check_collision_with_astar_v3(['human'], temp_position[detect_frames], temp_rotation[detect_frames], np.array([0.3, 0.3, 1.6]), state, detect_frames, temp_control_points.reshape(-1,4))
                if res is not None:
                    print(res)
                    return res

    def quick_check_object_collision(self, object_motion, state, object_name, object_points):
        # quick check collision for object
        for object in state['objects']:
            if object_name[0] == object['name']:
                object_size = np.array(object['sizes'])
                break
        if len(object_name) > 1:
            object_size[:2] = object_size[:2]*2

            middle_object_points = object_points[0].clone()
            middle_object_points[:,1:3] = (object_points[0][:,1:3] + object_points[1][:,1:3]) / 2
            diff_to_object = (object_points[0][0,1:] - middle_object_points[0,1:]).numpy()
        else:
            middle_object_points = object_points[0].clone()
            diff_to_object = None
        
        num_frames = object_motion.shape[1]
        detect_frames = [i for i in range(num_frames) if i % 10 == 0]
        if num_frames-1 not in detect_frames:
            detect_frames.append(num_frames-1)
        sampled_position = object_motion[0,detect_frames,:3].numpy()
        sampled_rotation = object_motion[0,detect_frames,3:].numpy().reshape(-1,3,3)
        
        res = self.quick_check_collision_with_astar_v3(object_name, sampled_position, sampled_rotation, object_size, state, detect_frames, middle_object_points.numpy(), diff_to_object=diff_to_object)
        if self.opt.multi_object_narrow_path and res is not None and object_size[2] < 0.9 and object_size[0] * object_size[1] > 0.2:
            # check if the object is a table or chair and if we could pass by lifting the object
            res = self.quick_check_collision_with_astar_v3(object_name, sampled_position, sampled_rotation, 
                                object_size, state, detect_frames, middle_object_points.numpy(), 
                                diff_to_object=diff_to_object, path_size=np.array([0.3, 0.3, 0.3]))
            self.target_lift_height = 1.5
        if res is not None:
            print(res)
            return res

    def adjust_frames(self, control_points, object_points_list, number_frames, init_position, high_thresh=0.04, low_thresh=0.02):
        number_frames = 0
        # currently if have object points, we will ignore control_points
        if object_points_list is not None:
            init_object_points_all = np.array([object_points[0][1:] for object_points in object_points_list])
            for obj_idx, object_points in enumerate(object_points_list):
                if obj_idx != 0:
                    if self.check_collision:
                        diff_dist = init_object_points_all[obj_idx] - init_object_points_all[0]
                        control_point = np.array(object_points_list[0]).reshape(-1,4)
                        control_point[:,1:] += diff_dist
                        object_points_list[obj_idx] = control_point
                    else:
                        control_point0 = np.array(object_points_list[0]).reshape(-1,4)
                        control_point1 = np.array(object_points_list[obj_idx]).reshape(-1,4)
                        control_point1[:,0] = control_point0[:,0]
                        object_points_list[obj_idx] = control_point1
                    continue
                
                control_point = np.array(object_points).reshape(-1,4)
                frame_idx = control_point[:,0]
                control_pos = control_point[:,1:]

                frame_diff = frame_idx[1:] - frame_idx[:-1]
                # add lift hight for the start and end point
                control_pos[2:-1,2] = self.target_lift_height
                distance = np.linalg.norm(control_pos[1:] - control_pos[:-1], axis=1)
                high_thresh_frame = np.ceil(distance/high_thresh).astype(int)
                low_thresh_frame = np.floor(distance/low_thresh).astype(int)
                new_frame_idx = frame_idx.copy()
                for i in range(len(frame_diff)):
                    if frame_diff[i] < high_thresh_frame[i]:
                        new_frame_idx[i+1:] += high_thresh_frame[i] - frame_diff[i]
                    elif i != 0 and frame_diff[i] > low_thresh_frame[i]:
                        new_frame_idx[i+1:] += low_thresh_frame[i] - frame_diff[i]
                
                # also consider the distance between the human and object
                dist_human_object = np.linalg.norm(init_object_points_all - init_position[0].reshape(-1,3), axis=1).max()
                high_thresh_frame = np.ceil(dist_human_object/high_thresh).astype(int)
                low_thresh_frame = np.floor(dist_human_object/low_thresh).astype(int)
                init_diff = new_frame_idx[1] - new_frame_idx[0] 
                if init_diff < high_thresh_frame:
                    new_frame_idx[1:] += high_thresh_frame - init_diff
                elif init_diff > low_thresh_frame:
                    new_frame_idx[1:] += low_thresh_frame - init_diff            

                new_frame_idx = new_frame_idx.astype(int)
                number_frames = max(number_frames, new_frame_idx[-1])
                new_object_points = np.concatenate([new_frame_idx.reshape(-1,1), control_pos], axis=-1)
                new_object_points = new_object_points.tolist()
                object_points_list[obj_idx] = new_object_points
                
            # change control points to init position
            new_control_points = [[[0] + init_position[0].tolist()] for _ in range(len(control_points))]
        else:
            if control_points is not None:
                new_control_points = []
                for control_point in control_points:
                    control_point = np.array(control_point).reshape(-1,4)
                    frame_idx = control_point[:,0]
                    control_pos = control_point[:,1:]

                    frame_diff = frame_idx[1:] - frame_idx[:-1]
                    distance = np.linalg.norm(control_pos[1:] - control_pos[:-1], axis=1)
                    high_thresh_frame = np.ceil(distance/high_thresh).astype(int)
                    low_thresh_frame = np.floor(distance/low_thresh).astype(int)

                    new_frame_idx = frame_idx.copy()
                    for i in range(len(frame_diff)):
                        if frame_diff[i] < high_thresh_frame[i]:
                            new_frame_idx[i+1:] += high_thresh_frame[i] - frame_diff[i]    
                        elif frame_diff[i] > low_thresh_frame[i]:
                            new_frame_idx[i+1:] += low_thresh_frame[i] - frame_diff[i]
                    new_frame_idx = new_frame_idx.astype(int)
                    number_frames = max(number_frames, new_frame_idx[-1])
                    new_control_point = np.concatenate([new_frame_idx.reshape(-1,1), control_pos], axis=-1)
                    new_control_points.append(new_control_point.tolist())
            else:
                new_control_points = control_points

        return new_control_points, object_points_list, number_frames
    
    def generate_human_object(self, control_joints=None, control_points=None, text=None, number_frames=120, task_index=0, object_name=None, object_points=None):
        if number_frames <= 30:
            print('Number_frames <= 30. Please skip or merge small tasks!')
            # self.detect_object()
            return 'Number_frames <= 30 Please skip or merge small tasks!'
        if object_name is None or len(object_name) == 0:
            object_points = None
            object_name = None

            for i in range(len(control_joints)):
                if control_joints[i] == 'pelvis':
                    dist = np.linalg.norm(np.array(control_points[i][-1])[1:] - np.array(control_points[i][0])[1:])
                    if dist < 1:
                        print('Initial and final points are too close. Please skip or merge small tasks!')
                        # self.detect_object()
                        return 'Initial and final points are too close. Please skip or merge small tasks!'
                    else:
                        control_joints = ['pelvis']
                        control_points = [control_points[i]]
                        break

        task_index -= 1
        if task_index > self.task_index + 1:
            task_index = self.task_index + 1
        
        # load last sample data
        init_position, init_rotation, _, init_object_pos_all, init_object_rot_all, prev_joint_point, prev_object_point_all, pose_aa = self.load_prev_npy_data(task_index)
        
        if object_points is not None:
            if not isinstance(object_name, list):
                object_name = [object_name]

        ## load the environment state
        if task_index == 0:
            state = json.load(open(f'{self.opt.data_folder}/environment_state/{self.seq_name}_pidx_{self.p_idx}.json', 'r'))
        else:
            state = json.load(open(os.path.join(self.save_motion_folder, self.seq_name+"_pidx_"+str(self.p_idx)+"_task_"+str(task_index-1)+".json"), 'r'))

        last_k_frame = min(self.last_k_frame, init_position.shape[1])

        if object_points is not None:
            new_object_name = []
            new_object_points = []
            for i, obj_point in enumerate(object_points):
                obj_point = np.array(obj_point).reshape(-1,4).tolist()
                # add init object point
                if 0 not in np.array(obj_point)[:,0]:
                    obj_point = [[0] + list(init_object_pos_all[object_name[i]][0,-1])] + obj_point

                if obj_point[0][1] - obj_point[1][1] > 0.01 or obj_point[0][2] - obj_point[1][2] > 0.01:
                    obj_point = [[1] + obj_point[0][1:]] + obj_point
                # sort object points by frame id
                obj_point = sorted(obj_point, key=lambda x: x[0])

                is_moving = False
                for point in np.array(obj_point)[:,1:]:
                    if np.sum((init_object_pos_all[object_name[i]][0,-1] - point)**2) > 0.01:
                        is_moving = True
                        break
                if is_moving:
                    new_object_name.append(object_name[i])
                    new_object_points.append(obj_point)
            if len(new_object_name) == 0:
                object_points = None
                object_name = None
            else:
                object_name = new_object_name
                object_points = new_object_points
        else:
            for i in range(len(control_joints)):
                if control_joints[i] == 'pelvis':
                    for j in range(len(control_points[i])):
                        control_points[i][j][3] = init_position[0,-1,0,2]

        control_points, object_points, number_frames = self.adjust_frames(control_points, object_points, number_frames, init_position[0,-1])

        number_frames += last_k_frame
        
        # quick check collision for human waypoints
        if object_points is None and self.check_collision:
            res = self.quick_check_human_collision(control_joints, control_points, number_frames - last_k_frame + 1, init_position, state)
            if res is not None:
                return res
        
        if object_points is not None:
            object_motion_list = []
            object_motion_all_list = []
            movement_start_all = [0]
            movement_stop_all = [0]
            object_points_new = []
            for obj_idx, name in enumerate(object_name):
                # set up control points
                init_object_pos = torch.from_numpy(init_object_pos_all[name]).float().reshape(1, -1, 3)
                init_object_rot = torch.from_numpy(init_object_rot_all[name]).float().reshape(1, -1, 4)
                obj_points = torch.from_numpy(np.array(object_points[obj_idx])).float().reshape(-1, 4)
                init_object_rot_mat = transforms.quaternion_to_matrix(init_object_rot).reshape(1, -1, 9)
                object_motion = torch.zeros((1, number_frames-last_k_frame+1, 3+9))
                for points in obj_points:
                    object_motion[:, points[0].long(), :3] = points[1:]
                object_motion[:,:1,:3] = init_object_pos[0,-1:]
                object_motion[:,:1,3:] = init_object_rot_mat[0,-1:]

                key_frame = np.array(obj_points, dtype=np.int32)[:,0].tolist()

                object_forward = torch.from_numpy(init_object_pos_all[name].reshape(-1,3)[-1] - init_position[0,-1].reshape(-1,3)[0])
                object_forward = object_forward.reshape(3)[:2]

                # object interpolation on initial z-up canonique
                object_motion, movement_start, movement_stop = interpolation_object_motion_new(object_motion.clone(), 
                                            key_frame=key_frame.copy(), forward=object_forward.to(self.device), rotate=len(object_name)<2)

                object_motion_list.append(object_motion.clone())

                # add initial object motion
                if last_k_frame != 1:
                    movement_start[0] += last_k_frame-1
                    movement_stop[0] += last_k_frame-1
                    key_frame = np.array(obj_points, dtype=np.int32)[:,0] + last_k_frame-1
                    key_frame = key_frame.tolist()
                    object_motion_init = torch.zeros((1, last_k_frame-1, 3+9)).float()
                    object_motion_init[:,:,:3] = init_object_pos[0,-last_k_frame+1:]
                    object_motion_init[:,:,3:] = init_object_rot_mat[0,-last_k_frame+1:]
                    object_motion = torch.cat([object_motion_init, object_motion], dim=1)

                object_motion_all_list.append(object_motion.clone())
                movement_start_all[0] = max(movement_start_all[0], movement_start[0])
                movement_stop_all[0] = max(movement_stop_all[0], movement_stop[0])
                object_points_new.append(obj_points)

            movement_start = movement_start_all
            movement_stop = movement_stop_all
            object_points = object_points_new

            # quick check collision for object waypoints
            if self.check_collision:
                object_motion = torch.cat(object_motion_list, dim=0).mean(dim=0, keepdim=True)
                res = self.quick_check_object_collision(object_motion, state, object_name, object_points)
                if res:
                    # self.detect_object()
                    return res
 

        # canonicalize the control points
        if object_points is not None:
            ori_object_pos_list = []
            ori_object_rot_quan_list = []
            for obj_idx, name in enumerate(object_name):
                object_motion = object_motion_all_list[obj_idx]
                ori_object_pos = object_motion[:,:,:3].clone().numpy()
                ori_object_rot_mat = object_motion[:,:,3:].clone().reshape(1,-1,3,3)
                ori_object_rot_quan = transforms.matrix_to_quaternion(ori_object_rot_mat).numpy()
                ori_object_pos_list.append(ori_object_pos)
                ori_object_rot_quan_list.append(ori_object_rot_quan)

            X, Q, new_control_points, yrot, trans2joint_list, new_object_pos_list, object_rot_mat_list = \
                self.rotate_at_frame_w_obj(init_position, init_rotation, control_points, n_past=init_position.shape[1]-last_k_frame+1
                        , y_up=self.y_up, object_pos_list=ori_object_pos_list.copy(), object_rot_list=ori_object_rot_quan_list.copy(), 
                        last_k_frame=last_k_frame, control_joint=control_joints)    
        else:
            X, Q, new_control_points, yrot, trans2joint_list = self.rotate_at_frame_w_obj(init_position, init_rotation, control_points, n_past=init_position.shape[1]-last_k_frame+1
                                                                                      , y_up=self.y_up, last_k_frame=last_k_frame, control_joint=control_joints)

        # interpolate new control points
        new_control_points_interpolate = []
        for control_idx, new_control_point in enumerate(new_control_points):
            control_joint_idx = joint_map[control_joints[control_idx]]
            key_times_init = np.arange(last_k_frame, dtype=int).tolist()
            key_pos_init = X[0,-last_k_frame:,control_joint_idx].tolist()
            new_key_times = np.array(new_control_point)[:,0].astype(int).tolist()
            new_key_pos = np.array(new_control_point)[:,1:].tolist()

            key_times = key_times_init.copy()
            key_poses = key_pos_init.copy()
            for key_time, key_pos in zip(new_key_times, new_key_pos):
                if key_time not in key_times:
                    key_times.append(key_time)
                    key_poses.append(key_pos)

            key_times_arr = np.arange(key_times[-1]-key_times[0]+1, dtype=int).reshape(-1,1)
            key_poses_arr = np.zeros((key_times[-1]-key_times[0]+1, 3))
            if key_times_arr.shape[0] != 1:
                for b in range(1):
                    for i in range(3): 
                        interp_func = PchipInterpolator(key_times, np.array(key_poses)[:,i])
                        t_interp = np.linspace(key_times[0], key_times[-1], key_times[-1]-key_times[0]+1)
                        interp = interp_func(t_interp)
                        key_poses_arr[:,i] = interp
            new_control_points_interpolate.append(np.concatenate([key_times_arr, key_poses_arr], axis=1).tolist())
        
        new_control_points = new_control_points_interpolate

        if object_points is not None:
            # get the contact points (currently only for hands)
            possible_contact_points_list = []
            for obj_idx, name in enumerate(object_name):
                possible_contact_points, my_seq_com_pos, my_obj_rot_mat, my_obj_mesh_verts = \
                    self.get_pred_object_mesh(new_object_pos_list[obj_idx], object_rot_mat_list[obj_idx], object_name=name, human_pos=X[0,-1,22:24])
            
                # change the hand height to self.target_lift_height when lifting the object
                height_offset = -(possible_contact_points[0,(movement_start[0]+movement_stop[0])//2,:,self.height_idx].max().item() - self.target_lift_height)
                object_motion = object_motion_all_list[obj_idx].clone()
                key_z = object_motion[:, :, 2].clone()
                for b in range(1):
                    start_idx = movement_start[b]
                    stop_idx = movement_stop[b]
                    key_z[b, :start_idx+1] = object_motion[b, 0, 2]  # Before movement, keep initial height
                    key_z[b, start_idx+1:stop_idx] = object_motion[b, start_idx+1:stop_idx, 2] + height_offset  # During movement, add lift height
                    key_z[b, stop_idx:] = object_motion[b, -1, 2]  # After movement, maintain last height
                key_z = key_z[:, key_frame].detach().cpu().numpy()
                interpolated_object_motion_ori = object_motion.clone()
                # interpolation again on z axis
                for b in range(1):
                    interp_func = PchipInterpolator(key_frame, key_z[b, :])
                    t_interp = np.linspace(key_frame[0], key_frame[-1], key_frame[-1]-key_frame[0]+1)
                    interp = interp_func(t_interp)
                    object_motion[b, key_frame[0]:key_frame[-1]+1, 2] = torch.tensor(interp, dtype=torch.float32)
                
                # change all the other variable height
                diff_height = object_motion[:,:,2:3] - interpolated_object_motion_ori[:,:,2:3]
                
                object_motion_all_list[obj_idx][:,:,2:3] += diff_height.to(object_motion_all_list[obj_idx].device)
                possible_contact_points[:,:,:,self.height_idx] += diff_height.to(possible_contact_points.device)

                possible_contact_points_list.append(possible_contact_points)
            
            # find the closest contact points
            if len(possible_contact_points_list) > 1:
                init_contact_points0 = possible_contact_points_list[0][0,0]
                init_contact_points1 = possible_contact_points_list[1][0,0]

                # left 0 and right 1
                diff_contact0 = np.linalg.norm(init_contact_points0[0].cpu().numpy() - X[0,-1,22]) + \
                                np.linalg.norm(init_contact_points1[0].cpu().numpy() - X[0,-1,23])
                diff_contact1 = np.linalg.norm(init_contact_points1[0].cpu().numpy() - X[0,-1,22]) + \
                                np.linalg.norm(init_contact_points0[0].cpu().numpy() - X[0,-1,23])
                possible_contact_points = possible_contact_points_list[0].clone()
                if diff_contact0 < diff_contact1:
                    possible_contact_points[0, :, 0] = possible_contact_points_list[0][0, :, 0]
                    possible_contact_points[0, :, 1] = possible_contact_points_list[1][0, :, 1]
                else:
                    possible_contact_points[0, :, 0] = possible_contact_points_list[1][0, :, 0]
                    possible_contact_points[0, :, 1] = possible_contact_points_list[0][0, :, 1]
            else:
                possible_contact_points = possible_contact_points_list[0]

            # todo modify other variables
            # my_seq_com_pos[:,:,1:2] += diff_height
            # my_obj_mesh_verts[:,:,:,2] += diff_height
            
            # interpolate hands from frame last_k_frame
            possible_contact_points[0,:last_k_frame] = torch.from_numpy(X[0,-last_k_frame:,22:24]).to(possible_contact_points.device)
            inter_end = movement_start[0]
            if inter_end != last_k_frame-1:
                key_times = [last_k_frame-1, inter_end]
                for b in range(1):
                    for joint_idx in [0, 1]:
                        for i in range(3): 
                            interp_func = PchipInterpolator(key_times, possible_contact_points[b, key_times, joint_idx, i].cpu().numpy())
                            t_interp = np.linspace(key_times[0], key_times[-1], inter_end+1)
                            interp = interp_func(t_interp)
                            possible_contact_points[b, :inter_end+1, joint_idx, i] = torch.tensor(interp, device=self.device, dtype=torch.float32)

            # overwrite the contact point generate by agent
            if self.use_sample_contact_point:
                sample_contact_point = possible_contact_points.cpu().numpy()
                frame_id = np.arange(number_frames).reshape(1, number_frames, 1, 1)
                frame_id = np.repeat(frame_id, sample_contact_point.shape[2], axis=2)
                sample_contact_point = np.concatenate([frame_id, sample_contact_point], axis=3)
                
                control_joints_temp = []
                new_control_points_temp = []
                contact_labels = torch.zeros((1, number_frames, 4), dtype=torch.float32).to(self.device)
                
                hand_dist = np.linalg.norm(sample_contact_point[0,movement_start[0]:movement_start[0]+1,0][0,1:] - sample_contact_point[0,movement_start[0]:movement_start[0]+1,1][0,1:])

                # temperally use 2 hands
                control_stop_idx = number_frames # movement_stop[0] # even if te object stops, still put hands on it
                if hand_dist > 0.0:
                    contact_labels[:,:control_stop_idx, 0] = 1
                    control_joints_temp.append('left_hand')
                    new_control_points_temp.append(sample_contact_point[0,:control_stop_idx,0].tolist())

                contact_labels[:,:control_stop_idx, 1] = 1
                control_joints_temp.append('right_hand')
                new_control_points_temp.append(sample_contact_point[0,:control_stop_idx,1].tolist())
                
                control_joints = control_joints_temp
                new_control_points = new_control_points_temp

        # create template for maskedmimic
        root_aa = transforms.quaternion_to_axis_angle(torch.from_numpy(Q))[:,-last_k_frame:]
        new_pose_aa = pose_aa[:, -last_k_frame:, :, :].copy()
        new_pose_aa[:,:,0:1] = root_aa.numpy()
        new_pose_aa = new_pose_aa
        new_trans = X[:,-last_k_frame:, 0]
        trans = np.zeros((number_frames+2, 3)) # maskedmimic will lose 2 frames
        poses = np.zeros((number_frames+2, 22, 3))

        trans[:last_k_frame] = new_trans[0].copy()
        poses[:last_k_frame] = new_pose_aa[0].copy()
        # padding the rest
        trans[last_k_frame:] = trans[last_k_frame-1:last_k_frame]
        poses[last_k_frame:] = poses[last_k_frame-1:last_k_frame]

        poses = poses.reshape(number_frames+2,-1)
        os.makedirs(os.path.join(self.template_npz_folder, f"task_{task_index}"), exist_ok=True)

        # process control points  
        os.makedirs(os.path.join(self.temp_control_points_folder, f"task_{task_index}"), exist_ok=True)
        if object_points is not None:
            # temporally only for hands
            temp_control_pos = np.zeros((number_frames+2, 2, 3))
            temp_control_pos[:number_frames] = np.array(new_control_points)[:,:,1:].transpose(1, 0, 2)
            temp_control_pos[number_frames:] = temp_control_pos[number_frames-1:number_frames]
            np.savez(os.path.join(self.temp_control_points_folder, f"task_{task_index}", f"{task_index}.npz"), 
                control_idx=[22,23], 
                positions=temp_control_pos
            )
            control_setting = 'hands'
        else:
            # temporally only for pelvis
            for contriol_id, (control_name, new_control_point) in enumerate(zip(control_joints, new_control_points)):
                if control_name == 'pelvis':
                    temp_control_pos = np.zeros((number_frames+2, 1, 3))

                    temp_control_pos[:number_frames] = np.array(new_control_point)[:,1:].reshape(-1,1,3)
                    temp_control_pos[number_frames:] = temp_control_pos[number_frames-1:number_frames]
                    np.savez(os.path.join(self.temp_control_points_folder, f"task_{task_index}", f"{task_index}.npz"), 
                        control_idx=[0], 
                        positions=temp_control_pos
                    )

                    # get root rotation
                    root_mat_all = transforms.axis_angle_to_matrix(torch.from_numpy(poses)[:,:3]).reshape(1, -1, 9).to(self.device)
                    root_trans = torch.from_numpy(temp_control_pos).reshape(1, -1, 3).to(self.device)
                    
                    keyframes_pelvis = np.arange(last_k_frame).tolist() + \
                        (np.array(control_points[contriol_id], dtype=int)[:,0]+last_k_frame-1).tolist() + [root_trans.shape[1]-1]
                    keyframes_pelvis = set(keyframes_pelvis)
                    keyframes_pelvis = sorted(list(keyframes_pelvis))

                    movement_start_pelvis = [1]
                    movement_stop_pelvis = [root_trans.shape[1]-1]
                    rotation_change_mask_pelvis = torch.ones((1, len(keyframes_pelvis)-1)).to(self.device)

                    root_mat_all_new = interpolate_rotations(root_mat_all, root_trans, keyframes_pelvis, movement_start_pelvis, movement_stop_pelvis,
                                          rotation_change_mask_pelvis)

                    root_aa = transforms.matrix_to_axis_angle(root_mat_all_new.reshape(-1,3,3)).cpu().numpy()
                    poses[:,:3] = root_aa.reshape(-1, 3)
                    poses[-2:,:3] = poses[-3:-2,:3]  # padding the last 2 frames
                    trans = temp_control_pos.reshape(-1,3).copy()
            control_setting = 'pelvis_position'

        np.savez(os.path.join(self.template_npz_folder, f"task_{task_index}", f"{task_index}.npz"),
                 trans=trans, poses=poses, mocap_framerate=30, betas=self.betas[0], text=text,gender=self.gender)
        
        input_folder = os.path.abspath(self.template_npz_folder)
        input_smpl_path = os.path.abspath(os.path.join(self.template_npz_folder, f"task_{task_index}", f"{task_index}.npz"))
        input_smpl_path = input_smpl_path.replace('.npz', '.npy').replace(f"/task_{task_index}/", f"/task_{task_index}-smpl/")
        input_control_path = os.path.abspath(os.path.join(self.temp_control_points_folder, f"task_{task_index}"))
        output_path = os.path.abspath(os.path.join(self.maskedmimic_res_folder, f"task_{task_index}"))

        # retarget npz file and run maskedmimic
        print(f"bash scripts/run_maskedmimic.sh {self.model_path} {control_setting} {input_folder} {input_smpl_path} {input_control_path} {output_path}")
        subprocess.call(f"bash scripts/run_maskedmimic.sh {self.model_path} {control_setting} {input_folder} {input_smpl_path} {input_control_path} {output_path}", 
                        shell=True, cwd='third-party/ProtoMotion_for_InterPose',stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL) 

        # load output
        maskedmimic_output_file = os.path.join(os.path.join(self.maskedmimic_res_folder, f"task_{task_index}"), "trajectory_pose_aa_0_0.npz")
        maskedmimic_output = np.load(maskedmimic_output_file)

        maskedmimic_rot_aa = torch.from_numpy(maskedmimic_output['pose']).float().to(self.device).reshape(1, -1, 24, 3)[:,:number_frames,:22]
        maskedmimic_trans = torch.from_numpy(maskedmimic_output['trans']).float().to(self.device)[:,:number_frames]
        betas = torch.from_numpy(self.betas).float().to(self.device)
        mesh_jnts, mesh_verts, mesh_faces = run_smplx_model(maskedmimic_trans, maskedmimic_rot_aa, betas, [self.gender], self.bm_dict)
        
        diff = maskedmimic_trans[0,0:1] - mesh_jnts[0,0,0:1]
        final_sample = mesh_jnts + diff
        # mesh_verts = mesh_verts + diff
        
        ## transform back to original coordinate
        final_sample = final_sample.cpu().numpy()

        # canoniclize the position using init_rotation
        final_sample = quat_mul_vec(yrot, final_sample)
        final_sample = final_sample - trans2joint_list[:, np.newaxis, np.newaxis, :]

        # swap y and z for root_quat
        root_aa = maskedmimic_rot_aa[:,:,0:1].clone()
        root_quat = transforms.axis_angle_to_quaternion(root_aa).cpu().numpy()
        # transform root_quat 
        root_quat = quat_mul(yrot, root_quat).reshape(1, -1, 1, 4)
        root_aa = transforms.quaternion_to_axis_angle(torch.from_numpy(root_quat).float().to(self.device))
        
        maskedmimic_rot_aa[:,:,0:1] = root_aa
        maskedmimic_rot_aa = maskedmimic_rot_aa.cpu().numpy()

        all_object_data = {}
        for name in init_object_pos_all:
            if object_points is not None and object_name is not None and name in object_name:
                object_motion = object_motion_all_list[object_name.index(name)]
                object_motion_pos = object_motion[:,:,:3].numpy().copy()
                object_motion_mat = object_motion[:,:,3:].reshape(1, -1, 3, 3).clone()
                object_motion_quan = transforms.matrix_to_quaternion(object_motion_mat).numpy()

                cur_object_points = np.array(object_points[object_name.index(name)])
                cur_object_points[:,0] += init_position.shape[1] - last_k_frame
                cur_object_points = cur_object_points.reshape(1, -1, 4)
            else:
                object_motion_pos = np.repeat(init_object_pos_all[name][:,-1:], number_frames, axis=1)
                object_motion_quan = np.repeat(init_object_rot_all[name][:,-1:], number_frames, axis=1)

                cur_object_points = np.array([]).reshape(1, -1, 4)
            
            all_object_data[name] = {
                "xyz": np.concatenate([init_object_pos_all[name][:,:-last_k_frame+1], object_motion_pos], axis=1), # z-up
                "rotation": np.concatenate([init_object_rot_all[name][:,:-last_k_frame+1], object_motion_quan], axis=1), # z-up
                "object_point": np.concatenate([prev_object_point_all[name], cur_object_points], axis=1), # z-up
            }

        flatten_control_points = []
        for control_point in new_control_points:
            flatten_control_points += control_point
        cur_joint_point = np.array(flatten_control_points)
        cur_joint_point[:,0] += init_position.shape[1] - last_k_frame
        cur_joint_point[:,1:] =  quat_mul_vec(yrot[:, 0, 0, :], cur_joint_point[:,1:])
        cur_joint_point[:,1:] = cur_joint_point[:,1:] - trans2joint_list[:, :]
        cur_joint_point = cur_joint_point.reshape(1, -1, 4)

        # save the motion data (human: joint_xyz, rotation, humanml3d; object: xyz, rotation)
        np.save(os.path.join(self.save_motion_folder, self.seq_name+"_pidx_"+str(self.p_idx)+"_task_"+str(task_index)+".npy"), 
        {
            "human": {
                "joint_xyz": np.concatenate([init_position[:,:-last_k_frame+1,:24], final_sample], axis=1), # z-up
                "orientation": np.concatenate([init_rotation[:,:-last_k_frame+1], root_quat], axis=1), # z-up
                "joint_point": np.concatenate([prev_joint_point, cur_joint_point], axis=1), # z-up
                "pose_aa": np.concatenate([pose_aa[:,:-last_k_frame+1], maskedmimic_rot_aa], axis=1)  # z-up
            },
                "object": all_object_data
        })

        # get pelvis, head, left hand, right hand, left foot, right foot
        init_pelvis_pos = final_sample[0,-1,0]
        init_left_hand_pos = final_sample[0,-1,22]
        init_right_hand_pos = final_sample[0,-1,23]

        init_human_mat = transforms.quaternion_to_matrix(torch.from_numpy(root_quat[0,0,0]))
        init_human_rot_euler = transforms.matrix_to_euler_angles(init_human_mat, "XYZ").numpy()

        state['human'] = {
            "pelvis": round_list(init_pelvis_pos.tolist()),
            "left_hand": round_list(init_left_hand_pos.tolist()),
            "right_hand": round_list(init_right_hand_pos.tolist()),
            "orientation": round_list(init_human_rot_euler.tolist())
        }
        for object in state['objects']:
            if object_name is not None and object['name'] in object_name:
                object['position'] = round_list(all_object_data[object['name']]["xyz"][0,-1].tolist())

                init_object_mat = transforms.quaternion_to_matrix(torch.from_numpy(all_object_data[object['name']]["rotation"][0,-1]))
                init_object_rot_euler = transforms.matrix_to_euler_angles(init_object_mat, "XYZ").numpy()
                object["orientation"] = round_list(init_object_rot_euler.tolist())

        # save the environment state
        json.dump(state, open(os.path.join(self.save_motion_folder, self.seq_name+"_pidx_"+str(self.p_idx)+"_task_"+str(task_index)+".json"), 'w'), indent=4)   
        self.task_index = task_index
        self.check_collision = self.opt.check_collision
        self.target_lift_height = 1.2
        self.detect_object()
        return 
    

if __name__ == "__main__":
    from HOI_agent.utils.config_utils import parse_opt
    opt = parse_opt()
    # single object
    model = protomotion_model(device='cuda', batch_size=1, vis_fps=15, save_motion_root='chois_output/multi_agent_mask', last_k_frame=10, opt=opt)

