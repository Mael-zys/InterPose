import argparse
import os
import numpy as np
import random 
import json 

from pathlib import Path

import wandb

import torch
from torch.cuda.amp import GradScaler
from torch.utils import data


import pytorch3d.transforms as transforms 

import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from manip.data.cano_traj_dataset import CanoObjectTrajDataset
from manip.data.unseen_obj_long_cano_traj_dataset import UnseenCanoObjectTrajDataset 
from manip.data.behave_obj_long_cano_traj_dataset import BehaveCanoObjectTrajDataset 
from manip.data.cano_traj_dataset_behave import CanoObjectTrajDataset_behave 

import random
from scipy.interpolate import PchipInterpolator
from utils.process_mesh_files import sample_point_trajectory, interpolation_object_motion_new

seed=1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


def run_smplx_model(root_trans, aa_rot_rep, betas, gender, bm_dict, return_joints24=True):
    # root_trans: BS X T X 3
    # aa_rot_rep: BS X T X 22 X 3 
    # betas: BS X 16
    # gender: BS 
    bs, num_steps, num_joints, _ = aa_rot_rep.shape
    if num_joints != 52:
        padding_zeros_hand = torch.zeros(bs, num_steps, 30, 3).to(aa_rot_rep.device) # BS X T X 30 X 3 
        aa_rot_rep = torch.cat((aa_rot_rep, padding_zeros_hand), dim=2) # BS X T X 52 X 3 

    aa_rot_rep = aa_rot_rep.reshape(bs*num_steps, -1, 3) # (BS*T) X n_joints X 3 
    betas = betas[:, None, :].repeat(1, num_steps, 1).reshape(bs*num_steps, -1) # (BS*T) X 16 
    gender = np.asarray(gender)[:, np.newaxis].repeat(num_steps, axis=1)
    gender = gender.reshape(-1).tolist() # (BS*T)

    smpl_trans = root_trans.reshape(-1, 3) # (BS*T) X 3  
    smpl_betas = betas # (BS*T) X 16
    smpl_root_orient = aa_rot_rep[:, 0, :] # (BS*T) X 3 
    smpl_pose_body = aa_rot_rep[:, 1:22, :].reshape(-1, 63) # (BS*T) X 63
    smpl_pose_hand = aa_rot_rep[:, 22:, :].reshape(-1, 90) # (BS*T) X 90 

    B = smpl_trans.shape[0] # (BS*T) 

    smpl_vals = [smpl_trans, smpl_root_orient, smpl_betas, smpl_pose_body, smpl_pose_hand]
    # batch may be a mix of genders, so need to carefully use the corresponding SMPL body model
    gender_names = ['male', 'female', "neutral"]
    pred_joints = []
    pred_verts = []
    prev_nbidx = 0
    cat_idx_map = np.ones((B), dtype=np.int64)*-1
    for gender_name in gender_names:
        gender_idx = np.array(gender) == gender_name
        nbidx = np.sum(gender_idx)

        cat_idx_map[gender_idx] = np.arange(prev_nbidx, prev_nbidx + nbidx, dtype=np.int64)
        prev_nbidx += nbidx

        gender_smpl_vals = [val[gender_idx] for val in smpl_vals]

        if nbidx == 0:
            # skip if no frames for this gender
            continue
        
        # reconstruct SMPL
        cur_pred_trans, cur_pred_orient, cur_betas, cur_pred_pose, cur_pred_pose_hand = gender_smpl_vals
        bm = bm_dict[gender_name]

        pred_body = bm(pose_body=cur_pred_pose, pose_hand=cur_pred_pose_hand, \
                betas=cur_betas, root_orient=cur_pred_orient, trans=cur_pred_trans)
        
        pred_joints.append(pred_body.Jtr)
        pred_verts.append(pred_body.v)

    # cat all genders and reorder to original batch ordering
    if return_joints24:
        x_pred_smpl_joints_all = torch.cat(pred_joints, axis=0) # () X 52 X 3 
        lmiddle_index= 28 
        rmiddle_index = 43 
        x_pred_smpl_joints = torch.cat((x_pred_smpl_joints_all[:, :22, :], \
            x_pred_smpl_joints_all[:, lmiddle_index:lmiddle_index+1, :], \
            x_pred_smpl_joints_all[:, rmiddle_index:rmiddle_index+1, :]), dim=1) 
    else:
        x_pred_smpl_joints = torch.cat(pred_joints, axis=0)[:, :num_joints, :]
        
    x_pred_smpl_joints = x_pred_smpl_joints[cat_idx_map] # (BS*T) X 22 X 3 

    x_pred_smpl_verts = torch.cat(pred_verts, axis=0)
    x_pred_smpl_verts = x_pred_smpl_verts[cat_idx_map] # (BS*T) X 6890 X 3 

    
    x_pred_smpl_joints = x_pred_smpl_joints.reshape(bs, num_steps, -1, 3) # BS X T X 22 X 3/BS X T X 24 X 3  
    x_pred_smpl_verts = x_pred_smpl_verts.reshape(bs, num_steps, -1, 3) # BS X T X 6890 X 3 

    mesh_faces = pred_body.f 
    
    return x_pred_smpl_joints, x_pred_smpl_verts, mesh_faces 

def cycle(dl):
    while True:
        for data in dl:
            yield data

class Trainer(object):
    def __init__(
        self,
        opt,
        diffusion_model,
        *,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=10000000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        ema_update_every=10,
        save_and_sample_every=40000,
        results_folder='./results',
        use_wandb=True, 
        args=None,
        diffusion=None 
    ):
        super().__init__()

        self.use_wandb = use_wandb  
        self.args = args       
        if self.use_wandb:
            # Loggers
            wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity, \
            name=opt.exp_name, dir=opt.save_dir)


        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps


        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)

        self.results_folder = results_folder

        self.vis_folder = results_folder.replace("weights", "vis_res")

        self.opt = opt 

        self.window = opt.window

        self.add_language_condition = self.opt.add_language_condition 

        self.use_random_frame_bps = self.opt.use_random_frame_bps 

        self.use_object_keypoints = self.opt.use_object_keypoints 

        self.add_semantic_contact_labels = self.opt.add_semantic_contact_labels 

        self.test_unseen_objects = self.opt.test_unseen_objects 

        self.test_behave = self.opt.test_behave 
        self.test_behave_motion = self.opt.test_behave_motion 

        self.save_res_folder = self.opt.save_res_folder 

        self.use_object_split = self.opt.use_object_split
        self.data_root_folder = self.opt.data_root_folder 
        self.prep_dataloader(window_size=opt.window)

        self.bm_dict = self.ds.bm_dict 

        self.test_on_train = self.opt.test_on_train 

        self.input_first_human_pose = self.opt.input_first_human_pose 

        self.use_guidance_in_denoising = self.opt.use_guidance_in_denoising 

        self.compute_metrics = self.opt.compute_metrics 

        self.loss_w_feet = self.opt.loss_w_feet 
        self.loss_w_fk = self.opt.loss_w_fk 
        self.loss_w_obj_pts = self.opt.loss_w_obj_pts 

        self.use_long_planned_path = self.opt.use_long_planned_path 
        self.test_object_name = self.opt.test_object_name 
        self.test_scene_name = self.opt.test_scene_name 
       

        self.hand_vertex_idxs, self.left_hand_vertex_idxs, self.right_hand_vertex_idxs = self.load_hand_vertex_ids() 
        
        if self.test_unseen_objects:
            if self.use_long_planned_path:
                test_long_seq = True 
            else:
                test_long_seq = False 

            self.unseen_seq_ds = UnseenCanoObjectTrajDataset(train=False, \
                data_root_folder=self.data_root_folder, \
                window=opt.window, use_object_splits=self.use_object_split, \
                input_language_condition=self.add_language_condition, \
                use_first_frame_bps=False, \
                use_random_frame_bps=self.use_random_frame_bps, \
                test_long_seq=test_long_seq, use_subset=opt.use_sub_dataset) 
        
        if self.test_behave:
            if self.use_long_planned_path:
                test_long_seq = True 
            else:
                test_long_seq = False 

            self.behave_seq_ds = BehaveCanoObjectTrajDataset(train=False, \
                data_root_folder=self.data_root_folder, \
                window=opt.window, use_object_splits=self.use_object_split, \
                input_language_condition=self.add_language_condition, \
                use_first_frame_bps=False, \
                use_random_frame_bps=self.use_random_frame_bps, \
                test_long_seq=test_long_seq, use_subset=opt.use_sub_dataset) 

        if self.test_behave_motion:
            if self.use_long_planned_path:
                test_long_seq = True 
            else:
                test_long_seq = False 

            self.behave_motion_seq_ds = CanoObjectTrajDataset_behave(train=False, \
                data_root_folder=self.data_root_folder, \
                window=opt.window, use_object_splits=self.use_object_split, \
                input_language_condition=self.add_language_condition, \
                use_random_frame_bps=self.use_random_frame_bps, \
                use_subset=opt.use_sub_dataset)
            
    def load_hand_vertex_ids(self):
        data_folder = "data/part_vert_ids"
        left_hand_npy_path = os.path.join(data_folder, "left_hand_vids.npy")
        right_hand_npy_path = os.path.join(data_folder, "right_hand_vids.npy")

        left_hand_vids = np.load(left_hand_npy_path)
        right_hand_vids = np.load(right_hand_npy_path) 

        hand_vids = np.concatenate((left_hand_vids, right_hand_vids), axis=0)

        return hand_vids, left_hand_vids, right_hand_vids  


    def load_object_sdf_data(self, object_name):
        if self.test_unseen_objects:
            data_folder = os.path.join(self.data_root_folder, "unseen_objects_data/selected_rotated_zeroed_obj_sdf_256_npy_files")
            sdf_npy_path = os.path.join(data_folder, object_name+".npy")
            sdf_json_path = os.path.join(data_folder, object_name+".json")
        elif self.test_behave:
            data_folder = os.path.join(self.data_root_folder, "behave_objects_data/selected_rotated_zeroed_obj_sdf_256_npy_files")
            sdf_npy_path = os.path.join(data_folder, object_name+".npy")
            sdf_json_path = os.path.join(data_folder, object_name+".json")
        elif self.test_behave_motion:
            data_folder = os.path.join(self.data_root_folder, "behave_objects_data/selected_rotated_zeroed_obj_sdf_256_npy_files")
            sdf_npy_path = os.path.join(data_folder, object_name+".npy")
            sdf_json_path = os.path.join(data_folder, object_name+".json")
        else:
            data_folder = os.path.join(self.data_root_folder, "rest_object_sdf_256_npy_files") 
            sdf_npy_path = os.path.join(data_folder, object_name+".ply.npy")
            sdf_json_path = os.path.join(data_folder, object_name+".ply.json")

        sdf = np.load(sdf_npy_path) # 256 X 256 X 256 
        sdf_json_data = json.load(open(sdf_json_path, 'r'))

        sdf_centroid = np.asarray(sdf_json_data['centroid']) # a list with 3 items -> 3 
        sdf_extents = np.asarray(sdf_json_data['extents']) # a list with 3 items -> 3 

        sdf = torch.from_numpy(sdf).float()[None].cuda()
        sdf_centroid = torch.from_numpy(sdf_centroid).float()[None].cuda()
        sdf_extents = torch.from_numpy(sdf_extents).float()[None].cuda() 

        return sdf, sdf_centroid, sdf_extents


    def prep_dataloader(self, window_size):
        # Define dataset
        train_dataset = CanoObjectTrajDataset(train=True, data_root_folder=self.data_root_folder, \
            window=window_size, use_object_splits=self.use_object_split, \
            input_language_condition=self.add_language_condition, \
            use_random_frame_bps=self.use_random_frame_bps, \
            use_object_keypoints=self.use_object_keypoints, use_subset=self.opt.use_sub_dataset)
        val_dataset = CanoObjectTrajDataset(train=False, data_root_folder=self.data_root_folder, \
            window=window_size, use_object_splits=self.use_object_split, \
            input_language_condition=self.add_language_condition, \
            use_random_frame_bps=self.use_random_frame_bps, \
            use_object_keypoints=self.use_object_keypoints, use_subset=self.opt.use_sub_dataset)

        self.ds = train_dataset 
        self.val_ds = val_dataset
        self.dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size, \
            shuffle=True, pin_memory=True, num_workers=4))
        self.val_dl = cycle(data.DataLoader(self.val_ds, batch_size=self.batch_size, \
            shuffle=False, pin_memory=True, num_workers=4))


    def prep_start_end_condition_mask_pos_only(self, data, actual_seq_len):
        # data: BS X T X D (3+9)
        # actual_seq_len: BS 
        tmp_mask = torch.arange(self.window).expand(data.shape[0], \
                self.window) == (actual_seq_len[:, None].repeat(1, self.window)-1)
                # BS X max_timesteps
        tmp_mask = tmp_mask.to(data.device)[:, :, None] # BS X T X 1

        # Missing regions are ones, the condition regions are zeros. 
        mask = torch.ones_like(data[:, :, :3]).to(data.device) # BS X T X 3
        mask = mask * (~tmp_mask) # Only the actual_seq_len frame is 0

        # Add rotation mask, only the first frame's rotation is given. 
        rotation_mask = torch.ones_like(data[:, :, 3:]).to(data.device)
        mask = torch.cat((mask, rotation_mask), dim=-1) 

        mask[:, 0, :] = torch.zeros(data.shape[0], data.shape[2]).to(data.device) # BS X D  

        return mask 
    
    def prep_mimic_A_star_path_condition_mask_pos_xy_only(self, data, actual_seq_len):
        # data: BS X T X D
        # actual_seq_len: BS 
        tmp_mask = torch.arange(self.window).expand(data.shape[0], \
                self.window) == (actual_seq_len[:, None].repeat(1, self.window)-1)
                # BS X max_timesteps
        tmp_mask = tmp_mask.to(data.device)[:, :, None] # BS X T X 1
        tmp_mask = (~tmp_mask)

        # Use fixed number of waypoints.
        random_steps = [30-1, 60-1, 90-1] 
        for selected_t in random_steps:
            if selected_t < self.window - 1:
                bs_selected_t = torch.from_numpy(np.asarray([selected_t])) # 1 
                bs_selected_t = bs_selected_t[None, :].repeat(data.shape[0], self.window) # BS X T 

                curr_tmp_mask = torch.arange(self.window).expand(data.shape[0], \
                    self.window) == (bs_selected_t)
                    # BS X max_timesteps
                curr_tmp_mask = curr_tmp_mask.to(data.device)[:, :, None] # BS X T X 1

                tmp_mask = (~curr_tmp_mask)*tmp_mask

        # Missing regions are ones, the condition regions are zeros. 
        mask = torch.ones_like(data[:, :, :2]).to(data.device) # BS X T X 2
        mask = mask * tmp_mask # Only the actual_seq_len frame is 0

        # Add rotation mask, only the first frame's rotation is given. 
        # Also, add z mask, only the first frane's z is given. 
        rotation_mask = torch.ones_like(data[:, :, 2:]).to(data.device)
        mask = torch.cat((mask, rotation_mask), dim=-1) 

        mask[:, 0, :] = torch.zeros(data.shape[0], data.shape[2]).to(data.device) # BS X D  

        return mask 


    def get_pred_object_mesh(self, object_motion, data_dict=None, curr_object_name=None, human_pos=None, only_mesh=False):
        bs = object_motion.shape[0]
        if self.use_random_frame_bps:
            reference_obj_rot_mat = data_dict['reference_obj_rot_mat'] # N X 1 X 3 X 3 
            pred_obj_rel_rot_mat = object_motion[:, :, 3:3+9].reshape(bs, -1, 3, 3) # N X T X 3 X 3

            if self.test_behave_motion:
                pred_obj_rot_mat = self.behave_motion_seq_ds.rel_rot_to_seq(pred_obj_rel_rot_mat, reference_obj_rot_mat)
            else:
                pred_obj_rot_mat = self.ds.rel_rot_to_seq(pred_obj_rel_rot_mat, reference_obj_rot_mat)
        else:
            pred_obj_rot_mat = object_motion[:, :, 3:3+9].reshape(bs, -1, 3, 3) # N X T X 3 X 3
        
        pred_seq_com_pos = object_motion[:, :, :3] # N X T X 3 
        sampled_points_list = []
        obj_mesh_verts_list = []
        for idx in range(bs):
            if curr_object_name is not None: 
                    object_name = curr_object_name[idx] 
            else:
                curr_seq_name = data_dict['seq_name'][0]
                object_name = data_dict['obj_name'][0]

            curr_obj_rot_mat = pred_obj_rot_mat[idx] # T X 3 X 3 
            curr_obj_quat = transforms.matrix_to_quaternion(curr_obj_rot_mat)
            curr_obj_rot_mat = transforms.quaternion_to_matrix(curr_obj_quat)
            if self.test_unseen_objects:
                # Get object verts 
                obj_rest_verts, obj_mesh_faces = self.unseen_seq_ds.load_rest_pose_object_geometry(object_name)
                obj_rest_verts = torch.from_numpy(obj_rest_verts)

                obj_mesh_verts = self.unseen_seq_ds.load_object_geometry_w_rest_geo(curr_obj_rot_mat, \
                            pred_seq_com_pos[idx], obj_rest_verts.float().to(pred_seq_com_pos.device))
            elif self.test_behave:
                # Get object verts 
                obj_rest_verts, obj_mesh_faces = self.behave_seq_ds.load_rest_pose_object_geometry(object_name)
                obj_rest_verts = torch.from_numpy(obj_rest_verts)

                obj_mesh_verts = self.behave_seq_ds.load_object_geometry_w_rest_geo(curr_obj_rot_mat, \
                            pred_seq_com_pos[idx], obj_rest_verts.float().to(pred_seq_com_pos.device))
            elif self.test_behave_motion:
                # Get object verts 
                obj_rest_verts, obj_mesh_faces = self.behave_motion_seq_ds.load_rest_pose_object_geometry(object_name)
                obj_rest_verts = torch.from_numpy(obj_rest_verts)

                obj_mesh_verts = self.behave_motion_seq_ds.load_object_geometry_w_rest_geo(curr_obj_rot_mat, \
                            pred_seq_com_pos[idx], obj_rest_verts.float().to(pred_seq_com_pos.device))
            else:
                # Get object verts 
                obj_rest_verts, obj_mesh_faces = self.ds.load_rest_pose_object_geometry(object_name)
                obj_rest_verts = torch.from_numpy(obj_rest_verts)

                obj_mesh_verts = self.ds.load_object_geometry_w_rest_geo(curr_obj_rot_mat, \
                            pred_seq_com_pos[idx], obj_rest_verts.float().to(pred_seq_com_pos.device))
            
            if not only_mesh:
                # init_center_z = pred_seq_com_pos[idx, 0, 2]
                max_height = obj_mesh_verts[0, :, 2].max()
                min_height = obj_mesh_verts[0, :, 2].min()
                height = max_height - min_height

                z_min = min_height + 0.0 * height if min_height > 1 else min_height + 0.3 * height
                z_max = min_height + 0.99 * height if max_height < 1.2 else min_height + 0.5 * height

                z_range = [z_min.item(), z_max.item()]

                # (T, 3)
                new_human_pos = human_pos.clone()
                new_human_pos[0,:2] += (human_pos[0, :2] - human_pos[1, :2])*3
                new_human_pos[1,:2] += (human_pos[1, :2] - human_pos[0, :2])*3
                sampled_points = sample_point_trajectory(obj_mesh_verts, obj_mesh_faces, z_range, new_human_pos[0], mode='nearest', height_idx=2)
                sampled_points2 = sample_point_trajectory(obj_mesh_verts, obj_mesh_faces, z_range, new_human_pos[1], mode='nearest', height_idx=2)
                sampled_points = torch.cat([sampled_points.unsqueeze(1),sampled_points2.unsqueeze(1)], dim=1)
                sampled_points_list.append(sampled_points.unsqueeze(0))

            obj_mesh_verts_list.append(obj_mesh_verts.unsqueeze(0))
        if only_mesh:
            return torch.concat(obj_mesh_verts_list, dim=0)
        # bs, T, n, 3
        return torch.concat(sampled_points_list, dim=0), pred_seq_com_pos, pred_obj_rot_mat, torch.concat(obj_mesh_verts_list, dim=0)

    def adjust_object_height_with_speed_control(
            self,
            interpolated_object_motion,
            possible_contact_points,
            movement_start,
            movement_stop,
            target_lift_height=1.2,
            max_speed_threshold=0.1
        ):
        """
        Adjust object height to ensure:
        1. Contact points during motion reach target_lift_height.
        2. The heights at movement_start and movement_stop remain unchanged.
        3. If required vertical speed exceeds max_speed_threshold, clamp speed at the threshold.
        4. Returns the full adjusted trajectory directly (no further interpolation needed).
        5. lift_frame and drop_frame are determined adaptively using max_speed_threshold.

        Args:
            interpolated_object_motion (torch.Tensor): Object motion trajectory [bs, frames, 3]
            possible_contact_points (torch.Tensor): Contact points [bs, frames, num_points, 3]
            movement_start (torch.Tensor): Motion start frame indices [bs]
            movement_stop (torch.Tensor): Motion end frame indices [bs]
            target_lift_height (float): Desired contact height
            max_speed_threshold (float): Maximum allowed vertical speed (units per frame)

        Returns:
            torch.Tensor: Adjusted full motion trajectory [bs, frames, 3]
        """
        bs = interpolated_object_motion.shape[0]
        total_frames = interpolated_object_motion.shape[1]
        device = interpolated_object_motion.device

        # Clone input to avoid modifying original tensor
        result_motion = interpolated_object_motion.clone()

        for b in range(bs):
            start_idx = movement_start[b]
            stop_idx = movement_stop[b]

            # --- Compute required height offset so that contact points hit target_lift_height ---
            mid_frame = (start_idx + stop_idx) // 2
            current_max_contact_height = possible_contact_points[b, mid_frame, :, 2].max().item()
            height_offset = target_lift_height - current_max_contact_height

            # Fixed reference heights (start/end must remain unchanged)
            initial_height = result_motion[b, 0, 2].item()
            final_height = result_motion[b, -1, 2].item()
            start_height = result_motion[b, 0, 2].item()
            stop_height = result_motion[b, -1, 2].item()

            # Object center height required to make contact points reach target_lift_height
            target_object_height = start_height + height_offset

            # --- Determine lift_frame and drop_frame adaptively ---
            height_diff_up = abs(target_object_height - start_height)
            height_diff_down = abs(stop_height - target_object_height)
            movement_frames = stop_idx - start_idx

            # Minimum frames required to stay under max_speed_threshold
            min_frames_to_lift = max(1, int(height_diff_up / max_speed_threshold))
            min_frames_to_drop = max(1, int(height_diff_down / max_speed_threshold))
            total_min_frames = min_frames_to_lift + min_frames_to_drop

            if total_min_frames >= movement_frames:
                # If motion is too short, compress timing proportionally
                ratio = movement_frames / total_min_frames
                lift_frames = max(1, int(min_frames_to_lift * ratio))
                lift_frame = start_idx + lift_frames
                drop_frame = lift_frame  # no plateau phase
            else:
                # Enough frames to allow lift, plateau, and drop
                lift_frame = start_idx + min_frames_to_lift
                drop_frame = stop_idx - min_frames_to_drop

                # Ensure drop_frame is not before lift_frame
                if drop_frame < lift_frame:
                    mid_point = start_idx + movement_frames // 2
                    lift_frame = mid_point
                    drop_frame = mid_point

            # Clamp boundaries
            lift_frame = max(start_idx + 1, min(lift_frame, stop_idx - 1))
            drop_frame = max(lift_frame, min(drop_frame, stop_idx - 1))

            # --- Define keyframes and heights ---
            keyframes = [0, start_idx, lift_frame, drop_frame, stop_idx, total_frames - 1]
            key_heights = [initial_height, start_height, target_object_height,
                        target_object_height, stop_height, final_height]

            # Remove duplicate frame indices
            unique_keyframes, unique_heights = [], []
            for frame, height in zip(keyframes, key_heights):
                if not unique_keyframes or frame > unique_keyframes[-1]:
                    unique_keyframes.append(frame)
                    unique_heights.append(height)
                elif frame == unique_keyframes[-1]:
                    # If frame is duplicated, overwrite with the later height
                    unique_heights[-1] = height

            # --- Adjust keyframe heights to obey max speed constraint ---
            adjusted_heights = []
            for i in range(len(unique_keyframes)):
                if i == 0:
                    adjusted_heights.append(unique_heights[i])
                    continue

                prev_frame = unique_keyframes[i-1]
                curr_frame = unique_keyframes[i]
                prev_height = adjusted_heights[i-1]
                target_height = unique_heights[i]

                frame_diff = curr_frame - prev_frame
                height_diff = target_height - prev_height

                if frame_diff > 0:
                    required_speed = abs(height_diff) / frame_diff
                    if required_speed <= max_speed_threshold:
                        adjusted_heights.append(target_height)
                    else:
                        direction = 1 if height_diff > 0 else -1
                        max_change = max_speed_threshold * frame_diff
                        adjusted_heights.append(prev_height + direction * max_change)
                else:
                    adjusted_heights.append(target_height)

            # --- Interpolate height curve with PCHIP ---
            try:
                if len(unique_keyframes) < 2:
                    # Not enough keyframes: fallback to linear interpolation
                    result_motion[b, :, 2] = torch.linspace(
                        initial_height, final_height, total_frames,
                        device=device, dtype=torch.float32
                    )
                else:
                    interp_func = PchipInterpolator(unique_keyframes, adjusted_heights)
                    all_frames = np.arange(total_frames)
                    interpolated_heights = interp_func(all_frames)

                    result_motion[b, :, 2] = torch.tensor(
                        interpolated_heights, device=device, dtype=torch.float32
                    )
            except Exception:
                # If interpolation fails, fallback to linear height change
                result_motion[b, :, 2] = torch.linspace(
                    initial_height, final_height, total_frames,
                    device=device, dtype=torch.float32
                )

        return result_motion

        
    def cond_sample_res(self):

        if self.test_on_train:
            test_loader = torch.utils.data.DataLoader(
                self.ds, batch_size=1, shuffle=False,
                num_workers=1, pin_memory=True, drop_last=False) 
        else:
            if self.test_unseen_objects:
                test_loader = torch.utils.data.DataLoader(
                    self.unseen_seq_ds, batch_size=1, shuffle=False,
                    num_workers=1, pin_memory=True, drop_last=False) 
            elif self.test_behave:
                test_loader = torch.utils.data.DataLoader(
                    self.behave_seq_ds, batch_size=1, shuffle=False,
                    num_workers=1, pin_memory=True, drop_last=False) 
            elif self.test_behave_motion:
                test_loader = torch.utils.data.DataLoader(
                    self.behave_motion_seq_ds, batch_size=1, shuffle=False,
                    num_workers=1, pin_memory=True, drop_last=False) 
            else:
                test_loader = torch.utils.data.DataLoader(
                    self.val_ds, batch_size=1, shuffle=False,
                    num_workers=1, pin_memory=True, drop_last=False) 


        for s_idx, val_data_dict in enumerate(test_loader):

            seq_name_list = val_data_dict['seq_name']

            object_name_list = val_data_dict['obj_name']
            start_frame_idx_list = val_data_dict['s_idx']
            end_frame_idx_list = val_data_dict['e_idx'] 

            val_human_data = val_data_dict['motion'].cuda() 
            val_obj_data = val_data_dict['obj_motion'].cuda()

            obj_bps_data = val_data_dict['input_obj_bps'].cuda().reshape(-1, 1, 1024*3)
            ori_data_cond = obj_bps_data # BS X 1 X (1024*3) 

            rest_human_offsets = val_data_dict['rest_human_offsets'].cuda() # BS X 24 X 3 
            
            if "contact_labels" in val_data_dict:
                contact_labels = val_data_dict['contact_labels'].cuda() # BS X T X 4 
            else:
                contact_labels = None 

            # Generate padding mask 
            actual_seq_len = val_data_dict['seq_len'] + 1 # BS, + 1 since we need additional timestep for noise level 
            tmp_mask = torch.arange(self.window+1).expand(val_obj_data.shape[0], \
            self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
            # BS X max_timesteps
            padding_mask = tmp_mask[:, None, :].to(val_obj_data.device)

            end_pos_cond_mask = self.prep_start_end_condition_mask_pos_only(val_obj_data, val_data_dict['seq_len'])
            cond_mask = self.prep_mimic_A_star_path_condition_mask_pos_xy_only(val_obj_data, val_data_dict['seq_len'])
            cond_mask = end_pos_cond_mask * cond_mask 

            human_cond_mask = torch.ones_like(val_human_data).to(val_human_data.device)
            if self.input_first_human_pose:
                human_cond_mask[:, 0, :] = 0 
                
            cond_mask = torch.cat((cond_mask, human_cond_mask), dim=-1) # BS X T X (3+6+24*3+22*6)

            if self.use_guidance_in_denoising:
                # Load current sequence's object SDF
                self.object_sdf, self.object_sdf_centroid, self.object_sdf_extents = \
                self.load_object_sdf_data(val_data_dict['obj_name'][0])


            if self.compute_metrics:
                num_samples_per_seq = 1 
            else:
                num_samples_per_seq = 1

            val_obj_data = val_obj_data.repeat(num_samples_per_seq, 1, 1) # BS X T X D 
            val_human_data = val_human_data.repeat(num_samples_per_seq, 1, 1) 
            cond_mask = cond_mask.repeat(num_samples_per_seq, 1, 1) 
            padding_mask = padding_mask.repeat(num_samples_per_seq, 1, 1) # BS X 1 X 121 
            ori_data_cond = ori_data_cond.repeat(num_samples_per_seq, 1, 1)  
            rest_human_offsets = rest_human_offsets.repeat(num_samples_per_seq, 1, 1)
           
            contact_data = torch.zeros(val_obj_data.shape[0], val_obj_data.shape[1], 4).to(val_obj_data.device) 
            data = torch.cat((val_obj_data, val_human_data, contact_data), dim=-1) 
            cond_mask = torch.cat((cond_mask, \
                    torch.ones_like(contact_data).to(cond_mask.device)), dim=-1) 
            
            if self.add_language_condition:
                text_anno_data = val_data_dict['text']
                
                
                if 'push' in text_anno_data[0].lower() or 'pull' in text_anno_data[0].lower() or 'sit' in text_anno_data[0].lower() or 'seat' in text_anno_data[0].lower() or 'occupy' in text_anno_data[0].lower():
                    lift_height = 0
                else:
                    lift_height = 0.2
                
                
                init_verts = self.get_pred_object_mesh(val_data_dict['ori_obj_motion'].to(device).clone()[:,:1], 
                    data_dict=val_data_dict, curr_object_name=object_name_list, only_mesh=True)
                
                # ------------------------------------------------------------------------------------------------
                # add CFG scale to batch
                hints = val_data_dict['ori_motion'][:,:,:24*3].clone().to(device) # should use real hands for control signal

                bs, temp_seq_len, dim = hints.shape
                hints = hints.reshape(bs, temp_seq_len, dim//3, 3) + val_data_dict['trans2joint'].clone().reshape(bs, 1, 1, 3).to(device)
                
                if self.test_behave_motion:
                    init_human_pos_all = hints[0,0,:].clone()-2*val_data_dict['trans2joint'].clone().reshape(1, 3).to(device)
                else:
                    init_human_pos_all = hints[0,0,:].clone()-val_data_dict['trans2joint'].clone().reshape(1, 3).to(device)
                
                init_human_pos = init_human_pos_all[22:24].clone()  # only hands
                init_human_root = init_human_pos_all[0].clone()
                init_object_pos = val_data_dict['ori_obj_motion'].to(device).clone()[0,0,:3]
                forward = (init_object_pos - init_human_root)[:2]

                # bs, T, 12
                interpolated_object_motion, movement_start, movement_stop = interpolation_object_motion_new(val_data_dict['ori_obj_motion'].to(device).clone(), 
                                                lift_height=lift_height, init_verts=init_verts[0,0], init_human_pos=init_human_pos, 
                                                forward=forward)
                
                possible_contact_points, my_seq_com_pos, my_obj_rot_mat, my_obj_mesh_verts = self.get_pred_object_mesh(interpolated_object_motion.clone(), 
                    data_dict=val_data_dict, curr_object_name=object_name_list, human_pos=init_human_pos.clone())

                # modify the lift height
                if 'push' in text_anno_data[0].lower() or 'pull' in text_anno_data[0].lower() or 'sit' in text_anno_data[0].lower() or 'seat' in text_anno_data[0].lower() or 'occupy' in text_anno_data[0].lower():
                    pass
                else:
                    interpolated_object_motion_ori = interpolated_object_motion.clone()
                    interpolated_object_motion = self.adjust_object_height_with_speed_control(
                        interpolated_object_motion.clone(), 
                        possible_contact_points.clone(), 
                        movement_start, 
                        movement_stop,
                        target_lift_height=1.2,  
                        max_speed_threshold=0.02,  
                    )

                    # change all the other variable height
                    diff_height = interpolated_object_motion[:,:,2:3] - interpolated_object_motion_ori[:,:,2:3]
                    possible_contact_points[:,:,:,2] += diff_height
                    my_seq_com_pos[:,:,2:3] += diff_height
                    my_obj_mesh_verts[:,:,:,2] += diff_height

                contact_positions = torch.zeros_like(hints).to(device)
                contact_labels = torch.zeros((bs, temp_seq_len, 4), dtype=torch.float32).to(device)

                contact_labels[:,movement_start[0]:movement_stop[0]+1, :2] = 1
                contact_positions[:,movement_start[0]:movement_stop[0]+1, 22:] = possible_contact_points[:,movement_start[0]:movement_stop[0]+1] + val_data_dict['trans2joint'].clone().reshape(bs, 1, 1, 3).to(device)
                contact_positions[:,movement_stop[0]+1:, 22:] = contact_positions[:,movement_stop[0]:movement_stop[0]+1, 22:]
                contact_positions[0,0,22:24] = init_human_pos + val_data_dict['trans2joint'].clone().reshape(1, 3).to(device)

                # interpolate hands from frame 0
                inter_end = movement_start[0]
                if inter_end != 0:
                    key_times = [0, inter_end]
                    for b in range(bs):
                        for joint_idx in [22, 23]:
                            for i in range(3): 
                                interp_func = PchipInterpolator(key_times, contact_positions[b, key_times, joint_idx, i].cpu().numpy())
                                t_interp = np.linspace(key_times[0], key_times[-1], inter_end+1)
                                interp = interp_func(t_interp)
                                contact_positions[b, :inter_end+1, joint_idx, i] = torch.tensor(interp, device=device, dtype=torch.float32)

                # release after movement_stop
                inter_end = movement_stop[0]
                if inter_end > 0 and inter_end != temp_seq_len-1:
                    rest_frames = temp_seq_len - inter_end
                    for b in range(bs):
                        for joint_idx in [22, 23]:
                            contact_positions[b, inter_end:inter_end+rest_frames, joint_idx] = \
                                    torch.flip(contact_positions[b, inter_end - rest_frames + 1 : inter_end + 1, joint_idx], dims=[0])
                                    
                if not self.use_random_frame_bps:
                    val_data_dict['reference_obj_rot_mat'] = torch.eye(3, dtype=torch.float32, device=contact_positions.device).unsqueeze(0).unsqueeze(0)

                if self.test_behave_motion:
                    save_folder = f'third-party/ProtoMotion_for_InterPose/data/Dataset/behave_data/test_set_hands_control_interpolation_position'
                else:
                    save_folder = f'third-party/ProtoMotion_for_InterPose/data/Dataset/omomo_data/test_set_hands_control_interpolation_position'

                os.makedirs(save_folder, exist_ok=True)
                # for maskedmimic, always use 2 hands
                if torch.norm(possible_contact_points[:,movement_start[0], 0] - possible_contact_points[:,movement_start[0], 1]).item() >= 0.0:
                    np.savez(os.path.join(save_folder, f"{seq_name_list[0]}.npz"), 
                        control_idx=[22,23], 
                        positions=contact_positions[0,:,22:].cpu().numpy(),
                        object_motion=interpolated_object_motion.cpu().numpy(),
                        reference_obj_rot_mat=val_data_dict['reference_obj_rot_mat'].cpu().numpy(),
                        init_trans=hints[0,0,0].cpu().numpy(),
                        init_rot=val_data_dict['ori_motion'][0,0,24*3:24*3+6].cpu().numpy(),
                    )
                else:
                    np.savez(os.path.join(save_folder, f"{seq_name_list[0]}.npz"), 
                        control_idx=[23], 
                        positions=contact_positions[0,:,23:].cpu().numpy(),
                        object_motion=interpolated_object_motion.cpu().numpy(),
                        reference_obj_rot_mat=val_data_dict['reference_obj_rot_mat'].cpu().numpy(),
                        init_trans=hints[0,0,0].cpu().numpy(),
                        init_rot=val_data_dict['ori_motion'][0,0,24*3:24*3+6].cpu().numpy(),
                    )


def run_sample(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'

    # Define model     
    repr_dim = 3 + 9 

    repr_dim += 24 * 3 + 22 * 6 

    if opt.use_object_keypoints:
        repr_dim += 4 
    
    trainer = Trainer(
        opt,
        None,
        train_batch_size=opt.batch_size, # 32
        train_lr=opt.learning_rate, # 1e-4
        train_num_steps=8000000,         # 700000, total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=True,                        # turn on mixed precision
        results_folder=str(wdir),
        use_wandb=False,
        args=None,
        diffusion=None
    )
    
    # only do inference
    trainer.cond_sample_res()

    torch.cuda.empty_cache()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--wandb_pj_name', type=str, default='chois_projects', help='project name')
    parser.add_argument('--entity', default='', help='W&B entity')
    parser.add_argument('--exp_name', default='chois', help='save to project/name')

    parser.add_argument('--device', default='0', help='cuda device')

    parser.add_argument('--window', type=int, default=120, help='horizon')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='generator_learning_rate')

    parser.add_argument('--pretrained_model', type=str, default="", help='checkpoint')

    parser.add_argument('--data_root_folder', type=str, default="", help='data root folder')

    parser.add_argument('--save_res_folder', type=str, default="", help='save res folder')

    parser.add_argument('--n_dec_layers', type=int, default=4, help='the number of decoder layers')
    parser.add_argument('--n_head', type=int, default=4, help='the number of heads in self-attention')
    parser.add_argument('--d_k', type=int, default=256, help='the dimension of keys in transformer')
    parser.add_argument('--d_v', type=int, default=256, help='the dimension of values in transformer')
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of intermediate representation in transformer')
    
    # For testing sampled results 
    parser.add_argument("--test_sample_res", action="store_true")

    # For testing sampled results w planned path 
    parser.add_argument("--use_long_planned_path", action="store_true")

    # For testing sampled results on training dataset 
    parser.add_argument("--test_on_train", action="store_true")

    # For quantitative evaluation. 
    parser.add_argument("--for_quant_eval", action="store_true")

    # Train and test on different objects. 
    parser.add_argument("--use_object_split", action="store_true")

    # Add language conditions. 
    parser.add_argument("--add_language_condition", action="store_true")

    # Input the first human pose, maybe can connect the windows better.  
    parser.add_argument("--input_first_human_pose", action="store_true")

    parser.add_argument("--use_guidance_in_denoising", action="store_true")

    parser.add_argument("--compute_metrics", action="store_true")

    # Add rest offsets for body shape information. 
    parser.add_argument("--use_random_frame_bps", action="store_true")

    parser.add_argument('--test_object_name', type=str, default="", help='object name for long sequence generation testing')
    parser.add_argument('--test_scene_name', type=str, default="", help='scene name for long sequence generation testing')

    parser.add_argument("--use_object_keypoints", action="store_true")

    parser.add_argument('--loss_w_feet', type=float, default=1, help='the loss weight for feet contact loss')
    parser.add_argument('--loss_w_fk', type=float, default=1, help='the loss weight for fk loss')
    parser.add_argument('--loss_w_obj_pts', type=float, default=1, help='the loss weight for fk loss')

    parser.add_argument("--add_semantic_contact_labels", action="store_true")

    parser.add_argument("--test_unseen_objects", action="store_true")

    parser.add_argument("--use_sub_dataset", action="store_true")

   
    parser.add_argument("--save_obj_only", action="store_true")

    parser.add_argument("--fps", type=int, default=30, help='fps')

    parser.add_argument("--test_behave", action="store_true")
    parser.add_argument("--test_behave_motion", action="store_true")


    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = os.path.join(opt.project, opt.exp_name)
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    if opt.test_sample_res:
        run_sample(opt, device)