import argparse
import os
import numpy as np
import random 
import json 

import trimesh 

from matplotlib import pyplot as plt
from pathlib import Path

import wandb

import torch
from torch.cuda.amp import GradScaler
from torch.utils import data

import torch.nn.functional as F

import pytorch3d.transforms as transforms 

from manip.data.cano_traj_dataset import CanoObjectTrajDataset, quat_ik_torch
from manip.data.unseen_obj_long_cano_traj_dataset import UnseenCanoObjectTrajDataset 
from manip.data.behave_obj_long_cano_traj_dataset import BehaveCanoObjectTrajDataset 
from manip.data.cano_traj_dataset_behave import CanoObjectTrajDataset_behave 

from manip.vis.blender_vis_mesh_motion import run_blender_rendering_and_save2video, save_verts_faces_to_mesh_file_w_object

from utils.evaluation_metrics import compute_metrics   

import random
seed=1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


def export_to_ply(points, filename='output.ply'):
    # Open the file in write mode
    with open(filename, 'w') as ply_file:
        # Write the PLY header
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("comment Created by YourProgram\n")
        ply_file.write(f"element vertex {len(points)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("end_header\n")
        
        # Write the points data
        for point in points:
            ply_file.write(f"{point[0]} {point[1]} {point[2]}\n")

def compute_signed_distances(
    sdf, sdf_centroid, sdf_extents,
    query_points):
    # sdf: 1 X 256 X 256 X 256 
    # sdf_centroid: 1 X 3, center of the bounding box.  
    # sdf_extents: 1 X 3, width, height, depth of the box.  
    # query_points: T X Nv X 3 

    query_pts_norm = (query_points - sdf_centroid[None, :, :]) * 2 / sdf_extents.cpu().detach().numpy().max() # Convert to range [-1, 1]
     
    query_pts_norm = query_pts_norm[...,[2, 1, 0]] # Switch the order to depth, height, width
    
    num_steps, nv, _ = query_pts_norm.shape # T X Nv X 3 

    query_pts_norm = query_pts_norm[None, :, None, :, :] # 1 X T X 1 X Nv X 3 

    signed_dists = F.grid_sample(sdf[:, None, :, :, :], query_pts_norm, \
    padding_mode='border', align_corners=True)
    # F.grid_sample: N X C X D_in X H_in X W_in, N X D_out X H_out X W_out X 3, output: N X C X D_out X H_out X W_out 
    # sdf: 1 X 1 X 256 X 256 X 256, query_pts: 1 X T X 1 X Nv X 3 -> 1 X 1 X T X 1 X Nv  

    signed_dists = signed_dists[0, 0, :, 0, :] * sdf_extents.cpu().detach().numpy().max() / 2. # T X Nv 
    
    return signed_dists

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
    ):
        super().__init__()

        self.use_wandb = use_wandb           
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
        self.maskmimic_path = self.opt.maskmimic_path
        self.hand_control_folder = self.opt.hand_control_folder


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

    def load_scene_sdf_data(self, scene_name):
        data_folder = os.path.join(self.data_root_folder, "replica_processed/replica_fixed_poisson_sdfs_res256")
        sdf_npy_path = os.path.join(data_folder, scene_name+"_sdf.npy")
        sdf_json_path = os.path.join(data_folder, scene_name+"_sdf_info.json")

        sdf = np.load(sdf_npy_path) # 256 X 256 X 256 
        sdf_json_data = json.load(open(sdf_json_path, 'r'))

        sdf_centroid = np.asarray(sdf_json_data['centroid']) # a list with 3 items -> 3 
        sdf_extents = np.asarray(sdf_json_data['extents']) # a list with 3 items -> 3 

        sdf = torch.from_numpy(sdf).float()[None].cuda()
        sdf_centroid = torch.from_numpy(sdf_centroid).float()[None].cuda()
        sdf_extents = torch.from_numpy(sdf_extents).float()[None].cuda() 

        return sdf, sdf_centroid, sdf_extents

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

    def append_new_value_to_metrics_list(self, lhand_jpe, rhand_jpe, hand_jpe, mpvpe, mpjpe, rot_dist, trans_err, \
            gt_contact_percent, contact_percent, gt_foot_sliding_jnts, foot_sliding_jnts, \
            contact_precision, contact_recall, contact_acc, contact_f1_score, obj_rot_dist, obj_com_pos_err, \
            start_obj_com_pos_err, end_obj_com_pos_err, waypoints_xy_pos_err, gt_penetration_score, penetration_score, \
            gt_hand_penetration_score, hand_penetration_score, gt_floor_height, pred_floor_height): 
        # Append new sequence's value to list. 
        self.lhand_jpe_list.append(lhand_jpe)
        self.rhand_jpe_list.append(rhand_jpe)
        self.hand_jpe_list.append(hand_jpe)
        self.mpvpe_list.append(mpvpe)
        self.mpjpe_list.append(mpjpe)
        self.rot_dist_list.append(rot_dist)
        self.trans_err_list.append(trans_err)
        
        self.gt_floor_height_list.append(gt_floor_height)
        self.floor_height_list.append(pred_floor_height)

        self.gt_foot_sliding_jnts_list.append(gt_foot_sliding_jnts)
        self.foot_sliding_jnts_list.append(foot_sliding_jnts)

        self.gt_contact_percent_list.append(gt_contact_percent)
        self.contact_percent_list.append(contact_percent)

        self.contact_precision_list.append(contact_precision)
        self.contact_recall_list.append(contact_recall)
        self.contact_acc_list.append(contact_acc)
        self.contact_f1_score_list.append(contact_f1_score)

        self.obj_rot_dist_list.append(obj_rot_dist)
        self.obj_com_pos_err_list.append(obj_com_pos_err)
        
        self.start_obj_com_pos_err_list.append(start_obj_com_pos_err)
        self.end_obj_com_pos_err_list.append(end_obj_com_pos_err)
        self.waypoints_xy_pos_err_list.append(waypoints_xy_pos_err)

        self.gt_penetration_list.append(gt_penetration_score)
        self.penetration_list.append(penetration_score) 

        self.gt_hand_penetration_list.append(gt_hand_penetration_score)
        self.hand_penetration_list.append(hand_penetration_score) 

    def print_evaluation_metrics(self, lhand_jpe_list, rhand_jpe_list, hand_jpe_list, mpvpe_list, mpjpe_list, \
                rot_dist_list, trans_err_list, gt_contact_percent_list, contact_percent_list, \
                gt_foot_sliding_jnts_list, foot_sliding_jnts_list, contact_precision_list, contact_recall_list, \
                contact_acc_list, contact_f1_score_list, obj_rot_dist_list, obj_com_pos_err_list, \
                start_obj_com_pos_err_list, end_obj_com_pos_err_list, waypoints_xy_pos_err_list, \
                gt_penetration_score_list, penetration_score_list, \
                gt_hand_penetration_score_list, hand_penetration_score_list, \
                gt_floor_height_list, pred_floor_height_list, \
                dest_metric_folder, seq_name=None): 
        
        mean_lhand_jpe = np.asarray(lhand_jpe_list).mean()
        mean_rhand_jpe = np.asarray(rhand_jpe_list).mean() 
        mean_hand_jpe = np.asarray(hand_jpe_list).mean() 
        mean_mpjpe = np.asarray(mpjpe_list).mean() 
        mean_mpvpe = np.asarray(mpvpe_list).mean() 
        mean_root_trans_err = np.asarray(trans_err_list).mean()
        mean_rot_dist = np.asarray(rot_dist_list).mean() 

        mean_fsliding_jnts = np.asarray(foot_sliding_jnts_list).mean()
        mean_gt_fsliding_jnts = np.asarray(gt_foot_sliding_jnts_list).mean() 
        
        mean_contact_percent = np.asarray(contact_percent_list).mean()
        mean_gt_contact_percent = np.asarray(gt_contact_percent_list).mean() 

        mean_contact_precision = np.asarray(contact_precision_list).mean()
        mean_contact_recall = np.asarray(contact_recall_list).mean() 
        mean_contact_acc = np.asarray(contact_acc_list).mean()
        mean_contact_f1_score = np.asarray(contact_f1_score_list).mean() 

        mean_obj_rot_dist = np.asarray(obj_rot_dist_list).mean() 
        mean_obj_com_pos_err = np.asarray(obj_com_pos_err_list).mean() 

        mean_start_obj_com_pos_err = np.asarray(start_obj_com_pos_err_list).mean() 
        mean_end_obj_com_pos_err = np.asarray(end_obj_com_pos_err_list).mean() 
        mean_waypoints_xy_pos_err = np.asarray(waypoints_xy_pos_err_list).mean()

        mean_penetration_score = np.asarray(penetration_score_list).mean()
        mean_gt_penetration_score = np.asarray(gt_penetration_score_list).mean() 

        mean_hand_penetration_score = np.asarray(hand_penetration_score_list).mean()
        mean_gt_hand_penetration_score = np.asarray(gt_hand_penetration_score_list).mean() 

        mean_gt_floor_height = np.asarray(gt_floor_height_list).mean() 
        mean_pred_floor_height = np.asarray(pred_floor_height_list).mean() 

        print("The number of sequences: {0}".format(len(mpjpe_list)))
        print("*********************************Human Motion Evaluation**************************************")
        print("Left Hand JPE: {0}, Right Hand JPE: {1}, Two Hands JPE: {2}".format(mean_lhand_jpe, mean_rhand_jpe, mean_hand_jpe))
        print("MPJPE: {0}, MPVPE: {1}, Root Trans: {2}, Global Rot Err: {3}".format(mean_mpjpe, mean_mpvpe, mean_root_trans_err, mean_rot_dist))
        print("Foot sliding jnts: {0}, GT Foot sliding jnts: {1}".format(mean_fsliding_jnts, mean_gt_fsliding_jnts))
        print("Floor Height: {0}, GT Floor Height: {1}".format(mean_pred_floor_height, mean_gt_floor_height))
        
        print("*********************************Object Motion Evaluation**************************************")
        print("Object com pos err: {0}, Object rotation err: {1}".format(mean_obj_com_pos_err, mean_obj_rot_dist))

        print("*********************************Interaction Evaluation**************************************")
        print("Hand-Object Penetration Score: {0}".format(mean_hand_penetration_score))
        print("GT Hand-Object Penetration Score: {0}".format(mean_gt_hand_penetration_score))
        print("Human-Object Penetration Score: {0}".format(mean_penetration_score))
        print("GT Human-Object Penetration Score: {0}".format(mean_gt_penetration_score))
        
        print("Contact precision: {0}, Contact recall: {1}".format(mean_contact_precision, mean_contact_recall))
        print("Contact Acc: {0}, Contact F1 score: {1}".format(mean_contact_acc, mean_contact_f1_score)) 
        print("Contact percentage: {0}, GT Contact percentage: {1}".format(mean_contact_percent, mean_gt_contact_percent))

        print("*********************************Condition Following Evaluation**************************************")
        print("Start obj_com_pos err: {0}, End obj_com_pos err: {1}".format(mean_start_obj_com_pos_err, mean_end_obj_com_pos_err))
        print("waypoints xy err: {0}".format(mean_waypoints_xy_pos_err)) 

        # Save the results to json files. 
        if not os.path.exists(dest_metric_folder):
            os.makedirs(dest_metric_folder) 
        if seq_name is not None: # number for all the testing data
            dest_metric_json_path = os.path.join(dest_metric_folder, seq_name+".json")
        else:
            dest_metric_json_path = os.path.join(dest_metric_folder, "evaluation_metrics_for_all_test_data.json")

        metric_dict = {}
        metric_dict['mean_lhand_jpe'] = mean_lhand_jpe
        metric_dict['mean_rhand_jpe'] = mean_rhand_jpe 
        metric_dict['mean_hand_jpe'] = mean_hand_jpe 
        metric_dict['mean_mpjpe'] = mean_mpjpe 
        metric_dict['mean_mpvpe'] = mean_mpvpe 
        metric_dict['mean_root_trans_err'] = mean_root_trans_err 
        metric_dict['mean_rot_dist'] = mean_rot_dist 

        metric_dict['mean_floor_height'] = mean_pred_floor_height
        metric_dict['mean_gt_floor_height'] = mean_gt_floor_height 

        metric_dict['mean_fsliding_jnts'] = mean_fsliding_jnts 
        metric_dict['mean_gt_fsliding_jnts'] = mean_gt_fsliding_jnts 

        metric_dict['mean_contact_percent'] = mean_contact_percent
        metric_dict['mean_gt_contact_percent'] = mean_gt_contact_percent  

        metric_dict['mean_contact_precision'] = mean_contact_precision 
        metric_dict['mean_contact_recall'] = mean_contact_recall 
        metric_dict['mean_contact_acc'] = mean_contact_acc 
        metric_dict['mean_contact_f1_score'] = mean_contact_f1_score 

        metric_dict['mean_obj_rot_dist'] = mean_obj_rot_dist 
        metric_dict['mean_obj_com_pos_err'] = mean_obj_com_pos_err

        metric_dict['mean_start_obj_com_pos_err'] = mean_start_obj_com_pos_err 
        metric_dict['mean_end_obj_com_pos_err'] = mean_end_obj_com_pos_err 
        metric_dict['mean_waypoints_xy_pos_err'] = mean_waypoints_xy_pos_err 

        metric_dict['mean_penetration_score'] = mean_penetration_score 
        metric_dict['mean_gt_penetration_score'] = mean_gt_penetration_score 

        metric_dict['mean_hand_penetration_score'] = mean_hand_penetration_score 
        metric_dict['mean_gt_hand_penetration_score'] = mean_gt_hand_penetration_score 

        # Convert all to float 
        for k in metric_dict:
            metric_dict[k] = float(metric_dict[k])

        json.dump(metric_dict, open(dest_metric_json_path, 'w'))

    def compute_hand_penetration_metric(self, object_name, ori_verts_pred, \
        pred_obj_com_pos, pred_obj_rot_mat, eval_fullbody=False):
        # ori_verts_pred: T X Nv X 3 
        # pred_obj_com_pos: T X 3
        # pred_obj_rot_mat: T X 3 X 3
        ori_verts_pred = ori_verts_pred[None] # 1 X T X Nv X 3 
        pred_obj_com_pos = pred_obj_com_pos[None] # 1 X T X 3 
        pred_obj_rot_mat = pred_obj_rot_mat[None] # 1 X T X 3 X 3 

        if not eval_fullbody:
            hand_verts = ori_verts_pred[:, :, self.hand_vertex_idxs, :] # BS X T X N_hand X 3
        else:
            hand_verts = ori_verts_pred 

        hand_verts_in_rest_frame = hand_verts - pred_obj_com_pos[:, :, None, :] # BS X T X N_hand X 3 
        hand_verts_in_rest_frame = torch.matmul(pred_obj_rot_mat[:, :, None, :, :].repeat(1, 1, \
                            hand_verts_in_rest_frame.shape[2], 1, 1), \
                            hand_verts_in_rest_frame[:, :, :, :, None]).squeeze(-1) # BS X T X N_hand X 3 

        curr_object_sdf, curr_object_sdf_centroid, curr_object_sdf_extents = \
        self.load_object_sdf_data(object_name)

        # Convert hand vertices to align with rest pose object. 
        signed_dists = compute_signed_distances(curr_object_sdf, curr_object_sdf_centroid, \
            curr_object_sdf_extents, hand_verts_in_rest_frame[0]) # we always use bs = 1 now!!!                          
        # signed_dists: T X N_hand (120 X 1535)

        penetration_score = torch.minimum(signed_dists, torch.zeros_like(signed_dists)).abs().mean() # The smaller, the better 
        # penetration_score = torch.minimum(signed_dists, torch.zeros_like(signed_dists)).abs().sum()
        return penetration_score.detach().cpu().numpy()  

    def prep_evaluation_metrics_list(self):
        self.lhand_jpe_list = [] 
        self.rhand_jpe_list = [] 
        self.hand_jpe_list = [] 
        self.mpvpe_list = [] 
        self.mpjpe_list = [] 
        self.rot_dist_list = [] 
        self.trans_err_list = [] 

        self.gt_floor_height_list = [] 
        self.floor_height_list = [] 
        
        self.gt_foot_sliding_jnts_list = []
        self.foot_sliding_jnts_list = [] 

        self.gt_contact_percent_list = [] 
        self.contact_percent_list = []
        self.contact_precision_list = [] 
        self.contact_recall_list = [] 
        self.contact_acc_list = [] 
        self.contact_f1_score_list = []

        self.obj_rot_dist_list = []
        self.obj_com_pos_err_list = [] 

        self.start_obj_com_pos_err_list = [] 
        self.end_obj_com_pos_err_list = [] 
        self.waypoints_xy_pos_err_list = []

        self.gt_penetration_list = []
        self.penetration_list = [] 

        self.gt_hand_penetration_list = []
        self.hand_penetration_list = [] 

    def prep_res_folders(self):
        res_root_folder = self.save_res_folder 
        # Prepare folder for saving npz results 
        dest_res_for_eval_npz_folder = os.path.join(res_root_folder, "res_npz_files")
        # Prepare folder for evaluation metrics 
        dest_metric_root_folder = os.path.join(res_root_folder, "evaluation_metrics_json")
        # Prepare folder for visualization 
        dest_out_vis_root_folder = os.path.join(res_root_folder, "single_window_cmp_settings")
        # Prepare folder for saving .obj files 
        dest_out_obj_root_folder = os.path.join(res_root_folder, "objs_single_window_cmp_settings")
       
        if self.test_unseen_objects:
            dest_res_for_eval_npz_folder += "_unseen_obj"
            dest_metric_root_folder += "_unseen_obj"
            dest_out_vis_root_folder += "_unseen_obj"
            dest_out_obj_root_folder += "_unseen_obj"

        if self.test_behave:
            dest_res_for_eval_npz_folder += "_behave_obj"
            dest_metric_root_folder += "_behave_obj"
            dest_out_vis_root_folder += "_behave_obj"
            dest_out_obj_root_folder += "_behave_obj"

        if self.test_behave_motion:
            dest_res_for_eval_npz_folder += "_behave_motion_obj"
            dest_metric_root_folder += "_behave_motion_obj"
            dest_out_vis_root_folder += "_behave_motion_obj"
            dest_out_obj_root_folder += "_behave_motion_obj"

        # Prepare folder for saving text json files 
        dest_out_text_json_folder = os.path.join(dest_out_vis_root_folder, "text_json_files")

        if self.use_guidance_in_denoising:
            dest_res_for_eval_npz_folder = os.path.join(dest_res_for_eval_npz_folder, "chois")
            dest_metric_folder = os.path.join(dest_metric_root_folder, "chois")
            dest_out_vis_folder = os.path.join(dest_out_vis_root_folder, "chois")
            dest_out_obj_folder = os.path.join(dest_out_obj_root_folder, "chois")
        else:
            if self.use_object_keypoints:
                dest_res_for_eval_npz_folder = os.path.join(dest_res_for_eval_npz_folder, "chois_wo_guidance")
                dest_metric_folder = os.path.join(dest_metric_root_folder, "chois_wo_guidance") 
                dest_out_vis_folder = os.path.join(dest_out_vis_root_folder, "chois_wo_guidance") 
                dest_out_obj_folder = os.path.join(dest_out_obj_root_folder, "chois_wo_guidance")
            else:
                dest_res_for_eval_npz_folder = os.path.join(dest_res_for_eval_npz_folder, "chois_wo_l_geo")  
                dest_metric_folder = os.path.join(dest_metric_root_folder, "chois_wo_l_geo")  
                dest_out_vis_folder = os.path.join(dest_out_vis_root_folder, "chois_wo_l_geo")           
                dest_out_obj_folder = os.path.join(dest_out_obj_root_folder, "chois_wo_l_geo")    
        
      
        # Create folders 
        if not os.path.exists(dest_metric_folder):
            os.makedirs(dest_metric_folder) 
        if not os.path.exists(dest_out_vis_folder):
            os.makedirs(dest_out_vis_folder) 
        if not os.path.exists(dest_res_for_eval_npz_folder):
            os.makedirs(dest_res_for_eval_npz_folder)
        if not os.path.exists(dest_out_obj_folder):
            os.makedirs(dest_out_obj_folder) 
        if not os.path.exists(dest_out_text_json_folder):
            os.makedirs(dest_out_text_json_folder)

        dest_out_gt_vis_folder = os.path.join(dest_out_vis_root_folder, "0_gt")
        if not os.path.exists(dest_out_gt_vis_folder):
            os.makedirs(dest_out_gt_vis_folder) 

        return dest_res_for_eval_npz_folder, dest_metric_folder, dest_out_vis_folder, \
            dest_out_gt_vis_folder, dest_out_obj_folder, dest_out_text_json_folder

    def cond_sample_res(self):
        milestone = "10" # 9, 10

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

        self.prep_evaluation_metrics_list()

        dest_res_for_eval_npz_folder, dest_metric_folder, dest_out_vis_folder, \
        dest_out_gt_vis_folder, dest_out_obj_folder, dest_out_text_json_folder = self.prep_res_folders() 
        self.maskmimic_id = -1
        import yaml
        if self.test_behave_motion:
            mask_mimic_id_list = yaml.load(open('third-party/ProtoMotion_for_InterPose/data/Dataset/behave_data/behave_test_smpl.yaml', "r"), Loader=yaml.FullLoader)["motions"]
        else:
            if self.opt.use_sub_dataset:
                mask_mimic_id_list = yaml.load(open('third-party/ProtoMotion_for_InterPose/data/Dataset/omomo_data/omomo_test_subset.yaml', "r"), Loader=yaml.FullLoader)["motions"]
            else:
                mask_mimic_id_list = yaml.load(open('third-party/ProtoMotion_for_InterPose/data/Dataset/omomo_data/omomo_test_smpl.yaml', "r"), Loader=yaml.FullLoader)["motions"]
        mask_mimic_name2inx = {}
        for mask_mimic_file in mask_mimic_id_list:
            mask_mimic_file_name = mask_mimic_file['file'].split('/')[-1].replace('.npy', '')
            mask_mimic_idx = mask_mimic_file['idx']
            mask_mimic_name2inx[mask_mimic_file_name] = mask_mimic_idx
        for s_idx, val_data_dict in enumerate(test_loader):

            seq_name_list = val_data_dict['seq_name']
            
            if seq_name_list[0] in mask_mimic_name2inx:
                self.maskmimic_id = mask_mimic_name2inx[seq_name_list[0]]
            else:
                print('not found')
                continue
            maskmimic_path = self.maskmimic_path
            maskmimic_file = os.path.join(maskmimic_path, f'trajectory_pose_aa_{self.maskmimic_id}_0.npz')
            if not os.path.exists(maskmimic_file):
                continue

            object_name_list = val_data_dict['obj_name']
            start_frame_idx_list = val_data_dict['s_idx']
            end_frame_idx_list = val_data_dict['e_idx'] 

            val_human_data = val_data_dict['motion'].cuda() 
            val_obj_data = val_data_dict['obj_motion'].cuda()

            obj_bps_data = val_data_dict['input_obj_bps'].cuda().reshape(-1, 1, 1024*3)
            ori_data_cond = obj_bps_data # BS X 1 X (1024*3) 

            rest_human_offsets = val_data_dict['rest_human_offsets'].cuda() # BS X 24 X 3 
            

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
           
           
            for_vis_gt_data = torch.cat((val_obj_data, val_human_data), dim=-1)

            sample_idx = 0
            vis_tag = str(milestone)+"_sidx_"+str(s_idx)+"_sample_cnt_"+str(sample_idx)

            if self.use_guidance_in_denoising:
                vis_tag = vis_tag + "_w_guidance"

            if self.test_on_train:
                vis_tag = vis_tag + "_on_train"

            if self.test_unseen_objects:
                vis_tag = vis_tag + "_on_unseen_objects"

            if self.test_behave:
                vis_tag = vis_tag + "_on_behave_objects"

            if self.test_behave_motion:
                vis_tag = vis_tag + "_on_behave_motion_objects"

            curr_seq_name_tag = seq_name_list[0] + "_" + object_name_list[0]+ "_sidx_" + \
                        str(start_frame_idx_list[0].detach().cpu().numpy()) +\
                        "_eidx_" + str(end_frame_idx_list[0].detach().cpu().numpy()) + \
                        "_sample_cnt_" + str(sample_idx)

            dest_text_json_path = os.path.join(dest_out_text_json_folder, curr_seq_name_tag+".json")
            dest_text_json_dict = {}
            dest_text_json_dict['text'] = val_data_dict['text'][0]
            if not os.path.exists(dest_text_json_path):
                json.dump(dest_text_json_dict, open(dest_text_json_path, 'w'))

            curr_dest_out_mesh_folder = os.path.join(dest_out_obj_folder, curr_seq_name_tag) 
            curr_dest_out_vid_path = os.path.join(dest_out_vis_folder, curr_seq_name_tag+".mp4")
            curr_dest_out_gt_vid_path = os.path.join(dest_out_gt_vis_folder, curr_seq_name_tag+".mp4")


            pred_human_verts_list, pred_human_jnts_list, pred_human_trans_list, pred_human_rot_list, \
            pred_obj_com_pos_list, pred_obj_rot_mat_list, pred_obj_verts_list, _, _, _ = \
            self.gen_vis_res_generic(for_vis_gt_data.clone(), val_data_dict, milestone, cond_mask, \
            curr_object_name=object_name_list[0], vis_tag=vis_tag, \
            dest_out_vid_path=curr_dest_out_vid_path, dest_mesh_vis_folder=curr_dest_out_mesh_folder, \
            save_obj_only=self.opt.save_obj_only, save_mesh_file=not self.opt.save_obj_only) # skip the rendering part


            gt_human_verts_list, gt_human_jnts_list, gt_human_trans_list, gt_human_rot_list, \
            gt_obj_com_pos_list, gt_obj_rot_mat_list, gt_obj_verts_list, human_faces_list, obj_faces_list, _ = \
            self.gen_vis_res_generic(for_vis_gt_data, val_data_dict, milestone, cond_mask, vis_gt=True, \
            curr_object_name=object_name_list[0], vis_tag=vis_tag, \
            dest_out_vid_path=curr_dest_out_gt_vid_path, \
            dest_mesh_vis_folder=curr_dest_out_mesh_folder, \
            save_obj_only=self.opt.save_obj_only, save_mesh_file=not self.opt.save_obj_only) # skip the rendering part 
           

            dest_gt_for_eval_npz_folder = os.path.join('/'.join(dest_res_for_eval_npz_folder.split('/')[:-1]), 'gt')
            if not os.path.exists(dest_gt_for_eval_npz_folder):
                os.makedirs(dest_gt_for_eval_npz_folder)

            tmp_bs = len(seq_name_list) 
            for tmp_bs_idx in range(tmp_bs):
                tmp_seq_name = seq_name_list[tmp_bs_idx]
                tmp_obj_name = object_name_list[tmp_bs_idx]
            
                curr_pred_global_jpos = pred_human_jnts_list[0].detach().cpu().numpy()
                if self.test_unseen_objects:
                    curr_seq_dest_res_npz_path = os.path.join(dest_res_for_eval_npz_folder, \
                                            tmp_seq_name+"_"+tmp_obj_name+".npz")
                elif self.test_behave:
                    curr_seq_dest_res_npz_path = os.path.join(dest_res_for_eval_npz_folder, \
                                            tmp_seq_name+"_"+tmp_obj_name+".npz")
                elif self.test_behave_motion:
                    curr_seq_dest_res_npz_path = os.path.join(dest_res_for_eval_npz_folder, \
                                            tmp_seq_name+".npz")
                else:
                    curr_seq_dest_res_npz_path = os.path.join(dest_res_for_eval_npz_folder, \
                                                        tmp_seq_name+".npz")
                np.savez(curr_seq_dest_res_npz_path, seq_name=tmp_seq_name, \
                        global_jpos=curr_pred_global_jpos) # T X 24 X 3 
                
                curr_gt_global_jpos = gt_human_jnts_list[0].detach().cpu().numpy()
                curr_seq_dest_gt_npz_path = os.path.join(dest_gt_for_eval_npz_folder, \
                                                        tmp_seq_name+".npz")
                np.savez(curr_seq_dest_gt_npz_path, seq_name=tmp_seq_name, \
                        global_jpos=curr_gt_global_jpos) # T X 24 X 3 

            
            for tmp_s_idx in range(num_samples_per_seq):
                # Compute evaluation metrics 
                lhand_jpe, rhand_jpe, hand_jpe, mpvpe, mpjpe, rot_dist, trans_err, \
                gt_contact_percent, contact_percent, \
                gt_foot_sliding_jnts, foot_sliding_jnts, \
                contact_precision, contact_recall, contact_acc, contact_f1_score, \
                obj_rot_dist, obj_com_pos_err, start_obj_com_pos_err, end_obj_com_pos_err, waypoints_xy_pos_err, \
                gt_floor_height, pred_floor_height = \
                        compute_metrics(gt_human_verts_list[tmp_s_idx], pred_human_verts_list[tmp_s_idx], \
                        gt_human_jnts_list[tmp_s_idx], pred_human_jnts_list[tmp_s_idx], \
                        human_faces_list[tmp_s_idx], \
                        gt_human_trans_list[tmp_s_idx], pred_human_trans_list[tmp_s_idx], \
                        gt_human_rot_list[tmp_s_idx], pred_human_rot_list[tmp_s_idx], \
                        gt_obj_com_pos_list[tmp_s_idx], pred_obj_com_pos_list[tmp_s_idx], \
                        gt_obj_rot_mat_list[tmp_s_idx], pred_obj_rot_mat_list[tmp_s_idx], \
                        gt_obj_verts_list[tmp_s_idx], pred_obj_verts_list[tmp_s_idx], \
                        obj_faces_list[tmp_s_idx], val_data_dict['seq_len'])

                pred_hand_penetration_score = self.compute_hand_penetration_metric(object_name_list[0], \
                                    pred_human_verts_list[tmp_s_idx], \
                                    pred_obj_com_pos_list[tmp_s_idx], pred_obj_rot_mat_list[tmp_s_idx])
                gt_hand_penetration_score = self.compute_hand_penetration_metric(object_name_list[0], \
                                    gt_human_verts_list[tmp_s_idx], \
                                    gt_obj_com_pos_list[tmp_s_idx], gt_obj_rot_mat_list[tmp_s_idx])

                pred_penetration_score = self.compute_hand_penetration_metric(object_name_list[0], \
                                    pred_human_verts_list[tmp_s_idx], \
                                    pred_obj_com_pos_list[tmp_s_idx], pred_obj_rot_mat_list[tmp_s_idx], eval_fullbody=True)
                gt_penetration_score = self.compute_hand_penetration_metric(object_name_list[0], \
                                    gt_human_verts_list[tmp_s_idx], \
                                    gt_obj_com_pos_list[tmp_s_idx], gt_obj_rot_mat_list[tmp_s_idx], eval_fullbody=True)
               
                self.append_new_value_to_metrics_list(lhand_jpe, rhand_jpe, hand_jpe, mpvpe, mpjpe, rot_dist, trans_err, \
                gt_contact_percent, contact_percent, gt_foot_sliding_jnts, foot_sliding_jnts, \
                contact_precision, contact_recall, contact_acc, contact_f1_score, \
                obj_rot_dist, obj_com_pos_err, \
                start_obj_com_pos_err, end_obj_com_pos_err, waypoints_xy_pos_err, \
                gt_penetration_score, pred_penetration_score, \
                gt_hand_penetration_score, pred_hand_penetration_score, \
                gt_floor_height, pred_floor_height) 

                # Print current seq's evaluation metrics. 
                curr_seq_name_tag = seq_name_list[0] + "_" + object_name_list[0]+ "_sidx_" + str(start_frame_idx_list[0].detach().cpu().numpy()) +\
                        "_eidx_" + str(end_frame_idx_list[0].detach().cpu().numpy()) + "_sample_cnt_" + str(tmp_s_idx)
                print("Current Sequence name:{0}".format(curr_seq_name_tag))
                self.print_evaluation_metrics([lhand_jpe], [rhand_jpe], [hand_jpe], [mpvpe], [mpjpe], \
                [rot_dist], [trans_err], \
                [gt_contact_percent], [contact_percent], \
                [gt_foot_sliding_jnts], [foot_sliding_jnts], \
                [contact_precision], [contact_recall], [contact_acc], [contact_f1_score], \
                [obj_rot_dist], [obj_com_pos_err], \
                [start_obj_com_pos_err], [end_obj_com_pos_err], [waypoints_xy_pos_err], \
                [gt_penetration_score], [pred_penetration_score], \
                [gt_hand_penetration_score], [pred_hand_penetration_score], \
                [gt_floor_height], [pred_floor_height], \
                dest_metric_folder, curr_seq_name_tag) # Assume batch size = 1 

            torch.cuda.empty_cache()

        self.print_evaluation_metrics(self.lhand_jpe_list, self.rhand_jpe_list, self.hand_jpe_list, self.mpvpe_list, self.mpjpe_list, \
            self.rot_dist_list, self.trans_err_list, self.gt_contact_percent_list, self.contact_percent_list, \
            self.gt_foot_sliding_jnts_list, self.foot_sliding_jnts_list, \
            self.contact_precision_list, self.contact_recall_list, \
            self.contact_acc_list, self.contact_f1_score_list, \
            self.obj_rot_dist_list, self.obj_com_pos_err_list, \
            self.start_obj_com_pos_err_list, self.end_obj_com_pos_err_list, self.waypoints_xy_pos_err_list, \
            self.gt_penetration_list, self.penetration_list, self.gt_hand_penetration_list, self.hand_penetration_list, \
            self.gt_floor_height_list, self.floor_height_list, \
            dest_metric_folder)   
        
        from utils.rank_sample_results import rank_sample
        rank_sample(dest_metric_folder)
          
    
    def create_ball_mesh(self, center_pos, ball_mesh_path):
        # center_pos: K X 3  
        ball_color = np.asarray([22, 173, 100]) # green 

        num_mesh = center_pos.shape[0]
        for idx in range(num_mesh):
            ball_mesh = trimesh.primitives.Sphere(radius=0.05, center=center_pos[idx])
            
            dest_ball_mesh = trimesh.Trimesh(
                vertices=ball_mesh.vertices,
                faces=ball_mesh.faces,
                vertex_colors=ball_color,
                process=False)

            result = trimesh.exchange.ply.export_ply(dest_ball_mesh, encoding='ascii')
            output_file = open(ball_mesh_path.replace(".ply", "_"+str(idx)+".ply"), "wb+")
            output_file.write(result)
            output_file.close()
    
    def create_ball_mesh_new(self, center_pos, ball_mesh_path, name, idx):
        # center_pos: K X 3  
        ball_color = np.asarray([22, 173, 100]) # green 

        
        ball_mesh = trimesh.primitives.Sphere(radius=0.05, center=center_pos)
        
        dest_ball_mesh = trimesh.Trimesh(
            vertices=ball_mesh.vertices,
            faces=ball_mesh.faces,
            vertex_colors=ball_color,
            process=False)

        result = trimesh.exchange.ply.export_ply(dest_ball_mesh, encoding='ascii')
        output_file = open(ball_mesh_path.replace(".ply", "_"+name+"_"+str(idx)+".ply"), "wb+")
        output_file.write(result)
        output_file.close()

    def export_to_mesh(self, mesh_verts, mesh_faces, mesh_path):
        dest_mesh = trimesh.Trimesh(
            vertices=mesh_verts,
            faces=mesh_faces,
            process=False)

        result = trimesh.exchange.ply.export_ply(dest_mesh, encoding='ascii')
        output_file = open(mesh_path, "wb+")
        output_file.write(result)
        output_file.close()

    def plot_arr(self, t_vec, pred_val, gt_val, dest_path):
        plt.plot(t_vec, gt_val, color='green', label="gt")
        plt.plot(t_vec, pred_val, color='red', label="pred")
        plt.legend(["gt", "pred"])
        plt.savefig(dest_path)
        plt.clf()
    
    def gen_vis_res_generic(self, all_res_list, data_dict, step, cond_mask, vis_gt=False, vis_tag=None, \
                planned_end_obj_com=None, move_to_planned_path=None, planned_waypoints_pos=None, \
                vis_long_seq=False, overlap_frame_num=10, planned_scene_names=None, \
                planned_path_floor_height=None, vis_wo_scene=False, text_anno=None, cano_quat=None, \
                gen_long_seq=False, curr_object_name=None, dest_out_vid_path=None, dest_mesh_vis_folder=None, \
                save_obj_only=False, save_mesh_file=False):

        # Prepare list used for evaluation. 
        human_jnts_list = []
        human_verts_list = [] 
        obj_verts_list = [] 
        trans_list = []
        human_mesh_faces_list = []
        obj_mesh_faces_list = [] 

        # all_res_list: N X T X (3+9) 
        num_seq = all_res_list.shape[0]

        # currently only works for batch_size=1

        if not vis_gt:
            if self.hand_control_folder is not None:
                if self.use_long_planned_path:
                    pid = vis_tag.split('_pidx_')[1].split('_sample_cnt_')[0]
                    extra_name1 = f"_pidx_{pid}"
                else:
                    extra_name1 = ""
                hand_control_file = os.path.join(self.hand_control_folder, f"{data_dict['seq_name'][0]}{extra_name1}.npz")
                interpolation_data = np.load(hand_control_file)
                control_idx = interpolation_data['control_idx'].tolist()
                hand_control_position = interpolation_data['positions'].reshape(1, -1, len(control_idx), 3) - data_dict['trans2joint'].cpu().numpy().reshape(1, 1, 1, 3)
                all_res_list[:,:,:12] = torch.from_numpy(interpolation_data['object_motion']).to(all_res_list.device)
            else:
                hand_control_position = data_dict['ori_motion'][:,:,:24*3].reshape(1, -1, 24, 3)[:,:,22:].cpu().numpy()
                control_idx = [22, 23]
   
        pred_normalized_obj_trans = all_res_list[:, :, :3] # N X T X 3 
        if not vis_gt and self.hand_control_folder is not None:
            pred_seq_com_pos = pred_normalized_obj_trans
        else:
            pred_seq_com_pos = self.ds.de_normalize_obj_pos_min_max(pred_normalized_obj_trans)

        if self.use_random_frame_bps:
            if not vis_gt and self.hand_control_folder is not None:
                reference_obj_rot_mat = torch.from_numpy(interpolation_data['reference_obj_rot_mat']).to(all_res_list.device)
            else:
                reference_obj_rot_mat = data_dict['reference_obj_rot_mat'] # N X 1 X 3 X 3 

            pred_obj_rel_rot_mat = all_res_list[:, :, 3:3+9].reshape(num_seq, -1, 3, 3) # N X T X 3 X 3
            pred_obj_rot_mat = self.ds.rel_rot_to_seq(pred_obj_rel_rot_mat, reference_obj_rot_mat)
        else:
            pred_obj_rot_mat = all_res_list[:, :, 3:3+9].reshape(num_seq, -1, 3, 3)
            
        num_joints = 24
    
        normalized_global_jpos = all_res_list[:, :, 3+9:3+9+num_joints*3].reshape(num_seq, -1, num_joints, 3)
        global_jpos = self.ds.de_normalize_jpos_min_max(normalized_global_jpos.reshape(-1, num_joints, 3))
        global_jpos = global_jpos.reshape(num_seq, -1, num_joints, 3) # N X T X 22 X 3 

        # For putting human into 3D scene 
        if move_to_planned_path is not None:
            pred_seq_com_pos = pred_seq_com_pos + move_to_planned_path
            global_jpos = global_jpos + move_to_planned_path[:, :, None, :]

        global_root_jpos = global_jpos[:, :, 0, :].clone() # N X T X 3 

        global_rot_6d = all_res_list[:, :, 3+9+24*3:3+9+24*3+22*6].reshape(num_seq, -1, 22, 6)
        global_rot_mat = transforms.rotation_6d_to_matrix(global_rot_6d) # N X T X 22 X 3 X 3 

        trans2joint = data_dict['trans2joint'].to(all_res_list.device).squeeze(1) # BS X  3 
        seq_len = data_dict['seq_len'] # BS, should only be used during for single window generation. 
        if all_res_list.shape[0] != trans2joint.shape[0]:
            trans2joint = trans2joint.repeat(num_seq, 1, 1) # N X 24 X 3 
            seq_len = seq_len.repeat(num_seq) # N 
        seq_len = seq_len.detach().cpu().numpy() # N 

        for idx in range(num_seq):
            curr_global_rot_mat = global_rot_mat[idx] # T X 22 X 3 X 3 
            curr_local_rot_mat = quat_ik_torch(curr_global_rot_mat) # T X 22 X 3 X 3 
            curr_local_rot_aa_rep = transforms.matrix_to_axis_angle(curr_local_rot_mat) # T X 22 X 3 
            
            curr_global_root_jpos = global_root_jpos[idx] # T X 3
     
            curr_trans2joint = trans2joint[idx:idx+1].clone() # 1 X 3 
            
            root_trans = curr_global_root_jpos + curr_trans2joint.to(curr_global_root_jpos.device) # T X 3 

            # Generate global joint position 
            bs = 1
            betas = data_dict['betas'][0]
            gender = data_dict['gender'][0]
            
            curr_gt_obj_rot_mat = data_dict['obj_rot_mat'][0] # T X 3 X 3
            curr_gt_obj_com_pos = data_dict['obj_com_pos'][0] # T X 3 
            
            curr_obj_rot_mat = pred_obj_rot_mat[idx] # T X 3 X 3 
            curr_obj_quat = transforms.matrix_to_quaternion(curr_obj_rot_mat)
            curr_obj_rot_mat = transforms.quaternion_to_matrix(curr_obj_quat) # Potentially avoid some prediction not satisfying rotation matrix requirements.

            if curr_object_name is not None: 
                object_name = curr_object_name 
            else:
                curr_seq_name = data_dict['seq_name'][0]
                object_name = data_dict['obj_name'][0]

            if not vis_gt:
                maskmimic_path = self.maskmimic_path
                maskmimic_file = os.path.join(maskmimic_path, f'trajectory_pose_aa_{self.maskmimic_id}_0.npz')
                maskmimic_data = np.load(maskmimic_file)
                best_poses = maskmimic_data['pose'][0,:all_res_list.shape[1]]

                required_length = curr_global_root_jpos.shape[0]
                trans_data = maskmimic_data['trans'][0,:all_res_list.shape[1]]
                trans_length = trans_data.shape[0]
                if trans_length < required_length:
                    padding = np.ones((required_length - trans_length, trans_data.shape[1])) * trans_data[-1:]
                    trans_data = np.concatenate([trans_data, padding], axis=0)
                root_transl = torch.from_numpy(trans_data).float().cuda()

                pose_data = best_poses
                pose_length = pose_data.shape[0]
                if pose_length < required_length:
                    padding = np.ones((required_length - pose_length, pose_data.shape[1])) * pose_data[-1:]
                    pose_data = np.concatenate([pose_data, padding], axis=0)
                
                curr_local_rot_aa_rep = torch.from_numpy(pose_data).float().reshape(-1, 24, 3)[:, :22].cuda()

                root_trans_init = root_transl[0:1].clone()
                root_trans_init[:, 2] = 0
                root_transl -= root_trans_init

                # Get human verts 
                mesh_jnts, mesh_verts, mesh_faces = \
                    run_smplx_model(root_transl[None].cuda(), curr_local_rot_aa_rep[None].cuda(), \
                    betas.cuda(), [gender], self.ds.bm_dict, return_joints24=True)
            else:
                # Get human verts 
                mesh_jnts, mesh_verts, mesh_faces = \
                    run_smplx_model(root_trans[None].cuda(), curr_local_rot_aa_rep[None].cuda(), \
                    betas.cuda(), [gender], self.ds.bm_dict, return_joints24=True)
            
            diff = root_trans[0:1] - curr_trans2joint.to(curr_global_root_jpos.device) - mesh_jnts[0,0,0:1]
            mesh_jnts = mesh_jnts + diff
            mesh_verts = mesh_verts + diff

            if self.test_unseen_objects:
                # Get object verts 
                obj_rest_verts, obj_mesh_faces = self.unseen_seq_ds.load_rest_pose_object_geometry(object_name)
                obj_rest_verts = torch.from_numpy(obj_rest_verts)

                gt_obj_mesh_verts = self.unseen_seq_ds.load_object_geometry_w_rest_geo(curr_gt_obj_rot_mat, \
                            curr_gt_obj_com_pos, obj_rest_verts.float())

                obj_mesh_verts = self.unseen_seq_ds.load_object_geometry_w_rest_geo(curr_obj_rot_mat, \
                            pred_seq_com_pos[idx], obj_rest_verts.float().to(pred_seq_com_pos.device))
            
            elif self.test_behave:
                # Get object verts 
                obj_rest_verts, obj_mesh_faces = self.behave_seq_ds.load_rest_pose_object_geometry(object_name)
                obj_rest_verts = torch.from_numpy(obj_rest_verts)

                gt_obj_mesh_verts = self.behave_seq_ds.load_object_geometry_w_rest_geo(curr_gt_obj_rot_mat, \
                            curr_gt_obj_com_pos, obj_rest_verts.float())

                obj_mesh_verts = self.behave_seq_ds.load_object_geometry_w_rest_geo(curr_obj_rot_mat, \
                            pred_seq_com_pos[idx], obj_rest_verts.float().to(pred_seq_com_pos.device))

            elif self.test_behave_motion:
                # Get object verts 
                obj_rest_verts, obj_mesh_faces = self.behave_motion_seq_ds.load_rest_pose_object_geometry(object_name)
                obj_rest_verts = torch.from_numpy(obj_rest_verts)

                gt_obj_mesh_verts = self.behave_motion_seq_ds.load_object_geometry_w_rest_geo(curr_gt_obj_rot_mat, \
                            curr_gt_obj_com_pos, obj_rest_verts.float())

                obj_mesh_verts = self.behave_motion_seq_ds.load_object_geometry_w_rest_geo(curr_obj_rot_mat, \
                            pred_seq_com_pos[idx], obj_rest_verts.float().to(pred_seq_com_pos.device))
                    
            else:
                # Get object verts 
                obj_rest_verts, obj_mesh_faces = self.ds.load_rest_pose_object_geometry(object_name)
                obj_rest_verts = torch.from_numpy(obj_rest_verts)

                gt_obj_mesh_verts = self.ds.load_object_geometry_w_rest_geo(curr_gt_obj_rot_mat, \
                            curr_gt_obj_com_pos, obj_rest_verts.float())
                obj_mesh_verts = self.ds.load_object_geometry_w_rest_geo(curr_obj_rot_mat, \
                            pred_seq_com_pos[idx], obj_rest_verts.float().to(pred_seq_com_pos.device))

            actual_len = seq_len[idx]

            human_jnts_list.append(mesh_jnts[0])
            human_verts_list.append(mesh_verts[0]) 
            obj_verts_list.append(obj_mesh_verts)
            trans_list.append(root_trans) 

            human_mesh_faces_list.append(mesh_faces)
            obj_mesh_faces_list.append(obj_mesh_faces) 

            if self.compute_metrics:
                continue 


            if dest_mesh_vis_folder is None:
                if vis_tag is None:
                    dest_mesh_vis_folder = os.path.join(self.vis_folder, "blender_mesh_vis", str(step))
                else:
                    dest_mesh_vis_folder = os.path.join(self.vis_folder, vis_tag, str(step))
            
            if not os.path.exists(dest_mesh_vis_folder):
                os.makedirs(dest_mesh_vis_folder)

            if vis_gt:
                ball_mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                "ball_objs_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt")
                mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                "objs_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt")
                out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                                "imgs_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt")
                out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                                "vid_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt.mp4")
            else:
                ball_mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                "ball_objs_step_"+str(step)+"_bs_idx_"+str(idx))
                mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                "objs_step_"+str(step)+"_bs_idx_"+str(idx))
                out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                                "imgs_step_"+str(step)+"_bs_idx_"+str(idx))
                out_sideview_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                                "sideview_imgs_step_"+str(step)+"_bs_idx_"+str(idx))
                
                out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                                "vid_step_"+str(step)+"_bs_idx_"+str(idx)+".mp4")
                out_sideview_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                                "sideview_vid_step_"+str(step)+"_bs_idx_"+str(idx)+".mp4")
                
                if vis_wo_scene:
                    ball_mesh_save_folder = ball_mesh_save_folder + "_vis_no_scene"
                    mesh_save_folder = mesh_save_folder + "_vis_no_scene"
                    out_rendered_img_folder = out_rendered_img_folder + "_vis_no_scene"
                    out_vid_file_path = out_vid_file_path.replace(".mp4", "_vis_no_scene.mp4") 

            if text_anno is not None:
                out_vid_file_path.replace(".mp4", "_"+text_anno.replace(" ", "_")+".mp4")

            start_obj_com_pos = data_dict['ori_obj_motion'][0, 0:1, :3] # 1 X 3 
            if planned_end_obj_com is None:
                end_obj_com_pos = data_dict['ori_obj_motion'][0, actual_len-1:actual_len, :3] # 1 X 3 
            else:
                end_obj_com_pos = planned_end_obj_com[idx].to(start_obj_com_pos.device) # 1 X 3 
            start_object_mesh = gt_obj_mesh_verts[0] # Nv X 3 
            if move_to_planned_path is not None:
                start_object_mesh += move_to_planned_path[idx].to(start_object_mesh.device) 
            end_object_mesh = gt_obj_mesh_verts[actual_len-1] # Nv X 3 
            if not os.path.exists(ball_mesh_save_folder):
                os.makedirs(ball_mesh_save_folder)
            ball_mesh_path = os.path.join(ball_mesh_save_folder, "conditions.ply")
            start_mesh_path = os.path.join(ball_mesh_save_folder, "start_object.ply")
            end_mesh_path = os.path.join(ball_mesh_save_folder, "end_object.ply") 
            if save_mesh_file:
                self.export_to_mesh(start_object_mesh, obj_mesh_faces, start_mesh_path)
           
            if planned_waypoints_pos is not None:
                if planned_path_floor_height is None:
                    num_waypoints = planned_waypoints_pos[idx].shape[0]
                    for tmp_idx in range(num_waypoints):
                        if (tmp_idx+1) % 4 != 0:
                            planned_waypoints_pos[idx, tmp_idx, 2] = 0.05 
                else:
                    planned_waypoints_pos[idx, :, 2] = planned_path_floor_height + 0.05 

                if move_to_planned_path is None:
                    ball_for_vis_data = torch.cat((start_obj_com_pos, \
                                    planned_waypoints_pos[idx].to(end_obj_com_pos.device), end_obj_com_pos), dim=0) 
                else:
                    ball_for_vis_data = torch.cat((start_obj_com_pos+move_to_planned_path[idx].to(start_obj_com_pos.device), \
                                    planned_waypoints_pos[idx].to(end_obj_com_pos.device), end_obj_com_pos), dim=0) 
                # ball_for_vis_data: K X 3 
                #  
                if cano_quat is not None:
                    cano_quat_for_ball = transforms.quaternion_invert(cano_quat[0:1].repeat(ball_for_vis_data.shape[0], \
                                                        1)) # K X 4 
                    ball_for_vis_data = transforms.quaternion_apply(cano_quat_for_ball.to(ball_for_vis_data.device), ball_for_vis_data)
    
                if save_mesh_file:
                    self.create_ball_mesh(ball_for_vis_data, ball_mesh_path)
            else:
                if vis_gt:
                    curr_cond_mask = cond_mask[idx, :, 0] # T 
                    waypoints_list = [start_obj_com_pos]
                    end_obj_com_pos_xy = end_obj_com_pos.clone()
                    # end_obj_com_pos_xy[:, 2] = 0.05
                    waypoints_list.append(end_obj_com_pos_xy)
                    curr_timesteps = curr_cond_mask.shape[0]
                    for t_idx in range(curr_timesteps):
                        if curr_cond_mask[t_idx] == 0 and t_idx != 0:
                            selected_waypoint = data_dict['ori_obj_motion'][idx, t_idx:t_idx+1, :3]
                            selected_waypoint[:, 2] = 0.05
                            waypoints_list.append(selected_waypoint)
                
                    ball_for_vis_data = torch.cat(waypoints_list, dim=0) # K X 3 
                    if save_mesh_file:
                        self.create_ball_mesh(ball_for_vis_data, ball_mesh_path)
                else:
                    waypoints_list = []
                    
                    temp_control = hand_control_position[idx,:].reshape(-1, len(control_idx), 3) # + curr_trans2joint.to(data_dict['ori_motion'].device).reshape(1, 1, 3)

                    for t_idx in range(actual_len):
                        if save_mesh_file:
                            if len(control_idx) > 1:
                                self.create_ball_mesh_new(temp_control[t_idx, 0], ball_mesh_path, name='lh', idx=t_idx)
                                self.create_ball_mesh_new(temp_control[t_idx, 1], ball_mesh_path, name='rh', idx=t_idx)
                            else:
                                self.create_ball_mesh_new(temp_control[t_idx, 0], ball_mesh_path, name='rh', idx=t_idx)


            # For faster debug visualization!!
            # mesh_verts = mesh_verts[:, ::30, :, :] # 1 X T X Nv X 3
            # obj_mesh_verts = obj_mesh_verts[::30, :, :] # T X Nv X 3 

            if cano_quat is not None:
                # mesh_verts: 1 X T X Nv X 3 
                # obj_mesh_verts: T X Nv' X 3 
                # cano_quat: K X 4 
                cano_quat_for_human = transforms.quaternion_invert(cano_quat[0:1][None].repeat(mesh_verts.shape[1], \
                                                            mesh_verts.shape[2], 1)) # T X Nv X 4 
                cano_quat_for_obj = transforms.quaternion_invert(cano_quat[0:1][None].repeat(obj_mesh_verts.shape[0], \
                                                            obj_mesh_verts.shape[1], 1)) # T X Nv X 4
                mesh_verts = transforms.quaternion_apply(cano_quat_for_human.to(mesh_verts.device), mesh_verts[0])
                obj_mesh_verts = transforms.quaternion_apply(cano_quat_for_obj.to(obj_mesh_verts.device), obj_mesh_verts) 

                if save_mesh_file:
                    save_verts_faces_to_mesh_file_w_object(mesh_verts.detach().cpu().numpy(), mesh_faces.detach().cpu().numpy(), \
                        obj_mesh_verts.detach().cpu().numpy(), obj_mesh_faces, mesh_save_folder)
            else:
                if gen_long_seq:
                    if save_mesh_file:
                        save_verts_faces_to_mesh_file_w_object(mesh_verts.detach().cpu().numpy()[0], \
                            mesh_faces.detach().cpu().numpy(), \
                            obj_mesh_verts.detach().cpu().numpy(), obj_mesh_faces, mesh_save_folder)
                else: # For single window
                    if save_mesh_file:
                        save_verts_faces_to_mesh_file_w_object(mesh_verts.detach().cpu().numpy()[0][:seq_len[idx]], \
                            mesh_faces.detach().cpu().numpy(), \
                            obj_mesh_verts.detach().cpu().numpy()[:seq_len[idx]], obj_mesh_faces, mesh_save_folder)

            # continue 
            if move_to_planned_path is not None:
                curr_scene_name = planned_scene_names.split("/")[-4]
                root_blend_file_folder = "/move/u/jiamanli/datasets/FullBodyManipCapture/processed_manip_data/replica_blender_files"
                
                # Top-down view visualization 
                curr_scene_blend_path = os.path.join(root_blend_file_folder, self.test_scene_name+"_topview.blend")
                # if not os.path.exists(dest_out_vid_path):
                if not save_obj_only:
                    run_blender_rendering_and_save2video(mesh_save_folder, out_rendered_img_folder, out_vid_file_path, \
                            condition_folder=ball_mesh_save_folder, vis_object=True, vis_condition=True, \
                            scene_blend_path=curr_scene_blend_path, fps=self.opt.fps) 
                
            else:
                floor_blend_path = os.path.join(self.data_root_folder, "blender_files/floor_colorful_mat.blend")
                if planned_end_obj_com is not None:
                    if dest_out_vid_path is None:
                        dest_out_vid_path = out_vid_file_path.replace(".mp4", "_wo_scene.mp4")

                    if not os.path.exists(dest_out_vid_path):
                        if not save_obj_only:
                            run_blender_rendering_and_save2video(mesh_save_folder, out_rendered_img_folder, dest_out_vid_path, \
                                condition_folder=ball_mesh_save_folder, vis_object=True, vis_condition=True, \
                                scene_blend_path=floor_blend_path, fps=self.opt.fps)
                    
                else:
                    if dest_out_vid_path is None:
                        dest_out_vid_path = out_vid_file_path.replace(".mp4", "_wo_scene.mp4")
                    if not os.path.exists(dest_out_vid_path):
                        if not vis_gt: # Skip GT visualiation 
                            if not save_obj_only:
                                run_blender_rendering_and_save2video(mesh_save_folder, out_rendered_img_folder, dest_out_vid_path, \
                                        condition_folder=ball_mesh_save_folder, vis_object=True, vis_condition=True, \
                                        scene_blend_path=floor_blend_path, fps=self.opt.fps)

                    if vis_gt: 
                        if not save_obj_only:
                            run_blender_rendering_and_save2video(mesh_save_folder, out_rendered_img_folder, dest_out_vid_path, \
                                    condition_folder=ball_mesh_save_folder, vis_object=True, vis_condition=True, \
                                    scene_blend_path=floor_blend_path, fps=self.opt.fps)
                    

            if idx > 1:
                break 

        return human_verts_list, human_jnts_list, trans_list, global_rot_mat, pred_seq_com_pos, pred_obj_rot_mat, \
        obj_verts_list, human_mesh_faces_list, obj_mesh_faces_list, dest_out_vid_path  

    
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
        use_wandb=False 
    )
   
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
    parser.add_argument("--fps", type=int, default=30, help='fps for visualization')
    parser.add_argument("--test_behave", action="store_true")
    parser.add_argument("--test_behave_motion", action="store_true") # currently only for window generation

    parser.add_argument("--maskmimic_path", type=str, default=None, help='maskmimic output path')

    parser.add_argument("--hand_control_folder", type=str, default=None, help='maskmimic output path')

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = os.path.join(opt.project, opt.exp_name)
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    if opt.test_sample_res:
        run_sample(opt, device)
    
