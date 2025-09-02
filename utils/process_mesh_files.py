import os
import sys
import numpy as np
import torch
import trimesh
import torch.nn.functional as F
import pytorch3d.transforms as transforms 
from scipy.interpolate import BSpline, interp1d, make_interp_spline, PchipInterpolator

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

def load_rest_pose_object_geometry(rest_object_geo_folder, object_name):
    rest_obj_path = os.path.join(rest_object_geo_folder, object_name+".ply")
    
    mesh = trimesh.load_mesh(rest_obj_path)
    rest_verts = np.asarray(mesh.vertices) # Nv X 3
    obj_mesh_faces = np.asarray(mesh.faces) # Nf X 3

    return rest_verts, obj_mesh_faces 
    
def load_object_geometry_w_rest_geo(obj_rot, obj_com_pos, rest_verts):
    # obj_scale: T, obj_rot: T X 3 X 3, obj_com_pos: T X 3, rest_verts: Nv X 3 
    rest_verts = rest_verts[None].repeat(obj_rot.shape[0], 1, 1)
    transformed_obj_verts = obj_rot.bmm(rest_verts.transpose(1, 2)) + obj_com_pos[:, :, None]
    transformed_obj_verts = transformed_obj_verts.transpose(1, 2) # T X Nv X 3 

    return transformed_obj_verts 

def create_ball_mesh_new(center_pos, ball_mesh_path, name, idx, ball_color=[22, 173, 100], radius=0.05):
    # center_pos: K X 3  
    ball_color = np.asarray(ball_color) # green 

    ball_mesh = trimesh.primitives.Sphere(radius=radius, center=center_pos)
    
    dest_ball_mesh = trimesh.Trimesh(
        vertices=ball_mesh.vertices,
        faces=ball_mesh.faces,
        vertex_colors=ball_color,
        process=False)

    result = trimesh.exchange.ply.export_ply(dest_ball_mesh, encoding='ascii')
    output_file = open(ball_mesh_path.replace(".ply", "_"+name+"_"+str(idx)+".ply"), "wb+")
    output_file.write(result)
    output_file.close()

def export_to_mesh(mesh_verts, mesh_faces, mesh_path):
    dest_mesh = trimesh.Trimesh(
        vertices=mesh_verts,
        faces=mesh_faces,
        process=False)

    result = trimesh.exchange.ply.export_ply(dest_mesh, encoding='ascii')
    output_file = open(mesh_path, "wb+")
    output_file.write(result)
    output_file.close()

def save_verts_faces_to_mesh_file_w_object(mesh_verts, mesh_faces, obj_verts, obj_faces, save_mesh_folder, object_name_all=None):
    # mesh_verts: T X Nv X 3 
    # mesh_faces: Nf X 3 
    if not os.path.exists(save_mesh_folder):
        os.makedirs(save_mesh_folder)

    if isinstance(mesh_verts, list):
        num_meshes = mesh_verts[0].shape[0]
    else:
        num_meshes = mesh_verts.shape[0]
    
    for idx in range(num_meshes):
        if isinstance(mesh_verts, list):
            for i in range(len(mesh_verts)):
                mesh = trimesh.Trimesh(vertices=mesh_verts[i][idx],
                                faces=mesh_faces[i])
                curr_mesh_path = os.path.join(save_mesh_folder, "%05d"%(idx)+f"_person{i}.ply")
                mesh.export(curr_mesh_path)
        else:
            mesh = trimesh.Trimesh(vertices=mesh_verts[idx],
                            faces=mesh_faces)
            curr_mesh_path = os.path.join(save_mesh_folder, "%05d"%(idx)+".ply")
            mesh.export(curr_mesh_path)

        if isinstance(obj_verts, list):
            for i in range(len(obj_verts)):
                obj_mesh = trimesh.Trimesh(vertices=obj_verts[i][idx],
                                faces=obj_faces[i])
                curr_obj_mesh_path = os.path.join(save_mesh_folder, "%05d"%(idx)+f"_object_{object_name_all[i]}.ply")
                obj_mesh.export(curr_obj_mesh_path)
        else:
            obj_mesh = trimesh.Trimesh(vertices=obj_verts[idx],
                            faces=obj_faces)
            curr_obj_mesh_path = os.path.join(save_mesh_folder, "%05d"%(idx)+"_object.ply")
            obj_mesh.export(curr_obj_mesh_path)


def get_default_keyframes(T):
    """Generate default keyframes for interpolation"""
    keyframes = [0, 1] + list(range(29, T, 30))
    if T - 1 not in keyframes:
        keyframes.append(T - 1)
    return keyframes


def detect_movement_boundaries(key_positions, keyframes):
    """
    Detect movement start and stop indices from keyframe positions
    
    Args:
        key_positions: (bs, num_keyframes, 3) positions at keyframes
        keyframes: list of keyframe indices
    
    Returns:
        movement_start: list of movement start indices
        movement_stop: list of movement stop indices
    """
    # Determine movement start and stop
    movement_mask = (key_positions[:, 1:, :2] - key_positions[:, :-1, :2]).norm(dim=-1) > 1e-2
    movement_start_temp = movement_mask.int().argmax(dim=-1, keepdim=True)  # First movement index
    movement_stop_temp = (movement_mask.int().cumsum(dim=-1) == movement_mask.int().sum(dim=-1, keepdim=True)).int().argmax(dim=-1, keepdim=True)  # Last movement index
    
    movement_start = [keyframes[movement_start_temp[0,0]]]
    if movement_mask[0,movement_stop_temp[0,0]]==False:
        movement_stop = [keyframes[movement_stop_temp[0,0]]]  
    else:
        movement_stop = [keyframes[-1]]
    
    return movement_start, movement_stop

def detect_exact_boundaries(movement_start, movement_stop, keyframes, 
                            key_positions, init_verts=None, init_human_pos=None, 
                            heuristic=True, start_ratio=0.5, end_ratio=0.5):
    # Find indices of the start and stop keyframes
    for idx, keyframe in enumerate(keyframes):
        if keyframe == movement_start[0]:
            start_idx = idx
        if keyframe == movement_stop[0]:
            stop_idx = idx

    # Adjust movement_start and movement_stop based on velocity
    if heuristic and start_idx < stop_idx and start_idx + 2 < len(keyframes):
        # Compute the distances between consecutive keyframes
        distance1 = (key_positions[:, start_idx + 1, :2] - key_positions[:, start_idx, :2]).norm(dim=-1)
        distance2 = (key_positions[:, start_idx + 2, :2] - key_positions[:, start_idx + 1, :2]).norm(dim=-1)
        # Compute velocity for the second segment
        velocity2 = distance2 / (keyframes[start_idx + 2] - keyframes[start_idx + 1])
        # Estimate the start frame
        estimated_frame = int(distance1 / (0.5 * velocity2)) + 1
        actual_start = max(keyframes[start_idx + 1] - estimated_frame, keyframes[start_idx])

        # Ensure start is not too early or too late
        actual_start = min(actual_start, keyframes[start_idx + 1] - int(distance1 / 0.03) + 1)
        actual_start = max(actual_start, keyframes[start_idx + 1] - int(distance1 / 0.02) + 1)
        actual_start = max(actual_start, keyframes[start_idx])
    elif not heuristic and start_idx < stop_idx and start_idx + 1 < len(keyframes):
        actual_start = keyframes[start_idx + 1] - int((keyframes[start_idx + 1] - keyframes[start_idx]) * start_ratio)
    else:
        actual_start = movement_start[0]

    # Adjust stop frame if vertical movement is small
    if stop_idx - 2 >= 0 and key_positions[0, -1, 2] - key_positions[0, 0, 2] < 0.05:
        distance1 = (key_positions[:, stop_idx - 1, :2] - key_positions[:, stop_idx, :2]).norm(dim=-1)
        distance2 = (key_positions[:, stop_idx - 2, :2] - key_positions[:, stop_idx - 1, :2]).norm(dim=-1)
        velocity2 = distance2 / (keyframes[stop_idx - 1] - keyframes[stop_idx - 2])
        estimated_frame = int(distance1 / (0.5 * velocity2)) + 1
        actual_stop = min(keyframes[stop_idx - 1] + estimated_frame, keyframes[stop_idx])

        # Ensure stop is not too early or too late
        actual_stop = max(actual_stop, keyframes[stop_idx - 1] + int(distance1 / 0.03) + 1)
        actual_stop = min(actual_stop, keyframes[stop_idx - 1] + int(distance1 / 0.02) + 1)
        actual_stop = min(actual_stop, keyframes[stop_idx])
    elif not heuristic and stop_idx - 1 >= 0 and key_positions[0, -1, 2] - key_positions[0, 0, 2] < 0.05:
        actual_stop = keyframes[stop_idx - 1] + int((keyframes[stop_idx] - keyframes[stop_idx - 1]) * end_ratio)
    else:
        actual_stop = movement_stop[0]

    # Adjust start based on initial human positions and object vertices
    if init_human_pos is not None and init_verts is not None:
        min_distances = []
        for hand_idx in range(init_human_pos.shape[0]):  # For each hand
            hand_pos = init_human_pos[hand_idx]  # (3,)
            # Compute distances from this hand to all object vertices
            distances_to_verts = torch.norm(init_verts - hand_pos.unsqueeze(0), dim=1)  # (num_verts,)
            min_dist = torch.min(distances_to_verts)
            min_distances.append(min_dist)
        
        farest_hand_dist = max(min_distances)
        actual_start = max(actual_start, int(farest_hand_dist / 0.04) + 1)  # Avoid starting too early
        actual_start = min(actual_start, int(farest_hand_dist / 0.02) + 1)  # Avoid starting too late

    movement_start = [actual_start]
    movement_stop = [actual_stop]
    return movement_start, movement_stop

    
def interpolate_positions(positions, key_pos, keyframes):
    """
    Interpolate positions using B-spline (PCHIP) interpolation.
    
    Args:
        positions: tensor of shape (batch_size, T, 3), original positions
        key_pos: tensor of shape (batch_size, num_keyframes, 3), positions at keyframes
        keyframes: list of keyframe indices corresponding to key_pos
    
    Returns:
        interp_positions: tensor of shape (batch_size, T, 3), interpolated positions
    """
    bs, T, _ = positions.shape
    device = positions.device
    
    # Initialize tensor for interpolated positions
    interp_positions = torch.zeros_like(positions, device=device)
    
    # Convert keyframe positions to NumPy for PCHIP
    key_pos = key_pos.detach().cpu().numpy()
    
    # Perform interpolation using keyframe data
    for b in range(bs):            # For each batch
        for i in range(3):         # For each coordinate x, y, z
            interp_func = PchipInterpolator(keyframes, key_pos[b, :, i])
            t_interp = np.linspace(keyframes[0], keyframes[-1], T)  # evenly spaced times
            interp = interp_func(t_interp)
            interp_positions[b, :, i] = torch.tensor(interp, device=device, dtype=torch.float32)
    
    return interp_positions



def compute_rotation_change_mask(key_positions, rotation_threshold, rotate=True):
    """
    Compute mask indicating which segments should have rotation changes
    
    Args:
        key_positions: (bs, num_keyframes, 3) positions at keyframes
        rotation_threshold: threshold for rotation changes
    
    Returns:
        rotation_change_mask: (bs, num_keyframes-1) boolean mask
    """
    # Check distances between consecutive keyframes
    keyframe_distances = (key_positions[:, 1:, :2] - key_positions[:, :-1, :2]).norm(dim=-1)  # (bs, num_keyframes-1)
    # Create mask for which segments should have rotation changes
    rotation_change_mask = keyframe_distances > rotation_threshold  # (bs, num_keyframes-1)

    rotation_change_mask = rotation_change_mask * ((key_positions[:, -1, :2] - key_positions[:, 0, :2]).norm(dim=-1) > 1) * rotate
    return rotation_change_mask


def detect_rotation_jumps(quaternions, threshold_degrees=30.0):
    if quaternions.shape[0] < 2:
        return [], []
    
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)
    
    q1 = quaternions[:-1]
    q2 = quaternions[1:]
    
    dot_products = torch.sum(q1 * q2, dim=-1)
    dot_products = torch.abs(dot_products)
    
    angles_rad = 2 * torch.acos(torch.clamp(dot_products, 0, 1))
    angles_deg = angles_rad * 180.0 / torch.pi
    
    jump_mask = angles_deg > threshold_degrees
    jump_indices = torch.where(jump_mask)[0].tolist()
    jump_angles = angles_deg[jump_mask].tolist()
    
    return jump_indices, jump_angles

def slerp(q1, q2, t):
    """Spherical linear interpolation between quaternions"""
    q1 = q1 / q1.norm(dim=-1, keepdim=True)
    q2 = q2 / q2.norm(dim=-1, keepdim=True)
    
    dot = torch.sum(q1 * q2, dim=-1, keepdim=True)
    q2 = torch.where(dot < 0, -q2, q2)
    dot = torch.abs(dot)
    
    linear_thresh = 0.95
    linear_interp = (1 - t) * q1 + t * q2
    linear_interp = linear_interp / linear_interp.norm(dim=-1, keepdim=True)
    
    theta = torch.acos(dot.clamp(0, 1))
    sin_theta = torch.sin(theta)
    safe_sin_theta = torch.where(sin_theta < 1e-6, torch.ones_like(sin_theta), sin_theta)
    
    w1 = torch.sin((1 - t) * theta) / safe_sin_theta
    w2 = torch.sin(t * theta) / safe_sin_theta
    spherical_interp = w1 * q1 + w2 * q2
    
    use_linear = dot > linear_thresh
    result = torch.where(use_linear, linear_interp, spherical_interp)
    
    return result

def smooth_rotation_jumps(quaternions, jump_indices, jump_angles, min_window=5, max_window=55):
    """Smooth detected rotational jumps using SLERP (Spherical Linear Interpolation)."""
    if not jump_indices:
        return quaternions.clone()
    
    T = quaternions.shape[0]
    smoothed_quaternions = quaternions.clone()
    
    for jump_idx, angle in zip(jump_indices, jump_angles):
        # Compute window size based on jump angle
        normalized_angle = min(angle / 180.0, 1.0)
        window_ratio = torch.sqrt(torch.tensor(normalized_angle))
        window_size = int(min_window + (max_window - min_window) * window_ratio)
        window_size = max(min_window, min(max_window, window_size))
        
        start_idx = max(0, jump_idx)
        end_idx = min(T, jump_idx + window_size)
        
        if end_idx - start_idx < 3:
            continue
        
        q_start = smoothed_quaternions[start_idx]
        q_end = smoothed_quaternions[end_idx - 1]
        
        num_frames = end_idx - start_idx
        t_values = torch.linspace(0, 1, num_frames, device=quaternions.device).unsqueeze(-1)
        
        q_start_expanded = q_start.unsqueeze(0).expand(num_frames, -1)
        q_end_expanded = q_end.unsqueeze(0).expand(num_frames, -1)
        
        interpolated_quats = slerp(q_start_expanded, q_end_expanded, t_values)
        smoothed_quaternions[start_idx:end_idx] = interpolated_quats
    
    return smoothed_quaternions


def compute_trajectory_directions_2d(interp_positions):
    """
    Compute trajectory directions using only the XY plane components.
    
    Args:
        interp_positions: (bs, T, 3) interpolated positions
    
    Returns:
        directions: (bs, T-1, 2) 2D trajectory direction vectors in XY plane
    """
    xy_positions = interp_positions[:, :, :2]  # take XY components
    
    # Compute 2D trajectory direction vectors
    directions = xy_positions[:, 1:, :] - xy_positions[:, :-1, :]
    
    # Normalize direction vectors
    directions_norm = torch.norm(directions, dim=-1, keepdim=True)
    directions_norm = torch.clamp(directions_norm, min=1e-8)
    directions = directions / directions_norm
    
    return directions


def apply_rotation_mask_to_directions_2d(directions, rotation_change_mask, keyframes, T):
    """
    Apply a rotation change mask to 2D trajectory directions.
    
    Args:
        directions: (bs, T-1, 2) 2D direction vectors
        rotation_change_mask: (bs, num_segments) boolean mask indicating segments where rotation can change
        keyframes: list of keyframe indices
        T: total number of frames
    
    Returns:
        Modified directions with rotations preserved for masked segments
    """
    bs = directions.shape[0]
    device = directions.device
    
    for b in range(bs):
        # Assign each time step to a keyframe segment
        frame_to_keyframe_segment = torch.zeros(T-1, dtype=torch.long, device=device)
        for i, kf_idx in enumerate(keyframes[:-1]):
            next_kf_idx = keyframes[i + 1]
            frame_to_keyframe_segment[kf_idx:next_kf_idx] = i
        
        # If a segment should not change rotation, keep the first frame's direction
        for t in range(T-1):
            seg_idx = frame_to_keyframe_segment[t]
            if not rotation_change_mask[b, seg_idx]:
                segment_start = keyframes[seg_idx]
                if segment_start == 0:
                    continue
                elif segment_start < T-1:
                    directions[b, t] = directions[b, segment_start-1]

    return directions


def extract_z_rotation_angle(rotation_matrix):
    """
    Extract the rotation angle around the Z-axis from a rotation matrix.
    Assumes rotation order ZYX (yaw-pitch-roll).
    
    Args:
        rotation_matrix: (3, 3) rotation matrix
    
    Returns:
        z_angle: rotation angle around the Z-axis (radians)
    """
    # Extract yaw angle (around Z-axis)
    z_angle = torch.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return z_angle


def decompose_rotation_preserve_xy(rotation_matrix):
    """
    Decompose a rotation matrix into XY-axis rotation (to preserve) and Z-axis rotation (to modify).
    
    Args:
        rotation_matrix: (3, 3) rotation matrix
    
    Returns:
        xy_base_rotation: (3, 3) base rotation containing only X and Y rotations
        original_z_angle: original rotation angle around Z-axis
    """
    # Extract original Z-axis rotation
    original_z_angle = extract_z_rotation_angle(rotation_matrix)
    
    # Create inverse Z rotation to remove original Z rotation
    cos_z = torch.cos(-original_z_angle)
    sin_z = torch.sin(-original_z_angle)
    
    inv_z_rotation = torch.tensor([
        [cos_z, -sin_z, 0.0],
        [sin_z,  cos_z, 0.0],
        [0.0,    0.0,   1.0]
    ], device=rotation_matrix.device, dtype=rotation_matrix.dtype)
    
    # Remove Z rotation to get base XY rotation
    xy_base_rotation = torch.mm(inv_z_rotation, rotation_matrix)
    
    return xy_base_rotation, original_z_angle


def compute_z_rotation_preserve_xy(forward_vec_2d, target_directions_2d, initial_rotation_matrix):
    """
    Compute new rotation matrices by modifying only the Z-axis rotation while preserving X and Y rotations.
    
    Args:
        forward_vec_2d: (2,) forward vector of the object in XY plane [x, y]
        target_directions_2d: (bs, T-1, 2) target direction vectors in XY plane
        initial_rotation_matrix: (3, 3) initial rotation matrix
    
    Returns:
        rotation_matrices: (bs, T-1, 3, 3) new rotation matrices
    """
    bs, T_minus_1, _ = target_directions_2d.shape
    device = target_directions_2d.device
    
    # Normalize forward vector
    forward_vec_2d = forward_vec_2d / torch.norm(forward_vec_2d)
    
    # Decompose initial rotation matrix
    xy_base_rotation, _ = decompose_rotation_preserve_xy(initial_rotation_matrix)
    
    rotation_matrices = torch.zeros(bs, T_minus_1, 3, 3, device=device, dtype=initial_rotation_matrix.dtype)
    
    for b in range(bs):
        for t in range(T_minus_1):
            target_dir_2d = target_directions_2d[b, t]
            
            # Check if target direction is valid
            if torch.norm(target_dir_2d) < 1e-8:
                rotation_matrices[b, t] = initial_rotation_matrix.clone()
                continue
            
            target_dir_2d = target_dir_2d / torch.norm(target_dir_2d)
            
            # Compute new Z-axis rotation angle
            forward_angle = torch.atan2(forward_vec_2d[1], forward_vec_2d[0])
            target_angle = torch.atan2(target_dir_2d[1], target_dir_2d[0])
            new_z_angle = target_angle - forward_angle
            
            # Construct new Z-axis rotation matrix
            cos_z = torch.cos(new_z_angle)
            sin_z = torch.sin(new_z_angle)
            
            new_z_rotation = torch.tensor([
                [cos_z, -sin_z, 0.0],
                [sin_z,  cos_z, 0.0],
                [0.0,    0.0,   1.0]
            ], device=device, dtype=initial_rotation_matrix.dtype)
            
            # Combine new rotation matrix: new Z rotation * XY base rotation
            rotation_matrices[b, t] = torch.mm(new_z_rotation, xy_base_rotation)
    
    return rotation_matrices


def interpolate_rotations(rotations, interp_positions, keyframes, movement_start, movement_stop, 
                                rotation_change_mask, forward_2d=None, smooth_jumps=True, jump_threshold=15.0, rotate=True):
    """
    只改变Z轴旋转的插值函数，保持X和Y轴旋转与初始状态一致
    
    Args:
        rotations: (bs, T, 9) 旋转矩阵（展平）
        interp_positions: (bs, T, 3) 插值位置
        keyframes: 关键帧索引列表
        movement_start: 运动开始索引列表
        movement_stop: 运动结束索引列表
        rotation_change_mask: (bs, num_keyframes-1) 布尔掩码
        forward_2d: (2,) 物体在XY平面的前向向量，默认为Y轴负方向 [0, -1]
        smooth_jumps: 是否进行跳跃平滑
        jump_threshold: 跳跃检测阈值（度）
    
    Returns:
        interp_rotations: (bs, T, 9) 插值后的旋转
    """
    bs, T, _ = rotations.shape
    device = rotations.device
    
    # 设置默认的2D前向向量
    if forward_2d is None or torch.norm(forward_2d) < 0.4:
        forward_2d = torch.tensor([0.0, -1.0], device=device)   # Y轴负方向
    else:
        forward_2d = forward_2d.to(device)

    forward_2d = forward_2d.to(rotations.dtype)
    
    # 初始化输出
    interp_rotations = torch.zeros_like(rotations, device=device)
    
    # 计算2D轨迹方向
    trajectory_directions_2d = compute_trajectory_directions_2d(interp_positions)
    trajectory_directions_2d = apply_rotation_mask_to_directions_2d(
        trajectory_directions_2d, rotation_change_mask, keyframes, T
    )
    
    for b in range(bs):
        start_idx = movement_start[b]
        stop_idx = movement_stop[b]
        
        # 获取初始旋转矩阵
        initial_rotation_matrix = rotations[b, 0].view(3, 3).clone()
        
        # 运动开始前保持初始旋转
        if start_idx > 0:
            interp_rotations[b, :start_idx+1, :] = rotations[b, 0:1, :].view(1, 9)
        
        if rotate:
            # 计算运动区间的旋转
            if start_idx < stop_idx and start_idx < T-1:
                # 获取运动区间的2D轨迹方向
                motion_directions_2d = trajectory_directions_2d[b, start_idx:min(stop_idx, T-1), :]
                
                if motion_directions_2d.shape[0] > 0:
                    # 计算只改变Z轴旋转的新旋转矩阵
                    alignment_rotations = compute_z_rotation_preserve_xy(
                        forward_2d, motion_directions_2d.unsqueeze(0), initial_rotation_matrix
                    ).squeeze(0)  # (T_motion, 3, 3)
                    
                    # 处理旋转变化掩码
                    for i in range(rotation_change_mask[b].shape[0]):
                        if not rotation_change_mask[b, i]:
                            # 如果该段不需要旋转变化，保持该段的第一个旋转
                            if i < len(keyframes) - 1:
                                seg_start = max(keyframes[i] - start_idx, 0)
                                seg_end = min(keyframes[i+1] - start_idx, alignment_rotations.shape[0])
                                if seg_start < alignment_rotations.shape[0] and seg_end > seg_start:
                                    if seg_start < alignment_rotations.shape[0]:
                                        constant_rotation = alignment_rotations[seg_start:seg_start+1]
                                        alignment_rotations[seg_start:seg_end] = constant_rotation.expand(seg_end-seg_start, -1, -1)
                    
                    # 将旋转矩阵存储到结果中
                    end_motion_idx = min(start_idx + alignment_rotations.shape[0], stop_idx)
                    interp_rotations[b, start_idx:end_motion_idx, :] = alignment_rotations[:end_motion_idx-start_idx].reshape(-1, 9)
                    
                    # 如果还有剩余的运动帧，用最后一个旋转填充
                    if end_motion_idx < stop_idx:
                        last_rotation = interp_rotations[b, end_motion_idx-1:end_motion_idx, :]
                        interp_rotations[b, end_motion_idx:stop_idx, :] = last_rotation.expand(stop_idx-end_motion_idx, -1)
        

            # 运动结束后保持最后的旋转
            if stop_idx < T:
                if stop_idx > 0:
                    last_rotation = interp_rotations[b, stop_idx-1:stop_idx, :]
                    interp_rotations[b, stop_idx:, :] = last_rotation.expand(T - stop_idx, -1)
                else:
                    # 如果没有运动，保持初始旋转
                    interp_rotations[b, stop_idx:, :] = rotations[b, 0:1, :].expand(T - stop_idx, -1)
            if start_idx > 0:
                interp_rotations[b, :start_idx+1, :] = rotations[b, 0:1, :].view(1, 9)

            # 角度突变检测和平滑处理
            if smooth_jumps and start_idx < stop_idx:
                motion_quaternions = transforms.matrix_to_quaternion(
                    interp_rotations[b, start_idx:stop_idx].view(-1, 3, 3)
                )
                
                if motion_quaternions.shape[0] > 1:
                    jump_indices, jump_angles = detect_rotation_jumps(motion_quaternions, jump_threshold)
                    
                    if jump_indices:
                        # print(f"Batch {b}: 检测到 {len(jump_indices)} 个旋转突变")
                        # for i, (idx, angle) in enumerate(zip(jump_indices, jump_angles)):
                        #     print(f"  突变 {i+1}: 位置 {start_idx + idx}, 角度 {angle:.1f}°")
                        
                        smoothed_motion_quaternions = smooth_rotation_jumps(
                            motion_quaternions, jump_indices, jump_angles
                        )
                        interp_rotations[b, start_idx:stop_idx, :] = transforms.quaternion_to_matrix(
                            smoothed_motion_quaternions
                        ).reshape(-1, 9)
        
        else:
            interp_rotations[b, :, :] = rotations[b, 0:1, :].view(1, 9)

    return interp_rotations


def interpolation_object_motion_new(object_motion, key_frame=None, rotation_threshold=0.5, lift_height=0., 
                                    init_verts=None, init_human_pos=None, cond_frame_idx_list=None, forward=None, rotate=True, heuristic=True, 
                                    start_ratio=0.5, end_ratio=0.5):
    """
    Interpolate object motion with position and rotation.
    
    Args:
        object_motion: (bs, T, 12) tensor containing positions and rotation matrices
        key_frame: list of keyframe indices (optional)
        rotation_threshold: threshold for detecting significant rotation changes
        lift_height: height to lift the object during motion
        init_verts: initial object vertices (num_verts, 3)
        init_human_pos: initial human hand positions (2, 3)
        cond_frame_idx_list: list of conditional frames for interpolation
        forward: optional forward direction information
        rotate: whether to apply rotation interpolation
        heuristic: whether to use heuristic for exact boundary detection
        start_ratio: ratio to adjust start index
        end_ratio: ratio to adjust stop index
    
    Returns:
        interpolated_object: (bs, T, 12) interpolated motion tensor
        movement_start: list of movement start indices  
        movement_stop: list of movement stop indices
    """
    # Extract necessary data
    bs, T, _ = object_motion.shape
    positions = object_motion[:, :, :3]  # Extract positions (bs, T, 3)
    rotations = object_motion[:, :, 3:]  # Extract rotation matrices (bs, T, 9)
    
    device = object_motion.device
    positions = positions.clone().to(dtype=torch.float32, device=device)
    
    # Define keyframes for interpolation
    if key_frame is None:
        if cond_frame_idx_list is not None:
            keyframes = [0, 1] + cond_frame_idx_list
        else:
            keyframes = get_default_keyframes(T)
        key_positions = positions[:, keyframes]
        key_positions[:,1] = key_positions[:,0]  # Ensure second keyframe matches the first
        key_positions[:,1:-1, 2] = key_positions[:,0:1, 2]  # Keep z-coordinate constant
    else:
        keyframes = sorted(set(key_frame))
        key_positions = positions[:, keyframes]
    
    # Detect movement boundaries
    movement_start, movement_stop = detect_movement_boundaries(key_positions, keyframes)
    
    if key_frame is None:
        # Further estimate exact start and stop indices
        movement_start, movement_stop = detect_exact_boundaries(
            movement_start, movement_stop, keyframes, key_positions, init_verts, init_human_pos, 
            heuristic=heuristic, start_ratio=start_ratio, end_ratio=end_ratio
        )
        
        keyframes = keyframes + [movement_start[0], movement_stop[0]]  # Add movement start and stop frames
        keyframes = sorted(set(keyframes))  # Remove duplicates and sort
        
        positions[:, :movement_start[0]+1] = positions[:, 0:1]  # Keep initial positions before movement
        positions[:, movement_stop[0]:] = positions[:, -1:]      # Keep final positions after movement
        key_positions = positions[:, keyframes]
        key_positions[:,1] = key_positions[:,0]  # Ensure second keyframe matches first

        key_z = positions[:, :, 2].clone()
        for b in range(bs):
            start_idx = movement_start[b]
            stop_idx = movement_stop[b]
            key_z[b, :start_idx+1] = positions[b, 0, 2]               # Before movement, keep initial height
            key_z[b, start_idx+1:stop_idx] = positions[b, 0, 2] + lift_height  # During movement, add lift height
            key_z[b, stop_idx:] = positions[b, -1, 2]                  # After movement, maintain last height
        key_z = key_z[:, keyframes]
        key_positions[:, :, 2] = key_z  # Update z-coordinates in key positions
    
    # Compute rotation change mask
    rotation_change_mask = compute_rotation_change_mask(key_positions, rotation_threshold, rotate=rotate)

    # Interpolate positions
    interp_positions = interpolate_positions(positions, key_positions, keyframes)
    
    # Interpolate rotations
    interp_rotations = interpolate_rotations(
        rotations, interp_positions, keyframes, 
        movement_start, movement_stop, rotation_change_mask, forward_2d=forward, rotate=rotate
    )

    # Combine results
    interpolated_object = torch.cat([interp_positions, interp_rotations], dim=-1)

    return interpolated_object, movement_start, movement_stop


def sample_point_trajectory(mesh, faces, z_range=None, human_pos=None, mode='sample', height_idx=1):
    """
    Select a point on the mesh surface at t=0 and obtain its trajectory from t=0 to T.
    
    Args:
        mesh: (T, Nv, 3) tensor of vertex coordinates
        faces: (Nf, 3) tensor or numpy array of triangle face indices
        z_range: (z_min, z_max) optional range to restrict face selection
        human_pos: (3,) tensor indicating human position
        mode: 'sample' or 'nearest' sampling mode
        height_idx: index of height dimension (default=1, y-axis)
    
    Returns:
        trajectory: (T, 3) tensor of the point trajectory
    """
    T, Nv, _ = mesh.shape
    
    # Ensure faces is a tensor
    if isinstance(faces, np.ndarray):
        faces = torch.from_numpy(faces).to(mesh.device)
    else:
        faces = faces.to(mesh.device)
    
    Nf = faces.shape[0]

    # Compute face centers at t=0
    v0 = mesh[0, faces[:, 0], :]
    v1 = mesh[0, faces[:, 1], :]
    v2 = mesh[0, faces[:, 2], :]
    face_centers = (v0 + v1 + v2) / 3  # (Nf, 3)

    # Filter faces based on z_range
    if z_range is not None:
        z_min, z_max = z_range
        valid_faces_mask = (face_centers[:, height_idx] >= z_min) & (face_centers[:, height_idx] <= z_max)
        valid_faces = faces[valid_faces_mask]
        valid_centers = face_centers[valid_faces_mask]
    else:
        valid_faces = faces
        valid_centers = face_centers

    if valid_faces.shape[0] == 0:
        raise ValueError(f"No faces found within z range {z_range}, please adjust the range!")

    # Select face based on distance to human or random
    if human_pos is not None:
        human_pos = human_pos.to(valid_centers.device)
        distances = torch.norm(valid_centers - human_pos, dim=1)
        
        if mode == 'nearest':
            face_idx = torch.argmin(distances)
        elif mode == 'sample':
            alpha = 10.0
            distances_norm = (distances - distances.min()) / (distances.max() - distances.min() + 1e-6)
            weights = torch.exp(-alpha * distances_norm)
            weights = weights / weights.sum()
            face_idx = torch.multinomial(weights, 1).item()
        else:
            raise ValueError(f"Unknown sampling mode: {mode}")
    else:
        face_idx = torch.randint(0, valid_faces.shape[0], (1,)).item()

    selected_face = valid_faces[face_idx]  # (3,)

    # Generate barycentric coordinates
    u = torch.rand((), device=mesh.device)
    v = torch.rand((), device=mesh.device)
    if u + v > 1:
        u, v = 1 - u, 1 - v
    w = 1 - u - v

    # Compute trajectory from t=0 to T
    trajectory = []
    for t in range(T):
        vt0 = mesh[t, selected_face[0]]
        vt1 = mesh[t, selected_face[1]]
        vt2 = mesh[t, selected_face[2]]
        point_t = u * vt0 + v * vt1 + w * vt2
        trajectory.append(point_t)

    return torch.stack(trajectory, dim=0)  # (T, 3)



def is_axis_separating(A, B, axis):
    """
    Determine whether the projections of two OBBs on the given axis are separated.
    A and B are the vertex sets of two OBBs.
    axis is the current separating axis.
    """
    if np.all(axis == 0):  # Ignore zero vectors
        return False

    axis = axis / np.linalg.norm(axis)  # Normalize the axis
    proj_a = np.dot(A, axis)
    proj_b = np.dot(B, axis)

    return proj_a.max() < proj_b.min() or proj_b.max() < proj_a.min()

def get_obb_vertices(center, rotation, extents):
    """
    Compute the 8 vertices of an OBB.
    """
    corners = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]) * extents
    rotated_corners = np.dot(corners, rotation.T)  # Rotate then translate
    return rotated_corners + center

def check_obb_intersection(obb1, obb2):
    """
    Check whether two 3D OBBs intersect.
    obb1 and obb2 are tuples in the format: (center, rotation matrix, half-extents)
    """

    center1, rotation1, extents1 = obb1
    center2, rotation2, extents2 = obb2

    # Get the 8 vertices of each OBB
    vertices1 = get_obb_vertices(center1, rotation1, extents1)
    vertices2 = get_obb_vertices(center2, rotation2, extents2)

    # The principal axes of each OBB (as row vectors)
    axes1 = [rotation1[i, :] for i in range(3)]
    axes2 = [rotation2[i, :] for i in range(3)]

    # Generate all 15 potential separating axes
    axes = []

    # 1. The 3 axes of each OBB
    axes.extend(axes1)
    axes.extend(axes2)

    # 2. Cross products between axes of the two OBBs (9 axes)
    for a1 in axes1:
        for a2 in axes2:
            cross_product = np.cross(a1, a2)
            if np.linalg.norm(cross_product) > 1e-6:  # Avoid near-zero vectors
                axes.append(cross_product)

    # Perform the Separating Axis Theorem (SAT) test
    for axis in axes:
        if is_axis_separating(vertices1, vertices2, axis):
            return False  # Found a separating axis, OBBs do not intersect

    return True  # No separating axis found, OBBs intersect


def round_list(input_list, around_number=3):
    return [round(x, around_number) for x in input_list]