#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_path=$1
training_data="InterPose_AMASS"
control_setting="hands"
success_threshold=0.5

source ~/miniconda3/etc/profile.d/conda.sh

conda activate interpose

# obtain hands waypoints by sampling from object mesh
python utils/get_hand_contacts_positions.py \
--window=120 \
--batch_size=1 \
--data_root_folder="./processed_data" \
--save_res_folder="./chois_output/temp" \
--input_first_human_pose \
--add_language_condition \
--use_object_keypoints \
--add_semantic_contact_labels \
--loss_w_feet=1 \
--loss_w_fk=0.5 \
--loss_w_obj_pts=1 \
--test_sample_res \
--use_guidance_in_denoising \
--for_quant_eval \
--fps=30 \
--save_obj_only 

python utils/get_hand_contacts_positions.py \
--window=120 \
--batch_size=1 \
--data_root_folder="./processed_data" \
--save_res_folder="./chois_output/temp" \
--input_first_human_pose \
--add_language_condition \
--use_object_keypoints \
--add_semantic_contact_labels \
--loss_w_feet=1 \
--loss_w_fk=0.5 \
--loss_w_obj_pts=1 \
--test_sample_res \
--use_guidance_in_denoising \
--for_quant_eval \
--fps=30 \
--test_behave_motion \
--save_obj_only 

# Use maskedmimic to generate human pose
cd third-party/ProtoMotion_for_InterPose
conda activate protomotions_interpose

sh scripts/zero_shot_HOI.sh  \
"${model_path}" "${training_data}" "${control_setting}" "${success_threshold}"


cd ..

conda activate interpose

# Calculate final metrics

python eval_zero_shot_HOI.py \
--window=120 \
--batch_size=32 \
--data_root_folder="./processed_data" \
--save_res_folder="./chois_output/maskedmimic_${training_data}_on_omomo_motion" \
--input_first_human_pose \
--add_language_condition \
--use_object_keypoints \
--add_semantic_contact_labels \
--loss_w_feet=1 \
--loss_w_fk=0.5 \
--loss_w_obj_pts=1 \
--test_sample_res \
--use_guidance_in_denoising \
--for_quant_eval \
--maskmimic_path third-party/ProtoMotion_for_InterPose/outputs/${training_data}_${control_setting}_omomo_${success_threshold}_interpolation \
--hand_control_folder "third-party/ProtoMotion_for_InterPose/data/Dataset/omomo_data/test_set_hands_control_interpolation_position" \
--save_obj_only \
--fps=30


python eval_zero_shot_HOI.py \
--window=120 \
--batch_size=32 \
--data_root_folder="./processed_data" \
--save_res_folder="./chois_output/maskedmimic_${training_data}_on_behave_motion" \
--input_first_human_pose \
--add_language_condition \
--use_object_keypoints \
--add_semantic_contact_labels \
--loss_w_feet=1 \
--loss_w_fk=0.5 \
--loss_w_obj_pts=1 \
--test_sample_res \
--use_guidance_in_denoising \
--for_quant_eval \
--maskmimic_path third-party/ProtoMotion_for_InterPose/outputs/${training_data}_${control_setting}_omomo_${success_threshold}_interpolation \
--hand_control_folder "third-party/ProtoMotion_for_InterPose/data/Dataset/behave_data/test_set_hands_control_interpolation_position" \
--save_obj_only \
--test_behave_motion \
--fps=30
