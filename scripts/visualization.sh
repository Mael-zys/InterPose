#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

training_data="InterPose_AMASS"
control_setting="hands"
success_threshold=0.5


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
--test_behave_motion \
--fps=30