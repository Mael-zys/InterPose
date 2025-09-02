
model_path=$1

# single agent
CUDA_VISIBLE_DEVICES=0 python run_HOI_agent.py \
--save_folder="./chois_output/HOI_agent" \
--data_folder 'processed_data/replica_new/frl_apartment_4' \
--model_path "${model_path}" \
--fps=15 \
--last_k_frame=2 \
--llm_model_name='gpt-4o' \
--use_sub_dataset \
--executor 'maskedmimic' \
--check_collision \
--cond_mode 'only_spatial'
