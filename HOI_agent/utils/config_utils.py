import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='0', help='cuda device')

    parser.add_argument('--llm_model_name', type=str, default='gpt-4', help='data root folder')

    parser.add_argument('--data_folder', type=str, default='processed_data/replica_new/frl_apartment_4', help='data root folder')

    parser.add_argument('--save_folder', type=str, default='chois_release/chois_output/test_agent', help='save res folder')
    
    parser.add_argument("--save_obj_only", action="store_true")

    parser.add_argument("--navigation_only", action="store_true")
    
    parser.add_argument("--multi_object_sequencial", action="store_true")

    parser.add_argument("--multi_object_together", action="store_true")

    parser.add_argument("--multi_object_narrow_path", action="store_true")

    parser.add_argument("--multi_person", action="store_true")
    
    parser.add_argument('--model_path', type=str, default=None, help='checkpoint')
    
    parser.add_argument("--fps", type=int, default=30, help='fps')

    parser.add_argument("--last_k_frame", type=int, default=10, help='last_k_frame')

    parser.add_argument("--use_sub_dataset", action="store_true")

    parser.add_argument("--cond_mode", default='both_text_spatial', type=str,
                       help="generation mode: both_text_spatial, only_text, only_spatial. Other words will be used as text prompt.")

    parser.add_argument("--executor", default='maskedmimic', choices=['maskedmimic'], type=str,
                       help="motion generator")
    
    parser.add_argument("--check_collision", action="store_true")
    
    opt, _ = parser.parse_known_args()
    return opt