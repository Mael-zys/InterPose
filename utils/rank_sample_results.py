import os
import json
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='chois', help='save to project/name')
    parser.add_argument('--project', default='chois_single_window_results', help='project/name')
    
    
    opt = parser.parse_args()
    return opt

def rank_sample(path):
    files_name = os.listdir(path)
    metric_list = ['mean_contact_f1_score', 'mean_fsliding_jnts', 'mean_floor_height', 'mean_hand_penetration_score', 'mean_penetration_score', 'mean_contact_precision', 'mean_contact_recall']
    res = {}
    for metric in metric_list:
        res[metric] = {}
    for file in files_name:
        if 'sub' not in file:
            continue
        with open(os.path.join(path, file), "r") as f:
            data = json.load(f)
        file_name_new = '_'.join(file.split('_')[:3])
        for metric in metric_list:
            res[metric][file_name_new] = data[metric]
    
    for metric in metric_list:
        res[metric] = dict(sorted(res[metric].items(), key=lambda x: x[1], reverse=True))
    
    with open(os.path.join(path, "sorted.json"), "w") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    opt = parse_opt()
    opt.files_root = os.path.join('chois_output', opt.project, 'evaluation_metrics_json', opt.exp_name)
    rank_sample(opt.files_root)
