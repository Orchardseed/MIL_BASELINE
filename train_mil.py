import argparse
from utils.yaml_utils import read_effective_yaml
from pathlib import Path
from process.process_all import process
import warnings
import os
from utils.general_utils import get_time,merge_k_fold_logs
warnings.filterwarnings('ignore')

def _preflight_run_outputs(args, yaml_path, output_dir, dataset_csv_path):
    output_dir = Path(output_dir)
    identity = f'seed{args.General.seed}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}'
    destinations = [
        output_dir / Path(dataset_csv_path).name,
        output_dir / Path(yaml_path).name,
        output_dir / f'Log_{identity}.csv',
        output_dir / f'Best_Log_{identity}.csv',
    ]
    # Best-checkpoint saving deletes every existing matching Best weight.
    # The last checkpoint epoch can vary with early stopping.
    destinations.extend(output_dir.glob('*Best*.pth'))
    destinations.extend(output_dir.glob('Last_EPOCH_*.pth'))
    conflicts = [str(path) for path in destinations if path.exists()]
    if conflicts:
        raise FileExistsError('Output already exists: ' + ', '.join(conflicts))

def main(arg):
    yaml_path = arg.yaml_path
    print(f"MIL-yaml path: {yaml_path}")
    args = read_effective_yaml(yaml_path, arg.input_path, arg.options)
    
    if args.Dataset.dataset_root_dir == {} and args.Dataset.dataset_csv_path != None:
        '''
        None-fold split
        '''
        log_root_dir = args.Logs.log_root_dir
        os.makedirs(log_root_dir,exist_ok=True)
        sub_dir = os.path.join(log_root_dir,args.Dataset.DATASET_NAME,args.General.MODEL_NAME)
        os.makedirs(sub_dir,exist_ok=True)
        args.Logs.now_log_dir = os.path.join(sub_dir,f'time_{get_time()}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}_seed_{args.General.seed}')
        _preflight_run_outputs(args, yaml_path, args.Logs.now_log_dir, args.Dataset.dataset_csv_path)
        process(args,yaml_path,arg.options)

    else:
        '''
        k-fold split
        '''
        dataset_root_dir = args.Dataset.dataset_root_dir
        k_fold_csv_paths = sorted([os.path.join(dataset_root_dir,path) for path in os.listdir(dataset_root_dir)])
        process_time = get_time()
        sub_dir = os.path.join(args.Logs.log_root_dir,args.Dataset.DATASET_NAME,args.General.MODEL_NAME)
        fold_total_log_dir = os.path.join(sub_dir,f'time_{process_time}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}_seed_{args.General.seed}')
        merge_destination = Path(fold_total_log_dir) / f'merge_{len(k_fold_csv_paths)}_fold_metrics.json'
        merge_destinations = set(Path(fold_total_log_dir).glob('merge_*_fold_metrics.json'))
        merge_destinations.add(merge_destination)
        merge_conflicts = sorted(str(path) for path in merge_destinations if path.exists())
        if merge_conflicts:
            raise FileExistsError('Output already exists: ' + ', '.join(merge_conflicts))
        # Check all selected folds before the first dispatch writes any input copy.
        for k_idx, k_fold_csv_path in enumerate(k_fold_csv_paths):
            output_dir = os.path.join(fold_total_log_dir, f'fold_{k_idx+1}')
            _preflight_run_outputs(args, yaml_path, output_dir, k_fold_csv_path)
        for k_idx,k_fold_csv_path in enumerate(k_fold_csv_paths):
            args.Dataset.dataset_csv_path = k_fold_csv_path
            now_fold = k_idx+1
            args.Dataset.now_fold = now_fold
            log_root_dir = args.Logs.log_root_dir
            os.makedirs(log_root_dir,exist_ok=True)
            sub_dir = os.path.join(log_root_dir,args.Dataset.DATASET_NAME,args.General.MODEL_NAME)
            os.makedirs(sub_dir,exist_ok=True)
            if now_fold != None:
                fold_dir = f'fold_{now_fold}'
                args.Logs.now_log_dir = os.path.join(sub_dir,f'time_{process_time}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}_seed_{args.General.seed}/{fold_dir}')
            os.makedirs(args.Logs.now_log_dir,exist_ok=True)
            _preflight_run_outputs(args, yaml_path, args.Logs.now_log_dir, args.Dataset.dataset_csv_path)
            process(args,yaml_path,arg.options)
            print(f'K-Fold:{k_idx+1} Done!')
        fold_total_log_dir = os.path.join(sub_dir,f'time_{process_time}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}_seed_{args.General.seed}')
        merge_k_fold_logs(fold_total_log_dir,args.General.process_pipeline)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path',type=str,default='/path/to/your/yaml',help='path to MIL-yaml file')
    parser.add_argument('--input_path',type=str,default=None,help='path to shared input settings')
    parser.add_argument('--options',nargs='+',help='override some settings in the used config, the key-value pair in xxx=yyy format will be merged into the yaml config file')
    arg = parser.parse_args()
    main(arg)
