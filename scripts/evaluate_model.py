import sys
import pickle
import argparse
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, '.')
from isegm.inference import utils
from isegm.utils.exp import load_config_file
from isegm.inference.predictors import get_predictor
from isegm.inference.evaluation import evaluate_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['NoBRS', 'RGB-BRS', 'DistMap-BRS',
                                         'f-BRS-A', 'f-BRS-B', 'f-BRS-C'],
                        help='')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')
    parser.add_argument('--datasets', type=str, default='GrabCut,Berkeley,DAVIS,COCO_MVal,SBD',
                        help='List of datasets on which the model should be tested. '
                             'Datasets are separated by a comma. Possible choices: '
                             'GrabCut, Berkeley, DAVIS, COCO_MVal, SBD')
    parser.add_argument('--n-clicks', type=int, default=20,
                        help='Maximum number of clicks for the NoC metric.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='ID of used GPU.')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use only CPU for inference.')
    parser.add_argument('--thresh', type=float, required=False, default=0.49,
                        help='The segmentation mask is obtained from the probability outputs using this threshold.')
    parser.add_argument('--target-iou', type=float, default=0.90,
                        help='Target IoU threshold for the NoC metric. (min possible value = 0.8)')
    parser.add_argument('--norm-radius', type=int, default=260)
    parser.add_argument('--clicks-limit', type=int, default=None)
    parser.add_argument('--config-path', type=str, default='./config.yml',
                        help='The path to the config file.')
    parser.add_argument('--logs-path', type=str, default='',
                        help='The path to the evaluation logs. Default path: cfg.EXPS_PATH/evaluation_logs.')

    args = parser.parse_args()
    if args.cpu:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f"cuda:{args.gpus.split(',')[0]}")
    args.target_iou = max(0.8, args.target_iou)

    cfg = load_config_file(args.config_path, return_edict=True)

    if args.logs_path == '':
        args.logs_path = Path(cfg.EXPS_PATH) / 'evaluation_logs'
    else:
        args.logs_path = Path(args.logs_path)

    return args, cfg


def main():
    args, cfg = parse_args()

    checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, args.checkpoint)
    model = utils.load_is_model(checkpoint_path, args.device, norm_radius=args.norm_radius)

    eval_exp_name = get_eval_exp_name(args)
    eval_exp_path = args.logs_path / eval_exp_name
    eval_exp_path.mkdir(parents=True, exist_ok=True)
    predictor_params = None
    if args.clicks_limit is not None:
        if args.clicks_limit == -1:
            args.clicks_limit = args.n_clicks
        predictor_params = {'net_clicks_limit': args.clicks_limit}

    print_header = True
    for dataset_name in args.datasets.split(','):
        dataset = utils.get_dataset(dataset_name, cfg)

        zoom_in_target_size = 600 if dataset_name == 'DAVIS' else 400
        predictor = get_predictor(model, args.mode, args.device,
                                  prob_thresh=args.thresh,
                                  zoom_in_params={'target_size': zoom_in_target_size},
                                  predictor_params=predictor_params)

        dataset_results = evaluate_dataset(dataset, predictor, pred_thr=args.thresh,
                                           max_iou_thr=args.target_iou,
                                           max_clicks=args.n_clicks)

        save_results(args, dataset_name, eval_exp_path, dataset_results,
                     print_header=print_header)
        print_header = False


def get_eval_exp_name(args):
    if ':' in args.checkpoint:
        model_name, checkpoint_prefix = args.checkpoint.split(':')
        model_name = model_name.split('/')[-1]

        return f"{model_name}_{checkpoint_prefix}"
    else:
        return Path(args.checkpoint).stem


def save_results(args, dataset_name, eval_exp_path, dataset_results, print_header=True):
    all_ious, elapsed_time = dataset_results
    mean_spc, mean_spi = utils.get_time_metrics(all_ious, elapsed_time)

    iou_thrs = np.arange(0.8, min(0.95, args.target_iou) + 0.001, 0.05).tolist()
    noc_list, over_max_list = utils.compute_noc_metric(all_ious, iou_thrs=iou_thrs, max_clicks=args.n_clicks)
    header, table_row = utils.get_results_table(noc_list, over_max_list, args.mode, dataset_name,
                                                mean_spc, elapsed_time, args.n_clicks,
                                                model_name=eval_exp_path.stem)
    target_iou_int = int(args.target_iou * 100)
    if target_iou_int not in [80, 85, 90]:
        noc_list, over_max_list = utils.compute_noc_metric(all_ious, iou_thrs=[args.target_iou],
                                                           max_clicks=args.n_clicks)
        table_row += f' NoC@{args.target_iou:.1%} = {noc_list[0]:.2f};'
        table_row += f' >={args.n_clicks}@{args.target_iou:.1%} = {over_max_list[0]}'

    if print_header:
        print(header)
    print(table_row)

    log_path = eval_exp_path / f'results_{args.mode}_{args.n_clicks}.txt'
    if log_path.exists():
        with open(log_path, 'a') as f:
            f.write(table_row + '\n')
    else:
        with open(log_path, 'w') as f:
            f.write(header + '\n')
            f.write(table_row + '\n')

    ious_path = eval_exp_path / 'all_ious'
    ious_path.mkdir(exist_ok=True)
    with open(ious_path / f'{dataset_name}_{args.mode}_{args.n_clicks}.pkl', 'wb') as fp:
        pickle.dump(all_ious, fp)


if __name__ == '__main__':
    main()
