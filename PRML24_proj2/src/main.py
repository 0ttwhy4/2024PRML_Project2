import os
import datetime
import json
import argparse
from tools.train import train, finetune
from tools.build import build_model, load_cfg, build_tools
from tools.logger import get_logger

def get_parser():
    parser = argparse.ArgumentParser()
    default_cfg = '/home/stu6/2024PRML_Project2/PRML24_proj2/src/config/teacher_config.py'
    parser.add_argument('--cfg', type=str, default=default_cfg)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--task', type=str, default='teacher', help="'teacher'/'student'")
    parser.add_argument('--job_name', type=str, default=None)
    return parser

if __name__ == '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = get_parser()
    args = parser.parse_args()
    cfg_path = args.cfg
    cfg = load_cfg(cfg_path)
    work_root = cfg['work_dir']
    job_name = args.job_name
    if job_name is not None:
        work_dir = os.path.join(work_root, job_name)
    else:
        work_dir = os.path.join(work_root, datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')) 
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    logger = get_logger(os.path.join(work_dir, 'logging.log'))
    logger.debug("config:\n %s", json.dumps(cfg, indent=4))
    
    task = args.task
    save_model = args.save_model
    model = build_model(cfg['model'])
    model = model.cuda()
    optimizer, criterion, train_loader, valid_loader, scheduler = build_tools(model, cfg, task)
    
    best_model = train(model, 
                       cfg,
                       optimizer,
                       criterion,
                       train_loader,
                       valid_loader,
                       scheduler,
                       work_dir, 
                       logger, 
                       save_model, 
                       task,
                       save_name='student_model')
    
    if 'finetune' in cfg.keys() and cfg['finetune']['finetuning']:
        finetune(best_model, cfg, logger, work_dir, save_model)