import os
import datetime
import json
import argparse
from tools.train_teacher import train_teacher
from tools.train_student import train_student, finetune
from tools.build import build_student_model, build_teacher_model, load_cfg
from tools.logger import get_logger

def get_parser():
    parser = argparse.ArgumentParser()
    default_cfg = '/home/stu6/2024PRML_Project2/PRML24_proj2/src/config/student_config.py'
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
    if task == 'teacher':
        model = build_teacher_model(cfg['model'])
        train_teacher(model, cfg, work_dir, logger, args.save_model, save_name='teacher_model')
    elif task == 'student':
        model = build_student_model(cfg['model'])
        best_model = train_student(model, cfg, work_dir, logger, save_model, save_name='student_model')
        if cfg['finetune']['finetuning']:
            finetune(best_model, cfg, logger, work_dir, save_model)