import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger

from train_model_jlf import train_detector

# 获得命令行的参数
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')  # 训练配置
    parser.add_argument('--work-dir', help='the dir to save logs and models')  # 保存日志和模型的地址
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')  # 在该checkpoint开始训练
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')  # 训练时是否评估checkpoint
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')  # 使用的gpu数量
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')  # 使用的gpu编号
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '  # 修改引用配置中的一些设置
             'in xxx=yyy format will be merged into config file (deprecate), '
             'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()  # 读取命令行参数
    cfg = Config.fromfile(args.config)  # 读取config

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # set cudnn_benchmark；如果需要输入的图片固定尺寸，则开启
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # 创建work_dir
    if args.work_dir is not None:  # 优先使用命令行设置的work_dir
        cfg.work_dir = args.work_dir  # 没有则使用config中的
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',  # 再没有则自动创建
                                osp.splitext(osp.basename(args.config))[0])

    # 从断点处继续训练的checkpoint
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    # 设置使用的gpu，不设置则使用第一个
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))                          # 创建work_dir
    # cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))             # 把config下载到work_dir中
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())            # 读取当前时间
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')                   # 并以当前时间命名log文件
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)    # 创建log文件

    meta = dict()  # 创建一个dict，用于记录一些重要信息，例如环境

    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)    # 记录环境及版本信息
    # meta['env_info'] = env_info
    # meta['config'] = cfg.pretty_text
    # logger.info(f'Distributed training: {distributed}')     # 记录是否分布式训练
    # logger.info(f'Config:\n{cfg.pretty_text}')              # 记录config

    # 设置随机种子
    # if args.seed is not None:
    # logger.info(f'Set random seed to {args.seed}, '
    #             f'deterministic: {args.deterministic}')
    # set_random_seed(args.seed, deterministic=args.deterministic)
    # cfg.seed = args.seed
    # meta['seed'] = args.seed
    # meta['exp_name'] = osp.basename(args.config)            # config名称
    print(cfg.log_level)
    # 创建模型
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]

    # # 如果workflow是两个，则把验证集也加入到训练中
    # if len(cfg.workflow) == 2:
    #     val_dataset = copy.deepcopy(cfg.data.val)
    #     val_dataset.pipeline = cfg.data.train.pipeline
    #     datasets.append(build_dataset(val_dataset))

    # 如果从
    # if cfg.checkpoint_config is not None:
    #     # save mmdet version, config file content and class names in
    #     # checkpoints as meta data
    #     cfg.checkpoint_config.meta = dict(
    #         mmdet_version=__version__ + get_git_hash()[:7],
    #         CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    # model.CLASSES = datasets[0].CLASSES

    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
