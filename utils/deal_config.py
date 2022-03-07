import os
from utils.deal_path import get_target_abs_path
from yacs.config import CfgNode as CN


PATH = os.path.abspath(__file__)

def read_config(name=None, default='default.yml'):
    """ 
        读取配置文件，先读取默认的配置文件，再将name对应的文件合并
        主要是为了方便做对比实验时减小改动，保证一致性

    Args:
        name (str): 配置文件的名字，需要包含文件格式，例如: big_batch_size.yml
        default (str): 默认配置文件的名字
    """
    
    default_path = get_target_abs_path(PATH, 2, f'config/{default}')
    with open(default_path) as f:
        config = CN.load_cfg(f)

    if name:
        cur_path = get_target_abs_path(PATH, 2, f'config/{name}')
        config.merge_from_file(cur_path)
    
    return config