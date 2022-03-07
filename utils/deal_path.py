import os

def get_target_abs_path(src_path, deep, relative_path):
    """ 获取以src_path去掉deep层目录为起点，相对路径为relative_path的绝对路径
        例如:
            src_path: /home/fyy/code/reid/train.py
            deep: 1
            =========> relative的起点路径为: /home/fyy/code/reid/
            relative_path: dataset/train

            拼接后的结果为: /home/fyy/code/reid/dataset/train

    Args:
        src_path (str): 绝对路径
        deep (int)): 需要去除的目录层数 
        relative_path (str): 相对于src_path去除deep后的路径
    """
    for i in range(deep):
        src_path = os.path.dirname(src_path)
    
    return os.path.join(src_path, relative_path)