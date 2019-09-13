import numpy as np


def parse_model_cfg(path):
    """
    配置文件定义了6种不同type
    'net': 相当于超参数,网络全局配置的相关参数
    {'convolutional', 'net', 'route', 'shortcut', 'upsample', 'yolo'}
    输入: 配置文件路径
    返回值: 列表对象,其中每一个元素为一个字典类型对应于一个要建立的神经网络模块（层）
    """
    # 加载文件并过滤掉文本中多余内容
    file = open(path, 'r')
    lines = file.read().split('\n') # store the lines in a list等价于readlines
    lines = [x for x in lines if x and not x.startswith('#')] # 去掉空行和以#开头的注释行
    lines = [x.rstrip().lstrip() for x in lines]  # 去掉左右两边的空格(rstricp是去掉右边的空格，lstrip是去掉左边的空格)

    mdefs = []  # module definitions
    for line in lines: # '[net]'
        if line.startswith('['):  # 这是cfg文件中一个层(块)的开始
            mdefs.append({})      # 添加一个字典
            mdefs[-1]['type'] = line[1:-1].rstrip() # 把cfg的[]中的块名作为键type的值  <class 'list'>: [{'type': 'net'}]
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1]['batch_normalize'] = 0    # pre-populate with zeros (may be overwritten later)
        else:
            key, val = line.split("=") # 按等号分割 key:'batch' val:16
            key = key.rstrip()         #  key(去掉右空格)

            if 'anchors' in key:
                # key:'anchors'
                # val: ' 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326'
                # val.split(','): [' 10', '13', '  16', '30', '  33', '23', '  30', '61', '  62', '45', '  59', '119', '  116', '90', '  156', '198', '  373', '326']
                # [float(x) for x in val.split(',')]: [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0]
                # np.array([float(x) for x in val.split(',')]): shape:(18,)
                # 将val以逗号划分元素，强制转成浮点数，形成了浮点数构成的列表，
                # 再转成numpy格式数组reshape为(9, 2)，因为两个元素代表一个anchor
                mdefs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))  # np anchors shape: (9, 2)
            else:
                # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
                mdefs[-1][key] = val.strip()

    return mdefs


def parse_data_cfg(path):
    """
    解析数据集的配置文件
    # 类别数目
    classes= 80
    # 训练集ID文件路径
    train=../coco/trainvalno5k.txt
    # 验证集ID文件路径
    valid=../coco/5k.txt
    # 类别名称文件路径
    names=data/coco.names
    backup=backup/
    eval=coco
    :param path: 'config/oxfordhand.data'
    :return:
    """
    # Parses the data configuration file
    options = dict()
    with open(path, 'r') as fp:
        lines = fp.readlines() # 返回列表，包含所有的行。

    for line in lines: # 'classes= 1\n'
        line = line.strip()
        if line == '' or line.startswith('#'): # 去掉空白行和以#开头的注释行
            continue
        key, val = line.split('=') # 按等号分割 key:'classes'  value:' 1'
        options[key.strip()] = val.strip()

    return options
