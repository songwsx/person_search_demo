# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset


class Market1501(BaseImageDataset):
    """
    Market1501
    命名规则：
    以 0001_c1s1_000151_01.jpg 为例
    1） 0001 表示每个人的标签编号，从0001到1501；
    2） c1 表示第一个摄像头(camera1)，共有6个摄像头；
    3） s1 表示第一个录像片段(sequece1)，每个摄像机都有数个录像段；
    4） 000151 表示 c1s1 的第000151帧图片，视频帧率25fps；
    5） 01 表示 c1s1_001051 这一帧上的第1个检测框，由于采用DPM检测器，
        对于每一帧上的行人可能会框出好几个bbox。00 表示手工标注框
    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    def __init__(self, root='query', verbose=True, **kwargs):
        super(Market1501, self).__init__()
        self.query_dir = root # 'query'

        query = self._process_dir(self.query_dir, relabel=False) # 对查询集进行预处理
        if verbose:
            self.print_dataset_statistics(query)
        self.query = query

        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)

    def _process_dir(self, dir_path, relabel=False):
        """
        对 dir_path 的数据进行处理
        :param dir_path: 'data\\market1501\\query'
        :param relabel:  训练时：True 测试时：False
        :return:
        """
        # 列表: 含有 n 个图片的路径
        # 'data\\market1501\\query\\0003_c1s6_015971_00.jpg'
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        # \d   : 匹配任意数字，等价于 [0-9]
        # [...]: 用来表示一组字符,单独列出：[amk] 匹配 'a'，'m'或'k'
        # re+	 : 匹配1个或多个的表达式。
        # (re) : 匹配括号内的表达式，也表示一个组
        # '-1_c6s1_078001_05.jpg',注意-1的情况，所以有 [-\d]
        # 由于是多个数字例如'0002_c1s1_000451_03.jpg'，0002，因此 [-\d]+
        # 因为要取出这个数字，因此加上括号作为一个 group
        pattern = re.compile(r'([-\d]+)_c(\d)')

        # set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
        pid_container = set()

        # query: 'data\\market1501\\query\\0003_c1s6_015971_00.jpg'
        for img_path in img_paths:
            # pid: 每个人的标签编号 3
            # _  : 摄像头号 1
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored 测试集 gallery 会有-1的图片，扔掉
            # query 一共 3 个 id： {1425, 3, 1452}
            # gallery一共 751 个:{0, 1, 3, 4, 5, 6, 8, 9, ... 1497, 1498, 1499, 1501}
            pid_container.add(pid)
        # query:  {1425: 0, 3: 1, 1452: 2}
        # gallery:{0: 0, 1: 1, 3: 2, 4: 3, 5: 4, 6: 5, ..., 1499: 749, 1501: 750}
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored pid为-1代表垃圾图片，忽略掉
            # assert 0 <= pid <= 1501  # pid == 0 means background pid为0代表背景图片
            # assert 1 <= camid <= 6   # 6 个摄像头 检查一下
            camid -= 1  # index starts from 0 摄像头号从零开始
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        # (img_path, pid, camid)
        # ('query\\0001_c1s1_001051_00.jpg', 1, 0)
        # ('query\\0001_c3s1_000551_00.jpg', 1, 2)
        return dataset
