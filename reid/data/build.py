# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader

from .collate_batch import  val_collate_fn
from .datasets import ImageDataset
from .transforms import build_transforms
from .datasets import Market1501

def make_data_loader(cfg):
    # 验证集的预处理
    val_transforms = build_transforms(cfg)
    num_workers = cfg.DATALOADER.NUM_WORKERS # 加载图像进程数 8
    dataset = Market1501(root=cfg.DATASETS.ROOT_DIR)

    val_set = ImageDataset(dataset.query, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return val_loader, len(dataset.query)
