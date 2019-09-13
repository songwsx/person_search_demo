from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cpu"
# ID number of GPU
_C.MODEL.DEVICE_ID = '-1'
# Name of backbone resnet50
_C.MODEL.NAME = 'resnet50_ibn_a'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = '/home/common/wangsong/weights/r50_ibn_a.pth'
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('query')

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8


# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'yes','no'
_C.TEST.RE_RANKING = 'no'
# Path to trained model
_C.TEST.WEIGHT = r"weights/719rank1.pth"
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = "output"
