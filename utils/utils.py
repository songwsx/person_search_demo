import glob
import os
import random
import shutil
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm

from . import torch_utils  # , google_utils

from termcolor import cprint
from PIL import Image, ImageDraw, ImageFont
matplotlib.rc('font', **{'size': 11})

# Set printoptions
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)


def floatn(x, n=3):  # format floats to n decimals
    return float(format(x, '.%gf' % n))


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)


def load_classes(path):
    """
    根据类别名称文件解析出类别，存储在列表中
    :param path: 类别名称文件路径
    :return: 得到类别名列表: ['person', 'bicycle'...]
    """
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class mAPs
    n = len(labels)
    class_counts = np.array([np.bincount(labels[i][:, 0].astype(np.int), minlength=nc) for i in range(n)])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.03)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.03)
        torch.nn.init.constant_(m.bias.data, 0.0)


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # 由center_x center_y width height转为左上角右下角xmin, ymin, xmax, ymax
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
    coords[:, [0, 2]] -= (img1_shape[1] - img0_shape[1] * gain) / 2  # x padding
    coords[:, [1, 3]] -= (img1_shape[0] - img0_shape[0] * gain) / 2  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=img_shape[1])  # clip x
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=img_shape[0])  # clip y

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    iou = inter_area / union_area  # iou
    if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
        c_x1, c_x2 = torch.min(b1_x1, b2_x1), torch.max(b1_x2, b2_x2)
        c_y1, c_y2 = torch.min(b1_y1, b2_y1), torch.max(b1_y2, b2_y2)
        c_area = (c_x2 - c_x1) * (c_y2 - c_y1)  # convex area
        return iou - (c_area - union_area) / c_area  # GIoU

    return iou


def wh_iou(box1, box2):
    # Returns the IoU of wh1 to wh2. wh1 is 2, wh2 is nx2
    box2 = box2.t()

    # w, h = box1
    w1, h1 = box1[0], box1[1]
    w2, h2 = box2[0], box2[1]

    # Intersection area
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)

    # Union Area
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area  # iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.5):
    """
    剔除物体置信度得分object confidence score低于'conf_thres'阈值的检测框
    再利用NMS进一步过滤筛选检测框
    :param prediction: 处理后的YOLO预测，相对于原图 torch.Size([1, 8190, 85])
    :param conf_thres: 置信度阈值
    :param nms_thres:  NMS阈值
    :return: Returns detections with shape:(x1, y1, x2, y2, object_conf, class_conf, class)
    """

    min_wh = 5  # (pixels) minimum box width and height

    output = [None] * len(prediction)
    for image_i, pred in enumerate(prediction):
        # x, y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        # Conf = pred[:, 4]
        # Cls pred = pred[:, 5:]

        # Multiply conf by class conf to get combined confidence
        # axis = 1 时为行方向的最值，也就是求每个预测框类别得分最高的分数
        # class_conf:每个预测框概率最高的那个类别的得分 torch.Size([8190])
        # class_pred:每个预测框概率最高的那个类别       torch.Size([8190])
        class_conf, class_pred = pred[:, 5:].max(1)
        # 对于测试阶段来说，网络直接输出 Pr(class/object)，就已经可以代表有物体存在的条件下类别概率。
        # 但是在测试阶段，作者还把这个概率乘上了confidence。
        # improves mAP from 0.549 to 0.551
        pred[:, 4] *= class_conf

        # Select only suitable predictions
        # 1.pred[:, 4] > conf_thres: 保留置信度阈值大于阈值的预测框
        # 2.(pred[:, 2:4] > min_wh).all(1) > min_wh: 保留w,h > min_wh = 2的预测框，太小的忽略不计
        # 3.torch.isfinite(pred).all(1): 保留预测值都正常的
        # torch.isfinite(pred) 返回一个新的张量，其布尔元素表示每个元素是否为+/-INF,是INF NAN则返回0
        i = (pred[:, 4] > conf_thres) & (pred[:, 2:4] > min_wh).all(1) & torch.isfinite(pred).all(1)
        pred = pred[i] # 经过NMS，只剩下29个预测框 torch.Size([29, 85])

        # If none are remaining => process next image
        if len(pred) == 0:
            continue

        # Select predicted classes
        # 进行NMS时，使用的是类别概率乘以confidence
        # 这里选择类别只用类别概率
        class_conf = class_conf[i]
        class_pred = class_pred[i].unsqueeze(1).float()

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4]) # torch.Size([29, 85])

        # Detections ordered as (x1y1x2y2, obj_conf, class_conf, class_pred)
        pred = torch.cat((pred[:, :5], class_conf.unsqueeze(1), class_pred), 1) # 通过这样NMS后，只剩下29个框torch.Size([29, 7])

        # Get detections sorted by decreasing confidence scores
        # 对于剩下的预测框安装置信度(类别概率乘以confidence)得分进行排序
        pred = pred[(-pred[:, 4]).argsort()]

        det_max = []
        ############# NMS方法选择 #############
        """
        'OR'   : 一般说的NMS都是OR方式
        'AND'  : 这个与OR整体类似，不同在于如果出现这个类别只有一个框，则认为无效。
                 可能是一般一个物体都会对应多个框，只有一个很有可能是误检了
        'MERGE': 综合利用了高于阈值的预测框，对于每个预测框的conf值来赋予权重，
                 然后求得x1y1x2y2的坐标的加权平均作为最后的预测框 
                 weighted mixture box精度更高，但速度较慢一些.
        'SOFT' : soft-NMS https://arxiv.org/abs/1704.04503
        """
        nms_style = 'MERGE'  # 'OR' (default), 'AND', 'MERGE' (experimental), 'SOFT'
        for c in pred[:, -1].unique():   # 没有80类一个个遍历，更高效
            dc = pred[pred[:, -1] == c]  # select class c torch.Size([21, 7]) 代表这个类别有21个预测框
            n = len(dc) # 当前类别有len(dc)=21个预测框
            if n == 1:
                det_max.append(dc)  # No NMS required if only 1 prediction
                continue
            elif n > 100:
                # 框太多只保留前100，一般情况下是OK的，不过密集场景可能得改一下
                dc = dc[:100]  # limit to first 100 boxes: https://github.com/ultralytics/yolov3/issues/117

            # Non-maximum suppression
            if nms_style == 'OR':  # default
                # torch.Size([21, 7]) 开始时21个框
                # dc.shape[0]也就是预测框的数目，如果预测框数目为0，则退出循环
                while dc.shape[0]: # 21->14->9->3->0
                    det_max.append(dc[:1])  # 保留conf最高的预测框 4
                    if len(dc) == 1:  # 如果只剩下一个预测框了，退出循环
                        break
                    iou = bbox_iou(dc[0], dc[1:]) # 计算conf得分最高的预测框与其他框的IoU
                    dc = dc[1:][iou < nms_thres]  # 移除与当前conf得分最高的预测框IoU大于阈值的预测框 remove ious > threshold

            elif nms_style == 'AND':  # requires overlap, single boxes erased
                while len(dc) > 1:    # 21->14->9->3->0
                    # 计算得分最高的预测框与其他框的IoU
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes dc[1:]: torch.Size([20])
                    if iou.max() > 0.5:            # 与当前conf得分最高的预测框IoU最大的如果大于0.5
                        det_max.append(dc[:1])     # 那么就将conf得分最高的预测框加入最终的det_max
                    dc = dc[1:][iou < nms_thres]   # remove ious > threshold

            elif nms_style == 'MERGE':  # weighted mixture box 默认采用，精度更高，但速度较慢一些
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    i = bbox_iou(dc[0], dc) > nms_thres  # 取大于NMS阈值的框
                    weights = dc[i, 4:5]      # 取出iou大于NMS阈值的框求得这些框的conf值作为weights torch.Size([7, 1])
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0] # 这一步也就是进行了筛选

            elif nms_style == 'SOFT':  # soft-NMS https://arxiv.org/abs/1704.04503
                sigma = 0.5  # soft-nms sigma parameter
                while len(dc):
                    if len(dc) == 1:
                        det_max.append(dc)
                        break
                    det_max.append(dc[:1])
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:]
                    dc[:, 4] *= torch.exp(-iou ** 2 / sigma)  # decay confidences
                    dc = dc[dc[:, 4] > nms_thres]  # new line per https://github.com/ultralytics/yolov3/issues/362

        if len(det_max): # 5
            det_max = torch.cat(det_max)  # concatenate torch.Size([5, 7])
            output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # sort

    return output


def get_yolo_layers(model):
    bool_vec = [x['type'] == 'yolo' for x in model.module_defs]
    return [i for i, x in enumerate(bool_vec) if x]  # [82, 94, 106] for yolov3


def print_model_biases(model):
    # prints the bias neurons preceding each yolo layer
    print('\nModel Bias Summary (per output layer):')
    for l in model.yolo_layers:  # print pretrained biases
        b = model.module_list[l - 1][0].bias.view(3, -1)  # bias 3x85
        print('regression: %5.2f+/-%-5.2f ' % (b[:, :4].mean(), b[:, :4].std()),
              'objectness: %5.2f+/-%-5.2f ' % (b[:, 4].mean(), b[:, 4].std()),
              'classification: %5.2f+/-%-5.2f' % (b[:, 5:].mean(), b[:, 5:].std()))


def strip_optimizer(f='weights/last.pt'):  # from utils.utils import *; strip_optimizer()
    # Strip optimizer from *.pt files for lighter files (reduced by 2/3 size)
    x = torch.load(f)
    x['optimizer'] = None
    torch.save(x, f)


def create_backbone(f='weights/last.pt'):  # from utils.utils import *; create_backbone()
    # create a backbone from a *.pt file
    x = torch.load(f)
    x['optimizer'] = None
    x['training_results'] = None
    x['epoch'] = -1
    for p in x['model'].values():
        try:
            p.requires_grad = True
        except:
            pass
    torch.save(x, 'weights/backbone.pt')


def coco_class_count(path='../coco/labels/train2014/'):
    # Histogram of occurrences per class
    nc = 80  # number classes
    x = np.zeros(nc, dtype='int32')
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        x += np.bincount(labels[:, 0].astype('int32'), minlength=nc)
        print(i, len(files))


def coco_only_people(path='../coco/labels/val2014/'):
    # Find images with only people
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        if all(labels[:, 0] == 0):
            print(labels.shape[0], file)


def select_best_evolve(path='evolve*.txt'):  # from utils.utils import *; select_best_evolve()
    # Find best evolved mutation
    for file in sorted(glob.glob(path)):
        x = np.loadtxt(file, dtype=np.float32, ndmin=2)
        fitness = x[:, 2] * 0.5 + x[:, 3] * 0.5  # weighted mAP and F1 combination
        print(file, x[fitness.argmax()])


def coco_single_class_labels(path='../coco/labels/train2014/', label_class=43):
    # Makes single-class coco datasets. from utils.utils import *; coco_single_class_labels()
    if os.path.exists('new/'):
        shutil.rmtree('new/')  # delete output folder
    os.makedirs('new/')  # make new output folder
    os.makedirs('new/labels/')
    os.makedirs('new/images/')
    for file in tqdm(sorted(glob.glob('%s/*.*' % path))):
        with open(file, 'r') as f:
            labels = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
        i = labels[:, 0] == label_class
        if any(i):
            img_file = file.replace('labels', 'images').replace('txt', 'jpg')
            labels[:, 0] = 0  # reset class to 0
            with open('new/images.txt', 'a') as f:  # add image to dataset list
                f.write(img_file + '\n')
            with open('new/labels/' + Path(file).name, 'a') as f:  # write label
                for l in labels[i]:
                    f.write('%g %.6f %.6f %.6f %.6f\n' % tuple(l))
            shutil.copyfile(src=img_file, dst='new/images/' + Path(file).name.replace('txt', 'jpg'))  # copy images


def kmeans_targets(path='../coco/trainvalno5k.txt', n=9, img_size=416):  # from utils.utils import *; kmeans_targets()
    # Produces a list of target kmeans suitable for use in *.cfg files
    from utils.datasets import LoadImagesAndLabels
    from scipy import cluster

    # Get label wh
    dataset = LoadImagesAndLabels(path, augment=True, rect=True)
    for s, l in zip(dataset.shapes, dataset.labels):
        l[:, [1, 3]] *= s[0]  # normalized to pixels
        l[:, [2, 4]] *= s[1]
        l[:, 1:] *= img_size / max(s)  # nominal img_size for training
    wh = np.concatenate(dataset.labels, 0)[:, 3:5]  # wh from cxywh

    # Kmeans calculation
    k = cluster.vq.kmeans(wh, n)[0]
    k = k[np.argsort(k.prod(1))]  # sort small to large

    # Measure IoUs
    iou = torch.stack([wh_iou(torch.Tensor(wh).T, torch.Tensor(x).T) for x in k], 0)
    biou = iou.max(0)[0]  # closest anchor IoU

    print((biou < 0.2635).float().mean())

    # Print
    print('kmeans anchors (n=%g, img_size=%g, IoU=%.2f/%.2f/%.2f-min/mean/best): ' %
          (n, img_size, biou.min(), iou.mean(), biou.mean()), end='')
    for i, x in enumerate(k):
        print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg

    # Plot
    # plt.hist(biou.numpy().ravel(), 100)


def print_mutation(hyp, results, bucket=''):
    # Print mutation results to evolve.txt (for use with train.py --evolve)
    a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%10.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%10.3g' * len(results) % results  # results (P, R, mAP, F1, test_loss)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    if bucket:
        os.system('gsutil cp gs://%s/evolve.txt .' % bucket)  # download evolve.txt

    with open('evolve.txt', 'a') as f:  # append result
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)  # load unique rows
    np.savetxt('evolve.txt', x[np.argsort(-fitness(x))], '%10.3g')  # save sort by fitness

    if bucket:
        os.system('gsutil cp evolve.txt gs://%s' % bucket)  # upload evolve.txt


def fitness(x):
    # Returns fitness (for use with results.txt or evolve.txt)
    return 0.50 * x[:, 2] + 0.50 * x[:, 3]  # fitness = 0.5 * mAP + 0.5 * F1


# Plotting functions ---------------------------------------------------------------------------------------------------
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def print_info(info, _type=None):
    if _type is not None:
        if isinstance(info,str):
            cprint(info, _type[0], attrs=[_type[1]])
        elif isinstance(info,list):
            for i in range(info):
                cprint(i, _type[0], attrs=[_type[1]])
    else:
        print(info)

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def plot_wh_methods():  # from utils.utils import *; plot_wh_methods()
    # Compares the two methods for width-height anchor multiplication
    # https://github.com/ultralytics/yolov3/issues/168
    x = np.arange(-4.0, 4.0, .1)
    ya = np.exp(x)
    yb = torch.sigmoid(torch.from_numpy(x)).numpy() * 2

    fig = plt.figure(figsize=(6, 3), dpi=150)
    plt.plot(x, ya, '.-', label='yolo method')
    plt.plot(x, yb ** 2, '.-', label='^2 power method')
    plt.plot(x, yb ** 2.5, '.-', label='^2.5 power method')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=0, top=6)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.legend()
    fig.tight_layout()
    fig.savefig('comparison.png', dpi=200)


def plot_images(imgs, targets, paths=None, fname='images.jpg'):
    # Plots training images overlaid with targets
    imgs = imgs.cpu().numpy()
    targets = targets.cpu().numpy()
    # targets = targets[targets[:, 1] == 21]  # plot only one class

    fig = plt.figure(figsize=(10, 10))
    bs, _, h, w = imgs.shape  # batch size, _, height, width
    bs = min(bs, 16)  # limit plot to 16 images
    ns = np.ceil(bs ** 0.5)  # number of subplots

    for i in range(bs):
        boxes = xywh2xyxy(targets[targets[:, 0] == i, 2:6]).T
        boxes[[0, 2]] *= w
        boxes[[1, 3]] *= h
        plt.subplot(ns, ns, i + 1).imshow(imgs[i].transpose(1, 2, 0))
        plt.plot(boxes[[0, 2, 2, 0, 0]], boxes[[1, 1, 3, 3, 1]], '.-')
        plt.axis('off')
        if paths is not None:
            s = Path(paths[i]).name
            plt.title(s[:min(len(s), 40)], fontdict={'size': 8})  # limit to 40 characters
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close()


def plot_test_txt():  # from utils.utils import *; plot_test()
    # Plot test.txt histograms
    x = np.loadtxt('test.txt', dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    fig.tight_layout()
    plt.savefig('hist2d.jpg', dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    fig.tight_layout()
    plt.savefig('hist1d.jpg', dpi=200)


def plot_targets_txt():  # from utils.utils import *; plot_targets_txt()
    # Plot test.txt histograms
    x = np.loadtxt('targets.txt', dtype=np.float32)
    x = x.T

    s = ['x targets', 'y targets', 'width targets', 'height targets']
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label='%.3g +/- %.3g' % (x[i].mean(), x[i].std()))
        ax[i].legend()
        ax[i].set_title(s[i])
    fig.tight_layout()
    plt.savefig('targets.jpg', dpi=200)


def plot_evolution_results(hyp):  # from utils.utils import *; plot_evolution_results(hyp)
    # Plot hyperparameter evolution results in evolve.txt
    x = np.loadtxt('evolve.txt', ndmin=2)
    f = fitness(x)
    weights = (f - f.min()) ** 2  # for weighted results
    fig = plt.figure(figsize=(12, 10))
    matplotlib.rc('font', **{'size': 8})
    for i, (k, v) in enumerate(hyp.items()):
        y = x[:, i + 5]
        # mu = (y * weights).sum() / weights.sum()  # best weighted result
        mu = y[f.argmax()]  # best single result
        plt.subplot(4, 5, i + 1)
        plt.plot(mu, f.max(), 'o', markersize=10)
        plt.plot(y, f, '.')
        plt.title('%s = %.3g' % (k, mu), fontdict={'size': 9})  # limit to 40 characters
        print('%15s: %.3g' % (k, mu))
    fig.tight_layout()
    plt.savefig('evolve.png', dpi=200)


def plot_results(start=0, stop=0):  # from utils.utils import *; plot_results()
    # Plot training results files 'results*.txt'
    fig, ax = plt.subplots(2, 5, figsize=(14, 7))
    ax = ax.ravel()
    s = ['GIoU', 'Objectness', 'Classification', 'Precision', 'Recall',
         'val GIoU', 'val Objectness', 'val Classification', 'mAP', 'F1']
    for f in sorted(glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')):
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        for i in range(10):
            y = results[i, x]
            if i in [0, 1, 2, 5, 6, 7]:
                y[y == 0] = np.nan  # dont show zero loss values
            ax[i].plot(x, y, marker='.', label=f.replace('.txt', ''))
            ax[i].set_title(s[i])
            if i in [5, 6, 7]:  # share train and val loss y axes
                ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])

    fig.tight_layout()
    ax[0].legend()
    fig.savefig('results.png', dpi=200)


def plot_results_overlay(start=0, stop=0):  # from utils.utils import *; plot_results_overlay()
    # Plot training results files 'results*.txt', overlaying train and val losses
    s = ['train', 'train', 'train', 'Precision', 'mAP', 'val', 'val', 'val', 'Recall', 'F1']  # legends
    t = ['GIoU', 'Objectness', 'Classification', 'P-R', 'mAP-F1']  # titles
    for f in sorted(glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')):
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        fig, ax = plt.subplots(1, 5, figsize=(14, 3.5))
        ax = ax.ravel()
        for i in range(5):
            for j in [i, i + 5]:
                y = results[j, x]
                if i in [0, 1, 2]:
                    y[y == 0] = np.nan  # dont show zero loss values
                ax[i].plot(x, y, marker='.', label=s[j])
            ax[i].set_title(t[i])
            ax[i].legend()
            ax[i].set_ylabel(f) if i == 0 else None  # add filename
        fig.tight_layout()
        fig.savefig(f.replace('.txt', '.png'), dpi=200)


def version_to_tuple(version):
    # Used to compare versions of library
    return tuple(map(int, (version.split("."))))
