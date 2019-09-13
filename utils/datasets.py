import glob
import math
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.utils import xyxy2xywh, xywh2xyxy

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif']
vid_formats = ['.mov', '.avi', '.mp4']

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        None

    return s


class LoadImages:  # for inference
    def __init__(self, path, img_size=416, half=False):
        """
        进行推理的图像预处理
        :param path: 需要检测的图片文件夹 'data/samples'
        :param img_size: 416
        :param half: 是否采用半精度推理 False
        """
        path = str(Path(path))  # os-agnostic
        files = []
        if os.path.isdir(path):
            # files：列表，包含了待检测的图片路径[图片1，图片2...]
            # 'data\\samples\\bus.jpg'
            # 'data\\samples\\zidane.jpg'
            files = sorted(glob.glob(os.path.join(path, '*.*')))
        elif os.path.isfile(path):
            files = [path]

        # os.path.splitext(“文件路径”)    分离文件名与扩展名；默认返回(fname,fextension)元组，可做分片操作
        # os.path.splitext(x): ('data\\samples\\bus', '.jpg')
        # os.path.splitext(x)[-1].lower(): '.jpg'
        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats] # 判断是否是支持的图片格式
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats] # 判断是否是支持的视频格式
        nI, nV = len(images), len(videos)

        self.img_size = img_size
        self.files = images + videos
        self.nF = nI + nV  # number of files 总共要检测的数目
        self.video_flag = [False] * nI + [True] * nV
        self.mode = 'images'
        self.half = half  # half precision fp16 images
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nF > 0, 'No images or videos found in ' + path

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        # 如果迭代次数等于图片数目，就停止迭代
        if self.count == self.nF:
            raise StopIteration
        path = self.files[self.count] # 得到第self.count张图片路径

        if self.video_flag[self.count]: # 如果有视频的话
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nF:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nF, self.frame, self.nframes, path), end='')

        else:
            # 迭代次数加一
            self.count += 1
            # Read image 读取图片
            img0 = cv2.imread(path)  # BGR HWC: (1080, 810, 3)
            assert img0 is not None, 'Image Not Found ' + path
            # image 1/2 data/samples/bus.jpg:
            print('image %g/%g %s: ' % (self.count, self.nF, path), end='')

        # Padded resize
        img, *_ = letterbox(img0, new_shape=self.img_size) # img经过padding后的最小输入矩形图: (416, 320, 3)

        # cv2.imshow('Padded Image', img)
        # cv2.waitKey()

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB  HWC2CHW: (3, 416, 320)
        # ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
        img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nF  # number of files


class LoadWebcam:  # for inference
    def __init__(self, img_size=416, half=False):
        self.img_size = img_size
        self.half = half  # half precision fp16 images

        pipe = 0  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login

        # https://answers.opencv.org/question/215996/changing-gstreamer-pipeline-to-opencv-in-pythonsolved/
        # pipe = '"rtspsrc location="rtsp://username:password@192.168.1.64/1" latency=10 ! appsink'  # GStreamer

        # https://answers.opencv.org/question/200787/video-acceleration-gstremer-pipeline-in-videocapture/
        # https://stackoverflow.com/questions/54095699/install-gstreamer-support-for-opencv-python-package  # install help
        # pipe = "rtspsrc location=rtsp://root:root@192.168.0.91:554/axis-media/media.amp?videocodec=h264&resolution=3840x2160 protocols=GST_RTSP_LOWER_TRANS_TCP ! rtph264depay ! queue ! vaapih264dec ! videoconvert ! appsink"  # GStreamer

        self.cap = cv2.VideoCapture(pipe)  # video capture object

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == 27:  # esc to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Read image
        ret_val, img0 = self.cap.read()
        assert ret_val, 'Webcam Error'
        img_path = 'webcam_%g.jpg' % self.count
        img0 = cv2.flip(img0, 1)  # flip left-right
        print('webcam %g: ' % self.count, end='')

        # Padded resize
        img, *_ = letterbox(img0, new_shape=self.img_size)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float16 if self.half else np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=416, batch_size=16, augment=False, hyp=None, rect=True, image_weights=False,
                 cache_images=False):
        """

        :param path: 得到训练集的ID文件路径 'data/train.txt'
        :param img_size: 网络输入分辨率 416
        :param batch_size: 2
        :param augment: 是否进行数据增广
        :param hyp: 数据增广的超参数
        :param rect: 是否采用矩形训练
        :param image_weights: False
        :param cache_images: True
        """
        path = str(Path(path))  # os-agnostic
        # 读取训练/验证txt文件的内容
        with open(path, 'r') as f:
            self.img_files = [x.replace('/', os.sep) for x in f.read().splitlines()  # os-agnostic
                              if os.path.splitext(x)[-1].lower() in img_formats]

        n = len(self.img_files) # 4807 图片的个数
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index [0 0 1 1 2 2...]
        nb = bi[-1] + 1  # number of batches 2404
        assert n > 0, 'No images found in %s' % path

        self.n = n
        self.batch = bi  # batch index of image
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect

        # 将图片与标注对应上，根据train.txt的图片路径得到对应的标注文件路径
        # 图片的images文件名替换为标注label所在的labels
        # 图片的后缀遇到.png或者.jpg则替换为标注文件后缀.txt
        self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt')
                            for x in self.img_files] # 读取train.txt记录的图片路径

        # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        if self.rect:
            # Read image shapes
            sp = 'data' + os.sep + path.replace('.txt', '.shapes').split(os.sep)[-1]  # shapefile path
            try:
                with open(sp, 'r') as f:  # read existing shapefile
                    s = [x.split() for x in f.read().splitlines()]
                    assert len(s) == n, 'Shapefile out of sync'
            except:
                s = [exif_size(Image.open(f)) for f in tqdm(self.img_files, desc='Reading image shapes')]
                np.savetxt(sp, s, fmt='%g')  # overwrites existing (if any)

            # Sort by aspect ratio
            s = np.array(s, dtype=np.float64)
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            i = ar.argsort()
            self.img_files = [self.img_files[i] for i in i]
            self.label_files = [self.label_files[i] for i in i]
            self.shapes = s[i]
            ar = ar[i]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / 32.).astype(np.int) * 32

        # Preload labels (required for weighted CE training)
        self.imgs = [None] * n
        self.labels = [None] * n
        if augment or image_weights:  # cache labels for faster training
            self.labels = [np.zeros((0, 5))] * n
            extract_bounding_boxes = False
            pbar = tqdm(self.label_files, desc='Reading labels') # Reading labels:   0%|          | 0/4807 [00:00<?, ?it/s]
            nm, nf, ne = 0, 0, 0  # number missing, number found, number empty
            for i, file in enumerate(pbar):
                try:
                    with open(file, 'r') as f: # 'data\\labels\\train\\Inria_319.txt'
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32) # 2代表两个目标物体: (2, 5)
                except:
                    nm += 1  # print('missing labels for image %s' % self.img_files[i])  # file missing
                    continue

                if l.shape[0]:
                    assert l.shape[1] == 5, '> 5 label columns: %s' % file
                    assert (l >= 0).all(), 'negative labels: %s' % file
                    assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                    self.labels[i] = l
                    nf += 1  # file found

                    # Extract object detection boxes for a second stage classifier
                    if extract_bounding_boxes:
                        p = Path(self.img_files[i])
                        img = cv2.imread(str(p))
                        h, w, _ = img.shape
                        for j, x in enumerate(l):
                            f = '%s%sclassifier%s%g_%g_%s' % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                            if not os.path.exists(Path(f).parent):
                                os.makedirs(Path(f).parent)  # make new output folder
                            box = xywh2xyxy(x[1:].reshape(-1, 4)).ravel()
                            b = np.clip(box, 0, 1)  # clip boxes outside of image
                            ret_val = cv2.imwrite(f, img[int(b[1] * h):int(b[3] * h), int(b[0] * w):int(b[2] * w)])
                            assert ret_val, 'Failure extracting classifier boxes'
                else:
                    ne += 1  # file empty

                pbar.desc = 'Reading labels (%g found, %g missing, %g empty for %g images)' % (nf, nm, ne, n)
            assert nf > 0, 'No labels found. Recommend correcting image and label paths.'

        # Cache images into memory for faster training (~5GB)
        # imread比较慢，因此这里直接先读取最多10000张图片，大概5GB，加快训练
        if cache_images and augment:  # if training
            for i in tqdm(range(min(len(self.img_files), 10000)), desc='Reading images'):  # max 10k images
                img_path = self.img_files[i]
                img = cv2.imread(img_path)  # BGR
                assert img is not None, 'Image Not Found ' + img_path
                r = self.img_size / max(img.shape)  # size ratio 长边缩放到416的缩放比例
                if self.augment and r < 1:  # if training (NOT testing), downsize to inference shape
                    h, w, _ = img.shape
                    img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)  # or INTER_AREA
                self.imgs[i] = img # 将等比例缩放后的图片存进去

        # Detect corrupted images https://medium.com/joelthchao/programmatically-detect-corrupted-image-8c1b2006c3d3
        # 判断图片是否下载下来正常，如果出现异常的图片就打印出来
        detect_corrupted_images = False
        if detect_corrupted_images:
            from skimage import io  # conda install -c conda-forge scikit-image
            for file in tqdm(self.img_files, desc='Detecting corrupted images'):
                try:
                    _ = io.imread(file)
                except:
                    print('Corrupted image detected: %s' % file)

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]

        img_path = self.img_files[index]
        label_path = self.label_files[index]
        hyp = self.hyp

        # Load image
        img = self.imgs[index]
        if img is None:
            img = cv2.imread(img_path)  # BGR
            assert img is not None, 'Image Not Found ' + img_path
            r = self.img_size / max(img.shape)  # size ratio
            if self.augment and r < 1:  # if training (NOT testing), downsize to inference shape
                h, w, _ = img.shape
                img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)  # INTER_LINEAR fastest

        # Augment colorspace
        augment_hsv = True
        if self.augment and augment_hsv:
            # SV augmentation by 50%
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # hue, sat, val
            S = img_hsv[:, :, 1].astype(np.float32)  # saturation
            V = img_hsv[:, :, 2].astype(np.float32)  # value

            a = random.uniform(-1, 1) * hyp['hsv_s'] + 1
            b = random.uniform(-1, 1) * hyp['hsv_v'] + 1
            S *= a
            V *= b

            img_hsv[:, :, 1] = S if a < 1 else S.clip(None, 255)
            img_hsv[:, :, 2] = V if b < 1 else V.clip(None, 255)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

        # Letterbox
        h, w, _ = img.shape
        if self.rect:
            shape = self.batch_shapes[self.batch[index]]
            img, ratiow, ratioh, padw, padh = letterbox(img, new_shape=shape, mode='rect')
        else:
            shape = self.img_size
            img, ratiow, ratioh, padw, padh = letterbox(img, new_shape=shape, mode='square')

        # Load labels
        labels = []
        if os.path.isfile(label_path):
            x = self.labels[index]
            if x is None:  # labels not preloaded
                with open(label_path, 'r') as f:
                    x = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)

            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratiow * w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = ratioh * h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 3] = ratiow * w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = ratioh * h * (x[:, 2] + x[:, 4] / 2) + padh

        # Augment image and labels
        if self.augment:
            img, labels = random_affine(img, labels,
                                        degrees=hyp['degrees'],
                                        translate=hyp['translate'],
                                        scale=hyp['scale'],
                                        shear=hyp['shear'])

        nL = len(labels)  # number of labels
        if nL:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() > 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() > 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Normalize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return torch.from_numpy(img), labels_out, img_path, (h, w)

    @staticmethod
    def collate_fn(batch):
        img, label, path, hw = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, hw


def letterbox(img, new_shape=416, color=(128, 128, 128), mode='auto'):
    """
    求得较长边缩放到416的比例，然后对图片wh按这个比例缩放，使得较长边达到416,
    再对较短边进行填充使得较短边满足32的倍数
    :param img: 需要处理的原始图片CHW
    :param new_shape: 网络的输入分辨率
    :param color: 进行pad时，填充的颜色(值)
    :param mode:需要进行填充的模式
    :return: 返回填充后wh都为32倍数的图片
    """
    # Resize a rectangular image to a 32 pixel multiple rectangle
    # https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width] (1080, 810)

    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape) # 416.0 / 1080 = 0.3851851851851852
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old
    ratiow, ratioh = ratio, ratio
    # round() 方法返回浮点数x的四舍五入值。
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio))) # WH:(312, 416)

    # Compute padding https://github.com/ultralytics/yolov3/issues/232
    if mode is 'auto':
        # 填充为符合条件的最小矩形minimum rectangle
        # 使得较长边达到416, 再对较短边进行填充使得较短边满足32的倍数
        dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding  4.0
        dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding 0.0
    elif mode is 'square':  # square
        # 直接填充为416x416的正方形
        dw = (new_shape - new_unpad[0]) / 2  # width padding
        dh = (new_shape - new_unpad[1]) / 2  # height padding
    elif mode is 'rect':  # square
        # 填充为指定形状new_shape=(320, 416)的矩形
        dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
        dh = (new_shape[0] - new_unpad[1]) / 2  # height padding
    elif mode is 'scaleFill':
        # resize到指定的416x416
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape, new_shape)
        ratiow, ratioh = new_shape / shape[1], new_shape / shape[0]

    if shape[::-1] != new_unpad:  # new_unpad: (312, 416)
        # 进行resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1)) # 0, 0
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1)) # 4, 4
    # 为图像扩边（填充）
    # 想为图像周围建一个边可以使用cv2.copyMakeBorder()函数。这经常在卷积运算或0填充时被用到。具体参数如下：
    # 5.1 src输入图像
    # 5.2 top,bottom,left,right对应边界的像素数目
    # 5.3 borderType要添加哪种类型的边界：
    # 5.3.1	cv2.BORDER_CONSTANT添加有颜色的常数值边界，还需要下一个参数（value）
    # 5.3.2	cv2.BORDER_REFLIECT边界元素的镜像。例如：fedcba | abcdefgh | hgfedcb
    # 5.3.3	cv2.BORDER_101或者cv2.BORDER_DEFAULT跟上面一样，但稍作改动，例如：gfedcb | abcdefgh | gfedcba
    # 5.3.4	cv2.BORDER_REPLICATE复后一个元素。例如: aaaaaa| abcdefgh|hhhhhhh
    # 5.3.5	cv2.BORDER_WRAP 不知怎么了, 就像样: cdefgh| abcdefgh|abcdefg
    # 5.3.6	value边界颜色
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratiow, ratioh, dw, dh


def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    if targets is None:
        targets = []
    border = 0  # width of added border (optional)
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_AREA,
                         borderValue=(128, 128, 128))  # BGR order borderValue

    # Return warped points also
    if len(targets) > 0:
        n = targets.shape[0]
        points = targets[:, 1:5].copy()
        area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return imw, targets


def convert_images2bmp():
    # cv2.imread() jpg at 230 img/s, *.bmp at 400 img/s
    for path in ['../coco/images/val2014/', '../coco/images/train2014/']:
        folder = os.sep + Path(path).name
        output = path.replace(folder, folder + 'bmp')
        if os.path.exists(output):
            shutil.rmtree(output)  # delete output folder
        os.makedirs(output)  # make new output folder

        for f in tqdm(glob.glob('%s*.jpg' % path)):
            save_name = f.replace('.jpg', '.bmp').replace(folder, folder + 'bmp')
            cv2.imwrite(save_name, cv2.imread(f))

    for label_path in ['../coco/trainvalno5k.txt', '../coco/5k.txt']:
        with open(label_path, 'r') as file:
            lines = file.read()
        lines = lines.replace('2014/', '2014bmp/').replace('.jpg', '.bmp').replace(
            '/Users/glennjocher/PycharmProjects/', '../')
        with open(label_path.replace('5k', '5k_bmp'), 'w') as file:
            file.write(lines)
