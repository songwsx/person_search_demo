import torch.nn.functional as F

from utils.parse_config import *
from utils.utils import *

def  create_modules(module_defs, img_size, arc):
    # Constructs module list of layer blocks from module configuration in module_defs

    hyperparams = module_defs.pop(0) # 存储了cfg中[net]的信息，它是一个字典，获取网络输入和预处理相关信息
    output_filters = [int(hyperparams['channels'])] # 初始值对应于输入数据3通道，我们不仅需要追踪前一层的卷积核数量，还需要追踪之前每个层。随着不断地迭代，我们将每个模块的输出卷积核数量添加到 output_filters 列表上。
    module_list = nn.ModuleList() # 一定要用ModuleList()才能被torch识别为module并进行管理，不能用list！
    routs = []  # list of layers which rout to deeper layes
    yolo_index = -1

    for i, mdef in enumerate(module_defs): # 遍历每一层网络配置
        modules = nn.Sequential()          # 这里每个块用nn.sequential()创建为了一个module,一个module有多个层

        if mdef['type'] == 'convolutional':
            ''' 1. 卷积层 '''
            bn = int(mdef['batch_normalize']) # 根据配置是否需要bn,默认是0不需要
            filters = int(mdef['filters'])    # 得到输出的通道数 32
            kernel_size = int(mdef['size'])   # 得到卷积核大小 3
            pad = (kernel_size - 1) // 2 if int(mdef['pad']) else 0 # 根据卷积核大小得到padding数
            # 开始创建并添加相应层
            modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                   out_channels=filters,
                                                   kernel_size=kernel_size,
                                                   stride=int(mdef['stride']),
                                                   padding=pad,
                                                   bias=not bn)) # 卷积层后无BN层就需要bias
            if bn: # Add the Batch Norm Layer
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))
            if mdef['activation'] == 'leaky':  # TODO: activation study https://github.com/ultralytics/yolov3/issues/441
                # Check the activation.
                # It is either Linear or a Leaky ReLU for YOLO
                # 给定参数负轴系数0.1
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
                # modules.add_module('activation', Swish())

        elif mdef['type'] == 'maxpool':
            kernel_size = int(mdef['size'])
            stride = int(mdef['stride'])
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            if kernel_size == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool

        elif mdef['type'] == 'upsample':
            '''
			2. upsampling layer
			没有使用 Bilinear2dUpsampling
			实际使用的为最近邻插值
			'''
            # 这个stride在cfg中就是2，所以下面的scale_factor写2或者stride是等价的
            modules = nn.Upsample(scale_factor=int(mdef['stride']), mode='nearest')

        # route layer -> Empty layer
        # route层的作用：当layer取值为正时，输出这个正数对应的层的特征，如果layer取值为负数，输出route层向后退layer层对应层的特征
        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = [int(x) for x in mdef['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            routs.extend([l if l > 0 else l + i for l in layers])
            # if mdef[i+1]['type'] == 'reorg3d':
            #     modules = nn.Upsample(scale_factor=1/float(mdef[i+1]['stride']), mode='nearest')  # reorg3d

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            filters = output_filters[int(mdef['from'])]
            layer = int(mdef['from'])
            routs.extend([i + layer if layer < 0 else layer])

        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            # torch.Size([16, 128, 104, 104])
            # torch.Size([16, 64, 208, 208]) <-- # stride 2 interpolate dimensions 2 and 3 to cat with prior layer
            pass

        elif mdef['type'] == 'yolo':
            yolo_index += 1
            mask = [int(x) for x in mdef['mask'].split(',')]   # anchor mask 解析此YOLO层需要的anchor: [6, 7, 8]
            modules = YOLOLayer(anchors=mdef['anchors'][mask], # anchor list 根据anchor_idxs得到所需要的anchor: [(116, 90), (156, 198), (373, 326)]
                                nc=int(mdef['classes']),       # number of classes 类别数量
                                img_size=img_size,             # (416, 416)
                                yolo_index=yolo_index,         # 0, 1 or 2 三个YOLO检测层
                                arc=arc)                       # yolo architecture

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                if arc == 'defaultpw':  # default with positive weights
                    b = [-4, -3.6]  # obj, cls
                elif arc == 'default':  # default no pw (40 cls, 80 obj)
                    b = [-5.5, -4.0]
                elif arc == 'uBCE':  # unified BCE (80 classes)
                    b = [0, -8.5]
                elif arc == 'uCE':  # unified CE (1 background + 80 classes)
                    b = [10, -0.1]
                elif arc == 'Fdefault':  # Focal default no pw (28 cls, 21 obj, no pw)
                    b = [-2.1, -1.8]
                elif arc == 'uFBCE':  # unified FocalBCE (5120 obj, 80 classes)
                    b = [0, -3.5]
                elif arc == 'uFCE':  # unified FocalCE (64 cls, 1 background + 80 classes)
                    b = [7, -0.1]

                bias = module_list[-1][0].bias.view(len(mask), -1)  # 255 to 3x85
                bias[:, 4] += b[0]  # obj
                bias[:, 5:] += b[1]  # cls
                # bias = torch.load('weights/yolov3-spp.bias.pt')[yolo_index]  # list of tensors [3x85, 3x85, 3x85]
                module_list[-1][0].bias = torch.nn.Parameter(bias.view(-1))
                # utils.print_model_biases(model)
            except:
                print('WARNING: smart bias initialization failure.')

        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters) # 存储经过卷积输出的channel数目 [3, 32, 64, 256, 128, 128, 384...]
    # <class 'list'>: [1, 5, 8, 12, 15, 18, 21, 24, 27, 30, 33, 37, 40, 43, 46, 49, 52, 55, 58, 62, 65, 68, 71, 79, 85, 61, 91, 97, 36]
    return module_list, routs


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class YOLOLayer(nn.Module):
    """Detection layer"""
    def __init__(self, anchors, nc, img_size, yolo_index, arc):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors) # 当前YOLO层的anchor设置: [(116, 90), (156, 198), (373, 326)]
        self.na = len(anchors)               # number of anchors (3)
        self.nc =  nc  # number of classes (80)
        self.nx = 0    # initialize number of x gridpoints
        self.ny = 0    # initialize number of y gridpoints
        self.arc = arc # 'default'

    def forward(self, p, img_size, var=None):
        """

        :param p: 输入的YOLO检测特征图 torch.Size([1, 255, 13, 10]) BCHW
        :param img_size: 输入尺寸torch.Size([416, 320])
        :param var:
        :return:
        """
        bs, ny, nx = p.shape[0], p.shape[-2], p.shape[-1]
        if (self.nx, self.ny) != (nx, ny): # 不用每次都计算，只有在输入图片大小第一次发生变化时计算
            create_grids(self, img_size, (nx, ny), p.device, p.dtype)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.nc + 5, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        else:  # inference
            # Get outputs
            # 这里的prediction（p）是初步的所有预测，在grid_size*grid_size个网格中，它表示每个网格都会有num_anchor（3）个anchor框
            # s = 1.5  # scale_xy  (pxy = pxy * s - (s - 1) / 2)
            io = p.clone()  # inference output
            # 针对每个网格cell的偏移量，每个网格的单位长度为1，而预测的中心点（x，y）是归一化的（0，1之间），所以可以直接相加
            io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + self.grid_xy # xy # Center xy 相对于特征图而不是原图
            # anchor_w的范围是[0,grid_size](416下)，浮点型变量
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh   # wh yolo method 相对于特征图而不是原图
            # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
            io[..., :4] *= self.stride # 乘上了特征图下采样的倍数，放大到最初输入的尺寸，使得由相对于特征图变成相对于输入图像（416）

            if 'default' in self.arc:  # seperate obj and cls
                torch.sigmoid_(io[..., 4:])
            elif 'BCE' in self.arc:  # unified BCE (80 classes)
                torch.sigmoid_(io[..., 5:])
                io[..., 4] = 1
            elif 'CE' in self.arc:  # unified CE (1 background + 80 classes)
                io[..., 4:] = F.softmax(io[..., 4:], dim=4)
                io[..., 4] = 1

            if self.nc == 1:
                io[..., 5] = 1  # single-class model https://github.com/ultralytics/yolov3/issues/235

            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
            return io.view(bs, -1, 5 + self.nc), p


class Darknet(nn.Module):
    # YOLOv3 object detection model

    def __init__(self, cfg, img_size=(416, 416), arc='default'):
        """
        构建YOLO-V3模型
        :param cfg: 模型配置文件 'cfg/yolov3-spp.cfg'
        :param img_size: 网络的输入分辨率
        :param arc: 'default'
        """
        super(Darknet, self).__init__()

        self.module_defs = parse_model_cfg(cfg)  # 解析模型配置文件，得到配置列表
        self.module_list, self.routs = create_modules(self.module_defs, img_size, arc)
        self.yolo_layers = get_yolo_layers(self) # 得到三个yolo检测层分别所在的层数: [82, 94, 106]

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training

    def forward(self, x, var=None):
        img_size = x.shape[-2:] # torch.Size([416, 320])
        layer_outputs = []
        output = []

        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = mdef['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'route':
                layers = [int(x) for x in mdef['layers'].split(',')]
                if len(layers) == 1: # 一个数字就代表取特定的层
                    # [-4] 代表去倒数第四个特征图
                    x = layer_outputs[layers[0]]
                else:
                    # 多个数字就代表进行特征图concat融合
                    try:
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
                    except:  # apply stride 2 for darknet reorg layer
                        layer_outputs[layers[1]] = F.interpolate(layer_outputs[layers[1]], scale_factor=[0.5, 0.5])
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
                    # print(''), [print(layer_outputs[i].shape) for i in layers], print(x.shape)
            elif mtype == 'shortcut':
                x = x + layer_outputs[int(mdef['from'])]
            elif mtype == 'yolo':
                x = module(x, img_size)
                output.append(x) # 保存三个YOLO层的输出结果
            layer_outputs.append(x if i in self.routs else []) # 将self.routs层的output都保存起来

        if self.training:
            return output
        else:
            # io代表进行了处理的YOLO层输出结果 torch.Size([1, 390, 85])       torch.Size([1, 1560, 85])      torch.Size([1, 6240, 85])
            # p代表YOLO层预测出来的结果        torch.Size([1, 3, 13, 10, 85]) torch.Size([1, 3, 26, 20, 85]) torch.Size([1, 3, 52, 40, 85])
            io, p = list(zip(*output))  # inference output, training output
            return torch.cat(io, 1), p

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        # model_info(self)  # yolov3-spp reduced from 225 to 152 layers


def get_yolo_layers(model):
    return [i for i, x in enumerate(model.module_defs) if x['type'] == 'yolo']  # [82, 94, 106] for yolov3


def create_grids(self, img_size=416, ng=(13, 13), device='cpu', type=torch.float32):
    nx, ny = ng  # x and y grid size 10,13
    self.img_size = max(img_size)         # 计算输入图片的较长边 416
    self.stride = self.img_size / max(ng) # 计算下采样倍数 32.0

    # build xy offsets
    # yv shape: torch.Size([13, 10])
    # xv shape: torch.Size([13, 10])
    # torch.arange(ny=13) = tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])
    # torch.arange(nx=10) = tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9])
    # 这里生成了每个格子的左上角坐标，生成的坐标为grid x grid的二维数组，
    # yv，xv分别对应这个二维矩阵的x,y坐标的数组，yv，xv的维度与grid维度一样。每个grid cell的尺寸均为1，
    # 故grid范围是[0,12]（假如当前的特征图13*13）
    # Calculate offsets for each grid
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view((1, 1, ny, nx, 2)) # torch.Size([1, 1, 13, 10, 2])

    # build wh gains
    # 图片缩小多少倍，对应的anchors也要缩小相应倍数，也就是相对于特征图的cell torch.Size([3, 2])
    self.anchor_vec = self.anchors.to(device) / self.stride
    # anchor_w的范围是[0, grid_size](416下),浮点型数值
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).to(device).type(type)
    self.ng = torch.Tensor(ng).to(device) # tensor([10., 13.])
    self.nx = nx
    self.ny = ny


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'
    # cutoff: save layers between 0 and cutoff (if cutoff = -1 all are saved)
    file = Path(weights).name

    # Try to download weights if not available locally
    msg = weights + ' missing, download from https://drive.google.com/drive/folders/1uxgUBemJVw9wZsdpboYbzUN4bcRhsuAI'
    if not os.path.isfile(weights):
        try:
            url = 'https://pjreddie.com/media/files/' + file
            print('Downloading ' + url)
            os.system('curl -f ' + url + ' -o ' + weights)
        except IOError:
            print(msg)
    assert os.path.exists(weights), msg  # download missing weights from Google Drive

    # Establish cutoffs
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

    ptr = 0
    for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if mdef['type'] == 'convolutional':
            conv_layer = module[0]
            if mdef['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w

    return cutoff


def save_weights(self, path='model.weights', cutoff=-1):
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, 'wb') as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version.tofile(f)  # (int32) version info: major, minor, revision
        self.seen.tofile(f)  # (int64) number of images seen during training

        # Iterate through layers
        for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if mdef['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if mdef['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(f)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(f)


def convert(cfg='cfg/yolov3-spp.cfg', weights='weights/yolov3-spp.weights'):
    # Converts between PyTorch and Darknet format per extension (i.e. *.weights convert to *.pt and vice versa)
    # from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')

    # Initialize model
    model = Darknet(cfg)

    # Load weights and save
    if weights.endswith('.pt'):  # if PyTorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        save_weights(model, path='converted.weights', cutoff=-1)
        print("Success: converted '%s' to 'converted.weights'" % weights)

    elif weights.endswith('.weights'):  # darknet format
        _ = load_darknet_weights(model, weights)

        chkpt = {'epoch': -1,
                 'best_fitness': None,
                 'training_results': None,
                 'model': model.state_dict(),
                 'optimizer': None}

        torch.save(chkpt, 'converted.pt')
        print("Success: converted '%s' to 'converted.pt'" % weights)

    else:
        print('Error: extension not supported.')
