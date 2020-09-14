import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import copy
from utils import VOC_CLASSES
from config import cfg

from PIL import Image
from scipy.io import loadmat

def remove_repetition(boxes):
    """
    remove repetited boxes
    :param boxes: [N, 4]
    :return: keep:
    """
    _, x1_keep = np.unique(boxes[:, 0], return_index=True)
    _, x2_keep = np.unique(boxes[:, 2], return_index=True)
    _, y1_keep = np.unique(boxes[:, 1], return_index=True)
    _, y2_keep = np.unique(boxes[:, 3], return_index=True)
 
    x_keep = np.union1d(x1_keep, x2_keep)
    y_keep = np.union1d(y1_keep, y2_keep)
    mask = np.union1d(x_keep, y_keep)
    return mask

def filter_small_boxes(boxes, min_size):
    """Filters out small boxes."""
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    mask = (w >= min_size) & (h >= min_size)
    return mask

def totensor(img):
#     t = transforms.ToTensor()
    t = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return t(img)

def hflip_box(boxes, w):
    for box in boxes:
        box[0], box[2] = w - box[2], w - box[0]

def hflip_img(img):
    """
    img : PIL Image
    """
#     fliper = transforms.RandomHorizontalFlip(1)
#     cflip(src, flipCode[, dst])
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def x2ychange_box(boxes):
    for box in boxes:
        box[0], box[1] = box[1], box[0]
        box[2], box[3] = box[3], box[2]

def resize_box(boxes, ratio):
    for box in boxes:
        box[0] = int(ratio * box[0])
        box[1] = int(ratio * box[1])
        box[2] = int(ratio * box[2])
        box[3] = int(ratio * box[3])
    
def resize_img_smallside(img, smallside):
    """
    img : PIL Image
    smallside : change the image small side length to smallside
    """
    w, h = img.size
    ratio = 0
    resizer = None
#     print(f'origin size: w * h:({w}, {h})')
    if w < h:
        if int(smallside*h/w) >= cfg.DATA.MAX_SIDE:
            ratio = cfg.DATA.MAX_SIDE/h
#                                             h                 w
            resizer = transforms.Resize((cfg.DATA.MAX_SIDE, int(w*ratio)))
        else:
            ratio = smallside/w
#                                             h                 w
            resizer = transforms.Resize((int(h*ratio), smallside))
    else: # h <= w
        if int(smallside*w/h) >= cfg.DATA.MAX_SIDE:
            ratio = cfg.DATA.MAX_SIDE/w
#                                             h                  w
            resizer = transforms.Resize((int(h*ratio), cfg.DATA.MAX_SIDE))
        else:
            ratio = smallside/h
#                                             h                  w
            resizer = transforms.Resize((smallside, int(w * ratio)))
    img = resizer(img)
    return img, ratio

class VOCAnnotationAnalyzer():
    """
    deal with annotation data (dict)
    
    Arguments:
        cls_to_idx (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """
    def __init__(self, cls_to_idx=None, keep_difficult=False):
        self.cls_to_idx = cls_to_idx or dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult
        
    def __call__(self, annotation: dict):
        w = int(annotation['size']['width'])
        h = int(annotation['size']['height'])
        # if img only contains one gt that annotation['object'] is just a dict, not a list
        objects = [annotation['object']] if type(annotation['object']) != list else annotation['object']
        res = [] # [xmin, ymin, xmax, ymax, label]
        for box in objects:
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            difficult = int(box['difficult'])
            if not self.keep_difficult and difficult:
                continue
            name = box['name']
            bnd = []
            for pt in pts:
                bnd.append(int(box['bndbox'][pt]))
            bnd.append(self.cls_to_idx[name])
            res.append(bnd)
            
        return res
    
class VOCDectectionDataset(data.Dataset):
    def __init__(self, root, year, image_set,
                 target_transform=VOCAnnotationAnalyzer(),
                 dataset_name='VOC07_12',
                 region_propose='selective_search',
                 use_corloc=False,
                 debug=False,
                 small_box=True,
                 over_box=True):
        super(VOCDectectionDataset, self).__init__()
        self.datas = datasets.VOCDetection(root, str(year), image_set, download=False)
        self.image_set = image_set
        self.name = dataset_name
        self.target_transform = target_transform # use for annotation
        self.debug = debug
        self.region_propose = region_propose
        self.box_mat = self.get_mat(year, image_set, region_propose)
        self.use_corloc = use_corloc
        self.small_box = small_box
        self.over_box = over_box
            
            
    def get_box_from_mat(self, index):
        return self.box_mat['boxes'][0][index].tolist()
    
    def get_mat(self, year, image_set, region_propose):
        """
        load the box generated
        """
        boxes = None
        boxes_score = None
        
        if str(year) == '2007' and image_set == 'trainval' and region_propose == 'selective_search':
            mat = loadmat("../region/SelectiveSearchVOC2007trainval.mat")
        elif str(year) == '2007' and image_set == 'test' and region_propose == 'selective_search':
            mat = loadmat("../region/SelectiveSearchVOC2007test.mat")
        return mat
            
    def __getitem__(self, index):
        img, gt = self.datas[index]
        region = self.get_box_from_mat(index)
        if self.target_transform:
            gt = self.target_transform(gt["annotation"])
        w, h = img.size
        
        region = np.array(region).astype(np.float32)
        
        if not self.small_box:
            region_filter = filter_small_boxes(region, 20)
            region = region[region_filter]
    
        if not self.over_box:
            unique_filter = remove_repetition(region)
            region = region[unique_filter]

        x2ychange_box(region)        
        
        gt = np.array(gt).astype(np.float32)

        # ----------------------------------------------------------------------------------
        if self.use_corloc:
            # use train data for SCLAES * 2 to get CorLoc
            assert(self.image_set == "trainval")
            n_images = []
            n_regions = []
            gt_box = gt[:, :4]
            # first change box's cor
            for flip in [0.0, 1.0]:            
                for scale in cfg.DATA.SCALES:
                    new_img = img.copy()
                    new_region = copy.deepcopy(region)
                    if flip == 1.0:
                        new_img = hflip_img(new_img)
                        hflip_box(new_region, w)
                    new_img, ratio = resize_img_smallside(new_img, scale)
                    resize_box(new_region, ratio)
                    n_images.append(totensor(new_img))
                    n_regions.append(new_region)
            return n_images, gt, n_regions, region   
        
        else:
            if self.debug == True:
                return img, gt, region
            # train normally
            elif self.image_set == "trainval":
                target = [0 for _ in range(len(VOC_CLASSES))]
                gt_target = gt[:, -1]
                for t in gt_target:
                    target[int(t)] += 1.0
                gt_box = gt[:, :4]
                gt_count = np.array(target).astype(np.float32)

                # follow by paper: randomly horiztontal flip and randomly resize
                if np.random.random() > 0.5: # then flip
                    img = hflip_img(img)
                    hflip_box(region, w)
                    hflip_box(gt_box, w)
                # then resize
                max_side = cfg.DATA.SCALES[np.random.randint(len(cfg.DATA.SCALES))]
    #             print()
                img, ratio = resize_img_smallside(img, max_side)
                resize_box(region, ratio)
                resize_box(gt_box, ratio)
                img = totensor(img)
                return img, gt_box, gt_count, region

            # ----------------------------------------------------------------------------------
            # test for map normally
            elif self.image_set == "test":
                n_images = []
                n_regions = []
                gt_box = gt[:, :4]
                # first change box's cor
                for flip in [0.0, 1.0]:            
                    for scale in cfg.DATA.SCALES:
                        new_img = img.copy()
                        new_region = copy.deepcopy(region)
                        if flip == 1.0:
                            new_img = hflip_img(new_img)
                            hflip_box(new_region, w)
                        new_img, ratio = resize_img_smallside(new_img, scale)
                        resize_box(new_region, ratio)
                        n_images.append(totensor(new_img))
                        n_regions.append(new_region)
                return n_images, gt, n_regions, region   
            else:
                raise ValueError(f"image_set can only be 'test' or 'trainval'")
    def __len__(self):
        return len(self.datas)
#         return 10
# 