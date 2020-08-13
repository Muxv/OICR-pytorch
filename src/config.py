import torch
import numpy as np
from easydict import EasyDict as edict

cfg = edict()
cfg.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cfg.DEVICE = torch.device("cpu")

cfg.K = 3
cfg.Groups = 64

cfg.DATA = edict()
# cfg.DATA.SCALES = (480, 576, 688, 864, 1200)
cfg.DATA.SCALES = (480, 576, 688)
cfg.DATA.MAX_SIDE = 1000

cfg.INIT = edict()
cfg.INIT.MEAN = 0
cfg.INIT.DEVI = 0.01
cfg.INIT.BAIS = 0

cfg.TRAIN = edict()
cfg.TRAIN.It = 0.5
cfg.TRAIN.NMSIoU = 0.3
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.BATCH_SIZE = 1
cfg.TRAIN.ITER_SIZE = 2
cfg.TRAIN.WD = 5e-4
cfg.TRAIN.BIAS_DOUBLE_LR = True
# cfg.TRAIN.LR = 1e-3
# 1e-3 will mislead the training process
cfg.TRAIN.LR = 1e-3
cfg.TRAIN.LR_STEP = 16
cfg.TRAIN.LR_MUL = 0.1
# cfg.TRAIN.OICR_LR = 1e-4
cfg.TRAIN.EPOCH = 30


cfg.PATH = edict()
cfg.PATH.PT_PATH = "../checkpoints/"
cfg.PATH.LOG_PATH = "../logs/"
cfg.PATH.ROI_PATH = "../region/"

