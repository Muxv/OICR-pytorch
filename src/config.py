import torch
import numpy as np
from easydict import EasyDict as edict

cfg = edict()
cfg.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cfg.K = 3

cfg.DATA = edict()
cfg.DATA.SCALES = (480, 576, 688, 864, 1200)
cfg.DATA.MAX_SIDE = 2000

cfg.INIT = edict()
cfg.INIT.MEAN = 0
cfg.INIT.DEVI = 0.01
cfg.INIT.BAIS = 0

cfg.TRAIN = edict()
cfg.TRAIN.It = 0.5
cfg.TRAIN.NMSIoU = 0.3
cfg.TRAIN.BATCH_SIZE = 2
cfg.TRAIN.WD = 5e-4
cfg.TRAIN.LR = 1e-3
cfg.TRAIN.LR_07STEP = 12
cfg.TRAIN.LR_07_MUL = 0.1
cfg.TRAIN.EPOCH_07 = 28

cfg.PATH = edict()
cfg.PATH.PT_PATH = "../checkpoints/"
cfg.PATH.LOG_PATH = "../logs/"
cfg.PATH.ROI_PATH = "../region/"

