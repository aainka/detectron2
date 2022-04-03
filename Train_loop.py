from codecs import open
import numpy as np
import yaml
from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm # progress bar
from datetime import datetime
import time
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import os, json, cv2, random
#import skimage.io as io
import copy
from pathlib import Path
from typing import Optional
import detectron2
from imantics import Polygons, Mask
import math
from tqdm import tqdm
import itertools
import shutil
#from labelme2coco import * 
#import All_Suscap_ml_randomforest as ar
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
#from copy_paste import CopyPaste
from fvcore.transforms.transform import TransformList, Transform, NoOpTransform

from glob import glob
#import numba
#from numba import jit
import glob
import os
import copy
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from matplotlib import cm
import argparse
import warnings
warnings.filterwarnings('ignore') #Ignore "future" warnings and Data-Frame-Slicing warnings.

#### detectron2
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import register_coco_instances

from torch.utils.tensorboard import SummaryWriter
from detectron2.utils.events import EventWriter, get_event_storage

from detectron2.engine import HookBase
from detectron2.data import build_detection_train_loader
import detectron2.utils.comm as comm

from detectron2.utils.visualizer import ColorMode

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
#from tools.swint import add_swint_config
import logging
setup_logger()
def set_init():
    ### CUDA 0번 사용 ###
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_yaml(file_path, encoding="utf-8"):
    assert os.path.isfile(file_path) 
    with open(file_path, "r", encoding=encoding) as f:
        dict_yaml = yaml.load(f, Loader=yaml.FullLoader)
        return dict_yaml

def createDirectory(directory): 
    try: 
        if not os.path.exists(directory): 
            os.makedirs(directory) 
    except OSError: 
        print("Error: Failed to create the directory.")

class Sharpness(Transform):
    
    def __init__(self, sharpness_stage=[0], p=0.5):
        super().__init__()
        
        self.sharpness_stage = sharpness_stage
        self.p = p
        
    def apply_image(self, img):
        sharpening_mask1 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
        sharpening_mask2 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpening_mask3 = np.array([[-1, -2, -1], [-1, 11, -1], [-1, -2, -1]])
        sharpening_mask4 = np.array([[-1, -2, -1], [-2, 13, -2], [-1, -2, -1]]) 
        sharpening_mask5 = np.array([[-2, -2, -2], [-2, 17, -2], [-2, -2, -2]])

        sharpning_list = [sharpening_mask1,sharpening_mask2,sharpening_mask3,sharpening_mask4,sharpening_mask5]
        stage_choice = random.choice(self.sharpness_stage)
        if random.random() > self.p:
            #h, w = img.shape[:2]
            img = cv2.filter2D(img,-1,sharpning_list[stage_choice])
        
        return np.asarray(img)
    
    def apply_coords(self, coords):
        return coords.astype(np.float32)

def set_transform_list(tr_state):
    index_transform = list(filter(lambda x:tr_state[x] == True, range(len(tr_state))))
    #list_t = list(map(lambda x:T.RandomFlip(prob=0.5, horizontal=False, vertical=True) if tr_state[0] == True else None,tr_state[0]))

    original_list = [
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomApply(T.RandomContrast(0.8, 1.2),prob=0.5),
        T.RandomApply(T.RandomBrightness(0.8, 1.2),prob=0.5),
        T.RandomApply(T.RandomSaturation(0.8, 1.2),prob=0.5),
        T.RandomApply(T.RandomLighting(0.8),prob=0.5),
        Sharpness(sharpness_stage=[0,1,2,3,4], p=0.5)
    ]
    new_list = []
    for i in range(len(index_transform)):
        new_list.append(original_list[index_transform[i]])

    return new_list

def custom_mapper(dataset_dict):
    """
    dataloader

    Augmentation 적용
    RandoFlip, RandomContrast, RadomBrigtness
    """

    transform_list = set_transform_list(Transform_State)
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    image, transforms = T.apply_transform_gens(transform_list, image)
    #print(image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict

class MyTrainer(DefaultTrainer):
    #"""
    # Trainer
    # Augmentation 적용 후 build_detection_train_loader 적용 
    
    # """
    @classmethod
    def build_train_loader(cls,cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

    def build_hooks(self):
        ret = super().build_hooks()
        ret.append(BestCheckpointer())   #Best checkpoint 적용
        return ret

class ValidationLoss(HookBase):

    """
    validation loss 정의
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        #self.cfg.DATASETS.TRAIN = cfg.DATASETS.VAL
        self._loader = iter(build_detection_train_loader(self.cfg))
        
    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)
            
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in 
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced, 
                                                 **loss_dict_reduced)

class BestCheckpointer(HookBase):
    """
    bestcheckpointer logger save
    """
    def before_train(self):
        self.best_metric = 0.0
        self.logger = logging.getLogger("detectron2.trainer")
        self.logger.info("######## Running best check pointer")

    def after_step(self):
        metric_name = 'val_total_loss'
        if metric_name in self.trainer.storage._history:
            eval_metric, batches = self.trainer.storage.history(metric_name)._data[-1]
            if self.best_metric < eval_metric:
                self.best_metric = eval_metric
                self.logger.info(f"######## New best metric: {self.best_metric}")
                self.trainer.checkpointer.save(f"model_best_{eval_metric:.4f}")



def main(arg):
    global Transform_State

    ##데이터 라벨 coco 라벨 형식으로 변환
    if not os.path.isfile(arg['PATH']['Train_path']+"/train.json"):
        print("train json making ....")
        labelme_json = glob.glob(os.path.join(arg['PATH']['Train_path'], "*.json"))
        labelme2coco(labelme_json, os.path.join(arg['PATH']['Train_path'],"train.json"))

    ####TRAIN, VAL 데이터셋 정의 ####
    register_coco_instances("Train_db", {}, arg['PATH']['Train_path']+"/train.json", arg['PATH']['Train_path'])
    if arg['Val']['eval']:
        register_coco_instances("Val_db", {}, arg['PATH']['Val_path']+"/val.json", arg['PATH']['Val_path'])
    thing_classes = arg['TRAIN']['Class'].split(',')
    metadata = MetadataCatalog.get("Train_db").thing_clasees = thing_classes
    
    #### cfg 정의 ###
    cfg = get_cfg()
    if arg['TRAIN']['MODEL'] == 'swinT':
        add_swint_config(cfg)
        cfg.merge_from_file("./configs/SwinT/mask_rcnn_swint_T_FPN_3x.yaml")
    elif arg['TRAIN']['MODEL'] =='MaskRCNN_R50':
        cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    elif arg['TRAIN']['MODEL'] =='CascadeRCNN_X152':
        cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"))


    cfg.DATASETS.TRAIN = ("Train_db",)
    if arg['Val']['eval']:
        cfg.DATASETS.TEST = ("Val_db",)
        cfg.DATASETS.VAL = ("Val_db",)
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.WEIGHTS = arg['PATH']['Pretrained_Weight_path']

    cfg.SOLVER.IMS_PER_BATCH = arg['TRAIN']['IMS_PER_BATCH']
    cfg.INPUT.MASK_FORMAT='bitmask'
    cfg.SOLVER.BASE_LR = arg['TRAIN']['BASE_LR']
    cfg.SOLVER.WARMUP_ITERS = arg['TRAIN']['WARMUP_ITER']  # base Lr에 도달하기 까지 얼마나 iteration을 갈 것인지 
    cfg.SOLVER.MAX_ITER = arg['TRAIN']['MAX_ITER'] #최종 Iteration
    cfg.SOLVER.CHECKPOINT_PERIOD = arg['TRAIN']['CHECKPOINT_PERIOD']
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)  # class 개수
    cfg.INPUT.MIN_SIZE_TEST = 720
    #cfg.MODEL.PIXEL_MEAN =  [163.13725043402778, 129.69134548611112, 97.49447916666666]
    #cfg.MODEL.PIXEL_MEAN = [140.0,145.0,150.0]
    #cfg.MODEL.PIXEL_STD = [58.395,57.120,57.375]
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.2, 0.5, 1.0, 2.0, 5.0]]  # [[0.5, 1.0, 2.0]]
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[9], [17], [31], [63], [127]]
    # minimum image size for the train set
    cfg.TEST_EVAL_PERIOD = 20
    #cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = []
    cfg.OUTPUT_DIR = arg['PATH']['Output_dir']
    #### train한 결과들을 저장하는 경로 만들기 ###
    cfg.MODEL.BACKBONE.FREEZE_AT = arg['TRAIN']['BACKBONE_FREEZE_AT']  #Backbone freeze 1 non freeze 0 
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = arg['TRAIN']['NMS_THRESH_TEST']
    #cfg.freeze()
    createDirectory(cfg.OUTPUT_DIR)

    #augmentation 정의
    RandomFlip_V = arg['AUGMENTATION']['RandomFlip_V']
    RandomFlip_H = arg['AUGMENTATION']['RandomFlip_H']
    RandomContrast = arg['AUGMENTATION']['RandomContrast']
    RandomBrightness = arg['AUGMENTATION']['RandomBrightness']
    RandomSaturation = arg['AUGMENTATION']['RandomSaturation']
    RandomLighting = arg['AUGMENTATION']['RandomLighting']
    Sharpness = arg['AUGMENTATION']['Sharpness']

    Transform_State = [RandomFlip_V,RandomFlip_H,RandomContrast,RandomBrightness,RandomSaturation,RandomLighting,Sharpness]

    #### trainning ####
    trainer = MyTrainer(cfg)

    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])

    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    if arg['Val']['eval']:
        #### Validaion ####
        evaluator = COCOEvaluator("Val_db", ("bbox", "segm"), False, output_dir="./output/")
        val_loader = build_detection_test_loader(cfg, "Val_db")
        print(inference_on_dataset(trainer.model, val_loader, evaluator))
        ap_data = inference_on_dataset(trainer.model, val_loader, evaluator)

        with open(f'{cfg.OUTPUT_DIR}/AP_result.json', 'w') as f:
            json.dump(ap_data, f)
    
if __name__ == "__main__":

    file_path = "./config.yml"
    config = parse_yaml(file_path, encoding="utf-8")
    #arg = parse_opt()
    set_init()
    main(config)
    print("######################")
    print("##Training complete!##")
    print("######################")

    """
    python3 ./Train.py --Train=True --Train_path='/content/hdd_0/SUSCAP/R104/Images/2022_02_04_Original93/Train'
 --Test_path='/content/hdd_0/SUSCAP/R104/Images/2022_02_03_IMAGES_Plus_Sharpness/Val' --Predict_path='/content/Scratch_Detect/R104/Predict/' --Predict=True --NMS=True --WARMUP_ITER=100 --MAX_ITER=300 --IMS_PER_BATCH=8  --Weight_path='/content/Scratch_Detect/R104/Output/output/model_final.pth' --Pretrained_Weight_path='/content/Scratch_Detect/R102/Output/output_oldsus_final/model_final.pth' --THRESH_TEST=0.8 --Val_path='/content/hdd_0/SUSCAP/R104/Images/2022_02_04_Original93/Val'
    """
    """
    python3 ./Train.py --Train=True --Train_path='/content/hdd_0/SUSCAP/R104/Images/OLD_SUS_PRETRAINED/Train' --Test_path='/content/hdd_0/SUSCAP/R104/Images/2022_02_03_IMAGES_Plus_Sharpness/Val' --Predict_path='/content/Scratch_Detect/R104/Predict/' --Predict=True --NMS=True --WARMUP_ITER=300 --MAX_ITER=1000 --IMS_PER_BATCH=32  --Weight_path='/content/Scratch_Detect/R104/Output/output/model_final.pth' --THRESH_TEST=0.8 --Val_path='/content/hdd_0/SUSCAP/R104/Images/OLD_SUS_PRETRAINED/Val'
    """

