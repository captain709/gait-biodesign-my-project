import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import pipeline as pipeline
import util as util
from collections import Counter
## dataset path
DATASET_DICT = {
    "LEFT": {
        "POS": "/mnt/ExpDrive/SparkLabLongRun/Data/peam_dataset/28-04-69/position/LeftFoot/Foot_to_Pelvis",
        "VEL": "/mnt/ExpDrive/SparkLabLongRun/Data/peam_dataset/28-04-69/velocity/LeftFoot/Foot_to_Pelvis"
    },
    "RIGHT": {
        "POS": "/mnt/ExpDrive/SparkLabLongRun/Data/peam_dataset/28-04-69/position/RightFoot/Foot_to_Pelvis",
        "VEL": "/mnt/ExpDrive/SparkLabLongRun/Data/peam_dataset/28-04-69/velocity/RightFoot/Foot_to_Pelvis"
    }
}

ROOT = "/mnt/ExpDrive/SparkLabLongRun/PeamProject/results"
# DATA_DIR = "C:\\Users\\noppa\\OneDrive\\เดสก์ท็อป\\SparkLab\\Code\\SPARK-Lab-IMU\\data\\pkl_peam_test\\position\\LeftFoot\\Foot_to_Pelvis"
DATA_DIR_POS = "/mnt/ExpDrive/SparkLabLongRun/Data/peam_dataset/28-04-69/position/RightFoot/Foot_to_Pelvis"
DATA_DIR_VEL = "/mnt/ExpDrive/SparkLabLongRun/Data/peam_dataset/28-04-69/velocity/RightFoot/Foot_to_Pelvis"
# DATA_DIR = "../data/Training_data/pkl/walking_left/LeftFoot"
DATA_ROOT = "/mnt/ExpDrive/SparkLabLongRun/Data/peam_dataset/28-04-69"


def label_counter(label_list):

    counter = {}

    for label in label_list:


        if label in counter.keys():

            counter[label] += 1

        else:
            
            counter[label] = 1

    return counter

def get_unique_data(data):

    counts_dict = Counter(data)

    unique_elements = sorted(counts_dict.keys())
    counts = [counts_dict[element] for element in unique_elements]
    
    return unique_elements, counts

## dummy datset

test_dataset = pipeline.GaitPhasingWithEventDataset(
    data_root=DATA_ROOT,
    data_dir=DATA_DIR_POS,
    img_dir=DATA_DIR_POS, 
    event_function_dict={"GaitEvent": util.detect_gait_events}, 
    trfm=None, 
    windowWing=3,
    regressWing=3,
    predictionGap=1,
)

for k, v in test_dataset.AllData.items():
    print(f"{k}: {type(v)}")
    
    if k == "centerPose_label":
        
        for event, label in v.items():
            print(f"Event: {event}, Label shape: {label}")
            values , counts = get_unique_data(label)
            print(f"Unique values: {values}")
            print(f"Counts: {counts}")
            # print(f"Label Unique: {label_counter(label)}")