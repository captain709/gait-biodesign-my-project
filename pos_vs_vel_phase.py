import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import os
from glob import glob


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pipeline as pipeline

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

## --dataset class -> paring position dataset and velocity dataset


class DatasetPosVel(Dataset):
    def __init__(
            self, 
            pos_dir, 
            vel_dir,
            event_function_dict=None, 
            trfm=None, 
            windowWing=3,
            regressWing=3,
            predictionGap=1,
            ):
        
        self.pos_dir = pos_dir
        self.vel_dir = vel_dir
        self.windowWing = windowWing

        ## getting position and velocity dataset

        self.pos_dataset = pipeline.GaitPhasingDataset(
            self.pos_dir, 
            trfm=trfm, 
            windowWing=windowWing,     
            windowWing=3,
            regressWing=3,
            predictionGap=1,
            rotationMatrixList=None,
            meanForPhaseExtraction=None,
            sdForPhaseExtraction=None,
        )

        self.vel_dataset = pipeline.GaitPhaseingDataset(
            self.vel_dir, 
            trfm=trfm,
            windowWing=windowWing,     
            windowWing=3,
            regressWing=3,
            predictionGap=1,
            rotationMatrixList=None,
            meanForPhaseExtraction=None,
            sdForPhaseExtraction=None,  
        )

        self.event_function_dict = event_function_dict # event function for extracting events from dataset
        self.events = self.get_events()

    def get_events(self):

        for event_function, input_field in self.event_function_dict.items():
            