import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
import csv

import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from typing import Literal, Callable, Optional

import os
import sys
from pathlib import Path
from glob import glob
import time

import model as model
import pipeline as pipeline
import util as util
from mpl_toolkits import mplot3d


# ==============================================================================
# VISUALIZATION HOOKS
# Default visualization functions — can be replaced or extended from outside
# ==============================================================================

def default_viz_epoch(
    step: int,
    pn,
    nowState_tensor,
    add_noise,
    centerPose_img,
    centerPose_position,
    EndNumberPlusOne,
    SubjectAmount,
    srcColorList,
    srcColorList_rat,
    imageFolder: str,
    imageFolderRat: str,
    DEVICE: str,
    **kwargs
):
    """
    Default per-epoch visualization function.
    Runs all util plot calls. Replace or extend by passing custom_viz_epoch to train().
    """
    with torch.no_grad():

        prePhase_distribution, phaseXY_distribution, phaseRad = pn(nowState_tensor)
        prePhaseMargin, _, _ = pn(add_noise(nowState_tensor).to(DEVICE))

        prePhase_distribution = prePhase_distribution.detach().cpu().numpy().T
        phaseXY_distribution  = phaseXY_distribution.detach().cpu().numpy().T
        phaseRad              = phaseRad.detach().cpu().numpy().T
        prePhaseMargin        = prePhaseMargin.detach().cpu().numpy().T

        fig0   = kwargs.get("fig0")
        fig    = kwargs.get("fig")
        figRatF1 = kwargs.get("figRatF1")

        util.saveMarginPlot(
            prePhase_distribution, phaseXY_distribution, prePhaseMargin,
            imageFolder + "pytorch_circle_" + str(step) + '.png',
            fig0, srcColorList
        )

        util.saveMarginPlot(
            prePhase_distribution[:, 0:EndNumberPlusOne[0]],
            phaseXY_distribution[:, 0:EndNumberPlusOne[0]],
            prePhaseMargin[:, 0:EndNumberPlusOne[0]],
            imageFolderRat + "pytorch_Rat1_prephase_" + str(step) + '.png',
            fig[0], srcColorList_rat[0]
        )

        for ipic in range(1, SubjectAmount):
            util.saveMarginPlot(
                prePhase_distribution[:, EndNumberPlusOne[ipic-1]:EndNumberPlusOne[ipic]],
                phaseXY_distribution[:, EndNumberPlusOne[ipic-1]:EndNumberPlusOne[ipic]],
                prePhaseMargin[:, EndNumberPlusOne[ipic-1]:EndNumberPlusOne[ipic]],
                imageFolderRat + "pytorch_Rat" + str(ipic+1) + "_prephase_" + str(step) + '.png',
                fig[ipic], srcColorList_rat[ipic]
            )

        phase = phaseRad % (2 * np.pi)

        phaseRat = [phaseRad[0:EndNumberPlusOne[0]] % (2 * np.pi)]
        for iphaseRat in range(1, SubjectAmount):
            phaseRat.append(phaseRad[EndNumberPlusOne[iphaseRat-1]:EndNumberPlusOne[iphaseRat]] % (2 * np.pi))

        util.saveRatPhaseRainbowPlot(
            centerPose_img[:, 0:EndNumberPlusOne[0]], phaseRat[0],
            imageFolderRat + "pytorch_Trajec_Rat1_F1_" + str(step) + '.png',
            figRatF1[0]
        )

        for iTrajec in range(1, SubjectAmount):
            util.saveRatPhaseRainbowPlot(
                centerPose_img[:, EndNumberPlusOne[iTrajec-1]:EndNumberPlusOne[iTrajec]],
                phaseRat[iTrajec],
                imageFolderRat + "pytorch_Trajec_Rat" + str(iTrajec+1) + "_F1_" + str(step) + '.png',
                figRatF1[iTrajec]
            )

        util.savePhasePerCycleSpiral(centerPose_img, phase, imageFolder + "pytorch_RatAll_F1_percycle_" + str(step) + '.png', fig=None)
        util.save1000PhaseSpiral(centerPose_img, phase, imageFolder + "pytorch_RatAll_F1_1000_" + str(step) + '.png', fig=None)
        util.saveFirstHalfPhaseSpiral(centerPose_img, phase, imageFolder + "pytorch_RatAll_F1_firsthalf_" + str(step) + '.png', fig=None)
        util.saveLateHalfPhaseSpiral(centerPose_img, phase, imageFolder + "pytorch_RatAll_F1_latehalf_" + str(step) + '.png', fig=None)

        HSpeaks, TOpeaks = util.evaluateplot(
            centerPose_img, centerPose_position, phase,
            imageFolder + "pytorch_RatAll_F1_evaluation_" + str(step) + '.png', fig=None
        )
        util.XZPhase(
            centerPose_img, centerPose_position, phase,
            imageFolder + "pytorch_position_and_phase_" + str(step) + '.png',
            HSpeaks, TOpeaks, fig=None
        )


def default_viz_loss(
    step: int,
    speed_penalty_log: list,
    singularity_penalty_log: list,
    distribution_penalty_log: list,
    margin_penalty_log: list,
    total_loss_log: list,
    imageFolder: str,
    **kwargs
):
    """
    Default loss visualization function.
    Plots and prints loss statistics. Replace by passing custom_viz_loss to train().
    """
    util.plot_loss_tracking(
        speed_penalty_log,
        singularity_penalty_log,
        distribution_penalty_log,
        margin_penalty_log,
        total_loss_log,
        save_path=imageFolder + "pytorch_loss_tracking_" + str(step) + '.png'
    )
    util.print_loss_statistics(
        speed_penalty_log,
        singularity_penalty_log,
        distribution_penalty_log,
        margin_penalty_log,
        total_loss_log
    )


# ==============================================================================
# TRAIN FUNCTION
# ==============================================================================

def train(
    # --- Input mode ---
    input_mode: Literal["position", "velocity"],

    # --- Data (pre-loaded, passed from outside) ---
    dataset_position,
    dataset_velocity,

    # --- Model & Optimizer ---
    pn,
    optimizer,
    loss_fn_dict: dict,

    # --- Training Config ---
    EPOCH: int,
    centerSigma: float,
    DEVICE: str,

    # --- Transforms ---
    to_tensor,
    add_noise,

    # --- Output Paths (from outside) ---
    imageFolder: str,
    imageFolderRat: str,
    sessionFile: str,

    # --- Visualization Functions (modular, replaceable from outside) ---
    viz_epoch_fn: Optional[Callable] = default_viz_epoch,
    viz_loss_fn: Optional[Callable] = default_viz_loss,

    # --- Training Misc ---
    crossValidateNo: int = -1,
    startTime: Optional[float] = None,
) -> dict:
    """
    Train PhaseNetwork for gait phase analysis.

    Supports two input modes:
        - "position": uses dataset_position for nowState / nextState
        - "velocity": uses dataset_velocity for nowState / nextState

    Visualization functions are fully modular:
        - viz_epoch_fn: called at epoch visualization checkpoints
        - viz_loss_fn:  called every 2000 epochs for loss plotting
        Pass None to disable either. Pass a custom callable to override.

    Args:
        input_mode:         "position" or "velocity"
        dataset_position:   Pre-loaded position GaitPhasingDataset
        dataset_velocity:   Pre-loaded velocity GaitPhasingDataset
        pn:                 PhaseNetwork model instance
        optimizer:          PyTorch optimizer
        loss_fn_dict:       Dict of loss functions and weights
        EPOCH:              Number of training epochs
        centerSigma:        Sigma value for singularity/margin loss
        DEVICE:             "cuda" or "cpu"
        to_tensor:          ToTensor transform callable
        add_noise:          AddNoise transform callable
        imageFolder:        Path to save global visualization images
        imageFolderRat:     Path to save per-subject visualization images
        sessionFile:        Path to save model checkpoint (.ckpt)
        viz_epoch_fn:       Epoch visualization callable (default: default_viz_epoch)
        viz_loss_fn:        Loss visualization callable (default: default_viz_loss)
        crossValidateNo:    Cross-validation index (for logging only)
        startTime:          Training start time (float from time.time())

    Returns:
        dict: {
            'speed_penalty_log':        list[float],
            'singularity_penalty_log':  list[float],
            'distribution_penalty_log': list[float],
            'margin_penalty_log':       list[float],
            'total_loss_log':           list[float],
            'model':                    trained pn,
            'optimizer':                optimizer
        }

    Example:
        results = train(
            input_mode="velocity",
            dataset_position=dataset_position,
            dataset_velocity=dataset_velocity,
            pn=pn,
            optimizer=optimizer,
            loss_fn_dict=loss_fn_dict,
            EPOCH=10_000,
            centerSigma=1,
            DEVICE=DEVICE,
            to_tensor=to_tensor,
            add_noise=add_noise,
            imageFolder=paths["img"],
            imageFolderRat=paths["subject"],
            sessionFile=sessionFile,
        )
    """

    if startTime is None:
        startTime = time.time()

    # --------------------------------------------------------------------------
    # Select dataset based on input mode
    # --------------------------------------------------------------------------

    if input_mode == "velocity":
        active_dataset = dataset_velocity
    elif input_mode == "position":
        active_dataset = dataset_position
    else:
        raise ValueError(f"input_mode must be 'position' or 'velocity', got '{input_mode}'")

    print(f"[train] Input mode: {input_mode.upper()}")
    print(f"[train] Training on: {DEVICE}")

    # --------------------------------------------------------------------------
    # Unpack data from active dataset
    # --------------------------------------------------------------------------

    nowState              = active_dataset.AllData["XY"][:, ...]
    nextState             = active_dataset.AllData["XY2"][:, ...]
    sessionSegmentMatrix  = active_dataset.AllData["SessionSegmentMatrix"]
    SubjectAmount         = active_dataset.SubjectAmount
    srcColorList          = active_dataset.AllData["srcColorList"]
    srcColorList_rat      = active_dataset.srcColorList_subject
    EndNumberPlusOne      = active_dataset.EndNumberPlusOne
    EndSessionPlusOne     = active_dataset.EndSessionPlusOne

    nowState_img              = active_dataset.AllData_img["XY"][0:2, ...]
    nextState_img             = active_dataset.AllData_img["XY2"][0:2, ...]
    sessionSegmentMatrix_img  = active_dataset.AllData_img["SessionSegmentMatrix"]
    centerPose_img            = active_dataset.AllData_img["centerPose"][0:3, ...]
    srcColorList_img          = active_dataset.AllData_img["srcColorList"]

    # centerPose_position always comes from position dataset for evaluation
    centerPose_position = dataset_position.AllData["centerPose"][0:3, ...]

    # --------------------------------------------------------------------------
    # Prepare model and session segment matrix
    # --------------------------------------------------------------------------

    sessionSegmentMatrix = to_tensor(sessionSegmentMatrix).to(DEVICE)
    pn = pn.float().to(DEVICE)

    # --------------------------------------------------------------------------
    # Pre-allocate visualization figures
    # --------------------------------------------------------------------------

    fig0     = plt.figure(figsize=(8, 8))
    fig      = [plt.figure(figsize=(8, 8)) for _ in range(SubjectAmount)]
    figRatF1 = [plt.figure(figsize=(8, 8)) for _ in range(SubjectAmount)]
    figRatF2 = [plt.figure(figsize=(8, 8)) for _ in range(SubjectAmount)]

    viz_kwargs = dict(
        fig0=fig0,
        fig=fig,
        figRatF1=figRatF1,
        figRatF2=figRatF2,
    )

    # --------------------------------------------------------------------------
    # Initial forward pass (log start cost)
    # --------------------------------------------------------------------------

    with torch.no_grad():
        nowState_tensor  = to_tensor(nowState)
        nextState_tensor = to_tensor(nextState)

        noisyInput       = add_noise(nowState_tensor).to(DEVICE)
        nowState_tensor  = nowState_tensor.to(DEVICE)
        nextState_tensor = nextState_tensor.to(DEVICE)

        prePhase3, phaseXY3, phaseRad3 = pn(noisyInput)
        prePhaseMargin                  = prePhase3
        prePhase,  phaseXY,  phaseRad  = pn(nowState_tensor)
        prePhase2, phaseXY2, phaseRad2 = pn(nextState_tensor)

        speed_penalty        = loss_fn_dict["SpeedPenalty"]["weight"]        * loss_fn_dict["SpeedPenalty"]["loss"](phaseXY, phaseXY2)
        singularity_penalty  = loss_fn_dict["SingularityPenalty"]["weight"]  * loss_fn_dict["SingularityPenalty"]["loss"](centerSigma, prePhase)
        distribution_penalty = loss_fn_dict["DistributionPenalty"]["weight"] * loss_fn_dict["DistributionPenalty"]["loss"](phaseXY, sessionSegmentMatrix)
        margin_penalty       = loss_fn_dict["MarginPenalty"]["weight"]        * loss_fn_dict["MarginPenalty"]["loss"](centerSigma, prePhaseMargin)
        loss                 = speed_penalty + singularity_penalty + distribution_penalty + margin_penalty

    print("start cost", f"[{loss.item():.6f}, {speed_penalty.item():.6f}, {distribution_penalty.item():.6f}, {singularity_penalty.item():.6f}, {margin_penalty.item():.6f}]")

    # --------------------------------------------------------------------------
    # Loss logging
    # --------------------------------------------------------------------------

    speed_penalty_log        = []
    singularity_penalty_log  = []
    distribution_penalty_log = []
    margin_penalty_log       = []
    total_loss_log           = []

    # --------------------------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------------------------

    for epoch in range(EPOCH + 1):

        step = epoch

        optimizer.zero_grad()

        # Convert to tensor
        nowState_tensor  = to_tensor(nowState)
        nextState_tensor = to_tensor(nextState)

        # Add noise
        noisyInput       = add_noise(nowState_tensor).to(DEVICE)
        nowState_tensor  = nowState_tensor.to(DEVICE)
        nextState_tensor = nextState_tensor.to(DEVICE)

        # Forward passes
        prePhase3, phaseXY3, phaseRad3 = pn(noisyInput)
        prePhaseMargin                  = prePhase3
        prePhase,  phaseXY,  phaseRad  = pn(nowState_tensor)
        prePhase2, phaseXY2, phaseRad2 = pn(nextState_tensor)

        # Losses
        speed_penalty        = loss_fn_dict["SpeedPenalty"]["weight"]        * loss_fn_dict["SpeedPenalty"]["loss"](phaseXY, phaseXY2)
        singularity_penalty  = loss_fn_dict["SingularityPenalty"]["weight"]  * loss_fn_dict["SingularityPenalty"]["loss"](centerSigma, prePhase)
        distribution_penalty = loss_fn_dict["DistributionPenalty"]["weight"] * loss_fn_dict["DistributionPenalty"]["loss"](phaseXY, sessionSegmentMatrix)
        margin_penalty       = loss_fn_dict["MarginPenalty"]["weight"]        * loss_fn_dict["MarginPenalty"]["loss"](centerSigma, prePhaseMargin)
        loss                 = speed_penalty + singularity_penalty + distribution_penalty + margin_penalty

        # Log losses
        speed_penalty_log.append(speed_penalty.item())
        singularity_penalty_log.append(singularity_penalty.item())
        distribution_penalty_log.append(distribution_penalty.item())
        margin_penalty_log.append(margin_penalty.item())
        total_loss_log.append(loss.item())

        # Backpropagation
        loss.backward()
        optimizer.step()

        # -- Console log every 50 epochs
        if epoch % 50 == 0:
            print(f"{epoch} [{loss.item():.6f}, {speed_penalty.item():.6f}, {distribution_penalty.item():.6f}, {singularity_penalty.item():.6f}, {margin_penalty.item():.6f}]")

        # -- Checkpoint every 500 epochs
        if epoch % 500 == 0:
            torch.save(pn.state_dict(), sessionFile)

        # -- Epoch visualization checkpoint
        viz_epoch_condition = (
            (epoch < 20000 and epoch % 1000 == 0) or
            (epoch < 2000  and epoch % 250  == 0) or
            (epoch < 250   and epoch % 50   == 0) or
            (epoch % 2000  == 0)
        )

        if viz_epoch_condition and viz_epoch_fn is not None:
            viz_epoch_fn(
                step=step,
                pn=pn,
                nowState_tensor=nowState_tensor,
                add_noise=add_noise,
                centerPose_img=centerPose_img,
                centerPose_position=centerPose_position,
                EndNumberPlusOne=EndNumberPlusOne,
                SubjectAmount=SubjectAmount,
                srcColorList=srcColorList,
                srcColorList_rat=srcColorList_rat,
                imageFolder=imageFolder,
                imageFolderRat=imageFolderRat,
                DEVICE=DEVICE,
                **viz_kwargs
            )

        # -- Loss visualization every 2000 epochs
        if epoch % 2000 == 0:
            print("CV:" + str(crossValidateNo))
            print("Time(min):", (time.time() - startTime) / 60)

            if viz_loss_fn is not None:
                viz_loss_fn(
                    step=step,
                    speed_penalty_log=speed_penalty_log,
                    singularity_penalty_log=singularity_penalty_log,
                    distribution_penalty_log=distribution_penalty_log,
                    margin_penalty_log=margin_penalty_log,
                    total_loss_log=total_loss_log,
                    imageFolder=imageFolder,
                )

    # --------------------------------------------------------------------------
    # Return logs and trained objects
    # --------------------------------------------------------------------------

    return {
        'speed_penalty_log':        speed_penalty_log,
        'singularity_penalty_log':  singularity_penalty_log,
        'distribution_penalty_log': distribution_penalty_log,
        'margin_penalty_log':       margin_penalty_log,
        'total_loss_log':           total_loss_log,
        'model':                    pn,
        'optimizer':                optimizer,
    }


# ==============================================================================
# ENTRYPOINT — All config lives here, nothing hardcoded inside train()
# ==============================================================================
def generate_gait_folder(root: str) -> dict:
    """Generate timestamped folder structure for gait analysis."""
    
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Root directory '{root}' does not exist.")
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    main_folder = os.path.join(root, timestamp)
    img_folder = os.path.join(main_folder, "img")
    subject_folder = os.path.join(img_folder, "subject")
    
    os.makedirs(subject_folder, exist_ok=True)
    
    return {
        'root': main_folder,
        'img': img_folder,
        'subject': subject_folder,
        'timestamp': timestamp
    }
if __name__ == "__main__":

    # ---- Paths ---------------------------------------------------------------
    ROOT        = "/mnt/ExpDrive/SparkLabLongRun/PeamProject/results"
    DATA_ROOT   = "/mnt/ExpDrive/SparkLabLongRun/Data/peam_dataset/28-04-69"
    DATA_DIR_POS = os.path.join(DATA_ROOT, "position/RightFoot/Foot_to_Pelvis")
    DATA_DIR_VEL = os.path.join(DATA_ROOT, "velocity/RightFoot/Foot_to_Pelvis")

    # ---- Device --------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(DEVICE)

    # ---- Output folders ------------------------------------------------------
    experiment_folders = generate_gait_folder(ROOT)
    imageFolder    = experiment_folders["img"]
    imageFolderRat = experiment_folders["subject"]

    # ---- Datasets (preprocessing lives HERE, not inside train()) -------------
    dataset_velocity = pipeline.GaitPhasingDataset(
        data_root=DATA_ROOT,
        data_dir=DATA_DIR_VEL,
        img_dir=DATA_DIR_VEL,
        windowWing=3,
        regressWing=3,
        predictionGap=1,
        rotationMatrixList=None,
        meanForPhaseExtraction=None,
        sdForPhaseExtraction=None,
    )

    dataset_position = pipeline.GaitPhasingDataset(
        data_root=DATA_ROOT,
        data_dir=DATA_DIR_POS,
        img_dir=DATA_DIR_POS,
        windowWing=3,
        regressWing=3,
        predictionGap=1,
        rotationMatrixList=None,
        meanForPhaseExtraction=None,
        sdForPhaseExtraction=None,
    )

    # ---- Transforms ----------------------------------------------------------
    to_tensor = model.ToTensor(dtype=torch.float, device=DEVICE)
    add_noise = model.AddNoise(mean=0, stddev=0.5)

    # ---- Model ---------------------------------------------------------------
    pn = model.PhaseNetwork(
        h=25,
        depth=4,
        D=2,
        windowWing=3,
        activation="tanh",
        device=DEVICE
    )
    optimizer = optim.Adam(pn.parameters())

    # ---- Loss Config ---------------------------------------------------------
    Ap = -2.3  * np.pi / 180
    Bp =  4    * np.pi / 180
    Cp = (-180 + 45) * np.pi / 105

    loss_fn_dict = {
        "SpeedPenalty": {
            "loss":   model.SpeedLoss(Ap=Ap, Bp=Bp, Cp=Cp),
            "weight": 1.0
        },
        "DistributionPenalty": {
            "loss":   model.DistributionLoss(kind="bad"),
            "weight": 0.45
        },
        "SingularityPenalty": {
            "loss":   model.SingularityLoss(),
            "weight": 0.55
        },
        "MarginPenalty": {
            "loss":   model.MarginalLoss(),
            "weight": 0.55
        }
    }

    # ---- Session File --------------------------------------------------------
    sessionFile = os.path.join(
        ROOT,
        f"{experiment_folders['timestamp']}_{dataset_velocity.trainName}phaseModel.ckpt"
    )

    # ---- Run Training --------------------------------------------------------
    # Switch between "position" and "velocity" here
    INPUT_MODE = "velocity"   # <-- change to "position" to use position dataset

    results = train(
        input_mode=INPUT_MODE,
        dataset_position=dataset_position,
        dataset_velocity=dataset_velocity,
        pn=pn,
        optimizer=optimizer,
        loss_fn_dict=loss_fn_dict,
        EPOCH=10_000,
        centerSigma=1,
        DEVICE=DEVICE,
        to_tensor=to_tensor,
        add_noise=add_noise,
        imageFolder=imageFolder,
        imageFolderRat=imageFolderRat,
        sessionFile=sessionFile,
        viz_epoch_fn=default_viz_epoch,   # swap with custom function or None to disable
        viz_loss_fn=default_viz_loss,     # swap with custom function or None to disable
        crossValidateNo=-1,
        startTime=time.time(),
    )

    print("\nTraining complete.")
    print(f"Final total loss: {results['total_loss_log'][-1]:.6f}")
