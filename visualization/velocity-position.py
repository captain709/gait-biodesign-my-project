import numpy as np
import pandas as pd
import os
import os.path as op
from glob import glob
import pickle
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import statistics
import math


import scipy.signal as signal
from scipy import stats


root_dir = "C:\\Users\\noppa\OneDrive\\เดสก์ท็อป\\SparkLab\\Code\\SPARK-Lab-IMU\\data\\Training_data\\peam_test"
left_acc_path = "C:\\Users\\noppa\OneDrive\\เดสก์ท็อป\\SparkLab\\Code\\SPARK-Lab-IMU\\data\\pkl_peam_test\\acceleration_graph\\LeftFoot"
left_angle_path = "C:\\Users\\noppa\OneDrive\\เดสก์ท็อป\\SparkLab\\Code\\SPARK-Lab-IMU\\data\\pkl_peam_test\\joint_angle_graph\\LeftFoot"
left_velocity_foot_path = "C:\\Users\\noppa\OneDrive\\เดสก์ท็อป\\SparkLab\\Code\\SPARK-Lab-IMU\\data\\pkl_peam_test\\velocity_graph\\LeftFoot"
left_velocity_upper_leg_path = "C:\\Users\\noppa\OneDrive\\เดสก์ท็อป\\SparkLab\\Code\\SPARK-Lab-IMU\\data\\pkl_peam_test\\velocity_graph\\LeftUpperLeg"
left_velocity_lower_leg_path = "C:\\Users\\noppa\OneDrive\\เดสก์ท็อป\\SparkLab\\Code\\SPARK-Lab-IMU\\data\\pkl_peam_test\\velocity_graph\\LeftLowerLeg"
left_velocity_toe_path = "C:\\Users\\noppa\OneDrive\\เดสก์ท็อป\\SparkLab\\Code\\SPARK-Lab-IMU\\data\\pkl_peam_test\\velocity_graph\\LeftToe"
left_velocity_shoulder_path = "C:\\Users\\noppa\OneDrive\\เดสก์ท็อป\\SparkLab\\Code\\SPARK-Lab-IMU\\data\\pkl_peam_test\\velocity_graph\\LeftShoulder"
left_velocity_forearm_path = "C:\\Users\\noppa\OneDrive\\เดสก์ท็อป\\SparkLab\\Code\\SPARK-Lab-IMU\\data\\pkl_peam_test\\velocity_graph\\LeftForearm"
left_velocity_upper_arm_path = "C:\\Users\\noppa\OneDrive\\เดสก์ท็อป\\SparkLab\\Code\\SPARK-Lab-IMU\\data\\pkl_peam_test\\velocity_graph\\LeftUpperArm"
left_velocity_hand_path = "C:\\Users\\noppa\OneDrive\\เดสก์ท็อป\\SparkLab\\Code\\SPARK-Lab-IMU\\data\\pkl_peam_test\\velocity_graph\\LeftHand"
left_position_foot_path = "C:\\Users\\noppa\OneDrive\\เดสก์ท็อป\\SparkLab\\Code\\SPARK-Lab-IMU\\data\\pkl_peam_test\\position_graph\\LeftFoot"
left_position_hand_path = "C:\\Users\\noppa\OneDrive\\เดสก์ท็อป\\SparkLab\\Code\\SPARK-Lab-IMU\\data\\pkl_peam_test\\position_graph\\LeftHand"




for i in range(len(all_files[162:])):
    data = pd.read_excel(all_files[i], sheet_name="Segment Position")["Left Foot x"].to_numpy()
    x = np.arange(len(data))
    # Different threshold methods
    # methods = ["mean_std", "mad", "percentile", "otsu"]

    # for method in methods:
    peaks, properties = detect_peaks(data, threshold_method="otsu", percentile=90)
        
        # Plot results
    fig = plt.figure()
    # plt.scatter(x, data, label='Data', s=0.5, alpha=0.5, c="grey")
    plt.plot(x, data, c='grey')
    plt.plot(x[peaks], data[peaks], "x", label=f'Peaks (otsu)', c="#FA69AB")
    plt.title(f'Peak Detection using otsu: {all_files[i].split("/")[-2]}-{Path(all_files[i]).stem.split("-")[-1]}', **H_FONT)
    plt.xticks(**C_FONT)
    plt.yticks(**C_FONT)
    plt.xlabel("Timesteps", **L_FONT)
    plt.ylabel("X")
    fig.savefig(f'img/peak_detection/{all_files[i].split("/")[-2]}_{Path(all_files[i]).stem.split("-")[-1]}.png')
    plt.clf()
    plt.close()


def choose_threshold(data, method="mean_std", percentile=90):
   
    if method == "mean_std":
        threshold = np.mean(data) + np.std(data)
    elif method == "mad":
        median = np.median(data)
        mad = np.median(np.abs(data - median))  # Median Absolute Deviation
        threshold = median + 3 * mad  # Adjust the multiplier as needed
    elif method == "percentile":
        threshold = np.percentile(data, percentile)
    elif method == "otsu":
        hist, bin_edges = np.histogram(data, bins=256)
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]
        mean1 = np.cumsum(hist * bin_mids) / weight1
        mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
        inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        index_of_max_variance = np.argmax(inter_class_variance)
        threshold = bin_mids[:-1][index_of_max_variance]
    else:
        raise ValueError("Invalid method. Choose from 'mean_std', 'mad', 'percentile', 'otsu'.")
    
    return threshold


def detect_peaks(data, threshold=None, distance=5, threshold_method="mean_std", percentile=90):

    if threshold is None:
        threshold = choose_threshold(data, method=threshold_method, percentile=percentile)
    
    # Detect peaks
    peaks, properties = signal.find_peaks(data, height=threshold, distance=distance)
    
    return peaks, properties


def interpolate_signal_window(signal, target_length):
    """
    Interpolates a signal window to a specified length.

    Parameters:
    signal (numpy.ndarray): The input signal window to be interpolated.
    target_length (int): The desired length of the interpolated signal.

    Returns:
    numpy.ndarray: The interpolated signal with the specified length.
    """
    original_length = len(signal)
    if original_length == target_length:
        return signal

    # Original indices
    original_indices = np.arange(original_length)

    # Target indices
    target_indices = np.linspace(0, original_length - 1, target_length)

    # Interpolate using numpy.interp
    interpolated_signal = np.interp(target_indices, original_indices, signal)

    return interpolated_signal
