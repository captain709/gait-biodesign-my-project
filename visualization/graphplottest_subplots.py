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

def hilbert_transform_with_phase_color(data_x):

    # fs = 1000  # Sampling frequency (Hz)
    # duration = 1.0  # Duration of the signal (seconds)
    # t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Generate a test signal (e.g., a sine wave with frequency modulation)
    # carrier_freq = 50  # Carrier frequency (Hz)
    # mod_freq = 5  # Modulation frequency (Hz)
    # mod_index = 0.5  # Modulation index

    # Modulated signal
    # signal = np.sin(2 * np.pi * carrier_freq * t + mod_index * np.sin(2 * np.pi * mod_freq * t))
    signal = data_x

    # Compute the analytic signal using the Hilbert transform
    analytic_signal = hilbert(signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))

    # Reduce phase to the range [0, 2π)
    instantaneous_phase_mod_2pi = np.mod(instantaneous_phase, 2 * np.pi)

    # Instantaneous frequency
    # instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs

    # colorized
    # bins = np.linspace(0, 2* np.pi, 8)
    # color_index = np.digitize(instantaneous_phase_mod_2pi, bins)
    # colorized = [UNI_PALETTE[color - 1] for color in color_index]
    colorized = instantaneous_phase_mod_2pi / np.pi

    return instantaneous_phase_mod_2pi, colorized

'''
folder_name = ['A001_M', 'A002_M', 'A003_F', 'A004_F','A005_F','A006_F','A007_F','A008_F','A010_M','A011_M']
for i in folder_name:
    file_location = glob(f'{root_dir}/{i}/*_treadmill-*.xlsx')

    # Getting Subplots for each subjects

    fig, axs = plt.subplots(5, 1, figsize=(50, 50))

    fig.suptitle(f"velocity_foot : {i}")


    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)

        excel_data_velocity_x = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Foot x','Pelvis x','Frame']]
        excel_data_velocity_x['LeftFootToPelvisx'] = excel_data_velocity_x['Left Foot x'] - excel_data_velocity_x['Pelvis x']
        excel_data_velocity_x = excel_data_velocity_x.iloc[:200]
        excel_data_velocity_x = excel_data_velocity_x.drop(['Left Foot x','Pelvis x'],axis=1)
        excel_data_velocity_x_numpy = excel_data_velocity_x['LeftFootToPelvisx'].to_numpy().T
        frame_velocity_x= excel_data_velocity_x['Frame'].to_numpy().T
         
        axs[j].plot(excel_data_velocity_x['Frame'], excel_data_velocity_x['LeftFootToPelvisx'], marker='o')
        axs[j].set_xlabel('Frame (X-axis)')
        axs[j].set_ylabel('LeftFootToPelvisx_vel (Y-axis)')
        axs[j].set_title(f'leftFootToPelvisx_vel Speed: {j}')
        # plt.savefig(op.join(left_acc_path, f"AccelerationLeftFoot_subplots_{i}.png"))
    
        # print(f"Plot saved at: {op.join(left_acc_path, f'AccelerationLeftFoot_subplots_{i}.png')}")

        # plt.clf()
        # excel_data_acceleration_z = pd.read_excel(file_name, sheet_name = 'Segment Acceleration')[['Left Foot z','Pelvis z','Frame']]
        # excel_data_acceleration_z['LeftFootToPelvisz'] = excel_data_acceleration_z['Left Foot z'] - excel_data_acceleration_z['Pelvis z']
        
        # excel_data_joint_angle = pd.read_excel(file_name, sheet_name = 'Joint Angles ZXY')[['Left Ball Foot Flexion/Extension','Frame']]
        
        # excel_data_velocity_x = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Foot x','Pelvis x','Frame']]
        # excel_data_velocity_x['LeftFootToPelvisx'] = excel_data_velocity_x['Left Foot x'] - excel_data_velocity_x['Pelvis x']
      
        
        # excel_data_velocity_z = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Foot z','Pelvis z','Frame']]
        # excel_data_velocity_z['LeftFootToPelvisx'] = excel_data_velocity_z['Left Foot z'] - excel_data_velocity_z['Pelvis z']

    fig.savefig(op.join(left_velocity_foot_path, f"VelocityLeftFoot_x_subplots_{i}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_foot_path, f'VelocityLeftFoot_x_subplots_{i}.png')}")

    plt.clf()
    fig, axs = plt.subplots(5, 1, figsize=(50, 50))

    fig.suptitle(f"velocity_foot : {i}")

    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)

        excel_data_velocity_y = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Foot y','Pelvis y','Frame']]
        excel_data_velocity_y['LeftFootToPelvisy'] = excel_data_velocity_y['Left Foot y'] - excel_data_velocity_y['Pelvis y']
        excel_data_velocity_y = excel_data_velocity_y.iloc[:200]
        excel_data_velocity_y = excel_data_velocity_y.drop(['Left Foot y','Pelvis y'],axis=1)
        excel_data_velocity_y_numpy = excel_data_velocity_y['LeftFootToPelvisy'].to_numpy().T
        frame_velocity_y= excel_data_velocity_y['Frame'].to_numpy().T
         
        axs[j].plot(excel_data_velocity_y['Frame'], excel_data_velocity_y['LeftFootToPelvisy'], marker='o')
        axs[j].set_xlabel('Frame (X-axis)')
        axs[j].set_ylabel('LeftFootToPelvisy_vel (Y-axis)')
        axs[j].set_title(f'leftFootToPelvisy_vel Speed: {j}')
       

    fig.savefig(op.join(left_velocity_foot_path, f"VelocityLeftFoot_y_subplots_{i}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_foot_path, f'VelocityLeftFoot_y_subplots_{i}.png')}")    

    plt.clf()
    fig, axs = plt.subplots(5, 1, figsize=(50, 50))

    fig.suptitle(f"velocity_foot : {i}")
        
    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)

        excel_data_velocity_z = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Foot z','Pelvis z','Frame']]
        excel_data_velocity_z['LeftFootToPelvisz'] = excel_data_velocity_z['Left Foot z'] - excel_data_velocity_z['Pelvis z']
        excel_data_velocity_z = excel_data_velocity_z.iloc[:200]
        excel_data_velocity_z = excel_data_velocity_z.drop(['Left Foot z','Pelvis z'],axis=1)
        excel_data_velocity_z_numpy = excel_data_velocity_z['LeftFootToPelvisz'].to_numpy().T
        frame_velocity_z= excel_data_velocity_z['Frame'].to_numpy().T
         
        axs[j].plot(excel_data_velocity_z['Frame'], excel_data_velocity_z['LeftFootToPelvisz'], marker='o')
        axs[j].set_xlabel('Frame (X-axis)')
        axs[j].set_ylabel('LeftFootToPelvisz_vel (Y-axis)')
        axs[j].set_title(f'leftFootToPelvisz_vel Speed: {j}')
       
    fig.savefig(op.join(left_velocity_foot_path, f"VelocityLeftFoot_z_subplots_{i}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_foot_path, f'VelocityLeftFoot_z_subplots_{i}.png')}")  

     


folder_name = ['A001_M', 'A002_M', 'A003_F', 'A004_F','A005_F','A006_F','A007_F','A008_F','A010_M','A011_M']
for i in folder_name:
    file_location = glob(f'{root_dir}/{i}/*_treadmill-*.xlsx')

    # Getting Subplots for each subjects

    fig, axs = plt.subplots(5, 1, figsize=(50, 50))

    fig.suptitle(f"velocity_uppperleg : {i}")


    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)

        excel_data_velocity_x = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Upper Leg x','Pelvis x','Frame']]
        excel_data_velocity_x['LeftUpperLegToPelvisx'] = excel_data_velocity_x['Left Upper Leg x'] - excel_data_velocity_x['Pelvis x']
        excel_data_velocity_x = excel_data_velocity_x.iloc[:200]
        excel_data_velocity_x = excel_data_velocity_x.drop(['Left Upper Leg x','Pelvis x'],axis=1)
        excel_data_velocity_x_numpy = excel_data_velocity_x['LeftUpperLegToPelvisx'].to_numpy().T
        frame_velocity_x= excel_data_velocity_x['Frame'].to_numpy().T

        axs[j].plot(excel_data_velocity_x['Frame'], excel_data_velocity_x['LeftUpperLegToPelvisx'], marker='o')
        axs[j].set_xlabel('Frame (X-axis)')
        axs[j].set_ylabel('LeftUpperLegToPelvisx_vel (Y-axis)')
        axs[j].set_title(f'leftUpperLegToPelvisx_vel Speed: {j}')
       

    fig.savefig(op.join(left_velocity_upper_leg_path, f"VelocityLeftUpperLeg_x_subplots_{i}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_upper_leg_path, f'VelocityLeftUpperLeg_x_subplots_{i}.png')}")

    plt.clf()
    fig, axs = plt.subplots(5, 1, figsize=(50, 50))

    fig.suptitle(f"velocity_upper_leg : {i}")

    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)

        excel_data_velocity_y = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Upper Leg y','Pelvis y','Frame']]
        excel_data_velocity_y['LeftUpperLegToPelvisy'] = excel_data_velocity_y['Left Upper Leg y'] - excel_data_velocity_y['Pelvis y']
        excel_data_velocity_y = excel_data_velocity_y.iloc[:200]
        excel_data_velocity_y = excel_data_velocity_y.drop(['Left Upper Leg y','Pelvis y'],axis=1)
        excel_data_velocity_y_numpy = excel_data_velocity_y['LeftUpperLegToPelvisy'].to_numpy().T
        frame_velocity_y= excel_data_velocity_y['Frame'].to_numpy().T

        axs[j].plot(excel_data_velocity_y['Frame'], excel_data_velocity_y['LeftUpperLegToPelvisy'], marker='o')
        axs[j].set_xlabel('Frame (X-axis)')
        axs[j].set_ylabel('LeftUpperLegToPelvisy_vel (Y-axis)')
        axs[j].set_title(f'leftUpperLegToPelvisy_vel Speed: {j}')
       

    fig.savefig(op.join(left_velocity_upper_leg_path, f"VelocityLeftUpperLeg_y_subplots_{i}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_upper_leg_path, f'VelocityLeftUpperLeg_y_subplots_{i}.png')}")
    

    plt.clf()
    fig, axs = plt.subplots(5, 1, figsize=(50, 50))

    fig.suptitle(f"velocity_upper_leg : {i}")
        
    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)

        excel_data_velocity_z = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Upper Leg z','Pelvis z','Frame']]
        excel_data_velocity_z['LeftUpperLegToPelvisz'] = excel_data_velocity_z['Left Upper Leg z'] - excel_data_velocity_z['Pelvis z']
        excel_data_velocity_z = excel_data_velocity_z.iloc[:200]
        excel_data_velocity_z = excel_data_velocity_z.drop(['Left Upper Leg z','Pelvis z'],axis=1)
        excel_data_velocity_z_numpy = excel_data_velocity_z['LeftUpperLegToPelvisz'].to_numpy().T
        frame_velocity_z= excel_data_velocity_z['Frame'].to_numpy().T

        axs[j].plot(excel_data_velocity_z['Frame'], excel_data_velocity_z['LeftUpperLegToPelvisz'], marker='o')
        axs[j].set_xlabel('Frame (X-axis)')
        axs[j].set_ylabel('LeftUpperLegToPelvisz_vel (Y-axis)')
        axs[j].set_title(f'leftUpperLegToPelvisz_vel Speed: {j}')
       

    fig.savefig(op.join(left_velocity_upper_leg_path, f"VelocityLeftUpperLeg_z_subplots_{i}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_upper_leg_path, f'VelocityLeftUpperLeg_z_subplots_{i}.png')}")



    
folder_name = ['A001_M', 'A002_M', 'A003_F', 'A004_F','A005_F','A006_F','A007_F','A008_F','A010_M','A011_M']
for i in folder_name:
    file_location = glob(f'{root_dir}/{i}/*_treadmill-*.xlsx')

    # Getting Subplots for each subjects

    fig, axs = plt.subplots(5, 1, figsize=(50, 50))

    fig.suptitle(f"velocity_lower_leg : {i}")


    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)

        excel_data_velocity_x = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Lower Leg x','Pelvis x','Frame']]
        excel_data_velocity_x['LeftLowerLegToPelvisx'] = excel_data_velocity_x['Left Lower Leg x'] - excel_data_velocity_x['Pelvis x']
        excel_data_velocity_x = excel_data_velocity_x.iloc[:200]
        excel_data_velocity_x = excel_data_velocity_x.drop(['Left Lower Leg x','Pelvis x'],axis=1)
        excel_data_velocity_x_numpy = excel_data_velocity_x['LeftLowerLegToPelvisx'].to_numpy().T
        frame_velocity_x= excel_data_velocity_x['Frame'].to_numpy().T

        axs[j].plot(excel_data_velocity_x['Frame'], excel_data_velocity_x['LeftLowerLegToPelvisx'], marker='o')
        axs[j].set_xlabel('Frame (X-axis)')
        axs[j].set_ylabel('LeftLowerLegToPelvisx_vel (Y-axis)')
        axs[j].set_title(f'leftLowerLegToPelvisx_vel Speed: {j}')
       

    fig.savefig(op.join(left_velocity_lower_leg_path, f"VelocityLeftLowerLeg_x_subplots_{i}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_lower_leg_path, f'VelocityLeftLowerLeg_x_subplots_{i}.png')}")

    plt.clf()
    fig, axs = plt.subplots(5, 1, figsize=(50, 50))

    fig.suptitle(f"velocity_lower_leg : {i}")
    

    for j, file_name in enumerate(file_location):

        print(str(j +1) + "." + i + ":" + file_name)

        excel_data_velocity_y = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Lower Leg y','Pelvis y','Frame']]
        excel_data_velocity_y['LeftLowerLegToPelvisy'] = excel_data_velocity_y['Left Lower Leg y'] - excel_data_velocity_y['Pelvis y']
        excel_data_velocity_y = excel_data_velocity_y.iloc[:200]
        excel_data_velocity_y = excel_data_velocity_y.drop(['Left Lower Leg y','Pelvis y'],axis=1)
        excel_data_velocity_y_numpy = excel_data_velocity_y['LeftLowerLegToPelvisy'].to_numpy().T
        frame_velocity_y= excel_data_velocity_y['Frame'].to_numpy().T

        axs[j].plot(excel_data_velocity_y['Frame'], excel_data_velocity_y['LeftLowerLegToPelvisy'], marker='o')
        axs[j].set_xlabel('Frame (X-axis)')
        axs[j].set_ylabel('LeftLowerLegToPelvisy_vel (Y-axis)')
        axs[j].set_title(f'leftLowerLegToPelvisy_vel Speed: {j}')
       

    fig.savefig(op.join(left_velocity_lower_leg_path, f"VelocityLeftLowerLeg_y_subplots_{i}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_lower_leg_path, f'VelocityLeftLowerLeg_y_subplots_{i}.png')}")
    

    plt.clf()
    fig, axs = plt.subplots(5, 1, figsize=(50, 50))

    fig.suptitle(f"velocity_lower_leg : {i}")
        
    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)

        excel_data_velocity_z = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Lower Leg z','Pelvis z','Frame']]
        excel_data_velocity_z['LeftLowerLegToPelvisz'] = excel_data_velocity_z['Left Lower Leg z'] - excel_data_velocity_z['Pelvis z']
        excel_data_velocity_z = excel_data_velocity_z.iloc[:200]
        excel_data_velocity_z = excel_data_velocity_z.drop(['Left Lower Leg z','Pelvis z'],axis=1)
        excel_data_velocity_z_numpy = excel_data_velocity_z['LeftLowerLegToPelvisz'].to_numpy().T
        frame_velocity_z= excel_data_velocity_z['Frame'].to_numpy().T

        axs[j].plot(excel_data_velocity_z['Frame'], excel_data_velocity_z['LeftLowerLegToPelvisz'], marker='o')
        axs[j].set_xlabel('Frame (X-axis)')
        axs[j].set_ylabel('LeftLowerLegToPelvisz_vel (Y-axis)')
        axs[j].set_title(f'leftLowerLegToPelvisz_vel Speed: {j}')
       

    fig.savefig(op.join(left_velocity_lower_leg_path, f"VelocityLeftLowerLeg_z_subplots_{i}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_lower_leg_path, f'VelocityLeftLowerLeg_z_subplots_{i}.png')}")



    folder_name = ['A001_M', 'A002_M', 'A003_F', 'A004_F','A005_F','A006_F','A007_F','A008_F','A010_M','A011_M']
for i in folder_name:
    file_location = glob(f'{root_dir}/{i}/*_treadmill-*.xlsx')

    # Getting Subplots for each subjects

    fig, axs = plt.subplots(5, 1, figsize=(50, 50))

    fig.suptitle(f"velocity_toe : {i}")


    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)

        excel_data_velocity_x = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Toe x','Pelvis x','Frame']]
        excel_data_velocity_x['LeftToeToPelvisx'] = excel_data_velocity_x['Left Toe x'] - excel_data_velocity_x['Pelvis x']
        excel_data_velocity_x = excel_data_velocity_x.iloc[:200]
        excel_data_velocity_x = excel_data_velocity_x.drop(['Left Toe x','Pelvis x'],axis=1)
        excel_data_velocity_x_numpy = excel_data_velocity_x['LeftToeToPelvisx'].to_numpy().T
        frame_velocity_x= excel_data_velocity_x['Frame'].to_numpy().T

        axs[j].plot(excel_data_velocity_x['Frame'], excel_data_velocity_x['LeftToeToPelvisx'], marker='o')
        axs[j].set_xlabel('Frame (X-axis)')
        axs[j].set_ylabel('LeftToeToPelvisx_vel (Y-axis)')
        axs[j].set_title(f'leftToeToPelvisx_vel Speed: {j}')
       

    fig.savefig(op.join(left_velocity_toe_path, f"VelocityLeftToe_x_subplots_{i}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_toe_path, f'VelocityLeftToe_x_subplots_{i}.png')}")

    plt.clf()
    fig, axs = plt.subplots(5, 1, figsize=(50, 50))

    fig.suptitle(f"velocity_toe : {i}")

    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)

        excel_data_velocity_y = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Toe y','Pelvis y','Frame']]
        excel_data_velocity_y['LeftToeToPelvisy'] = excel_data_velocity_y['Left Toe y'] - excel_data_velocity_y['Pelvis y']
        excel_data_velocity_y = excel_data_velocity_y.iloc[:200]
        excel_data_velocity_y = excel_data_velocity_y.drop(['Left Toe y','Pelvis y'],axis=1)
        excel_data_velocity_y_numpy = excel_data_velocity_y['LeftToeToPelvisy'].to_numpy().T
        frame_velocity_y= excel_data_velocity_y['Frame'].to_numpy().T

        axs[j].plot(excel_data_velocity_y['Frame'], excel_data_velocity_y['LeftToeToPelvisy'], marker='o')
        axs[j].set_xlabel('Frame (X-axis)')
        axs[j].set_ylabel('LeftToeToPelvisy_vel (Y-axis)')
        axs[j].set_title(f'leftToeToPelvisy_vel Speed: {j}')
       

    fig.savefig(op.join(left_velocity_toe_path, f"VelocityLeftToe_y_subplots_{i}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_toe_path, f'VelocityLeftToe_y_subplots_{i}.png')}")
    

    plt.clf()
    fig, axs = plt.subplots(5, 1, figsize=(50, 50))

    fig.suptitle(f"velocity_toe : {i}")
        
    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)

        excel_data_velocity_z = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Toe z','Pelvis z','Frame']]
        excel_data_velocity_z['LeftToeToPelvisz'] = excel_data_velocity_z['Left Toe z'] - excel_data_velocity_z['Pelvis z']
        excel_data_velocity_z = excel_data_velocity_z.iloc[:200]
        excel_data_velocity_z = excel_data_velocity_z.drop(['Left Toe z','Pelvis z'],axis=1)
        excel_data_velocity_z_numpy = excel_data_velocity_z['LeftToeToPelvisz'].to_numpy().T
        frame_velocity_z= excel_data_velocity_z['Frame'].to_numpy().T

        axs[j].plot(excel_data_velocity_z['Frame'], excel_data_velocity_z['LeftToeToPelvisz'], marker='o')
        axs[j].set_xlabel('Frame (X-axis)')
        axs[j].set_ylabel('LeftToeToPelvisz_vel (Y-axis)')
        axs[j].set_title(f'leftToeToPelvisz_vel Speed: {j}')
       

    fig.savefig(op.join(left_velocity_toe_path, f"VelocityLeftToe_z_subplots_{i}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_toe_path, f'VelocityLeftToe_z_subplots_{i}.png')}")


    folder_name = ['A001_M', 'A002_M', 'A003_F', 'A004_F','A005_F','A006_F','A007_F','A008_F','A010_M','A011_M']
for i in folder_name:
    file_location = glob(f'{root_dir}/{i}/*_treadmill-*.xlsx')

    # Getting Subplots for each subjects

    fig, axs = plt.subplots(5, 1, figsize=(50, 50))

    fig.suptitle(f"velocity_shoulder : {i}")


    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)

        excel_data_velocity_x = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Shoulder x','T8 x','Frame']]
        excel_data_velocity_x['LeftShoulderToSternumx'] = excel_data_velocity_x['Left Shoulder x'] - excel_data_velocity_x['T8 x']
        excel_data_velocity_x = excel_data_velocity_x.iloc[:200]
        excel_data_velocity_x = excel_data_velocity_x.drop(['Left Shoulder x','T8 x'],axis=1)
        excel_data_velocity_x_numpy = excel_data_velocity_x['LeftShoulderToSternumx'].to_numpy().T
        frame_velocity_x= excel_data_velocity_x['Frame'].to_numpy().T

        axs[j].plot(excel_data_velocity_x['Frame'], excel_data_velocity_x['LeftShoulderToSternumx'], marker='o')
        axs[j].set_xlabel('Frame (X-axis)')
        axs[j].set_ylabel('LeftShoulderToSternumx_vel (Y-axis)')
        axs[j].set_title(f'leftShoulderToSternumx_vel Speed: {j}')
       

    fig.savefig(op.join(left_velocity_shoulder_path, f"VelocityLeftShoulder_x_subplots_{i}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_shoulder_path, f'VelocityLeftShoulder_x_subplots_{i}.png')}")

    plt.clf()
    fig, axs = plt.subplots(5, 1, figsize=(50, 50))

    fig.suptitle(f"velocity_shoulder : {i}")

    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)

        excel_data_velocity_y = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Shoulder y','T8 y','Frame']]
        excel_data_velocity_y['LeftShoulderToSternumy'] = excel_data_velocity_y['Left Shoulder y'] - excel_data_velocity_y['T8 y']
        excel_data_velocity_y = excel_data_velocity_y.iloc[:200]
        excel_data_velocity_y = excel_data_velocity_y.drop(['Left Shoulder y','T8 y'],axis=1)
        excel_data_velocity_y_numpy = excel_data_velocity_y['LeftShoulderToSternumy'].to_numpy().T
        frame_velocity_y= excel_data_velocity_y['Frame'].to_numpy().T

        axs[j].plot(excel_data_velocity_y['Frame'], excel_data_velocity_y['LeftShoulderToSternumy'], marker='o')
        axs[j].set_xlabel('Frame (X-axis)')
        axs[j].set_ylabel('LeftShoulderToSternumy_vel (Y-axis)')
        axs[j].set_title(f'leftShoulderToSternumy_vel Speed: {j}')
       
    fig.savefig(op.join(left_velocity_shoulder_path, f"VelocityLeftShoulder_y_subplots_{i}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_shoulder_path, f'VelocityLeftShoulder_y_subplots_{i}.png')}")

    plt.clf()
    fig, axs = plt.subplots(5, 1, figsize=(50, 50))

    fig.suptitle(f"velocity_shoulder : {i}")
        
    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)

        excel_data_velocity_z = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Shoulder z','T8 z','Frame']]
        excel_data_velocity_z['LeftShoulderToSternumz'] = excel_data_velocity_z['Left Shoulder z'] - excel_data_velocity_z['T8 z']
        excel_data_velocity_z = excel_data_velocity_z.iloc[:200]
        excel_data_velocity_z = excel_data_velocity_z.drop(['Left Shoulder z','T8 z'],axis=1)
        excel_data_velocity_z_numpy = excel_data_velocity_z['LeftShoulderToSternumz'].to_numpy().T
        frame_velocity_z= excel_data_velocity_z['Frame'].to_numpy().T

        axs[j].plot(excel_data_velocity_z['Frame'], excel_data_velocity_z['LeftShoulderToSternumz'], marker='o')
        axs[j].set_xlabel('Frame (X-axis)')
        axs[j].set_ylabel('LeftShoulderToSternumz_vel (Y-axis)')
        axs[j].set_title(f'leftShoulderToSternumz_vel Speed: {j}')
       
    fig.savefig(op.join(left_velocity_shoulder_path, f"VelocityLeftShoulder_z_subplots_{i}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_shoulder_path, f'VelocityLeftShoulder_z_subplots_{i}.png')}")


    folder_name = ['A001_M', 'A002_M', 'A003_F', 'A004_F','A005_F','A006_F','A007_F','A008_F','A010_M','A011_M']
for i in folder_name:
    file_location = glob(f'{root_dir}/{i}/*_treadmill-*.xlsx')

    # Getting Subplots for each subjects

    fig, axs = plt.subplots(5, 1, figsize=(50, 50))

    fig.suptitle(f"velocity_upper_arm : {i}")


    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)

        excel_data_velocity_x = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Upper Arm x','T8 x','Frame']]
        excel_data_velocity_x['LeftUpperArmToSternumx'] = excel_data_velocity_x['Left Upper Arm x'] - excel_data_velocity_x['T8 x']
        excel_data_velocity_x = excel_data_velocity_x.iloc[:200]
        excel_data_velocity_x = excel_data_velocity_x.drop(['Left Upper Arm x','T8 x'],axis=1)
        excel_data_velocity_x_numpy = excel_data_velocity_x['LeftUpperArmToSternumx'].to_numpy().T
        frame_velocity_x= excel_data_velocity_x['Frame'].to_numpy().T

        axs[j].plot(excel_data_velocity_x['Frame'], excel_data_velocity_x['LeftUpperArmToSternumx'], marker='o')
        axs[j].set_xlabel('Frame (X-axis)')
        axs[j].set_ylabel('LeftUpperArmToSternumx_vel (Y-axis)')
        axs[j].set_title(f'leftUpperArmToSternumx_vel Speed: {j}')
       

    fig.savefig(op.join(left_velocity_upper_arm_path, f"VelocityLeftUpperArm_x_subplots_{i}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_upper_arm_path, f'VelocityLeftUpperArm_x_subplots_{i}.png')}")

    plt.clf()
    fig, axs = plt.subplots(5, 1, figsize=(50, 50))

    fig.suptitle(f"velocity_upper_arm : {i}")

    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)

        excel_data_velocity_y = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Upper Arm y','T8 y','Frame']]
        excel_data_velocity_y['LeftUpperArmToSternumy'] = excel_data_velocity_y['Left Upper Arm y'] - excel_data_velocity_y['T8 y']
        excel_data_velocity_y = excel_data_velocity_y.iloc[:200]
        excel_data_velocity_y = excel_data_velocity_y.drop(['Left Upper Arm y','T8 y'],axis=1)
        excel_data_velocity_y_numpy = excel_data_velocity_y['LeftUpperArmToSternumy'].to_numpy().T
        frame_velocity_y= excel_data_velocity_y['Frame'].to_numpy().T

        axs[j].plot(excel_data_velocity_y['Frame'], excel_data_velocity_y['LeftUpperArmToSternumy'], marker='o')
        axs[j].set_xlabel('Frame (X-axis)')
        axs[j].set_ylabel('LeftUpperArmToSternumy_vel (Y-axis)')
        axs[j].set_title(f'leftUpperArmToSternumy_vel Speed: {j}')
       
    fig.savefig(op.join(left_velocity_upper_arm_path, f"VelocityLeftUpperArm_y_subplots_{i}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_upper_arm_path, f'VelocityLeftUpperArm_y_subplots_{i}.png')}")

    plt.clf()
    fig, axs = plt.subplots(5, 1, figsize=(50, 50))

    fig.suptitle(f"velocity_upper_arm : {i}")
        
    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)

        excel_data_velocity_z = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Upper Arm z','T8 z','Frame']]
        excel_data_velocity_z['LeftUpperArmToSternumz'] = excel_data_velocity_z['Left Upper Arm z'] - excel_data_velocity_z['T8 z']
        excel_data_velocity_z = excel_data_velocity_z.iloc[:200]
        excel_data_velocity_z = excel_data_velocity_z.drop(['Left Upper Arm z','T8 z'],axis=1)
        excel_data_velocity_z_numpy = excel_data_velocity_z['LeftUpperArmToSternumz'].to_numpy().T
        frame_velocity_z= excel_data_velocity_z['Frame'].to_numpy().T

        axs[j].plot(excel_data_velocity_z['Frame'], excel_data_velocity_z['LeftUpperArmToSternumz'], marker='o')
        axs[j].set_xlabel('Frame (X-axis)')
        axs[j].set_ylabel('LeftUpperArmToSternumz_vel (Y-axis)')
        axs[j].set_title(f'leftUpperArmToSternumz_vel Speed: {j}')
       
    fig.savefig(op.join(left_velocity_upper_arm_path, f"VelocityLeftUpperArm_z_subplots_{i}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_upper_arm_path, f'VelocityLeftUpperArm_z_subplots_{i}.png')}")

'''


folder_name = ['A001_M', 'A002_M', 'A003_F', 'A004_F','A005_F','A006_F','A007_F','A008_F','A010_M','A011_M']

for i in folder_name:
    file_location = glob(f'{root_dir}/{i}/*_treadmill-*.xlsx')

    # Getting Subplots for each subjects

    fig, axs = plt.subplots(5, 1, figsize=(50, 50))

    fig.suptitle(f"velocity_forearm : {i}")


    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)

        excel_data_velocity_x = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Forearm x','T8 x','Frame']]
        excel_data_velocity_x['LeftForearmToSternumx'] = excel_data_velocity_x['Left Forearm x'] - excel_data_velocity_x['T8 x']
        excel_data_velocity_x = excel_data_velocity_x.iloc[:200]
        excel_data_velocity_x = excel_data_velocity_x.drop(['Left Forearm x','T8 x'],axis=1)
        excel_data_velocity_x_numpy = excel_data_velocity_x['LeftForearmToSternumx'].to_numpy().T
        frame_velocity_x= excel_data_velocity_x['Frame'].to_numpy().T

        axs[j].plot(excel_data_velocity_x['Frame'], excel_data_velocity_x['LeftForearmToSternumx'], marker='o')
        axs[j].set_xlabel('Frame (X-axis)')
        axs[j].set_ylabel('LeftForearmToSternumx_vel (Y-axis)')
        axs[j].set_title(f'leftForearmToSternumx_vel Speed: {j}')
       

    fig.savefig(op.join(left_velocity_forearm_path, f"VelocityLeftForearm_x_subplots_{i}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_forearm_path, f'VelocityLeftForearm_x_subplots_{i}.png')}")

    plt.clf()
    fig, axs = plt.subplots(5, 1, figsize=(50, 50))

    fig.suptitle(f"velocity_forearm : {i}")

    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)

        excel_data_velocity_y = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Forearm y','T8 y','Frame']]
        excel_data_velocity_y['LeftForearmToSternumy'] = excel_data_velocity_y['Left Forearm y'] - excel_data_velocity_y['T8 y']
        excel_data_velocity_y = excel_data_velocity_y.iloc[:200]
        excel_data_velocity_y = excel_data_velocity_y.drop(['Left Forearm y','T8 y'],axis=1)
        excel_data_velocity_y_numpy = excel_data_velocity_y['LeftForearmToSternumy'].to_numpy().T
        frame_velocity_y= excel_data_velocity_y['Frame'].to_numpy().T

        axs[j].plot(excel_data_velocity_y['Frame'], excel_data_velocity_y['LeftForearmToSternumy'], marker='o')
        axs[j].set_xlabel('Frame (X-axis)')
        axs[j].set_ylabel('LeftForearmToSternumy_vel (Y-axis)')
        axs[j].set_title(f'leftForearmToSternumy_vel Speed: {j}')
       
    fig.savefig(op.join(left_velocity_forearm_path, f"VelocityLeftForearm_y_subplots_{i}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_forearm_path, f'VelocityLeftForearm_y_subplots_{i}.png')}")

    plt.clf()
    fig, axs = plt.subplots(5, 1, figsize=(50, 50))

    fig.suptitle(f"velocity_forearm : {i}")
        
    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)

        excel_data_velocity_z = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Forearm z','T8 z','Frame']]
        excel_data_velocity_z['LeftForearmToSternumz'] = excel_data_velocity_z['Left Forearm z'] - excel_data_velocity_z['T8 z']
        excel_data_velocity_z = excel_data_velocity_z.iloc[:200]
        excel_data_velocity_z = excel_data_velocity_z.drop(['Left Forearm z','T8 z'],axis=1)
        excel_data_velocity_z_numpy = excel_data_velocity_z['LeftForearmToSternumz'].to_numpy().T
        frame_velocity_z= excel_data_velocity_z['Frame'].to_numpy().T

        axs[j].plot(excel_data_velocity_z['Frame'], excel_data_velocity_z['LeftForearmToSternumz'], marker='o')
        axs[j].set_xlabel('Frame (X-axis)')
        axs[j].set_ylabel('LeftForearmToSternumz_vel (Y-axis)')
        axs[j].set_title(f'leftForearmToSternumz_vel Speed: {j}')
       
    fig.savefig(op.join(left_velocity_forearm_path, f"VelocityLeftForearm_z_subplots_{i}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_forearm_path, f'VelocityLeftForearm_z_subplots_{i}.png')}")



    folder_name = ['A001_M', 'A002_M', 'A003_F', 'A004_F','A005_F','A006_F','A007_F','A008_F','A010_M','A011_M']
for i in folder_name:
    file_location = glob(f'{root_dir}/{i}/*_treadmill-*.xlsx')

    # Getting Subplots for each subjects

    fig, axs = plt.subplots(5, 1, figsize=(50, 50))

    fig.suptitle(f"velocity_hand : {i}")


    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)

        excel_data_velocity_x = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Hand x','T8 x','Frame']]
        excel_data_velocity_x['LeftHandToSternumx'] = excel_data_velocity_x['Left Hand x'] - excel_data_velocity_x['T8 x']
        excel_data_velocity_x = excel_data_velocity_x.iloc[:200]
        excel_data_velocity_x = excel_data_velocity_x.drop(['Left Hand x','T8 x'],axis=1)
        excel_data_velocity_x_numpy = excel_data_velocity_x['LeftHandToSternumx'].to_numpy().T
        frame_velocity_x= excel_data_velocity_x['Frame'].to_numpy().T

        axs[j].plot(excel_data_velocity_x['Frame'], excel_data_velocity_x['LeftHandToSternumx'], marker='o')
        axs[j].set_xlabel('Frame (X-axis)')
        axs[j].set_ylabel('LeftHandToSternumx_vel (Y-axis)')
        axs[j].set_title(f'leftHandToSternumx_vel Speed: {j}')
       

    fig.savefig(op.join(left_velocity_hand_path, f"VelocityLeftHand_x_subplots_{i}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_hand_path, f'VelocityLeftHand_x_subplots_{i}.png')}")

    plt.clf()
    fig, axs = plt.subplots(5, 1, figsize=(50, 50))

    fig.suptitle(f"velocity_hand : {i}")

    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)

        excel_data_velocity_y = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Hand y','T8 y','Frame']]
        excel_data_velocity_y['LeftHandToSternumy'] = excel_data_velocity_y['Left Hand y'] - excel_data_velocity_y['T8 y']
        excel_data_velocity_y = excel_data_velocity_y.iloc[:200]
        excel_data_velocity_y = excel_data_velocity_y.drop(['Left Hand y','T8 y'],axis=1)
        excel_data_velocity_y_numpy = excel_data_velocity_y['LeftHandToSternumy'].to_numpy().T
        frame_velocity_y= excel_data_velocity_y['Frame'].to_numpy().T

        axs[j].plot(excel_data_velocity_y['Frame'], excel_data_velocity_y['LeftHandToSternumy'], marker='o')
        axs[j].set_xlabel('Frame (X-axis)')
        axs[j].set_ylabel('LeftHandToSternumy_vel (Y-axis)')
        axs[j].set_title(f'leftHandToSternumy_vel Speed: {j}')
       
    fig.savefig(op.join(left_velocity_hand_path, f"VelocityLeftHand_y_subplots_{i}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_hand_path, f'VelocityLeftHand_y_subplots_{i}.png')}")

    plt.clf()
    fig, axs = plt.subplots(5, 1, figsize=(50, 50))

    fig.suptitle(f"velocity_hand : {i}")
        
    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)

        excel_data_velocity_z = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Hand z','T8 z','Frame']]
        excel_data_velocity_z['LeftHandToSternumz'] = excel_data_velocity_z['Left Hand z'] - excel_data_velocity_z['T8 z']
        excel_data_velocity_z = excel_data_velocity_z.iloc[:200]
        excel_data_velocity_z = excel_data_velocity_z.drop(['Left Hand z','T8 z'],axis=1)
        excel_data_velocity_z_numpy = excel_data_velocity_z['LeftHandToSternumz'].to_numpy().T
        frame_velocity_z= excel_data_velocity_z['Frame'].to_numpy().T

        axs[j].plot(excel_data_velocity_z['Frame'], excel_data_velocity_z['LeftHandToSternumz'], marker='o')
        axs[j].set_xlabel('Frame (X-axis)')
        axs[j].set_ylabel('LeftHandToSternumz_vel (Y-axis)')
        axs[j].set_title(f'leftHandToSternumz_vel Speed: {j}')
       
    fig.savefig(op.join(left_velocity_hand_path, f"VelocityLeftHand_z_subplots_{i}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_hand_path, f'VelocityLeftHand_z_subplots_{i}.png')}")