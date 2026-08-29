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
import matplotlib

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

folder_name = ['A001_M', 'A002_M', 'A003_F', 'A004_F','A005_F','A006_F','A007_F','A008_F','A010_M','A011_M']
for i in folder_name:
    file_location = glob(f'{root_dir}/{i}/*_treadmill-*.xlsx')
    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)
        
        excel_data_acceleration_x = pd.read_excel(file_name, sheet_name = 'Segment Acceleration')[['Left Foot x','Pelvis x','Frame']]
        excel_data_acceleration_x['LeftFootToPelvisx'] = excel_data_acceleration_x['Left Foot x'] - excel_data_acceleration_x['Pelvis x']
        excel_data_acceleration_x = excel_data_acceleration_x.iloc[:200]
        excel_data_acceleration_x = excel_data_acceleration_x.drop(['Pelvis x'],axis=1)
        # excel_data_acceleration_x = excel_data_acceleration_x.to_numpy().T
        excel_data_acceleration_x_numpy = excel_data_acceleration_x['LeftFootToPelvisx'].to_numpy().T
        frame_acc_x = excel_data_acceleration_x['Frame'].to_numpy().T
        # x = np.arange(len(excel_data_acceleration_x_numpy))
        # fig, axs = plt.subplots(len(folder_name))
        # fig.suptitle('Vertically stacked subplots')
         

        plt.plot(excel_data_acceleration_x['Frame'], excel_data_acceleration_x['Left Foot x'], marker='x')
        plt.xlabel('Frame (X-axis)')
        plt.ylabel('LeftFootx_acc (Y-axis)')
        plt.title('leftFootx_acc')
        plt.savefig(op.join(left_acc_path, f"AccelerationLeftFoot_X_defult_{i}_{j}.png"))
    
        print(f"Plot saved at: {op.join(left_acc_path, f"AccelerationLeftFoot_X_defult_{i}_{j}.png")}")

        plt.clf()

        
        plt.plot(excel_data_acceleration_x['Frame'], excel_data_acceleration_x['LeftFootToPelvisx'], marker='o')
        plt.xlabel('Frame (X-axis)')
        plt.ylabel('LeftFootToPelvisx_acc (Y-axis)')
        plt.title('leftFootToPelvisx_acc')
        plt.savefig(op.join(left_acc_path, f"AccelerationLeftFoot_X_{i}_{j}.png"))
    
        print(f"Plot saved at: {op.join(left_acc_path, f"AccelerationLeftFoot_X_{i}_{j}.png")}")

        plt.clf()
'''
        
        phase, colorized = hilbert_transform_with_phase_color(excel_data_acceleration_x_numpy)

        plt.rcParams['image.cmap'] = 'gist_rainbow'
        # Plot results
        fig = plt.figure()
    
        plt.scatter(frame_acc_x, excel_data_acceleration_x_numpy, c=colorized, s=1, alpha=0.75)
        # plt.plot(x[peaks], data[peaks], "x", label=f'Peaks (otsu)', c="#FA69AB")
        plt.title(f'Loop Position with Hilbert Phase X')
        plt.xlabel('Frame (X-axis)')
        plt.ylabel('LeftFootToPelvisx_acc (Y-axis) Hilbert transform')
        cb=plt.colorbar(format=matplotlib.ticker.FormatStrFormatter('%g $\pi$'))
        plt.savefig(op.join(left_acc_path, f"AccelerationLeftFoot_Hilbert_transform_X_{i}_{j}.png"))
        
        print(f"Plot saved at: {op.join(left_acc_path, f"AccelerationLeftFoot_Hilbert_transform_X_{i}_{j}.png")}")

        plt.clf()

        excel_data_acceleration_z = pd.read_excel(file_name, sheet_name = 'Segment Acceleration')[['Left Foot z','Pelvis z','Frame']]
        excel_data_acceleration_z['LeftFootToPelvisz'] = excel_data_acceleration_z['Left Foot z'] - excel_data_acceleration_z['Pelvis z']
        excel_data_acceleration_z = excel_data_acceleration_z.iloc[:200]
        excel_data_acceleration_z = excel_data_acceleration_z.drop(['Left Foot z','Pelvis z'],axis=1)
        excel_data_acceleration_z_numpy = excel_data_acceleration_z['LeftFootToPelvisz'].to_numpy().T
        frame_acc_z = excel_data_acceleration_z['Frame'].to_numpy().T
        
        plt.plot(excel_data_acceleration_z['Frame'], excel_data_acceleration_z['LeftFootToPelvisz'], marker='o')
        plt.xlabel('Frame (X-axis)')
        plt.ylabel('LeftFootToPelvisz_acc (Y-axis)')
        plt.title('leftFootToPelvisz_acc')
        plt.savefig(op.join(left_acc_path, f"AccelerationLeftFoot_Z_{i}_{j}.png"))
    
        print(f"Plot saved at: {op.join(left_acc_path, f"AccelerationLeftFoot_Z_{i}_{j}.png")}")

        plt.clf()

        phase, colorized = hilbert_transform_with_phase_color(excel_data_acceleration_z_numpy)

        plt.rcParams['image.cmap'] = 'gist_rainbow'
        # Plot results
        fig = plt.figure()
    
        plt.scatter(frame_acc_z, excel_data_acceleration_z_numpy, c=colorized, s=1, alpha=0.75)
        # plt.plot(x[peaks], data[peaks], "x", label=f'Peaks (otsu)', c="#FA69AB")
        plt.title(f'Loop Position with Hilbert Phase Z')
        plt.xlabel('Frame (X-axis)')
        plt.ylabel('LeftFootToPelvisz_acc (Y-axis) Hilbert transform')
        cb=plt.colorbar(format=matplotlib.ticker.FormatStrFormatter('%g $\pi$'))
        plt.savefig(op.join(left_acc_path, f"AccelerationLeftFoot_Hilbert_transform_Z_{i}_{j}.png"))
        
        print(f"Plot saved at: {op.join(left_acc_path, f"AccelerationLeftFoot_Hilbert_transform_Z_{i}_{j}.png")}")

        plt.clf()

        excel_data_joint_angle = pd.read_excel(file_name, sheet_name = 'Joint Angles ZXY')[['Left Ball Foot Flexion/Extension','Frame']]
        excel_data_joint_angle = excel_data_joint_angle.iloc[:200]
        excel_data_joint_angle_numpy = excel_data_joint_angle['Left Ball Foot Flexion/Extension'].to_numpy().T
        frame_joint_angle = excel_data_joint_angle['Frame'].to_numpy().T

        plt.plot(excel_data_joint_angle['Frame'], excel_data_joint_angle['Left Ball Foot Flexion/Extension'], marker='o')
        plt.xlabel('Frame (X-axis)')
        plt.ylabel('LeftFoot angle Flex/Extend (Y-axis)')
        plt.title('LeftFoot angle Flex/Extend')
        plt.savefig(op.join(left_angle_path, f"LeftFoot_angle_Flex_Extend_{i}_{j}.png"))
    
        print(f"Plot saved at: {op.join(left_angle_path, f"LeftFoot_angle_Flex_Extend_{i}_{j}.png")}")

        plt.clf()

        # phase, colorized = hilbert_transform_with_phase_color(excel_data_joint_angle)

        plt.rcParams['image.cmap'] = 'gist_rainbow'
        # Plot results
        # fig = plt.figure()
    
        # plt.scatter(frame_joint_angle, excel_data_joint_angle_numpy, c=colorized, s=1, alpha=0.75)
        # plt.plot(x[peaks], data[peaks], "x", label=f'Peaks (otsu)', c="#FA69AB")
        # plt.title(f'Loop Position with Hilbert foot joint angle')
        # plt.xlabel('Frame (X-axis)')
        # plt.ylabel('LeftFoot angle Flex/Extend (Y-axis) Hilbert transform')
        # cb=plt.colorbar(format=matplotlib.ticker.FormatStrFormatter('%g $\pi$'))
        # plt.savefig(op.join(left_angle_path, f"LeftFoot angle Flex_Extend_{i}_{j}.png"))
        
        # print(f"Plot saved at: {op.join(left_angle_path, f"LeftFoot angle Flex_Extend_{i}_{j}.png")}")

        # plt.clf()

        
        #Left foot
        excel_data_velocity_x = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Foot x','Pelvis x','Frame']]
        excel_data_velocity_x['LeftFootToPelvisx'] = excel_data_velocity_x['Left Foot x'] - excel_data_velocity_x['Pelvis x']
        excel_data_velocity_x = excel_data_velocity_x.iloc[:200]
        excel_data_velocity_x = excel_data_velocity_x.drop(['Left Foot x','Pelvis x'],axis=1)
        excel_data_velocity_x_numpy = excel_data_velocity_x['LeftFootToPelvisx'].to_numpy().T
        frame_velocity_x= excel_data_velocity_x['Frame'].to_numpy().T
        
        plt.plot(excel_data_velocity_x['Frame'], excel_data_velocity_x['LeftFootToPelvisx'], marker='o')
        plt.xlabel('Frame (X-axis)')
        plt.ylabel('LeftFootToPelvisx_vel (Y-axis)')
        plt.title('leftFootToPelvisx_vel')
        plt.savefig(op.join(left_velocity_foot_path, f"LeftFoot_velocity_X_{i}_{j}.png"))
    
        print(f"Plot saved at: {op.join(left_velocity_foot_path, f"LeftFoot_velocity_X_{i}_{j}.png")}")

        plt.clf()

        phase, colorized = hilbert_transform_with_phase_color(excel_data_velocity_x_numpy)

        plt.rcParams['image.cmap'] = 'gist_rainbow'
        # Plot results
        fig = plt.figure()
    
        plt.scatter(frame_velocity_x, excel_data_velocity_x_numpy, c=colorized, s=1, alpha=0.75)
        # plt.plot(x[peaks], data[peaks], "x", label=f'Peaks (otsu)', c="#FA69AB")
        plt.title(f'Loop Position with Hilbert Phase X')
        plt.xlabel('Frame (X-axis)')
        plt.ylabel('LeftFootToPelvisx_vel (Y-axis) Hilbert transform')
        cb=plt.colorbar(format=matplotlib.ticker.FormatStrFormatter('%g $\pi$'))
        plt.savefig(op.join(left_velocity_foot_path, f"VelocityLeftFoot_Hilbert_transform_X_{i}_{j}.png"))
        
        print(f"Plot saved at: {op.join(left_velocity_foot_path, f"VelocityLeftFoot_Hilbert_transform_X_{i}_{j}.png")}")

        plt.clf()

        excel_data_velocity_y = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Foot y','Pelvis y','Frame']]
        excel_data_velocity_y['LeftFootToPelvisy'] = excel_data_velocity_y['Left Foot y'] - excel_data_velocity_y['Pelvis y']
        excel_data_velocity_y = excel_data_velocity_y.iloc[:200]
        excel_data_velocity_y = excel_data_velocity_y.drop(['Left Foot y','Pelvis y'],axis=1)
        excel_data_velocity_y_numpy = excel_data_velocity_y['LeftFootToPelvisy'].to_numpy().T
        frame_velocity_y= excel_data_velocity_y['Frame'].to_numpy().T
        
        plt.plot(excel_data_velocity_y['Frame'], excel_data_velocity_y['LeftFootToPelvisy'], marker='o')
        plt.xlabel('Frame (X-axis)')
        plt.ylabel('LeftFootToPelvisy_vel (Y-axis)')
        plt.title('leftFootToPelvisy_vel')
        plt.savefig(op.join(left_velocity_foot_path, f"LeftFoot_velocity_Y_{i}_{j}.png"))
    
        print(f"Plot saved at: {op.join(left_velocity_foot_path, f"LeftFoot_velocity_Y_{i}_{j}.png")}")

        plt.clf()

        phase, colorized = hilbert_transform_with_phase_color(excel_data_velocity_y_numpy)

        plt.rcParams['image.cmap'] = 'gist_rainbow'
        # Plot results
        fig = plt.figure()
    
        plt.scatter(frame_velocity_y, excel_data_velocity_y_numpy, c=colorized, s=1, alpha=0.75)
        # plt.plot(x[peaks], data[peaks], "x", label=f'Peaks (otsu)', c="#FA69AB")
        plt.title(f'Loop Position with Hilbert Phase Y')
        plt.xlabel('Frame (X-axis)')
        plt.ylabel('LeftFootToPelvisy_vel (Y-axis) Hilbert transform')
        cb=plt.colorbar(format=matplotlib.ticker.FormatStrFormatter('%g $\pi$'))
        plt.savefig(op.join(left_velocity_foot_path, f"VelocityLeftFoot_Hilbert_transform_Y_{i}_{j}.png"))
        
        print(f"Plot saved at: {op.join(left_velocity_foot_path, f"VelocityLeftFoot_Hilbert_transform_Y_{i}_{j}.png")}")

        plt.clf()        

        
        excel_data_velocity_z = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Foot z','Pelvis z','Frame']]
        excel_data_velocity_z['LeftFootToPelvisz'] = excel_data_velocity_z['Left Foot z'] - excel_data_velocity_z['Pelvis z']
        excel_data_velocity_z = excel_data_velocity_z.iloc[:200]
        excel_data_velocity_z = excel_data_velocity_z.drop(['Left Foot z','Pelvis z'],axis=1)
        excel_data_velocity_z_numpy = excel_data_velocity_z['LeftFootToPelvisz'].to_numpy().T
        frame_velocity_z= excel_data_velocity_z['Frame'].to_numpy().T
        
        plt.plot(excel_data_velocity_z['Frame'], excel_data_velocity_z['LeftFootToPelvisz'], marker='o')
        plt.xlabel('Frame (X-axis)')
        plt.ylabel('LeftFootToPelvisz_vel (Y-axis)')
        plt.title('leftFootToPelvisz_vel')
        plt.savefig(op.join(left_velocity_foot_path, f"LeftFoot_velocity_Z_{i}_{j}.png"))
    
        print(f"Plot saved at: {op.join(left_velocity_foot_path, f"LeftFoot_velocity_Z_{i}_{j}.png")}")

        plt.clf()

        phase, colorized = hilbert_transform_with_phase_color(excel_data_velocity_z_numpy)

        plt.rcParams['image.cmap'] = 'gist_rainbow'
        # Plot results
        fig = plt.figure()
    
        plt.scatter(frame_velocity_z, excel_data_velocity_z_numpy, c=colorized, s=1, alpha=0.75)
        # plt.plot(x[peaks], data[peaks], "x", label=f'Peaks (otsu)', c="#FA69AB")
        plt.title(f'Loop Position with Hilbert Phase Z')
        plt.xlabel('Frame (X-axis)')
        plt.ylabel('LeftFootToPelvisz_vel (Y-axis) Hilbert transform')
        cb=plt.colorbar(format=matplotlib.ticker.FormatStrFormatter('%g $\pi$'))
        plt.savefig(op.join(left_velocity_foot_path, f"VelocityLeftFoot_Hilbert_transform_Z_{i}_{j}.png"))
        
        print(f"Plot saved at: {op.join(left_velocity_foot_path, f"VelocityLeftFoot_Hilbert_transform_Z_{i}_{j}.png")}")

        plt.clf()        

        #upperleg

        excel_data_velocity_x = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Upper Leg x','Pelvis x','Frame']]
        excel_data_velocity_x['LeftUpperLegToPelvisx'] = excel_data_velocity_x['Left Upper Leg x'] - excel_data_velocity_x['Pelvis x']
        excel_data_velocity_x = excel_data_velocity_x.iloc[:200]
        excel_data_velocity_x = excel_data_velocity_x.drop(['Left Upper Leg x','Pelvis x'],axis=1)
        excel_data_velocity_x_numpy = excel_data_velocity_x['LeftUpperLegToPelvisx'].to_numpy().T
        frame_velocity_x= excel_data_velocity_x['Frame'].to_numpy().T
        
        plt.plot(excel_data_velocity_x['Frame'], excel_data_velocity_x['LeftUpperLegToPelvisx'], marker='o')
        plt.xlabel('Frame (X-axis)')
        plt.ylabel('LeftUpperLegToPelvisx_vel (Y-axis)')
        plt.title('leftUpperLegToPelvisx_vel')
        plt.savefig(op.join(left_velocity_upper_leg_path, f"LeftUpperLeg_velocity_X_{i}_{j}.png"))
    
        print(f"Plot saved at: {op.join(left_velocity_upper_leg_path, f"LeftUpperLeg_velocity_X_{i}_{j}.png")}")

        plt.clf()

        phase, colorized = hilbert_transform_with_phase_color(excel_data_velocity_x_numpy)

        plt.rcParams['image.cmap'] = 'gist_rainbow'
        # Plot results
        fig = plt.figure()
    
        plt.scatter(frame_velocity_x, excel_data_velocity_x_numpy, c=colorized, s=1, alpha=0.75)
        # plt.plot(x[peaks], data[peaks], "x", label=f'Peaks (otsu)', c="#FA69AB")
        plt.title(f'Loop Position with Hilbert Phase X')
        plt.xlabel('Frame (X-axis)')
        plt.ylabel('LeftUpperLegToPelvisx_vel (Y-axis) Hilbert transform')
        cb=plt.colorbar(format=matplotlib.ticker.FormatStrFormatter('%g $\pi$'))
        plt.savefig(op.join(left_velocity_upper_leg_path, f"VelocityLeftUpperLeg_Hilbert_transform_X_{i}_{j}.png"))
        
        print(f"Plot saved at: {op.join(left_velocity_upper_leg_path, f"VelocityLeftUpperLeg_Hilbert_transform_X_{i}_{j}.png")}")

        plt.clf()

        excel_data_velocity_y = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Upper Leg y','Pelvis y','Frame']]
        excel_data_velocity_y['LeftUpperLegToPelvisy'] = excel_data_velocity_y['Left Upper Leg y'] - excel_data_velocity_y['Pelvis y']
        excel_data_velocity_y = excel_data_velocity_y.iloc[:200]
        excel_data_velocity_y = excel_data_velocity_y.drop(['Left Upper Leg y','Pelvis y'],axis=1)
        excel_data_velocity_y_numpy = excel_data_velocity_y['LeftUpperLegToPelvisy'].to_numpy().T
        frame_velocity_y= excel_data_velocity_y['Frame'].to_numpy().T
        
        plt.plot(excel_data_velocity_y['Frame'], excel_data_velocity_y['LeftUpperLegToPelvisy'], marker='o')
        plt.xlabel('Frame (X-axis)')
        plt.ylabel('LeftUpperLegToPelvisy_vel (Y-axis)')
        plt.title('leftUpperLegToPelvisy_vel')
        plt.savefig(op.join(left_velocity_upper_leg_path, f"LeftUpperLeg_velocity_Y_{i}_{j}.png"))
    
        print(f"Plot saved at: {op.join(left_velocity_upper_leg_path, f"LeftUpperLeg_velocity_Y_{i}_{j}.png")}")

        plt.clf()

        phase, colorized = hilbert_transform_with_phase_color(excel_data_velocity_y_numpy)

        plt.rcParams['image.cmap'] = 'gist_rainbow'
        # Plot results
        fig = plt.figure()
    
        plt.scatter(frame_velocity_y, excel_data_velocity_y_numpy, c=colorized, s=1, alpha=0.75)
        # plt.plot(x[peaks], data[peaks], "x", label=f'Peaks (otsu)', c="#FA69AB")
        plt.title(f'Loop Position with Hilbert Phase Y')
        plt.xlabel('Frame (X-axis)')
        plt.ylabel('LeftUpperLegToPelvisy_vel (Y-axis) Hilbert transform')
        cb=plt.colorbar(format=matplotlib.ticker.FormatStrFormatter('%g $\pi$'))
        plt.savefig(op.join(left_velocity_upper_leg_path, f"VelocityLeftUpperLeg_Hilbert_transform_Y_{i}_{j}.png"))
        
        print(f"Plot saved at: {op.join(left_velocity_upper_leg_path, f"VelocityLeftUpperLeg_Hilbert_transform_Y_{i}_{j}.png")}")

        plt.clf()        

        
        excel_data_velocity_z = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Upper Leg z','Pelvis z','Frame']]
        excel_data_velocity_z['LeftUpperLegToPelvisz'] = excel_data_velocity_z['Left Upper Leg z'] - excel_data_velocity_z['Pelvis z']
        excel_data_velocity_z = excel_data_velocity_z.iloc[:200]
        excel_data_velocity_z = excel_data_velocity_z.drop(['Left Upper Leg z','Pelvis z'],axis=1)
        excel_data_velocity_z_numpy = excel_data_velocity_z['LeftUpperLegToPelvisz'].to_numpy().T
        frame_velocity_z= excel_data_velocity_z['Frame'].to_numpy().T
        
        plt.plot(excel_data_velocity_z['Frame'], excel_data_velocity_z['LeftUpperLegToPelvisz'], marker='o')
        plt.xlabel('Frame (X-axis)')
        plt.ylabel('LeftUpperLegToPelvisz_vel (Y-axis)')
        plt.title('leftUpperLegToPelvisz_vel')
        plt.savefig(op.join(left_velocity_upper_leg_path, f"LeftUpperLeg_velocity_Z_{i}_{j}.png"))
    
        print(f"Plot saved at: {op.join(left_velocity_upper_leg_path, f"LeftUpperLeg_velocity_Z_{i}_{j}.png")}")

        plt.clf()

        phase, colorized = hilbert_transform_with_phase_color(excel_data_velocity_z_numpy)

        plt.rcParams['image.cmap'] = 'gist_rainbow'
        # Plot results
        fig = plt.figure()
    
        plt.scatter(frame_velocity_z, excel_data_velocity_z_numpy, c=colorized, s=1, alpha=0.75)
        # plt.plot(x[peaks], data[peaks], "x", label=f'Peaks (otsu)', c="#FA69AB")
        plt.title(f'Loop Position with Hilbert Phase Z')
        plt.xlabel('Frame (X-axis)')
        plt.ylabel('LeftUpperLegToPelvisz_vel (Y-axis) Hilbert transform')
        cb=plt.colorbar(format=matplotlib.ticker.FormatStrFormatter('%g $\pi$'))
        plt.savefig(op.join(left_velocity_upper_leg_path, f"VelocityLeftUpperLeg_Hilbert_transform_Z_{i}_{j}.png"))
        
        print(f"Plot saved at: {op.join(left_velocity_upper_leg_path, f"VelocityLeftUpperLeg_Hilbert_transform_Z_{i}_{j}.png")}")

        plt.clf()        

'''

        