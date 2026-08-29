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


# root_dir = "C:\\Users\\noppa\OneDrive\\เดสก์ท็อป\\SparkLab\\Code\\SPARK-Lab-IMU\\data\\Training_data\\peam_test"
root_dir = "../peam_test"
# left_acc_path = "C:\\Users\\noppa\OneDrive\\เดสก์ท็อป\\SparkLab\\Code\\SPARK-Lab-IMU\\data\\pkl_peam_test\\acceleration_graph\\LeftFoot"
left_acc_path = "../pkl_peam_test/acceleration_graph/LeftFoot"
# left_angle_path = "C:\\Users\\noppa\OneDrive\\เดสก์ท็อป\\SparkLab\\Code\\SPARK-Lab-IMU\\data\\pkl_peam_test\\joint_angle_graph\\LeftFoot"
left_angle_path = "../pkl_peam_test/joint_angle_graph/LeftFoot"
# left_velocity_foot_path = "C:\\Users\\noppa\OneDrive\\เดสก์ท็อป\\SparkLab\\Code\\SPARK-Lab-IMU\\data\\pkl_peam_test\\velocity_graph\\LeftFoot"
left_velocity_foot_path = "../pkl_peam_test/velocity_graph/LeftFoot"
# left_velocity_upper_leg_path = "C:\\Users\\noppa\OneDrive\\เดสก์ท็อป\\SparkLab\\Code\\SPARK-Lab-IMU\\data\\pkl_peam_test\\velocity_graph\\LeftUpperLeg"
left_velocity_upper_leg_path = "../pkl_peam_test/velocity_graph/LeftUpperLeg"
# left_velocity_lower_leg_path = "C:\\Users\\noppa\OneDrive\\เดสก์ท็อป\\SparkLab\\Code\\SPARK-Lab-IMU\\data\\pkl_peam_test\\velocity_graph\\LeftLowerLeg"
left_velocity_lower_leg_path = "../pkl_peam_test/velocity_graph/LeftLowerLeg"
# left_velocity_toe_path = "C:\\Users\\noppa\OneDrive\\เดสก์ท็อป\\SparkLab\\Code\\SPARK-Lab-IMU\\data\\pkl_peam_test\\velocity_graph\\LeftToe"
left_velocity_toe_path = "../pkl_peam_test/velocity_graph/LeftToe"
# left_velocity_shoulder_path = "C:\\Users\\noppa\OneDrive\\เดสก์ท็อป\\SparkLab\\Code\\SPARK-Lab-IMU\\data\\pkl_peam_test\\velocity_graph\\LeftShoulder"
left_velocity_shoulder_path = "../pkl_peam_test/velocity_graph/LeftShoulder"
# left_velocity_forearm_path = "C:\\Users\\noppa\OneDrive\\เดสก์ท็อป\\SparkLab\\Code\\SPARK-Lab-IMU\\data\\pkl_peam_test\\velocity_graph\\LeftForearm"
left_velocity_forearm_path = "../pkl_peam_test/velocity_graph/LeftForearm"
# left_velocity_upper_arm_path = "C:\\Users\\noppa\OneDrive\\เดสก์ท็อป\\SparkLab\\Code\\SPARK-Lab-IMU\\data\\pkl_peam_test\\velocity_graph\\LeftUpperArm"
left_velocity_upper_arm_path = "../pkl_peam_test/velocity_graph/LeftUpperArm"
# left_velocity_hand_path = "C:\\Users\\noppa\OneDrive\\เดสก์ท็อป\\SparkLab\\Code\\SPARK-Lab-IMU\\data\\pkl_peam_test\\velocity_graph\\LeftHand"
left_velocity_hand_path = "../pkl_peam_test/velocity_graph/LeftHand"

folder_name = ['A001_M', 'A002_M', 'A003_F', 'A004_F','A005_F','A006_F','A007_F','A008_F','A010_M','A011_M']
treadmill_speed = ['001','002','003','004','005']


for speedy in treadmill_speed:
    
    fig, axs = plt.subplots(5, 2, figsize=(20, 50))
    
    for ax, f_name in zip(axs.ravel(), folder_name):
    
        file_location = glob(f'{root_dir}/{f_name}/*_treadmill-' + speedy + '.xlsx')

        file_name = file_location[0]

        print(file_name)

      

        fig.suptitle(f"velocity_foot : {f_name}")

       
        excel_data_velocity_x = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Foot x','Pelvis x','Frame']]
        excel_data_velocity_x['LeftFootToPelvisx'] = excel_data_velocity_x['Left Foot x'] - excel_data_velocity_x['Pelvis x']
        excel_data_velocity_x = excel_data_velocity_x.iloc[60:260]
        excel_data_velocity_x = excel_data_velocity_x.drop(['Left Foot x','Pelvis x'],axis=1)
        excel_data_velocity_x_numpy = excel_data_velocity_x['LeftFootToPelvisx'].to_numpy().T

        

        ax.plot(excel_data_velocity_x['Frame'], excel_data_velocity_x['LeftFootToPelvisx'], marker='o')
        ax.set_xlabel('Frame (X-axis)')
        ax.set_ylabel('LeftFootToPelvisx_vel (Y-axis)')
        ax.set_title(f'leftFootToPelvisx_vel SUBJ: {Path(file_name).stem}')

    fig.savefig(op.join(left_velocity_foot_path, f"Test2_VelocityLeftFoot_x_subplots_{speedy}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_foot_path, f'Test2_VelocityLeftFoot_x_subplots_{speedy}.png')}")



for speedy in treadmill_speed:
    
    fig, axs = plt.subplots(5, 2, figsize=(20, 50))
    
    for ax, f_name in zip(axs.ravel(), folder_name):
    
        file_location = glob(f'{root_dir}/{f_name}/*_treadmill-' + speedy + '.xlsx')

        file_name = file_location[0]

        print(file_name)

      

        fig.suptitle(f"velocity_foot : {f_name}")

       
        excel_data_velocity_y = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Foot y','Pelvis y','Frame']]
        excel_data_velocity_y['LeftFootToPelvisy'] = excel_data_velocity_y['Left Foot y'] - excel_data_velocity_y['Pelvis y']
        excel_data_velocity_y = excel_data_velocity_y.iloc[60:260]
        excel_data_velocity_y = excel_data_velocity_y.drop(['Left Foot y','Pelvis y'],axis=1)
        excel_data_velocity_y_numpy = excel_data_velocity_y['LeftFootToPelvisy'].to_numpy().T

        

        ax.plot(excel_data_velocity_y['Frame'], excel_data_velocity_y['LeftFootToPelvisy'], marker='o')
        ax.set_xlabel('Frame (X-axis)')
        ax.set_ylabel('LeftFootToPelvisy_vel (Y-axis)')
        ax.set_title(f'leftFootToPelvisy_vel SUBJ: {Path(file_name).stem}')

    fig.savefig(op.join(left_velocity_foot_path, f"Test2_VelocityLeftFoot_y_subplots_{speedy}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_foot_path, f'Test2_VelocityLeftFoot_y_subplots_{speedy}.png')}")

    
for speedy in treadmill_speed:
    
    fig, axs = plt.subplots(5, 2, figsize=(20, 50))
    
    for ax, f_name in zip(axs.ravel(), folder_name):
    
        file_location = glob(f'{root_dir}/{f_name}/*_treadmill-' + speedy + '.xlsx')

        file_name = file_location[0]

        print(file_name)

      

        fig.suptitle(f"velocity_foot : {f_name}")

       
        excel_data_velocity_z = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Foot z','Pelvis z','Frame']]
        excel_data_velocity_z['LeftFootToPelvisz'] = excel_data_velocity_z['Left Foot z'] - excel_data_velocity_z['Pelvis z']
        excel_data_velocity_z = excel_data_velocity_z.iloc[60:260]
        excel_data_velocity_z = excel_data_velocity_z.drop(['Left Foot z','Pelvis z'],axis=1)
        excel_data_velocity_z_numpy = excel_data_velocity_z['LeftFootToPelvisz'].to_numpy().T

        

        ax.plot(excel_data_velocity_z['Frame'], excel_data_velocity_z['LeftFootToPelvisz'], marker='o')
        ax.set_xlabel('Frame (X-axis)')
        ax.set_ylabel('LeftFootToPelvisz_vel (Y-axis)')
        ax.set_title(f'leftFootToPelvisz_vel SUBJ: {Path(file_name).stem}')

    fig.savefig(op.join(left_velocity_foot_path, f"Test2_VelocityLeftFoot_z_subplots_{speedy}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_foot_path, f'Test2_VelocityLeftFoot_z_subplots_{speedy}.png')}")


for speedy in treadmill_speed:
    
    fig, axs = plt.subplots(5, 2, figsize=(20, 50))
    
    for ax, f_name in zip(axs.ravel(), folder_name):
    
        file_location = glob(f'{root_dir}/{f_name}/*_treadmill-' + speedy + '.xlsx')

        file_name = file_location[0]

        print(file_name)

      

        fig.suptitle(f"velocity_foot : {f_name}")

       
        excel_data_velocity_x = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Upper Leg x','Pelvis x','Frame']]
        excel_data_velocity_x['LeftUpperLegToPelvisx'] = excel_data_velocity_x['Left Upper Leg x'] - excel_data_velocity_x['Pelvis x']
        excel_data_velocity_x = excel_data_velocity_x.iloc[60:260]
        excel_data_velocity_x = excel_data_velocity_x.drop(['Left Upper Leg x','Pelvis x'],axis=1)
        excel_data_velocity_x_numpy = excel_data_velocity_x['LeftUpperLegToPelvisx'].to_numpy().T

        

        ax.plot(excel_data_velocity_x['Frame'], excel_data_velocity_x['LeftUpperLegToPelvisx'], marker='o')
        ax.set_xlabel('Frame (X-axis)')
        ax.set_ylabel('LeftUpperLegToPelvisx_vel (Y-axis)')
        ax.set_title(f'leftUpperLegToPelvisx_vel SUBJ: {Path(file_name).stem}')

    fig.savefig(op.join(left_velocity_upper_leg_path, f"Test2_VelocityLeftUpperLeg_x_subplots_{speedy}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_upper_leg_path, f'Test2_VelocityLeftUpperLeg_x_subplots_{speedy}.png')}")

for speedy in treadmill_speed:
    
    fig, axs = plt.subplots(5, 2, figsize=(20, 50))
    
    for ax, f_name in zip(axs.ravel(), folder_name):
    
        file_location = glob(f'{root_dir}/{f_name}/*_treadmill-' + speedy + '.xlsx')

        file_name = file_location[0]

        print(file_name)

      

        fig.suptitle(f"velocity_foot : {f_name}")

       
        excel_data_velocity_z = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Upper Leg z','Pelvis z','Frame']]
        excel_data_velocity_z['LeftUpperLegToPelvisz'] = excel_data_velocity_z['Left Upper Leg z'] - excel_data_velocity_z['Pelvis z']
        excel_data_velocity_z = excel_data_velocity_z.iloc[60:260]
        excel_data_velocity_z = excel_data_velocity_z.drop(['Left Upper Leg z','Pelvis z'],axis=1)
        excel_data_velocity_z_numpy = excel_data_velocity_z['LeftUpperLegToPelvisz'].to_numpy().T

        

        ax.plot(excel_data_velocity_z['Frame'], excel_data_velocity_z['LeftUpperLegToPelvisz'], marker='o')
        ax.set_xlabel('Frame (X-axis)')
        ax.set_ylabel('LeftUpperLegToPelvisz_vel (Y-axis)')
        ax.set_title(f'leftUpperLegToPelvisz_vel SUBJ: {Path(file_name).stem}')

    fig.savefig(op.join(left_velocity_upper_leg_path, f"Test2_VelocityLeftUpperLeg_z_subplots_{speedy}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_upper_leg_path, f'Test2_VelocityLeftUpperLeg_z_subplots_{speedy}.png')}")


for speedy in treadmill_speed:
    
    fig, axs = plt.subplots(5, 2, figsize=(20, 50))
    
    for ax, f_name in zip(axs.ravel(), folder_name):
    
        file_location = glob(f'{root_dir}/{f_name}/*_treadmill-' + speedy + '.xlsx')

        file_name = file_location[0]

        print(file_name)

      

        fig.suptitle(f"velocity_foot : {f_name}")

       
        excel_data_velocity_x = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Lower Leg x','Pelvis x','Frame']]
        excel_data_velocity_x['LeftLowerLegToPelvisx'] = excel_data_velocity_x['Left Lower Leg x'] - excel_data_velocity_x['Pelvis x']
        excel_data_velocity_x = excel_data_velocity_x.iloc[60:260]
        excel_data_velocity_x = excel_data_velocity_x.drop(['Left Lower Leg x','Pelvis x'],axis=1)
        excel_data_velocity_x_numpy = excel_data_velocity_x['LeftLowerLegToPelvisx'].to_numpy().T

        

        ax.plot(excel_data_velocity_x['Frame'], excel_data_velocity_x['LeftLowerLegToPelvisx'], marker='o')
        ax.set_xlabel('Frame (X-axis)')
        ax.set_ylabel('LeftLowerLegToPelvisx_vel (Y-axis)')
        ax.set_title(f'leftLowerLegToPelvisx_vel SUBJ: {Path(file_name).stem}')

    fig.savefig(op.join(left_velocity_lower_leg_path, f"Test2_VelocityLeftLowerLeg_x_subplots_{speedy}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_lower_leg_path, f'Test2_VelocityLeftLowerLeg_x_subplots_{speedy}.png')}")

for speedy in treadmill_speed:
    
    fig, axs = plt.subplots(5, 2, figsize=(20, 50))
    
    for ax, f_name in zip(axs.ravel(), folder_name):
    
        file_location = glob(f'{root_dir}/{f_name}/*_treadmill-' + speedy + '.xlsx')

        file_name = file_location[0]

        print(file_name)

      

        fig.suptitle(f"velocity_foot : {f_name}")

       
        excel_data_velocity_z = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Lower Leg z','Pelvis z','Frame']]
        excel_data_velocity_z['LeftLowerLegToPelvisz'] = excel_data_velocity_z['Left Lower Leg z'] - excel_data_velocity_z['Pelvis z']
        excel_data_velocity_z = excel_data_velocity_z.iloc[60:260]
        excel_data_velocity_z = excel_data_velocity_z.drop(['Left Lower Leg z','Pelvis z'],axis=1)
        excel_data_velocity_z_numpy = excel_data_velocity_z['LeftLowerLegToPelvisz'].to_numpy().T

        

        ax.plot(excel_data_velocity_z['Frame'], excel_data_velocity_z['LeftLowerLegToPelvisz'], marker='o')
        ax.set_xlabel('Frame (X-axis)')
        ax.set_ylabel('LeftLowerLegToPelvisz_vel (Y-axis)')
        ax.set_title(f'leftLowerLegToPelvisz_vel SUBJ: {Path(file_name).stem}')

    fig.savefig(op.join(left_velocity_lower_leg_path, f"Test2_VelocityLeftLowerLeg_z_subplots_{speedy}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_lower_leg_path, f'Test2_VelocityLeftLowerLeg_z_subplots_{speedy}.png')}")

for speedy in treadmill_speed:
    
    fig, axs = plt.subplots(5, 2, figsize=(20, 50))
    
    for ax, f_name in zip(axs.ravel(), folder_name):
    
        file_location = glob(f'{root_dir}/{f_name}/*_treadmill-' + speedy + '.xlsx')

        file_name = file_location[0]

        print(file_name)

      

        fig.suptitle(f"velocity_foot : {f_name}")

       
        excel_data_velocity_x = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Toe x','Pelvis x','Frame']]
        excel_data_velocity_x['LeftToeToPelvisx'] = excel_data_velocity_x['Left Toe x'] - excel_data_velocity_x['Pelvis x']
        excel_data_velocity_x = excel_data_velocity_x.iloc[60:260]
        excel_data_velocity_x = excel_data_velocity_x.drop(['Left Toe x','Pelvis x'],axis=1)
        excel_data_velocity_x_numpy = excel_data_velocity_x['LeftToeToPelvisx'].to_numpy().T

        

        ax.plot(excel_data_velocity_x['Frame'], excel_data_velocity_x['LeftToeToPelvisx'], marker='o')
        ax.set_xlabel('Frame (X-axis)')
        ax.set_ylabel('LeftToeToPelvisx_vel (Y-axis)')
        ax.set_title(f'leftToeToPelvisx_vel SUBJ: {Path(file_name).stem}')

    fig.savefig(op.join(left_velocity_toe_path, f"Test2_VelocityLeftToe_x_subplots_{speedy}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_toe_path, f'Test2_VelocityLeftToe_x_subplots_{speedy}.png')}")

    
for speedy in treadmill_speed:
    
    fig, axs = plt.subplots(5, 2, figsize=(20, 50))
    
    for ax, f_name in zip(axs.ravel(), folder_name):
    
        file_location = glob(f'{root_dir}/{f_name}/*_treadmill-' + speedy + '.xlsx')

        file_name = file_location[0]

        print(file_name)

      

        fig.suptitle(f"velocity_foot : {f_name}")

       
        excel_data_velocity_z = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Toe z','Pelvis z','Frame']]
        excel_data_velocity_z['LeftToeToPelvisz'] = excel_data_velocity_z['Left Toe z'] - excel_data_velocity_z['Pelvis z']
        excel_data_velocity_z = excel_data_velocity_z.iloc[60:260]
        excel_data_velocity_z = excel_data_velocity_z.drop(['Left Toe z','Pelvis z'],axis=1)
        excel_data_velocity_z_numpy = excel_data_velocity_z['LeftToeToPelvisz'].to_numpy().T

        

        ax.plot(excel_data_velocity_z['Frame'], excel_data_velocity_z['LeftToeToPelvisz'], marker='o')
        ax.set_xlabel('Frame (X-axis)')
        ax.set_ylabel('LeftToeToPelvisz_vel (Y-axis)')
        ax.set_title(f'leftToeToPelvisz_vel SUBJ: {Path(file_name).stem}')

    fig.savefig(op.join(left_velocity_toe_path, f"Test2_VelocityLeftToe_z_subplots_{speedy}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_toe_path, f'Test2_VelocityLeftToe_z_subplots_{speedy}.png')}")

for speedy in treadmill_speed:
    
    fig, axs = plt.subplots(5, 2, figsize=(20, 50))
    
    for ax, f_name in zip(axs.ravel(), folder_name):
    
        file_location = glob(f'{root_dir}/{f_name}/*_treadmill-' + speedy + '.xlsx')

        file_name = file_location[0]

        print(file_name)

      

        fig.suptitle(f"velocity_foot : {f_name}")

       
        excel_data_velocity_x = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Forearm x','T8 x','Frame']]
        excel_data_velocity_x['LeftForearmToPelvisx'] = excel_data_velocity_x['Left Forearm x'] - excel_data_velocity_x['T8 x']
        excel_data_velocity_x = excel_data_velocity_x.iloc[60:260]
        excel_data_velocity_x = excel_data_velocity_x.drop(['Left Forearm x','T8 x'],axis=1)
        excel_data_velocity_x_numpy = excel_data_velocity_x['LeftForearmToPelvisx'].to_numpy().T

        

        ax.plot(excel_data_velocity_x['Frame'], excel_data_velocity_x['LeftForearmToPelvisx'], marker='o')
        ax.set_xlabel('Frame (X-axis)')
        ax.set_ylabel('LeftForearmToPelvisx_vel (Y-axis)')
        ax.set_title(f'leftForearmToPelvisx_vel SUBJ: {Path(file_name).stem}')

    fig.savefig(op.join(left_velocity_forearm_path, f"Test2_VelocityLeftForearm_x_subplots_{speedy}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_forearm_path, f'Test2_VelocityLeftForearm_x_subplots_{speedy}.png')}")



for speedy in treadmill_speed:
    
    fig, axs = plt.subplots(5, 2, figsize=(20, 50))
    
    for ax, f_name in zip(axs.ravel(), folder_name):
    
        file_location = glob(f'{root_dir}/{f_name}/*_treadmill-' + speedy + '.xlsx')

        file_name = file_location[0]

        print(file_name)

      

        fig.suptitle(f"velocity_foot : {f_name}")

       
        excel_data_velocity_x = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Hand x','T8 x','Frame']]
        excel_data_velocity_x['LeftHandToPelvisx'] = excel_data_velocity_x['Left Hand x'] - excel_data_velocity_x['T8 x']
        excel_data_velocity_x = excel_data_velocity_x.iloc[60:260]
        excel_data_velocity_x = excel_data_velocity_x.drop(['Left Hand x','T8 x'],axis=1)
        excel_data_velocity_x_numpy = excel_data_velocity_x['LeftHandToPelvisx'].to_numpy().T

        

        ax.plot(excel_data_velocity_x['Frame'], excel_data_velocity_x['LeftHandToPelvisx'], marker='o')
        ax.set_xlabel('Frame (X-axis)')
        ax.set_ylabel('LeftHandToPelvisx_vel (Y-axis)')
        ax.set_title(f'leftHandToPelvisx_vel SUBJ: {Path(file_name).stem}')

    fig.savefig(op.join(left_velocity_hand_path, f"Test2_VelocityLeftHand_x_subplots_{speedy}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_hand_path, f'Test2_VelocityLeftHand_x_subplots_{speedy}.png')}")


for speedy in treadmill_speed:
    
    fig, axs = plt.subplots(5, 2, figsize=(20, 50))
    
    for ax, f_name in zip(axs.ravel(), folder_name):
    
        file_location = glob(f'{root_dir}/{f_name}/*_treadmill-' + speedy + '.xlsx')

        file_name = file_location[0]

        print(file_name)

      

        fig.suptitle(f"velocity_foot : {f_name}")

       
        excel_data_velocity_y = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Hand y','T8 y','Frame']]
        excel_data_velocity_y['LeftHandToPelvisy'] = excel_data_velocity_y['Left Hand y'] - excel_data_velocity_y['T8 y']
        excel_data_velocity_y = excel_data_velocity_y.iloc[60:260]
        excel_data_velocity_y = excel_data_velocity_y.drop(['Left Hand y','T8 y'],axis=1)
        excel_data_velocity_y_numpy = excel_data_velocity_y['LeftHandToPelvisy'].to_numpy().T

        

        ax.plot(excel_data_velocity_y['Frame'], excel_data_velocity_y['LeftHandToPelvisy'], marker='o')
        ax.set_xlabel('Frame (X-axis)')
        ax.set_ylabel('LeftHandToPelvisy_vel (Y-axis)')
        ax.set_title(f'leftHandToPelvisy_vel SUBJ: {Path(file_name).stem}')

    fig.savefig(op.join(left_velocity_hand_path, f"Test2_VelocityLeftHand_y_subplots_{speedy}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_hand_path, f'Test2_VelocityLeftHand_y_subplots_{speedy}.png')}")


for speedy in treadmill_speed:
    
    fig, axs = plt.subplots(5, 2, figsize=(20, 50))
    
    for ax, f_name in zip(axs.ravel(), folder_name):
    
        file_location = glob(f'{root_dir}/{f_name}/*_treadmill-' + speedy + '.xlsx')

        file_name = file_location[0]

        print(file_name)

      

        fig.suptitle(f"velocity_foot : {f_name}")

       
        excel_data_velocity_z = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Hand z','T8 z','Frame']]
        excel_data_velocity_z['LeftHandToPelvisz'] = excel_data_velocity_z['Left Hand z'] - excel_data_velocity_z['T8 z']
        excel_data_velocity_z = excel_data_velocity_z.iloc[60:260]
        excel_data_velocity_z = excel_data_velocity_z.drop(['Left Hand z','T8 z'],axis=1)
        excel_data_velocity_z_numpy = excel_data_velocity_z['LeftHandToPelvisz'].to_numpy().T

        

        ax.plot(excel_data_velocity_z['Frame'], excel_data_velocity_z['LeftHandToPelvisz'], marker='o')
        ax.set_xlabel('Frame (X-axis)')
        ax.set_ylabel('LeftHandToPelvisz_vel (Y-axis)')
        ax.set_title(f'leftHandToPelvisz_vel SUBJ: {Path(file_name).stem}')

    fig.savefig(op.join(left_velocity_hand_path, f"Test2_VelocityLeftHand_z_subplots_{speedy}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_hand_path, f'Test2_VelocityLeftHand_z_subplots_{speedy}.png')}")

    #name pelvis but actually t8