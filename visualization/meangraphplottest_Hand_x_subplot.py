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
all_mean_data_final =[]
all_sd_data_final = []

n = 71

for speedy in treadmill_speed:

    list_of_excel_data_velocity_x = []
    
    for person in folder_name:

        file_location = glob(f'{root_dir}/{person}/*_treadmill-' + speedy + '.xlsx')

        file_name = file_location[0]
        
        excel_data_velocity_x = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Hand x','T8 x','Frame']]
        excel_data_velocity_x['LeftHandToSternumx'] = excel_data_velocity_x['Left Hand x'] - excel_data_velocity_x['T8 x']
        excel_data_velocity_x = excel_data_velocity_x.iloc[60:260]
        excel_data_velocity_x = excel_data_velocity_x.drop(['Left Hand x','T8 x'],axis=1)
        
        # excel_data_velocity_x['LeftFootToPelvisx'] = [float(i) for i in excel_data_velocity_x['LeftFootToPelvisx']]
        excel_data_velocity_x_list = excel_data_velocity_x['LeftHandToSternumx'].values.tolist()
        print(type(excel_data_velocity_x_list))
        print(len(excel_data_velocity_x_list))
        print('x' + person+ speedy + 'done')
        list_of_excel_data_velocity_x.append(excel_data_velocity_x_list)
        print(len(list_of_excel_data_velocity_x))
        
   
    # print(list_of_excel_data_velocity_x(0))
    mean_of_excel_data_velocity_x = []
    for i in range(0,n):
        sum_of_num = 0
        for j in range(0,10):
            
            sum_of_num = sum_of_num + float(list_of_excel_data_velocity_x[j][i])
            mean_of_num = sum_of_num/len(folder_name)
        mean_of_excel_data_velocity_x.append(mean_of_num)
        print('mean_x' + speedy + 'done')
    
    all_mean_data_final.append(mean_of_excel_data_velocity_x)


    

    sd_of_excel_data_velocity_x = []
    for i in range (0,n):
        what_the_sigma = 0
        for j in range(0,10):
            what_the_sigma = what_the_sigma + ((list_of_excel_data_velocity_x[j][i]- mean_of_excel_data_velocity_x[i])**2)
            total_sd = math.sqrt(what_the_sigma / n)
        sd_of_excel_data_velocity_x.append(total_sd)
        print('sd_x' + speedy + 'done' )

    all_sd_data_final.append(sd_of_excel_data_velocity_x)

print(len(all_mean_data_final))
print(len(mean_of_excel_data_velocity_x))
print(len(all_sd_data_final))
print(len(sd_of_excel_data_velocity_x))


for k in range(0,5):
    excel_frame_velocity_x = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Frame']]
    excel_frame_velocity_x = excel_frame_velocity_x.iloc[60:131]
    excel_frame_velocity_x_list = excel_frame_velocity_x['Frame'].values.tolist()

    plt.plot(excel_frame_velocity_x_list, all_mean_data_final[k], marker='o', color='red')
    plt.xlabel('Frame (X-axis)')
    plt.ylabel('MeanLeftHandToSternumx_vel (Y-axis)')
    plt.title('meanLeftHandToSternumx_vel')
    plt.savefig(op.join(left_velocity_hand_path, f"MeanLeftHandToSternumx_{k + 1}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_hand_path, f"MeanLeftHandToSternumx_{k +1 }.png")}")

    plt.clf()

for k in range(0,5):
    excel_frame_velocity_x = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Frame']]
    excel_frame_velocity_x = excel_frame_velocity_x.iloc[60:131]
    excel_frame_velocity_x_list = excel_frame_velocity_x['Frame'].values.tolist()

    plt.plot(excel_frame_velocity_x_list, all_sd_data_final[k], marker='o', color='green')
    plt.xlabel('Frame (X-axis)')
    plt.ylabel('SDLLeftHandToSternumx_vel (Y-axis)')
    plt.title('SDLeftHandToSternumx_vel')
    plt.savefig(op.join(left_velocity_hand_path, f"SDLeftHandToSternumx_{k + 1}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_hand_path, f"SDLeftHandToSternumx_{k +1 }.png")}")

    plt.clf()

# mean_plus_sd = [a + b for a, b in zip(all_mean_data_final,all_sd_data_final)]
# mean_minus_sd = [a - b for a, b in zip(all_mean_data_final,all_sd_data_final)]


for k in range (0,5):
    excel_frame_velocity_x = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Frame']]
    excel_frame_velocity_x = excel_frame_velocity_x.iloc[60:131]
    excel_frame_velocity_x_list = excel_frame_velocity_x['Frame'].values.tolist() 

    mean_plus_sd = [a + b for a, b in zip(all_mean_data_final[k],all_sd_data_final[k])]
    mean_minus_sd = [a - b for a, b in zip(all_mean_data_final[k],all_sd_data_final[k])]

    plt.plot(excel_frame_velocity_x_list, all_mean_data_final[k], marker='o', color='red')
    plt.plot(excel_frame_velocity_x_list, mean_plus_sd, marker='o', color='salmon')
    plt.plot(excel_frame_velocity_x_list, mean_minus_sd, marker='o', color='indianred')
    plt.xlabel('Frame (X-axis)')
    plt.ylabel('MeanLeftHandToSternumx_vel (Y-axis)')
    plt.title('meanLeftHandToSternumx_vel')
    plt.savefig(op.join(left_velocity_hand_path, f"MeanLeftHandToSternumx_with_SD_X_{k + 1}.png"))
    
    print(f"Plot saved at: {op.join(left_velocity_hand_path, f"MeanLeftHandToSternumx_with_SD_X_{k +1 }.png")}")

    plt.clf()


