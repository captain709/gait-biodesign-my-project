import numpy
import pandas as pd
import os
from glob import glob
import pickle
from tqdm import tqdm
from pathlib import Path

print(os.getcwd())
# root_dir = "C:\\Users\\Admin\\Desktop\\peam_biodis_gait\\peam_test"
root_dir = "/mnt/ExpDrive/SparkLabLongRun/Data/Biodesign_Data"
# left_acc_path = "C:\\\\Users\\\\noppa\\OneDrive\\\\เดสก์ท็อป\\\\SparkLab\\\\Code\\\\SPARK-Lab-IMU\\\\data\\\\pkl_peam_test\\\\acceleration\\\\LeftFoot"

# left_angle_path = "C:\\\\Users\\\\noppa\\OneDrive\\\\เดสก์ท็อป\\\\SparkLab\\\\Code\\\\SPARK-Lab-IMU\\\\data\\\\pkl_peam_test\\\\joint_angle\\\\LeftFoot"
# left_velocity_foot_to_pelvis_path = "C:\\\\Users\\\\noppa\\OneDrive\\\\เดสก์ท็อป\\\\SparkLab\\\\Code\\\\SPARK-Lab-IMU\\\\data\\\\pkl_peam_test\\\\velocity\\\\LeftFoot\\\\Foot_to_Pelvis"
# right_velocity_foot_to_pelvis_path = "C:\\\\Users\\\\noppa\\OneDrive\\\\เดสก์ท็อป\\\\SparkLab\\\\Code\\\\SPARK-Lab-IMU\\\\data\\\\pkl_peam_test\\\\velocity\\\\RightFoot\\\\Foot_to_Pelvis"
# left_position_foot_to_pelvis_path = "C:\\Users\\Admin\\Desktop\\peam_biodis_gait\\pkl_peam_test\\position\\LeftFoot\\Foot_to_Pelvis"
# right_position_foot_to_pelvis_path = "C:\\Users\\Admin\\Desktop\\peam_biodis_gait\\pkl_peam_test\\position\\RightFoot\\Foot_to_Pelvis"

# left_position_foot_to_pelvis_path = "C:\\Users\\Admin\\Desktop\\peam_biodis_gait\\pkl_peam_test\\position\\LeftFoot\\Foot_to_Pelvis"
# right_position_foot_to_pelvis_path = "C:\\Users\\Admin\\Desktop\\peam_biodis_gait\\pkl_peam_test\\position\\RightFoot\\Foot_to_Pelvis"
# left_velocity_foot_to_pelvis_path = "C:\\Users\\Admin\\Desktop\\peam_biodis_gait\\pkl_peam_test\\velocity\\LeftFoot\\Foot_to_Pelvis"
# right_velocity_foot_to_pelvis_path = "C:\\Users\\Admin\\Desktop\\peam_biodis_gait\\pkl_peam_test\\velocity\\RightFoot\\Foot_to_Pelvis"

left_position_foot_to_pelvis_path = "/mnt/ExpDrive/SparkLabLongRun/Data/peam_dataset/28-04-69/position/LeftFoot/Foot_to_Pelvis"
right_position_foot_to_pelvis_path = "/mnt/ExpDrive/SparkLabLongRun/Data/peam_dataset/28-04-69/position/RightFoot/Foot_to_Pelvis"
left_velocity_foot_to_pelvis_path = "/mnt/ExpDrive/SparkLabLongRun/Data/peam_dataset/28-04-69/velocity/LeftFoot/Foot_to_Pelvis"
right_velocity_foot_to_pelvis_path = "/mnt/ExpDrive/SparkLabLongRun/Data/peam_dataset/28-04-69/velocity/RightFoot/Foot_to_Pelvis"

All_path =  [left_position_foot_to_pelvis_path,right_position_foot_to_pelvis_path,left_velocity_foot_to_pelvis_path ,right_velocity_foot_to_pelvis_path]
#print(os.listdir())

for i in All_path:
    if not os.path.exists(i):
        os.makedirs(i)
        print(f"Folder created at: {i}")
    else:
        print(f"Folder already exists: {i}")
#position
# folder_name = ['A001_M', 'A007_F', 'A013_M', 'B001_M', 'B007_F', 'B013_F', 'C001_F', 'C007_M', 'C013_F',
# 'A002_M', 'A008_F', 'A014_M', 'B002_F', 'B008_F', 'B014_M', 'C002_M', 'C008_M', 'C014_F',
# 'A003_F', 'A009_M', 'A015_F', 'B003_F', 'B009_M', 'B015_M', 'C003_F', 'C009_M', 'C015_F',
# 'A004_F', 'A010_M', 'A016_M', 'B004_F', 'B010_F', 'B016_M', 'C004_F', 'C010_F', 'C016_M',
# 'A005_F', 'A011_M', 'A017_F', 'B005_M', 'B011_F', 'B017_M', 'C005_M', 'C011_F', 'C017_M',
# 'A006_F', 'A012_F', 'A018_M', 'B006_F', 'B012_M', 'B018_M', 'C006_F', 'C012_M', 'C018_M']
folder_name = ['A001_M', 'A002_M', 'A003_F', 'A004_F','A005_F','A006_F','A007_F','A008_F','A010_M','A011_M']
for i in tqdm(folder_name):
    file_location = glob(f'{root_dir}/{i}/*_treadmill-005*.xlsx')
    print(file_location)
    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)
        # excel_data_acceleration = pd.read_excel(file_name, sheet_name = 'Segment Acceleration')[['Left Foot x','Left Foot z']]
        # excel_data_acceleration = excel_data_acceleration.to_numpy().T
        # excel_data_joint_angle = pd.read_excel(file_name, sheet_name = 'Joint Angles ZXY')[['Left Ball Foot Abduction/Adduction','Left Ball Foot Internal/External Rotation','Left Ball Foot Flexion/Extension']]
        # excel_data_joint_angle = excel_data_joint_angle.to_numpy().T
        excel_data_velocity = pd.read_excel(file_name, sheet_name = 'Segment Position')[['Left Foot x','Pelvis x','Left Foot y','Pelvis y', 'Left Foot z','Pelvis z','Frame']]
        excel_data_velocity['Foot_to_pelvis_x'] = excel_data_velocity['Left Foot x'] - excel_data_velocity['Pelvis x']
        excel_data_velocity['Foot_to_pelvis_y'] = excel_data_velocity['Left Foot y'] - excel_data_velocity['Pelvis y']
        excel_data_velocity['Foot_to_pelvis_z'] = excel_data_velocity['Left Foot z'] - excel_data_velocity['Pelvis z']
        excel_data_velocity = excel_data_velocity.drop(['Left Foot x','Pelvis x','Left Foot y','Pelvis y','Left Foot z','Pelvis z'],axis=1)
        # excel_data_velocity = excel_data_velocity[['Foot_to_pelvis_x', 'Foot_to_pelvis_z','Frame','Foot_to_pelvis_y']]
        excel_data_velocity = excel_data_velocity[['Foot_to_pelvis_x', 'Foot_to_pelvis_z','Frame','Foot_to_pelvis_y']]
        # print(excel_data_velocity.head())
        excel_data_velocity = excel_data_velocity.to_numpy().T
        # data_dict_acceleration = {'data': excel_data_acceleration}
        # data_dict_joint_angle = {'data': excel_data_joint_angle}
        data_dict_velocity = {'data':excel_data_velocity}
        # with open(os.path.join(left_acc_path,i + '_' + Path(file_name).stem + '_'+ "acc" + ".pkl"),"wb") as f :
        #     pickle.dump(data_dict_acceleration,f)
        # with open(os.path.join(left_angle_path,i + '_' + Path(file_name).stem + '_'+ "angle" + ".pkl"),"wb") as f :
        #     pickle.dump(data_dict_joint_angle,f)
        with open(os.path.join(left_position_foot_to_pelvis_path,i + '_' + Path(file_name).stem + '_'+ "angle"+'_5' + ".pkl"),"wb") as f :
            pickle.dump(data_dict_velocity,f)

        # print(excel_data_velocity.shape)
        # print('velocity size')
        # print(excel_data_acceleration.shape)
        # print('acc size'
        # print('done')
# print(file_location)


for i in tqdm(folder_name):
    file_location = glob(f'{root_dir}/{i}/*_treadmill-005*.xlsx')
    print(file_location)
    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)
        # excel_data_acceleration = pd.read_excel(file_name, sheet_name = 'Segment Acceleration')[['Left Foot x','Left Foot z']]
        # excel_data_acceleration = excel_data_acceleration.to_numpy().T
        # excel_data_joint_angle = pd.read_excel(file_name, sheet_name = 'Joint Angles ZXY')[['Left Ball Foot Abduction/Adduction','Left Ball Foot Internal/External Rotation','Left Ball Foot Flexion/Extension']]
        # excel_data_joint_angle = excel_data_joint_angle.to_numpy().T
        excel_data_velocity = pd.read_excel(file_name, sheet_name = 'Segment Position')[['Right Foot x','Pelvis x','Right Foot y','Pelvis y', 'Right Foot z','Pelvis z','Frame']]
        excel_data_velocity['Foot_to_pelvis_x'] = excel_data_velocity['Right Foot x'] - excel_data_velocity['Pelvis x']
        excel_data_velocity['Foot_to_pelvis_y'] = excel_data_velocity['Right Foot y'] - excel_data_velocity['Pelvis y']
        excel_data_velocity['Foot_to_pelvis_z'] = excel_data_velocity['Right Foot z'] - excel_data_velocity['Pelvis z']
        excel_data_velocity = excel_data_velocity.drop(['Right Foot x','Pelvis x','Right Foot y','Pelvis y','Right Foot z','Pelvis z'],axis=1)
        # excel_data_velocity = excel_data_velocity[['Foot_to_pelvis_x','Foot_to_pelvis_z','Frame','Foot_to_pelvis_y']]
        excel_data_velocity = excel_data_velocity[['Foot_to_pelvis_x','Foot_to_pelvis_z','Frame','Foot_to_pelvis_y']]
        # print(excel_data_velocity.head())


        excel_data_velocity = excel_data_velocity.to_numpy().T
        # data_dict_acceleration = {'data': excel_data_acceleration}
        # data_dict_joint_angle = {'data': excel_data_joint_angle}
        data_dict_velocity = {'data':excel_data_velocity}
        # with open(os.path.join(left_acc_path,i + '_' + Path(file_name).stem + '_'+ "acc" + ".pkl"),"wb") as f :
        #     pickle.dump(data_dict_acceleration,f)
        # with open(os.path.join(left_angle_path,i + '_' + Path(file_name).stem + '_'+ "angle" + ".pkl"),"wb") as f :
        #     pickle.dump(data_dict_joint_angle,f)
        with open(os.path.join(right_position_foot_to_pelvis_path,i + '_' + Path(file_name).stem + '_'+ "angle"+'_5'+'xzyframe' + ".pkl"),"wb") as f :
            pickle.dump(data_dict_velocity,f)

        # print(excel_data_velocity.shape)
        # print('velocity size')
        # print(excel_data_acceleration.shape)
        # print('acc size')
        # print('done')
# print(file_location)

#velocty

# folder_name = ['A001_M', 'A007_F', 'A013_M', 'B001_M', 'B007_F', 'B013_F', 'C001_F', 'C007_M', 'C013_F',
# 'A002_M', 'A008_F', 'A014_M', 'B002_F', 'B008_F', 'B014_M', 'C002_M', 'C008_M', 'C014_F',
# 'A003_F', 'A009_M', 'A015_F', 'B003_F', 'B009_M', 'B015_M', 'C003_F', 'C009_M', 'C015_F',
# 'A004_F', 'A010_M', 'A016_M', 'B004_F', 'B010_F', 'B016_M', 'C004_F', 'C010_F', 'C016_M',
# 'A005_F', 'A011_M', 'A017_F', 'B005_M', 'B011_F', 'B017_M', 'C005_M', 'C011_F', 'C017_M',
# 'A006_F', 'A012_F', 'A018_M', 'B006_F', 'B012_M', 'B018_M', 'C006_F', 'C012_M', 'C018_M']
folder_name = ['A001_M', 'A002_M', 'A003_F', 'A004_F','A005_F','A006_F','A007_F','A008_F','A010_M','A011_M']
for i in tqdm(folder_name):
    file_location = glob(f'{root_dir}/{i}/*_treadmill-005*.xlsx')
    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)
        # excel_data_acceleration = pd.read_excel(file_name, sheet_name = 'Segment Acceleration')[['Left Foot x','Left Foot z']]
        # excel_data_acceleration = excel_data_acceleration.to_numpy().T
        # excel_data_joint_angle = pd.read_excel(file_name, sheet_name = 'Joint Angles ZXY')[['Left Ball Foot Abduction/Adduction','Left Ball Foot Internal/External Rotation','Left Ball Foot Flexion/Extension']]
        # excel_data_joint_angle = excel_data_joint_angle.to_numpy().T
        excel_data_velocity = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Left Foot x','Pelvis x','Left Foot y','Pelvis y', 'Left Foot z','Pelvis z','Frame']]
        excel_data_velocity['Foot_to_pelvis_x'] = excel_data_velocity['Left Foot x'] - excel_data_velocity['Pelvis x']
        excel_data_velocity['Foot_to_pelvis_y'] = excel_data_velocity['Left Foot y'] - excel_data_velocity['Pelvis y']
        excel_data_velocity['Foot_to_pelvis_z'] = excel_data_velocity['Left Foot z'] - excel_data_velocity['Pelvis z']
        excel_data_velocity = excel_data_velocity.drop(['Left Foot x','Pelvis x','Left Foot y','Pelvis y','Left Foot z','Pelvis z'],axis=1)
        # excel_data_velocity = excel_data_velocity[['Foot_to_pelvis_x', 'Foot_to_pelvis_z','Frame','Foot_to_pelvis_y']]
        excel_data_velocity = excel_data_velocity[['Foot_to_pelvis_x', 'Foot_to_pelvis_z','Frame','Foot_to_pelvis_y']]
        # print(excel_data_velocity.head())
        excel_data_velocity = excel_data_velocity.to_numpy().T
        # data_dict_acceleration = {'data': excel_data_acceleration}
        # data_dict_joint_angle = {'data': excel_data_joint_angle}
        data_dict_velocity = {'data':excel_data_velocity}
        # with open(os.path.join(left_acc_path,i + '_' + Path(file_name).stem + '_'+ "acc" + ".pkl"),"wb") as f :
        #     pickle.dump(data_dict_acceleration,f)
        # with open(os.path.join(left_angle_path,i + '_' + Path(file_name).stem + '_'+ "angle" + ".pkl"),"wb") as f :
        #     pickle.dump(data_dict_joint_angle,f)
        with open(os.path.join(left_velocity_foot_to_pelvis_path,i + '_' + Path(file_name).stem + '_'+ "angle"+'_5' + ".pkl"),"wb") as f :
            pickle.dump(data_dict_velocity,f)

        # print(excel_data_velocity.shape)
        # print('velocity size')
        # print(excel_data_acceleration.shape)
        # print('acc size')
        # print('done')
# print(file_location)


for i in tqdm(folder_name):
    file_location = glob(f'{root_dir}/{i}/*_treadmill-005*.xlsx')
    for j, file_name in enumerate(file_location):
        print(str(j +1) + "." + i + ":" + file_name)
        # excel_data_acceleration = pd.read_excel(file_name, sheet_name = 'Segment Acceleration')[['Left Foot x','Left Foot z']]
        # excel_data_acceleration = excel_data_acceleration.to_numpy().T
        # excel_data_joint_angle = pd.read_excel(file_name, sheet_name = 'Joint Angles ZXY')[['Left Ball Foot Abduction/Adduction','Left Ball Foot Internal/External Rotation','Left Ball Foot Flexion/Extension']]
        # excel_data_joint_angle = excel_data_joint_angle.to_numpy().T
        excel_data_velocity = pd.read_excel(file_name, sheet_name = 'Segment Velocity')[['Right Foot x','Pelvis x','Right Foot y','Pelvis y', 'Right Foot z','Pelvis z','Frame']]
        excel_data_velocity['Foot_to_pelvis_x'] = excel_data_velocity['Right Foot x'] - excel_data_velocity['Pelvis x']
        excel_data_velocity['Foot_to_pelvis_y'] = excel_data_velocity['Right Foot y'] - excel_data_velocity['Pelvis y']
        excel_data_velocity['Foot_to_pelvis_z'] = excel_data_velocity['Right Foot z'] - excel_data_velocity['Pelvis z']
        excel_data_velocity = excel_data_velocity.drop(['Right Foot x','Pelvis x','Right Foot y','Pelvis y','Right Foot z','Pelvis z'],axis=1)
        # excel_data_velocity = excel_data_velocity[['Foot_to_pelvis_x','Foot_to_pelvis_z','Frame','Foot_to_pelvis_y']]
        excel_data_velocity = excel_data_velocity[['Foot_to_pelvis_x','Foot_to_pelvis_z','Frame','Foot_to_pelvis_y']]
        # print(excel_data_velocity.head())


        excel_data_velocity = excel_data_velocity.to_numpy().T
        # data_dict_acceleration = {'data': excel_data_acceleration}
        # data_dict_joint_angle = {'data': excel_data_joint_angle}
        data_dict_velocity = {'data':excel_data_velocity}
        # with open(os.path.join(left_acc_path,i + '_' + Path(file_name).stem + '_'+ "acc" + ".pkl"),"wb") as f :
        #     pickle.dump(data_dict_acceleration,f)
        # with open(os.path.join(left_angle_path,i + '_' + Path(file_name).stem + '_'+ "angle" + ".pkl"),"wb") as f :
        #     pickle.dump(data_dict_joint_angle,f)
        with open(os.path.join(right_velocity_foot_to_pelvis_path,i + '_' + Path(file_name).stem + '_'+ "angle"+'_5' + ".pkl"),"wb") as f :
            pickle.dump(data_dict_velocity,f)

        # print(excel_data_velocity.shape)
        # print('velocity size')
        # print(excel_data_acceleration.shape)
        # print('acc size')
        # print('done')
# print(file_location)
