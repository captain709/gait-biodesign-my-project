cd /mnt/ExpDrive/SparkLabLongRun/Data/peam_dataset/28-04-69

mkdir "position"
mkdir "acceleration"
mkdir "velocity"

for path in $(ls -d */); do
    # if [[ $path == "position/" ]]; then
    #     for file in $(ls $path); do
    #         mv "$path$file" "position/$file"
    #     done
    # elif [[ $path == "acceleration/" ]]; then
    #     for file in $(ls $path); do
    #         mv "$path$file" "acceleration/$file"
    #     done
    # elif [[ $path == "velocity/" ]]; then
    #     for file in $(ls $path); do
    #         mv "$path$file" "velocity/$file"
    #     done
    # fi
    mkdir "$path/LeftFoot"
    mkdir "$path/RightFoot"
    mkdir "$path/LeftFoot/Foot_to_Pelvis"
    mkdir "$path/RightFoot/Foot_to_Pelvis"

done