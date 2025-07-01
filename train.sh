model_name=$1
device=$2

# Table 1 and 2 training scheme
python train.py --model_name $1 --train_data_path nail_dataset_v5_train --dataset nail_dataset_v5_train --device $2 --why "Evalution purpose"

# # To test it
bash test.sh $1 $2
