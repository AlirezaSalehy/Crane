model_name=$1
device=$2

# Use bellow alternatively for the woDattn model
# base_command="python test.py --devices $device --epoch 5 --dino_model none --features_list 6 12 18 24"
# base_command="python test.py --devices $device --epoch 5 --dino_model dinov2"
# command="$base_command --dataset mvtec --model_name trained_on_visa_$model_name"
# eval $command

# Use bellow alternatively for the woDattn model
# base_command="python test.py --devices $device --epoch 5 --dino_model none --features_list 6 12 18 24"
base_command="python test.py --devices $device --epoch 5 --dino_model dinov2"
for dataset in nail_dataset_v5_test hard_test_case; do
    command="$base_command --dataset $dataset --model_name trained_on_nail_dataset_v5_train_$model_name"
    eval $command
    command="$base_command --dataset $dataset --model_name trained_on_mvtec_default"
    eval $command
    command="$base_command --dataset $dataset --model_name trained_on_visa_default"
    eval $command
done
