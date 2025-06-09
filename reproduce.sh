# First train a model on MVTec and a model on VisA 
# Then use model train on MVTec for testing on the rest of datasets and the model train on VisA to test on MVTec
# for both version Crane and Crane+
model_name=$1
device=$2

run_for_trained_on_mvtec() {
    local base_command="$1"
    shift
    local datasets=("$@")

    for dataset in "${datasets[@]}"; do
        local command="$base_command --dataset $dataset --model_name trained_on_mvtec_$cur_model_name"
        eval "$command"
    done
}

# Table 1 Training Scheme 
# Crane (woD-Attn)
cur_model_name="${model_name}_crane"
echo "The name for base version (Crane) is: $cur_model_name"

python train.py --model_name "$cur_model_name" --dataset mvtec --device "$device" --features_list 6 12 18 24 --dino_model none --why "Evaluation purpose"
python train.py --model_name "$cur_model_name" --dataset visa  --device "$device" --features_list 6 12 18 24 --dino_model none --why "Evaluation purpose"

base_command="python test.py --devices $device --epoch 5 --dino_model none --soft_mean True --features_list 6 12 18 24 --visualize False"
eval "$base_command --dataset mvtec --model_name trained_on_visa_$cur_model_name"
run_for_trained_on_mvtec "$base_command" visa mpdd sdd btad dtd dagm
run_for_trained_on_mvtec "$base_command" brainmri headct br35h isic tn3k cvc-colondb cvc-clinicdb

# Table 1 Training Scheme 
# Crane+
cur_model_name="${model_name}_cranep"
echo "The name for enhanced version (Crane+) is: $cur_model_name"

python train.py --model_name "$cur_model_name" --dataset mvtec --device "$device" --features_list 24 --why "Evaluation purpose"
python train.py --model_name "$cur_model_name" --dataset visa  --device "$device" --features_list 24 --why "Evaluation purpose"

base_command="python test.py --devices $device --epoch 5 --dino_model dinov2 --soft_mean True --features_list 24 --visualize False"
eval "$base_command --dataset mvtec --model_name trained_on_visa_$cur_model_name"
run_for_trained_on_mvtec "$base_command" visa mpdd sdd btad dtd
eval "$base_command --dataset dagm --model_name trained_on_visa_$cur_model_name --soft_mean True "
base_command="python test.py --devices $device --epoch 1 --dino_model dinov2 --soft_mean True --features_list 24 --visualize False"
run_for_trained_on_mvtec "$base_command" brainmri headct br35h isic tn3k cvc-colondb cvc-clinicdb