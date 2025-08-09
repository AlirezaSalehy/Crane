# To test the current checkpoints (reported in paper)
device=$1

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
# Crane 
cur_model_name="crane"
base_command="python test.py --devices $device --epoch 5 --dino_model none --soft_mean True --features_list 6 12 18 24 --visualize False"
eval "$base_command --dataset mvtec --model_name trained_on_visa_$cur_model_name"
run_for_trained_on_mvtec "$base_command" visa mpdd sdd btad dtd dagm
run_for_trained_on_mvtec "$base_command" brainmri headct br35h isic tn3k cvc-colondb cvc-clinicdb

# Table 1 Training Scheme  
# Crane+ (with D-Atten)
cur_model_name="cranep"
# MVTec
base_command="python test.py --devices $device --epoch 5 --dino_model dinov2 --features_list 24 --visualize False"
eval "$base_command --dataset mvtec --model_name trained_on_visa_$cur_model_name"
# visa mpdd sdd btad dtd
run_for_trained_on_mvtec "$base_command" visa mpdd sdd btad dtd
eval "$base_command --dataset dagm --soft_mean True --model_name trained_on_mvtec_$cur_model_name"
# DAGM
base_command="python test.py --devices $device --epoch 1 --dino_model dinov2 --soft_mean True --features_list 24 --visualize False"
# Medicals
run_for_trained_on_mvtec "$base_command" brainmri headct br35h isic tn3k cvc-colondb cvc-clinicdb
