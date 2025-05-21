model_name=$1
device="0"
# Use bellow alternatively for the woDattn model
# base_command="python test.py --device $device --epoch 5 --dino_model none --features_list 6 12 18 24"
base_command="python test.py --device $device --epoch 5 --dino_model dinov2"

for dataset in visa mpdd sdd btad dtd; do
    command="$base_command --dataset $dataset --model_name trained_on_mvtec_$model_name"
    eval $command
done

command="$base_command --dataset mvtec --model_name trained_on_visa_$model_name"
eval $command
command="$base_command --dataset dagm --model_name trained_on_mvtec_$model_name"
eval $command

# Use bellow alternatively for the woDattn model
# base_command="python test.py --device $device --epoch 5 --dino_model none --features_list 6 12 18 24 --soft_mean True"
base_command="python test.py --device $device --epoch 1 --dino_model dinov2 --soft_mean True"
for dataset in brainmri headct br35h isic tn3k cvc-colondb cvc-clinicdb; do
    command="$base_command --dataset $dataset --model_name trained_on_mvtec_$model_name"
    eval $command
done 