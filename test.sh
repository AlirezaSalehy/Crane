model_name=$1
device="0"
base_command="python test.py --device $device --epoch 5"

for dataset in visa mpdd sdd btad dtd; do
    command="$base_command --dataset $dataset --model_name trained_on_mvtec_$model_name"
    eval $command
done

command="$base_command --dataset mvtec --model_name trained_on_visa_$model_name"
eval $command
command="$base_command --dataset dagm --model_name trained_on_mvtec_$model_name --soft_mean True"
eval $command

base_command="python test.py --device $device --epoch 1 --soft_mean True"
for dataset in brainmri headct br35h isic tn3k cvc-colondb cvc-clinicdb; do
    command="$base_command --dataset $dataset --model_name mvtec_$model_name"
    eval $command
done 