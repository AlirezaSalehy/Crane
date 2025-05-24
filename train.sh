model_name=$1
device=$2

# Table 1 and 2 training scheme
python train.py --model_name $1 --dataset mvtec --device $2 --why "Evalution purpose"
python train.py --model_name $1 --dataset visa --device $2 --why "Evalution purpose"

# To test it
bash test.sh $1 $2
