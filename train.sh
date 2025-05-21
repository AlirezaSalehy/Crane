model_name=$1
device=$2

# Table 1 and 2 training scheme
python train.py --model_name $1 --dataset mvtec --device $3 --why "Evalution purpose"
python train.py --model_name $1 --dataset visa --device $3 --why "Evalution purpose"

# Table 5 (Appendix D) for medical visualizations
python train.py --model_name $1 --dataset cvc-colondb endo-cls-test --device $3 --why "Evalution purpose"
python train.py --model_name $1 --dataset cvc-clinicdb endo-cls-test --device $3 --why "Evalution purpose"

# To test it
bash test.sh $1 mvtec visa
