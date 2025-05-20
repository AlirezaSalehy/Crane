
# Capture start time
start_time=$(date +%s)
echo $start_time

python test.py --devices 0 --epochs 5 --mean_all_layers false --dino_model dinov2 --model_name trained_mvtec_default 

# Capture end time
end_time=$(date +%s)
echo $end_time

# Calculate elapsed time
elapsed=$((end_time - start_time))

# Print result
echo "Execution time: $elapsed seconds"