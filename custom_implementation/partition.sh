# This script automates running the split inference experiment for Models
# The loop iterates from 1 to n, which is the valid range of split indices for different model's feature layers.
for SPLIT_INDEX in {1..n}; do
    echo "🚀 Starting run for Models(MobilenetV2, AlexNet, ResNet18,....) with split index: $SPLIT_INDEX"
    # partition 1
    sshpass -p '123456' ssh -o StrictHostKeyChecking=no cc@10.100.117.4 \
        "python3 /home/cc/pareto/partition/partition1.py --model-name MobileNetV2 --split-index $SPLIT_INDEX" &.
    # Reduced from 100s to 5s, which is usually sufficient and much faster.
    sleep 5

    #partition 2 ---
    sshpass -p '1976@1976' ssh -o StrictHostKeyChecking=no new_username@10.100.117.1 \
        "source ~/miniconda3/etc/profile.d/conda.sh && conda activate torch121 && python /home/new_username/pareto/partition/delay/partition2.py --model-name MobileNetV2 --host 10.100.117.4 --split-index $SPLIT_INDEX --output-dir '/home/new_username/pareto/partition/delay/mobilenetv2_G' --network-delay-ms 0.094" &

    # Wait for both the server and client background processes to finish before starting the next iteration.
    wait
    echo "✅ Completed run for split index: $SPLIT_INDEX"
    echo "------------------------------------"
done

echo "🎉 All runs completed."
