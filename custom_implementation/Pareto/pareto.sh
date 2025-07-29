for SPLIT_INDEX in {1..18}; do
    echo "🚀 Starting run for split index: $SPLIT_INDEX"

    # --- Start Server ---
    sshpass -p '123456' ssh -o StrictHostKeyChecking=no cc@10.100.117.4 \
        "python3 /home/cc/pareto/partition/part1_mobile_200.py --split-index $SPLIT_INDEX" &

    # Give the server a moment to start
    sleep 100

    # --- Start Client ---
    sshpass -p '1976@1976' ssh -o StrictHostKeyChecking=no new_username@10.100.117.1 \
        "source ~/miniconda3/etc/profile.d/conda.sh && conda activate torch121 && python /home/new_username/pareto/partition/delay/part2_mobile_200.py --host 10.100.117.4 --split-index $SPLIT_INDEX --output-dir '/home/new_username/pareto/partition/delay/mobilenetv2_G' --network-delay-ms 0.094" &

    # Wait for the client/server pair to finish before starting the next loop
    wait
    echo "✅ Completed run for split index: $SPLIT_INDEX"
    echo "------------------------------------"
done

echo "🎉 All runs completed."
