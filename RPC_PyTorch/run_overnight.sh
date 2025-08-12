#!/bin/bash

# Simple overnight test runner
nohup bash -c '
python automated_split_tester.py --model resnet18 --runs 5 --wait-time 10
python automated_split_tester.py --model resnet50 --runs 5 --wait-time 10
python automated_split_tester.py --model alexnet --runs 5 --wait-time 10
python automated_split_tester.py --model mobilenetv2 --runs 5 --wait-time 10
python automated_split_tester.py --model inceptionv3 --runs 5 --wait-time 10
python automated_split_tester.py --model vgg16 --runs 5 --wait-time 10
' > overnight.log 2>&1 &

echo "Started tests in background. You can close terminal now."
echo "Check progress: tail -f overnight.log"
