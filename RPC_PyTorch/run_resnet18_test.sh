#!/bin/bash
# Test ResNet18 with only split_0 (architectural split)
python automated_split_tester.py --model resnet18 --splits 0 --runs 5 --wait-time 45