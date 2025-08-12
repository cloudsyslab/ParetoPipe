# Model Splitting Guide for Distributed Inference

## Overview
This document explains how different neural network architectures need to be split for distributed inference, and the specific considerations for each model type.

## Key Principles
1. **Preserve architectural integrity** - Don't split in the middle of residual blocks, inception modules, or other architectural units
2. **Respect data dependencies** - Ensure skip connections and parallel branches stay together
3. **Balance compute load** - Try to split at points that create roughly equal workloads
4. **Minimize communication** - Prefer splits after pooling layers or where feature maps are smaller

## Model-Specific Splitting Rules

### 1. MobileNetV2
**Architecture**: Sequential blocks of InvertedResidual layers
**Split Strategy**: Can split between any InvertedResidual blocks
**Key Considerations**:
- Each InvertedResidual block is self-contained
- No skip connections between different blocks
- Currently uses split_0 which works correctly

### 2. ResNet18/ResNet50
**Architecture**: Residual blocks with skip connections
**Split Strategy**: MUST split only between complete residual blocks
**Key Considerations**:
- Each residual block contains: conv→bn→relu→conv→bn + skip connection
- The skip connection (downsample) must stay with its block
- Layer naming pattern: `layer{N}.{M}.{component}` where N=layer group, M=block number
- Valid split points:
  - Between layer1 and layer2
  - Between layer2 and layer3
  - Between layer3 and layer4
  - Between layer4 and avgpool
  - After complete blocks within a layer (e.g., after layer1.0, layer1.1)

**What went wrong**: The original splitter was cutting at layer 60 for ResNet50, which fell in the middle of a residual block, breaking the skip connection.

### 3. VGG16
**Architecture**: Simple sequential model with conv→relu→pool patterns
**Split Strategy**: Can split after any ReLU or MaxPool layer
**Key Considerations**:
- No skip connections or complex dependencies
- Best to split after pooling layers to minimize data transfer
- Features section has clear boundaries at pooling layers
- Classifier section should generally stay together

### 4. AlexNet
**Architecture**: Similar to VGG, sequential conv→relu→pool
**Split Strategy**: Can split after ReLU or MaxPool layers
**Key Considerations**:
- Simple sequential architecture
- Features and classifier sections
- No complex dependencies between layers

### 5. InceptionV3
**Architecture**: Complex parallel branches within "Mixed" modules
**Split Strategy**: MUST split only between complete Mixed modules
**Key Considerations**:
- Each Mixed module has multiple parallel branches (1x1, 3x3, 5x5, pool)
- All branches within a module must stay together
- Layer naming pattern: `Mixed_{ID}.branch{TYPE}.{component}`
- Valid split points:
  - Between different Mixed modules (e.g., after Mixed_5d, before Mixed_6a)
  - After the stem layers before first Mixed module
  - Before/after auxiliary classifiers if present

**What went wrong**: The original splitter was cutting in the middle of Mixed_6a module, separating its parallel branches.

## Implementation Details

### Modified `intelligent_splitter.py`
Added three key methods:

1. `_in_same_resnet_block()` - Checks if two layers belong to the same residual block
2. `_in_same_inception_module()` - Checks if two layers belong to the same inception module  
3. Updated `_is_architectural_boundary()` - Only allows splits at proper boundaries for each architecture

### Testing Strategy
1. Generate shards for each model
2. Load shards locally and verify:
   - Shapes match between shard outputs and inputs
   - No missing layers
   - No architectural units are split
3. Test with actual distributed inference
4. Monitor for dimension mismatch errors

## Common Error Patterns
- `mat1 and mat2 shapes cannot be multiplied` - Usually means a residual block or inception module was split
- `running_mean should contain X elements not Y` - BatchNorm layer expects different channels, indicating architectural mismatch
- `expected input[B,C1,H,W] to have C2 channels` - Conv layer expects different input channels than provided