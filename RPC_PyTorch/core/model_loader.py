#!/usr/bin/env python3
"""
Model loading utilities for the enhanced distributed inference system.
Handles loading and preparing various model architectures.
"""

import torch
import torch.nn as nn
import torchvision.models as torchvision_models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import logging
from typing import Tuple, Dict, Any
import os


class ModelLoader:
    """Handles loading and preparing models for distributed inference."""
    
    SUPPORTED_MODELS = {
        "mobilenetv2": {
            "model_class": torchvision_models.mobilenet_v2,
            "weight_file": "mobilenetv2_100epochs_jul15.pth",
            "input_size": (224, 224),
            "last_channel_attr": "last_channel"
        },
        "inceptionv3": {
            "model_class": torchvision_models.inception_v3,
            "weight_file": "inception_100epochs_jul16.pth",
            "input_size": (299, 299),
            "init_kwargs": {"aux_logits": False}
        },
        "alexnet": {
            "model_class": torchvision_models.alexnet,
            "weight_file": "alexnet_100epochs_jul16.pth",
            "input_size": (224, 224)
        },
        "resnet18": {
            "model_class": torchvision_models.resnet18,
            "weight_file": "resnet18_100epochs_jul16.pth",
            "input_size": (224, 224)
        },
        "resnet50": {
            "model_class": torchvision_models.resnet50,
            "weight_file": "resnet50_100epochs_jul17.pth",
            "input_size": (224, 224)
        },
        "vgg16": {
            "model_class": torchvision_models.vgg16,
            "weight_file": "vgg16_100epochs_jul16.pth",
            "input_size": (224, 224)
        }
    }
    
    def __init__(self, models_dir: str = "./models"):
        """
        Initialize model loader.
        
        Args:
            models_dir: Directory containing model weight files
        """
        self.models_dir = models_dir
        self.logger = logging.getLogger(__name__)
    
    def load_model(self, model_type: str, num_classes: int = 10) -> nn.Module:
        """
        Load a model with pre-trained weights.
        
        Args:
            model_type: Type of model to load
            num_classes: Number of output classes
            
        Returns:
            Loaded PyTorch model
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_type}. Supported models: {list(self.SUPPORTED_MODELS.keys())}")
        
        model_config = self.SUPPORTED_MODELS[model_type]
        
        # Create model instance
        init_kwargs = model_config.get("init_kwargs", {})
        model = model_config["model_class"](weights=None, **init_kwargs)
        
        # Modify classifier for CIFAR-10
        self._modify_classifier(model, model_type, num_classes)
        
        # Load pre-trained weights
        weight_file = os.path.join(self.models_dir, model_config["weight_file"])
        if os.path.exists(weight_file):
            try:
                state_dict = torch.load(weight_file, map_location="cpu")
                model.load_state_dict(state_dict)
                self.logger.info(f"Loaded weights for {model_type} from {weight_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load weights for {model_type}: {e}")
        else:
            self.logger.warning(f"Weight file not found: {weight_file}")
        
        model.eval()
        return model
    
    def _modify_classifier(self, model: nn.Module, model_type: str, num_classes: int):
        """Modify the classifier layer for the specified number of classes."""
        if model_type == "mobilenetv2":
            model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        
        elif model_type == "inceptionv3":
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        elif model_type == "alexnet":
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        
        elif model_type == "resnet18":
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        elif model_type == "resnet50":
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        elif model_type == "vgg16":
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    
    def get_input_transform(self, model_type: str) -> transforms.Compose:
        """
        Get the appropriate input transform for a model.
        
        Args:
            model_type: Type of model
            
        Returns:
            Transform pipeline
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_type}")
        
        input_size = self.SUPPORTED_MODELS[model_type]["input_size"]
        
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def get_sample_input(self, model_type: str, batch_size: int = 1) -> torch.Tensor:
        """
        Get a sample input tensor for a model.
        
        Args:
            model_type: Type of model
            batch_size: Batch size for the sample
            
        Returns:
            Sample input tensor
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_type}")
        
        input_size = self.SUPPORTED_MODELS[model_type]["input_size"]
        return torch.randn(batch_size, 3, input_size[0], input_size[1])
    
    def load_dataset(self, dataset_name: str, model_type: str, 
                    batch_size: int = 16, dataset_path: str = None) -> torch.utils.data.DataLoader:
        """
        Load a dataset with appropriate transforms.
        
        Args:
            dataset_name: Name of dataset ("cifar10" or "dummy")
            model_type: Type of model (for transform selection)
            batch_size: Batch size for DataLoader
            dataset_path: Path to dataset (for CIFAR-10)
            
        Returns:
            DataLoader instance
        """
        transform = self.get_input_transform(model_type)
        
        if dataset_name == "cifar10":
            if dataset_path is None:
                dataset_path = os.getenv("CIFAR10_PATH", "/export/datasets/cifar10")
            
            test_dataset = datasets.CIFAR10(
                root=dataset_path,
                train=False,
                download=False,
                transform=transform
            )
            
            return torch.utils.data.DataLoader(
                test_dataset, 
                batch_size=batch_size, 
                shuffle=True
            )
        
        elif dataset_name == "dummy":
            class DummyDataset(torch.utils.data.Dataset):
                def __init__(self, num_samples: int, input_shape: tuple, num_classes: int):
                    self.num_samples = num_samples
                    self.input_shape = input_shape
                    self.num_classes = num_classes
                
                def __len__(self):
                    return self.num_samples
                
                def __getitem__(self, idx):
                    # Generate random image and label
                    image = torch.randn(*self.input_shape)
                    label = torch.randint(0, self.num_classes, (1,)).item()
                    return image, label
            
            input_size = self.SUPPORTED_MODELS[model_type]["input_size"]
            dummy_dataset = DummyDataset(
                num_samples=1000,
                input_shape=(3, input_size[0], input_size[1]),
                num_classes=10
            )
            
            return torch.utils.data.DataLoader(
                dummy_dataset,
                batch_size=batch_size,
                shuffle=True
            )
        
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """
        Get information about a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary with model information
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_type}")
        
        config = self.SUPPORTED_MODELS[model_type]
        return {
            "model_type": model_type,
            "input_size": config["input_size"],
            "weight_file": config["weight_file"],
            "supported": True
        }
    
    @classmethod
    def list_supported_models(cls) -> list:
        """Get list of supported model types."""
        return list(cls.SUPPORTED_MODELS.keys())