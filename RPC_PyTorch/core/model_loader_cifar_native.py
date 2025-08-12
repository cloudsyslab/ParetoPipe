"""
Model loader with native CIFAR-10 32x32 input size for massive speedup.
"""

from .model_loader import ModelLoader
import torchvision.transforms as transforms

class ModelLoaderCifarNative(ModelLoader):
    """Model loader that uses native CIFAR-10 32x32 size instead of resizing to 224x224."""
    
    # Override with CIFAR-10 native sizes
    SUPPORTED_MODELS = {
        "mobilenetv2": {
            "model_class": ModelLoader.SUPPORTED_MODELS["mobilenetv2"]["model_class"],
            "weight_file": "mobilenetv2_cifar10.pth",
            "input_size": (32, 32),  # Native CIFAR-10 size!
            "last_channel_attr": "last_channel"
        },
        "resnet18": {
            "model_class": ModelLoader.SUPPORTED_MODELS["resnet18"]["model_class"],
            "weight_file": "resnet18_cifar10.pth",
            "input_size": (32, 32),  # Native CIFAR-10 size!
        },
        "resnet50": {
            "model_class": ModelLoader.SUPPORTED_MODELS["resnet50"]["model_class"],
            "weight_file": "resnet50_100epochs_jul17.pth",
            "input_size": (32, 32),  # Native CIFAR-10 size!
        },
        # Keep others at original size for compatibility
        "inceptionv3": ModelLoader.SUPPORTED_MODELS["inceptionv3"],
        "alexnet": ModelLoader.SUPPORTED_MODELS["alexnet"],
        "vgg16": ModelLoader.SUPPORTED_MODELS["vgg16"]
    }
    
    def get_input_transform(self, model_type: str):
        """Get input transform - no resize for CIFAR-10 native models!"""
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_type}")
        
        input_size = self.SUPPORTED_MODELS[model_type]["input_size"]
        
        # For 32x32 models, skip resize entirely
        if input_size == (32, 32):
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            # Fall back to parent class for other models
            return super().get_input_transform(model_type)