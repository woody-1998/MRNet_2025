# MRNet_2025
## requirement
monai
pytorch
torchvision

## project structure
MRNet_2025/
├── Dataloader/
│ ├── figs/ # Visualization or figure assets used during dataloading
│ └── utils/ # Dataloader utilities
│ └── init.py
│
├── Dataset/
│ ├── MRNet-v1.0/ # Unzipped MRNet dataset
│ └── MRNet-v1.0.zip # Original dataset archive
│
├── Model/
│ ├── figs/ # Architecture figures or attention maps
│ ├── utils/ # Model utility files
│ ├── AlexNet.py # AlexNet with optional Slice Attention and BN
│ └── ResNet18CBAM.py # ResNet18 architecture with CBAM (spatial/channel attention)
│
├── Tool/
│ ├── alexnet_checkpoints/ # Saved checkpoints for AlexNet models
│ ├── figs/ # Training figures (loss/accuracy curves)
│ ├── test1/ # Test outputs, intermediate runs
│ │ ├── init.py
│ │ ├── accuracy.png
│ │ ├── loss.png
│ │ └── AlexNet_SliceAttention_50_epochs.ipynb
│ ├── EDA_augmentation_loader_25_epochs.ipynb
│ ├── EDA_augmentation_loader_100_epochs.ipynb
│ └── train.py # Main training script
│
└── README.md # Project documentation

## command example
 python3 train.py --model_saving optimal --epochs 1 --device mps --checkpoint_path test1 --batch_size 1