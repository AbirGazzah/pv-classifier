# pv-classifier
This repository contains Python code for training and validating a PV cell defect classification model. The dataset used contains three classes.

### Installation
PyTorch and TorchVision (with CUDA 11.5) were used in this project. The code also runs on CPU.

```bash
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install argparse
pip install torch==1.11.0 torchvision==0.12.0 --index-url https://download.pytorch.org/whl/cu115
```

### Running the Code
The code comprises three python scripts:
- model_EfficientnetB2.py: Contains the classifier definition
- train_EfficienetB2.py: Is the main script. It contains the training and validation
code.
- utils.py: Contains several function definitions, such as the training loop and
metrics calculation.

To run the training and validation process:
```bash
python train_EfficienetB2.py --epochs [number_of_epochs] --lr [learning_rate_value] --dataset-path [dataset_path] --out-dir [path_to_output_directory]
```
