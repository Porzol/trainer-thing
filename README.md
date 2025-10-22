# Driver Distraction Classification Training System

A comprehensive training system for driver distraction classification using multiple deep learning architectures.


## Dataset Structure

The system expects data in the following structure:
```
├── C1_Drive_Safe/
├── C2_Sleep/
├── ... (22 classes total)
├── driver_train.txt
├── driver_val.txt
└── driver_test.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. **Create a configuration file** (see `config_example.json`):
```json
{
  "run_name": "experiment_name",
  "model": "efficientnet",
  "batch_size": 32,
  "epochs": 100,
  "learning_rate": 0.001,
  "patience": 10,
  "dropout": 0.5,
  "checkpoint_interval": 5,
  "activation_function": "relu"
}
```

2. **Run training**:
```bash
# Train from the created config file
python train.py your_config.json

# Force CPU usage for testing (even if GPU is available)
python train.py your_config.json --cpu

```

## Configuration Parameters

- `run_name`: Unique identifier for the experiment
- `model`: Model architecture __("efficientnet", "mobilenetv3", "regnet", "resnetv2")__
- `batch_size`: Training batch size in powers of 2, __(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)__
- `epochs`: Maximum number of training epochs __(50 is a good start)__
- `learning_rate`: Initial learning rate in negative exponents of 10, __(0.01, 0.001, 0.0001)__
- `patience`: Early stopping patience when no improvement __(10% of epoch is ok)__
- `dropout`: Dropout rate for regularization __(0.5 is fine)__
- `checkpoint_interval`: How often to save model checkpoints, __(5% of epoch is ok)__
- `activation_function`: Activation function __("relu", "gelu", "swish", "leaky_relu")__
- `dataset_path`: Path to dataset directory, point to either Cam2 or Cam4

## Output Structure

Results are saved in `results/{run_name}_{timestamp}/`:
```
results/experiment_20241002_143022/
├── epoch_05/
│   ├── model.pth
│   ├── model.onnx
│   └── visualizations/
├── epoch_10/
├── best/
│   ├── model.pth
│   ├── model.onnx
│   └── visualizations/
├── final_results/
│   ├── test_confusion_matrix.png
│   └── test_per_class_metrics.png
└── training_stats.json
```

## Supported Models

1. **EfficientNet-B0**: Efficient architecture with compound scaling
2. **MobileNetV3-Small**: Lightweight model optimized for mobile deployment
3. **RegNetY-200MF**: Regular network with optimized design space
4. **ResNet-18**: Deep residual network with improved training

## Metrics and Visualizations

The system automatically generates:
- **Training Progress**: Loss, accuracy, F1-score, precision, recall over time
- **Confusion Matrix**: Detailed classification performance per class
- **Per-Class Metrics**: Precision and recall breakdown by class
- **Sample Predictions**: Visual examples of model predictions on test data

All metrics are saved in JSON format for further analysis.