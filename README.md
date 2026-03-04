# CNN for MNIST Digit Recognition

A simple Convolutional Neural Network implementation in PyTorch for MNIST handwritten digit classification.

## Files

- **`cnn.py`** - Core CNN model and training pipeline. Trains on MNIST dataset and generates:
  - Training loss curves
  - Test accuracy metrics  
  - Sample predictions visualization
  
- **`cnn_feature_extractor.py`** - Feature map visualization tool. Extracts and displays activations from convolutional layers to understand what features the network learns.

## Installation

```bash
pip install torch torchvision matplotlib
```

## Usage

**Train CNN:**
```bash
python cnn.py
```

**Visualize Feature Maps:**
```bash
python cnn_feature_extractor.py
python cnn_feature_extractor.py --epochs 2 --num-images 4 --maps-per-layer 4
```

## Outputs

Generated visualizations are saved to `outputs/`:
- `training_metrics.png` - Loss and accuracy curves
- `sample_predictions.png` - Model predictions on test images
- `cnn_feature_maps.png` - Feature map activations from conv layers

## Architecture

SimpleCNN: 2 conv layers (16→32 filters) + 2 FC layers (128→10 outputs)
