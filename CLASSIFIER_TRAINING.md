# Plastic/Non-Plastic Classifier Training Guide

This guide explains how to train a custom classifier model using your labeled dataset to improve plastic/non-plastic classification accuracy.

## Dataset Preparation

### Option 1: Simple Folder Structure (Recommended)

Organize your dataset in the following structure:

```
data/
  classifier/
    plastic/
      image1.jpg
      image2.jpg
      image3.jpg
      ...
    non_plastic/
      image1.jpg
      image2.jpg
      image3.jpg
      ...
```

**What images should you include?**

- **Plastic folder**: Images of plastic items (bottles, bags, covers, containers, etc.)
- **Non-plastic folder**: Images of non-plastic items (wood, metal, glass, organic materials, etc.)

**Tips:**
- Use cropped images of individual debris items (not full scene images)
- Include diverse examples: different colors, sizes, lighting conditions
- Aim for at least 50-100 images per class (more is better)
- Images can be of any size (they will be resized during training)

### Option 2: Using Detected Debris Regions

If you have already detected debris using the main YOLO model, you can:
1. Extract bounding box regions from detected debris
2. Manually label them as plastic or non-plastic
3. Save them in the appropriate folders

## Training the Classifier

### Step 1: Prepare Your Dataset

1. Create the folder structure:
   ```bash
   mkdir -p data/classifier/plastic
   mkdir -p data/classifier/non_plastic
   ```

2. Copy your labeled images into the respective folders:
   - Plastic images → `data/classifier/plastic/`
   - Non-plastic images → `data/classifier/non_plastic/`

### Step 2: Update Configuration (Optional)

Edit `configs/config.yaml` to customize training parameters:

```yaml
classifier:
  plastic_dir: "data/classifier/plastic"
  non_plastic_dir: "data/classifier/non_plastic"
  model_path: "models/classifier/plastic_classifier_best.pt"
  epochs: 20          # Number of training epochs
  batch_size: 32     # Batch size
  learning_rate: 0.001
  device: auto       # auto, cuda, or cpu
```

### Step 3: Run Training

Activate your virtual environment and run:

```bash
.venv\Scripts\python src/train_classifier.py
```

The training script will:
- Load images from both folders
- Split into training (80%) and validation (20%) sets
- Train a ResNet18-based classifier
- Save the best model to `models/classifier/plastic_classifier_best.pt`

### Step 4: Use the Trained Model

Once training is complete, the inference code will automatically:
- Detect the trained classifier model
- Use it instead of heuristic-based classification
- Provide more accurate plastic/non-plastic predictions

**Note:** If no trained model is found, the system will fall back to heuristic-based classification.

## Training Output

After training, you'll see:
- Training progress with loss and accuracy metrics
- Best model saved to `models/classifier/plastic_classifier_best.pt`
- Training info saved to `models/classifier/classifier_info.json`

## Tips for Better Accuracy

1. **More Data = Better Model**: Aim for 100+ images per class
2. **Diverse Examples**: Include various colors, shapes, lighting conditions
3. **Balanced Dataset**: Try to have similar numbers of plastic and non-plastic images
4. **Quality Images**: Use clear, well-lit images of individual items
5. **Augmentation**: The training script automatically applies data augmentation (flips, rotations, color jitter)

## Troubleshooting

**Error: "No images found"**
- Check that your folder paths in `config.yaml` are correct
- Ensure images are in `.jpg`, `.jpeg`, `.png`, or `.bmp` format

**Low accuracy**
- Add more training images
- Ensure images are properly labeled
- Try increasing epochs or adjusting learning rate

**Out of memory**
- Reduce batch_size in config
- Use smaller images or reduce image resolution

## Example Workflow

1. Collect debris images and label them
2. Organize into `plastic/` and `non_plastic/` folders
3. Run training: `python src/train_classifier.py`
4. Test in the Streamlit app - it will automatically use the trained model!

