# Data Augmentation and Visualization Pipeline

This repository provides tools for preprocessing, splitting, augmenting, and visualizing image datasets for machine learning tasks.

## Features

- **Dataset Splitting**: Automatically splits datasets into training, validation, and test sets.
- **Image Augmentation**: Applies transformations like rotation, flipping, cropping, color jitter, and noise addition.
- **Visualization**: Generates bar charts, heatmaps, and sample image displays for dataset analysis.

## File Overview

- `augmentation_to_image.py`: Script for augmenting images in the training set and saving the results.
- `dataset_visualization.ipynb`: Notebook for splitting datasets, generating visualizations, and displaying sample images.
- `.gitignore`: Specifies files and directories to exclude from version control.

## How to Use

### 0. Download Dataset

Use the `dataset_visualization.ipynb` notebook to:

* Download dataset

### 1. Dataset Splitting

1. Place your dataset in the `dataset/` folder, structured with subfolders for each class.
2. Run `dataset_visualization.ipynb` to split the dataset into `train`, `validation`, and `test` sets.

### 2. Image Augmentation

Run the `augmentation_to_image.py` script:

```bash
python augmentation_to_image.py
```

Augmented images will be saved in `split_data/augmented_train`.

### 3. Visualization

Use the `dataset_visualization.ipynb` notebook to:

- Generate bar charts and heatmaps for dataset distribution.
- Display random sample images from each dataset split.

### 4. Training

For folders containing `main.py`, use the script to train models and generate outputs.

- **Note**: Ensure the dataset paths in `main.py` are adjusted according to your directory structure before running.

### 5. Evaluation

To evaluate the results:

1. Run the `load_and_test.py` script in the respective folders.
2. This script loads the trained model and evaluates its performance on the test dataset.

## Dependencies

Install the required Python libraries with:

```bash
pip install -r requirements.txt
```

## Directory Structure

```
code_repo/
├── augmentation_to_image.py
├── dataset_visualization.ipynb
├── .gitignore
├── requirements.txt
├── split_data/
│   ├── train/
│   ├── validation/
│   ├── test/
│   ├── augmented_train/
└── dataset/
    └── Labeled Data/
```

## Notes

- Modify directory paths in scripts as needed for your environment.
- Ensure the dataset folder is organized with subfolders for each class.

## License

This project is licensed under the MIT License.
