# Skin Lesion Classification: Data Augmentation and Visualization Pipeline  

This repository provides tools for preprocessing, splitting, augmenting, and visualizing image datasets for machine learning tasks, specifically designed for skin lesion classification using the HAM10000 dataset.  

## Features  
- **Dataset Splitting:** Automatically splits datasets into training, validation, and test sets with proportional representation by class.  
- **Image Augmentation:** Enhances dataset diversity with transformations such as rotation, flipping, cropping, color jitter, and noise addition.  
- **Visualization:** Generates bar charts, heatmaps, and sample image displays for dataset analysis.  

## File Overview  
- `augmentation_to_image.py`: Script for applying image augmentation to the training set and saving the results.  
- `dataset_visualization.ipynb`: Notebook for dataset preprocessing, splitting, visualization, and sample image display.  
- `.gitignore`: Excludes unnecessary files and directories from version control.  

## How to Use  

### 0. Clone the Repository and Download the Dataset  
- Clone the repository:  
  ```bash  
  git clone https://github.com/zxia545/csml-final-project  
  ```  
- Download the HAM10000 dataset from [Kaggle](https://www.kaggle.com/datasets/rauf41/skin-cancer-image-dataset).  

### 1. Dataset Splitting  
- Organize your dataset into `dataset/` with subfolders for each class.  
- Run `dataset_visualization.ipynb` to split the dataset into training, validation, and test sets.  

### 2. Image Augmentation  
- Augment the training data by running:  
  ```bash  
  python augmentation_to_image.py  
  ```  
- Augmented images will be saved in `split_data/augmented_train/`.  

### 3. Visualization  
- Use `dataset_visualization.ipynb` to:  
  - Generate bar charts and heatmaps for class distribution.  
  - Display random sample images from training, validation, and test sets.  

### 4. Model Training  
- Use `main.py` (if present) to train models. Adjust dataset paths in the script to match your directory structure.  

### 5. Evaluation  
- Evaluate models using `load_and_test.py`, ensuring it aligns with your trained model and test dataset structure.  

## Dependencies  
Install required Python libraries using:  
```bash  
pip install -r requirements.txt  
```  

## Directory Structure  
```
csml-final-project/
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
- Modify paths in scripts as necessary for your environment.  
- Ensure datasets are organized with subfolders for each class.  
- For vision-language model experiments (CLIP, BLIP), include appropriate pre-trained weights and scripts.  

## License  
This project is licensed under the MIT License.  

## Acknowledgments  
This repository supports the research in *"Exploring Vision-Language and Deep Learning Models for Skin Cancer Classification"* by Zheyuan Xiao, utilizing HAM10000 for experimentation with methods like Random Forest, ResNet, Custom CNNs, and vision-language models (CLIP, BLIP).  
