import os
import cv2
import numpy as np
from skimage import io, color
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # For progress bars
import logging
import joblib  # Import joblib for model saving and loading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Define feature extraction functions
def extract_hog_features(image):
    image_gray = color.rgb2gray(image)
    features = hog(
        image_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,  # Set to False
        block_norm='L2-Hys'
    )
    return features

def extract_color_histogram(image, bins=(8, 8, 8)):
    # Compute a 3D color histogram in RGB color space
    hist = cv2.calcHist(
        [image], [0, 1, 2], None, bins,
        [0, 256, 0, 256, 0, 256]
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def feature_extractor(image):
    hog_feat = extract_hog_features(image)
    color_hist = extract_color_histogram(image)
    # Concatenate features
    combined_features = np.concatenate([hog_feat, color_hist])
    return combined_features

def load_dataset(image_dir, classes):
    features = []
    labels = []
    for cls_idx, cls in enumerate(classes):
        class_dir = os.path.join(image_dir, cls)
        if not os.path.isdir(class_dir):
            logging.warning(f"Class directory not found: {class_dir}")
            continue
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        if not image_files:
            logging.warning(f"No image files found in: {class_dir}")
            continue
        logging.info(f"Processing class '{cls}' with {len(image_files)} images.")
        for img_name in tqdm(image_files, desc=f"Processing {cls}", unit="image"):
            img_path = os.path.join(class_dir, img_name)
            try:
                # Read the image using skimage.io.imread
                image = io.imread(img_path)
                if image is None:
                    logging.error(f"Failed to read image: {img_path}")
                    continue
                # If image has an alpha channel, remove it
                if image.shape[-1] == 4:
                    image = color.rgba2rgb(image)
                # Extract features
                feature = feature_extractor(image)
                features.append(feature)
                labels.append(cls_idx)
            except Exception as e:
                logging.error(f"Error processing image {img_path}: {e}")
    if not features:
        logging.error("No features extracted. Please check your dataset and feature extraction functions.")
    return np.array(features), np.array(labels)

def plot_confusion_matrix(cm, classes, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.show()

def main():
    # Define classes
    classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    
    # Paths (update these paths accordingly)
    train_dir = '/data/huzhengyu/github_repo/tony_csml/csml-final-project/split_data/train'
    val_dir = '/data/huzhengyu/github_repo/tony_csml/csml-final-project/split_data/validation'
    test_dir = '/data/huzhengyu/github_repo/tony_csml/csml-final-project/split_data/test'
    
    # Load datasets with progress bars
    # Load datasets
    logging.info("Loading training data...")
    X_train, y_train = load_dataset(train_dir, classes)
    logging.info("Loading validation data...")
    X_val, y_val = load_dataset(val_dir, classes)
    logging.info("Loading test data...")
    X_test, y_test = load_dataset(test_dir, classes)
    
    # Print shapes to verify
    logging.info(f"X_train shape: {X_train.shape}")
    logging.info(f"y_train shape: {y_train.shape}")
    logging.info(f"X_val shape: {X_val.shape}")
    logging.info(f"y_val shape: {y_val.shape}")
    logging.info(f"X_test shape: {X_test.shape}")
    logging.info(f"y_test shape: {y_test.shape}")
    
    # Check if datasets are non-empty
    if X_train.size == 0 or y_train.size == 0:
        logging.error("Training data is empty. Exiting the script.")
        return
    if X_val.size == 0 or y_val.size == 0:
        logging.error("Validation data is empty. Exiting the script.")
        return
    if X_test.size == 0 or y_test.size == 0:
        logging.error("Test data is empty. Exiting the script.")
        return
    
    # Combine training and validation data for better cross-validation
    X_combined = np.vstack((X_train, X_val))
    y_combined = np.hstack((y_train, y_val))
    
    # Resample training data to handle class imbalance
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Define the pipeline
    from imblearn.pipeline import Pipeline as ImbPipeline
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=500, random_state=42)),
        ('rf', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            verbose=2,
            n_jobs=-1
        ))
    ])
    
    # Train the model
    logging.info("Training the Random Forest model...")
    pipeline.fit(X_train_resampled, y_train_resampled)
    logging.info("Model training completed.")

    # Save the trained model
    model_save_path = 'random_forest_pipeline.pkl'  # Define your desired save path
    joblib.dump(pipeline, model_save_path)
    logging.info(f"Trained model saved to '{model_save_path}'.")
    
    # Evaluate on validation set
    logging.info("Evaluating on validation set...")
    y_val_pred = pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    logging.info(f"Validation Accuracy: {val_accuracy:.4f}")
    logging.info(f"Validation F1-Score: {val_f1:.4f}")
    print(classification_report(y_val, y_val_pred, target_names=classes, zero_division=0))
    
    
    
    # Confusion Matrix - Validation Set
    cm_val = confusion_matrix(y_val, y_val_pred)
    plot_confusion_matrix(cm_val, classes, 'Confusion Matrix - Validation Set')
    
    # Evaluate on test set
    logging.info("Evaluating on test set...")
    y_test_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    logging.info(f"Test Accuracy: {test_accuracy:.4f}")
    logging.info(f"Test F1-Score: {test_f1:.4f}")
    print(classification_report(y_test, y_test_pred, target_names=classes, zero_division=0))
    
    
    # Confusion Matrix - Test Set
    cm_test = confusion_matrix(y_test, y_test_pred)
    plot_confusion_matrix(cm_test, classes, 'Confusion Matrix - Test Set')

if __name__ == '__main__':
    main()
