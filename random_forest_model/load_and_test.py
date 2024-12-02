import joblib
import numpy as np
import os
from skimage.feature import hog  # Import hog function
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load saved model
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'")
    model = joblib.load(model_path)
    print(f"Model loaded from '{model_path}'")
    return model

# Define a function to load test data
def load_test_data(test_dir, classes):
    from skimage import io, color
    import cv2
    from tqdm import tqdm

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

    features = []
    labels = []
    for cls_idx, cls in enumerate(classes):
        class_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(class_dir):
            print(f"Warning: Class directory not found: {class_dir}")
            continue
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        if not image_files:
            print(f"Warning: No image files found in: {class_dir}")
            continue
        print(f"Processing class '{cls}' with {len(image_files)} images.")
        for img_name in tqdm(image_files, desc=f"Processing {cls}", unit="image"):
            img_path = os.path.join(class_dir, img_name)
            try:
                # Read the image using skimage.io.imread
                image = io.imread(img_path)
                if image is None:
                    print(f"Error: Failed to read image: {img_path}")
                    continue
                # If image has an alpha channel, remove it
                if image.shape[-1] == 4:
                    image = color.rgba2rgb(image)
                # Extract features
                feature = feature_extractor(image)
                features.append(feature)
                labels.append(cls_idx)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
    if not features:
        raise ValueError("No features extracted. Please check your dataset and feature extraction functions.")
    return np.array(features), np.array(labels)

# Plot confusion matrix function
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', normalize=False, save_path='confusion_matrix.png'):
    plt.figure(figsize=(10, 8))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.savefig(save_path)
    print(f"Confusion matrix saved to '{save_path}'")
    plt.show()

# Main function to load the model, test data, evaluate, and visualize results
def main():
    # Define classes (same as the training set)
    classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    
    # Paths (update these paths accordingly)
    model_path = 'random_forest_pipeline.pkl'  # Model save path
    test_dir = '/data/huzhengyu/github_repo/tony_csml/csml-final-project/split_data/test'  # Test dataset directory
    
    # Load the model
    model = load_model(model_path)
    
    # Load test data
    print("Loading test data...")
    X_test, y_test = load_test_data(test_dir, classes)
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Evaluate on the test set
    print("Evaluating on test set...")
    y_test_pred = model.predict(X_test)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    # Print metrics
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    print(classification_report(y_test, y_test_pred, target_names=classes, zero_division=0))
    
    # Confusion Matrix - Test Set
    cm_test = confusion_matrix(y_test, y_test_pred)
    plot_confusion_matrix(cm_test, classes, 'Confusion Matrix - Test Set', normalize=True, save_path='confusion_matrix_rf.png')

if __name__ == '__main__':
    main()
