import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

DATASET_PATH = r"C:\Users\VIRUS\Desktop\Study-PLAN\Machine-Learning\Jan2026\260202-road-signs\road-signs"
IMG_SIZE = (50, 50)

def load_data(data_path, grayscale=False):
    images = []
    labels = []
    class_names = []

    class_names = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    print(f"Found {len(class_names)} classes.")

    for class_index, class_name in enumerate(class_names):
        class_path = os.path.join(data_path, class_name)

        for split in ['train', 'test']:
            split_path = os.path.join(class_path, split)
            if not os.path.exists(split_path):
                continue

            for img_file in os.listdir(split_path):
                img_path = os.path.join(split_path, img_file)
                img = cv2.imread(img_path)

                if img is None:
                    continue

                img = cv2.resize(img, IMG_SIZE)

                if grayscale:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                images.append(img)
                labels.append(class_index)

    return np.array(images), np.array(labels), class_names

print("Loading Color Data...")
X_color, y, class_names = load_data(DATASET_PATH, grayscale=False)

print("Loading Grayscale Data...")
X_gray, _, _ = load_data(DATASET_PATH, grayscale=True)

n_samples_color = X_color.shape[0]
X_color_flat = X_color.reshape(n_samples_color, -1) / 255.0

n_samples_gray = X_gray.shape[0]
X_gray_flat = X_gray.reshape(n_samples_gray, -1) / 255.0

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_color_flat, y, test_size=0.2, random_state=42
)

Xg_train, Xg_test, yg_train, yg_test = train_test_split(
    X_gray_flat, y, test_size=0.2, random_state=42
)

print(f"Data Ready. Train shape: {Xc_train.shape}")

print("\n--- Training Color Model ---")
mlp_color = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42, verbose=True)
mlp_color.fit(Xc_train, yc_train)
pred_color = mlp_color.predict(Xc_test)
acc_color = accuracy_score(yc_test, pred_color)
print(f"Color Model Accuracy: {acc_color*100:.2f}%")

print("\n--- Training Grayscale Model ---")
mlp_gray = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42, verbose=True)
mlp_gray.fit(Xg_train, yg_train)
pred_gray = mlp_gray.predict(Xg_test)
acc_gray = accuracy_score(yg_test, pred_gray)
print(f"Grayscale Model Accuracy: {acc_gray*100:.2f}%")

plt.figure(figsize=(10, 5))
for i in range(4):
    plt.subplot(2, 4, i + 1)
    plt.imshow(Xc_test[i].reshape(IMG_SIZE[0], IMG_SIZE[1], 3))
    plt.title(f"Pred: {class_names[pred_color[i]]}")
    plt.axis('off')

    plt.subplot(2, 4, i + 5)
    plt.imshow(Xg_test[i].reshape(IMG_SIZE[0], IMG_SIZE[1]), cmap='gray')
    plt.title(f"Pred: {class_names[pred_gray[i]]}")
    plt.axis('off')

plt.suptitle("Comparison: Color vs Grayscale Predictions")
plt.show()

print("\nFinal Comparison:")
print(f"Color Accuracy:    {acc_color:.2%}")
print(f"Grayscale Accuracy:{acc_gray:.2%}")