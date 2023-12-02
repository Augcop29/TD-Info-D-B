import numpy as np
from PIL import Image
import random
from collections import Counter

# Étape 1: Charger et analyser les couleurs de l'image
def load_and_analyze_image(image_path):
    img = Image.open(image_path)
    img_data = np.array(img)

    colors = Counter([tuple(colors) for row in img_data for colors in row])
    return colors, img

# Étape 5: Amélioration de la palette avec K-means
def k_means_clustering(data, k):
    centroids = data[random.sample(range(0, data.shape[0]), k)]
    while True:
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            min_distance_index = distances.index(min(distances))
            clusters[min_distance_index].append(point)

        new_centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters if cluster])
        if np.array_equal(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids

def improved_palette(img_data, k):
    reshaped_img_data = img_data.reshape(-1, 3)
    new_palette = k_means_clustering(reshaped_img_data, k)
    return new_palette.astype('uint8')

# Étape 3: Recolorier l'image
def find_nearest_color(color, palette):
    return min(palette, key=lambda x: sum((c1 - c2) ** 2 for c1, c2 in zip(color, x)))

def recolor_image(img_data, palette):
    recolored_img_data = np.array([[find_nearest_color(pixel, palette) for pixel in row] for row in img_data])
    recolored_img = Image.fromarray(recolored_img_data.astype('uint8'), 'RGB')
    return recolored_img

# Étape 4: Calcul de l'erreur
def calculate_error(original_img, recolored_img):
    original_data = np.array(original_img)
    recolored_data = np.array(recolored_img)
    error = np.mean((original_data - recolored_data) ** 2)
    return error

# Exécution du script
image_path = 'TD5/IMG_3135.png'
colors, original_img = load_and_analyze_image(image_path)
print("Couleurs uniques et leurs fréquences:", colors)

k = 8  # Nombre de couleurs dans la nouvelle palette
new_palette = improved_palette(np.array(original_img), k)
print("Nouvelle palette:", new_palette)

recolored_img = recolor_image(np.array(original_img), new_palette)
recolored_img.show()

error = calculate_error(original_img, recolored_img)
print(f"Erreur: {error}")
