from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import random

#im = Image.open("TD5/IMG_1964.png")
#im = Image.open("TD5/IMG_3135.png")
im = Image.open("TD5/figures/color-rainbow.png")
im = im.convert("RGB")  # important pour bien avoir 3 couleurs
px = im.load()
w , h = im.size

#fonction pour faire la liste des couleurs d'une image et leur fréquence

def list_couleur(px , w , h) :
    lst = {}
    for x in range (w) :
        for y in range (h) :
            if px[x,y] in lst :
                lst[px[x,y]] += 1
            else :
                lst[px[x,y]] = 1
    return lst

#Distance euclidienne

def dist_eu2(c1,c2) :
    return ( (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2 )

#fonction qui prend les k couleurs les plus fréquentes d'une image et s'en sert comme palette

def palette_k(dict , k) :
    pal = []
    for i in range (k) :
        max = 0
        key_max = (0 , 0 , 0)
        for key in dict :
            if dict[key] > max :
                max = dict[key]
                key_max = key
        dict[key_max] = -1
        pal.append(key_max)
    return pal



# Amélioration de la fonction K-means
def Kmeans(dict, k):
    # Initialisation des centroids
    keys = list(dict.keys())
    centroids = [keys[i] for i in random.sample(range(len(keys)), k)]
    print (centroids)
    while True:
        clusters = [[] for _ in range(k)]
        new_centroids = []

        # Assignation des couleurs aux clusters
        for color in keys:
            distances = [dist_eu2(color, centroid) for centroid in centroids]
            closest_centroid_index = distances.index(min(distances))
            clusters[closest_centroid_index].append(color)

        # Mise à jour des centroids
        for cluster in clusters:
            if len(cluster) > 0:
                new_centroid = tuple(np.mean(cluster, axis=0).astype(int))
                new_centroids.append(new_centroid)
            else:
                # Dans le cas d'un cluster vide, reprendre un centroid au hasard
                new_centroids.append(keys[random.randint(0, len(keys) - 1)])

        # Vérification de la convergence
        if set(new_centroids) == set(centroids):
            break
        else:
            centroids = new_centroids

    return centroids

# Vous pouvez maintenant utiliser la fonction Kmeans pour créer une palette
#palette_kmeans = Kmeans(list_couleur(px, w, h), 8)

#print("Palette K-means:", palette_kmeans)



#Afficher la palette

def plot_color_palette(rgb_values):
    rgb_values = np.array(rgb_values) / 255.0

    custom_cmap = ListedColormap(rgb_values)

    data = np.arange(len(rgb_values)).reshape(1, -1)

    fig, ax = plt.subplots(figsize=(len(rgb_values), 1))

    cax = ax.pcolormesh(data, cmap=custom_cmap, edgecolors='w', linewidth=0.5)

    ax.set_yticks([])
    ax.set_xticks(np.arange(0.5, len(rgb_values), 1))
    ax.set_xticklabels([])

    plt.show()

#plot_color_palette(palette_kmeans)


def recolorier_pal(im , pal) :
    im.show()
    erreur = 0
    px = im.load()
    im_2 = im
    px_2 = im_2.load()
    w , h = im_2.size
    k = len(pal)
    for i in range (w) :
        for j in range (h) :
            c = px_2[i,j]
            min = dist_eu2(c , pal[0])
            l_min = 0
            for l in range (k) :
                if dist_eu2(px_2[i,j] , pal[l]) < min :
                    min = dist_eu2(c , pal[l])
                    l_min = l
            px[i,j] = pal[l_min]
            erreur += min
    print("L'erreur est : ", erreur//(w*h))
    im_2.show()    
#recolorier_pal(im , palette_kmeans)


def recolorier_image_km(im , k) :
    
    erreur = 0
    px = im.load()
    im_2 = im
    px_2 = im_2.load()
    w , h = im_2.size
    pal = Kmeans(list_couleur(px, w, h), k)
    for i in range (w) :
        for j in range (h) :
            c = px_2[i,j]
            min = dist_eu2(c , pal[0])
            l_min = 0
            for l in range (k) :
                if dist_eu2(px_2[i,j] , pal[l]) < min :
                    min = dist_eu2(c , pal[l])
                    l_min = l
            px[i,j] = pal[l_min]
            erreur += min
    print("L'erreur est : ", erreur//(w*h))
    return(im_2) 

#=========================================================================================================
#=============================            Question 6           ===========================================
#=========================================================================================================

#Fonction pour comparer 2 images : 
def compare_images(original, recolored):
    original_data = np.array(original)
    recolored_data = np.array(recolored)

    # Calcul de l'erreur moyenne par pixel
    error = np.mean(np.abs(original_data - recolored_data))
    return error

#Quatification de couleur de PIL 
def pil_quantize_image(image, num_colors):
    quantized = image.quantize(colors=num_colors)
    return quantized

#Test est comparaison des deux algorythmes

# Charger une image de test

# Appliquer votre méthode de quantification des couleurs
num_colors = 10  # Par exemple, réduire à 10 couleurs
your_recolored_image = recolorier_image(im, num_colors)

# Appliquer la quantification de couleur de PIL
pil_recolored_image = pil_quantize_image(im, num_colors)

# Comparer les images
your_error = compare_images(original_image, your_recolored_image)
pil_error = compare_images(original_image, pil_recolored_image)

print(f"Erreur avec votre méthode: {your_error}")
print(f"Erreur avec la méthode PIL: {pil_error}")

# Afficher les images pour une comparaison visuelle
original_image.show()
your_recolored_image.show()
pil_recolored_image.show()




def convolution(px, W: int, H: int, m: list)
    w = len(m)
    h = len(m[0])
    wp = int((w - 1) / 2)
    hp = int((h - 1) / 2)

    for x in range(wp, W - wp):
        for y in range(hp, H - hp):
            sum = 0
            for a in range(-wp, wp + 1):
                for b in range(-hp, hp + 1):
                    sum += px[x + a, y + b][0] * m[a + wp][b + hp]
            v = int(sum / somme_matrice(m))
            px[x, y] = v, v, v