from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import random
from copy import deepcopy

from skimage.color import rgb2lab, deltaE_ciede2000

#im = Image.open("TD5/figures/color-rainbow.png")
#im = Image.open("TD5/IMG_0741.jpg")
im = Image.open("TD5/IMG_0742.jpg")
#im = Image.open("TD5/figures/color-rainbow.png")
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

#=========================================================================================================
#=============================        Methode des Kmeans       ===========================================
#=========================================================================================================



def Kmeans(colors , k):
    keys = list(colors.keys())
    centroids = [keys[i] for i in random.sample(range(len(keys)), k)]
    variation = True

    while variation:
        centroids2 = []
        cluster = [[] for _ in range(k)]
#On compare la distance de la couleur "key" a chaque centroid
        for key in keys :

#Indice du centroid duquel la couleur qu'on considere est le plus proche
 
            i_min = 0
            for i in range (k) :
                if dist_eu2(centroids[i_min],key) > dist_eu2(centroids[i],key):
                    i_min = i           
            cluster[i_min].append(key)

#Calcul de la moyenne de la couleur de chaque groupe
        '''
        for i in range (k):
            n_i = len(cluster[i])
           
            r_moy = sum(cluster[i][:][0])//n_i
            v_moy = sum(cluster[i][:][1])//n_i
            b_moy = sum(cluster[i][:][2])//n_i
            centroids2.append((r_moy , v_moy , b_moy))
        '''
#Calcul de la moyenne de la couleur de chaque groupe (à la main)

        for i in range (k):
            n_i = len(cluster[i])
            S_r = 0
            S_v = 0
            S_b = 0
            for j in range (n_i):
                S_r += cluster[i][j][0]
                S_v += cluster[i][j][1]
                S_b += cluster[i][j][2]
            moy_r = S_r//n_i
            moy_v = S_v//n_i
            moy_b = S_b//n_i
            centroids2.append((moy_r , moy_v , moy_b))

        
#Si les nouveaux centroides sont egaux aux anciens, alors il n'y a plus de variation

        if set(centroids2) == set(centroids) :
            variation = False
        else :
            centroids = centroids2
    
    return centroids        
        
#palette_kmeans = Kmeans(list_couleur(px, w, h), 8)
        

        

                







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

#plot_color_palette(pal)



#fonction pour recolorier l'image avec la palette naïve

def recolorier(im , k) :
    im.show()
    erreur = 0
    px = im.load()
    w1 , h1 = im.size
    dict = list_couleur(px , w1 , h1)
    pal = palette_k2(dict ,5 , k)
    im_2 = im
    px_2 = im_2.load()
    w , h = im_2.size
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


#=========================================================================================================
#=============================            Question 6           ===========================================
#=========================================================================================================

def recolorier_image_km(im , k) :
    
    erreur = 0
    px = im.load()
    im_2 = im.copy()
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
            px_2[i,j] = pal[l_min]
    return(im_2) 

#Fonction pour comparer 2 images : 
def compare_images(original, recolored):
    # Convertir les deux images en format RGB pour assurer la compatibilité
    original_rgb = original.convert('RGB')
    recolored_rgb = recolored.convert('RGB')

    original_data = np.array(original_rgb)
    recolored_data = np.array(recolored_rgb)

    # Calcul de l'erreur moyenne par pixel
    error = np.mean(np.abs(original_data - recolored_data))
    return error

#Quatification de couleur de PIL 
def pil_quantize_image(image, num_colors):
    quantized = image.quantize(colors=num_colors)
    return quantized

#Test est comparaison des deux algorythmes
'''
# Appliquer votre méthode de quantification des couleurs
num_colors = 10  # Par exemple, réduire à 10 couleurs
your_recolored_image = recolorier_image_km(im.copy(), num_colors)

# Appliquer la quantification de couleur de PIL
pil_recolored_image = pil_quantize_image(im.copy(), num_colors)

# Comparer les images
your_error = compare_images(im, your_recolored_image)
pil_error = compare_images(im, pil_recolored_image)

print(f"Erreur avec votre méthode: {your_error}")
print(f"Erreur avec la méthode PIL: {pil_error}")

# Afficher les images pour une comparaison visuelle
if pil_recolored_image.mode != 'RGB':
    pil_recolored_image = pil_recolored_image.convert('RGB')
pil_recolored_image.show()


im.show()
your_recolored_image.show()
pil_recolored_image.show()
'''
#=========================================================================================================
#=============================            Question 7           ===========================================
#=========================================================================================================
#Les operations etant coupteuse, nous passons en niveau de gris

def conversion_gris(px, W: int, H: int) -> None:
    for x in range(0, W):
        for y in range(0, H):
            m = int(0.30*px[x, y][0] + 0.59*px[x, y][1] + 0.11*px[x, y][2])
            px[x, y] = (m, m, m)



def somme_matrice(m: list = []):
    s = 0
    for row in m:
        s += sum(row)
    return s

gauss3 = [[1,2,1],
          [2,4,2],
          [1,2,1]]

gauss7 = [[1,1,2,2,2,1,1],
          [1,2,2,4,2,2,1],
          [2,2,4,8,4,2,2],
          [2,4,8,16,8,4,2],
          [2,2,4,8,4,2,2],
          [1,2,2,4,2,2,1],
          [1,1,2,2,2,1,1]]

def convolution(px, W: int, H: int, m: list):
    
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

#On applique le flou gaussien à notre image et ensuite on créé notre palette et on recolorie l'image

'''im_gris = im.copy()
im_gris_original = im.copy()
px_go = im_gris_original.load()
px_g = im_gris.load()
w , h = im_gris.size
conversion_gris(px_g, w, h)
conversion_gris(px_go, w, h)
convolution(px_g,w,h,gauss3)
recolored_image_g = recolorier_image_km(im_gris, 20)
recolored_image_sans_g = recolorier_image_km(im_gris_original.copy(), 20)
error_g = compare_images(im_gris_original, recolored_image_g)
error_sans_g = compare_images(im_gris_original, recolored_image_sans_g)

print(f"Erreur avec gauss: {error_g}")
print(f"Erreur sans gauss: {error_sans_g}")


im_gris.show()
recolored_image_g.show()
recolored_image_sans_g.show()'''

#=========================================================================================================
#=============================            Question 8           ===========================================
#=========================================================================================================
#On utilise l'espace de couleur lab pour un calcul plus proche de ce que l'oeil humain percois:

#Conversion rgb vers lab :
def rgb_to_lab_color(r, g, b):
    # Convertir de RGB à Lab Color
    rgb = np.array([[[r / 255, g / 255, b / 255]]])
    lab = rgb2lab(rgb)[0][0]
    return lab

def color_distance_ciede2000(color1_rgb, color2_rgb):
    # Convertir les couleurs RGB en Lab
    color1_lab = rgb_to_lab_color(*color1_rgb)
    color2_lab = rgb_to_lab_color(*color2_rgb)

    # Calculer la distance CIEDE2000
    delta_e = deltaE_ciede2000(color1_lab, color2_lab)
    return delta_e

def Kmeans_lab(colors , k):
    keys = list(colors.keys())
    centroids = [keys[i] for i in random.sample(range(len(keys)), k)]
    variation = True

    while variation:
        centroids2 = []
        cluster = [[] for _ in range(k)]
#On compare la distance de la couleur "key" a chaque centroid
        for key in keys :

#Indice du centroid duquel la couleur qu'on considere est le plus proche
 
            i_min = 0
            for i in range (k) :
                if color_distance_ciede2000(centroids[i_min],key) > color_distance_ciede2000(centroids[i],key):
                    i_min = i           
            cluster[i_min].append(key)

#Calcul de la moyenne de la couleur de chaque groupe (à la main)

        for i in range (k):
            n_i = len(cluster[i])
            S_r = 0
            S_v = 0
            S_b = 0
            for j in range (n_i):
                S_r += cluster[i][j][0]
                S_v += cluster[i][j][1]
                S_b += cluster[i][j][2]
            moy_r = S_r//n_i
            moy_v = S_v//n_i
            moy_b = S_b//n_i
            centroids2.append((moy_r , moy_v , moy_b))

        
#Si les nouveaux centroides sont egaux aux anciens, alors il n'y a plus de variation

        if set(centroids2) == set(centroids) :
            variation = False
        else :
            centroids = centroids2
    
    return centroids

def recolorier_image_km_ciede2000(im , k) :
    px = im.load()
    im_2 = im.copy()
    px_2 = im_2.load()
    w , h = im_2.size
    pal = Kmeans(list_couleur(px, w, h), k)
    for i in range (w) :
        for j in range (h) :
            c = px_2[i,j]
            min = color_distance_ciede2000(c , pal[0])
            l_min = 0
            for l in range (k) :
                if color_distance_ciede2000(px_2[i,j] , pal[l]) < min :
                    min = color_distance_ciede2000(c , pal[l])
                    l_min = l
            px_2[i,j] = pal[l_min]
    return(im_2) 

im_ciede2000 = recolorier_image_km_ciede2000(im,10)
im_eucl = recolorier_image_km(im,10)

error_ciede2000 = compare_images(im, im_ciede2000)
error_eucl = compare_images(im, im_eucl)

print(f"Erreur avec ciede2000: {error_ciede2000}")
print(f"Erreur avec eucl: {error_eucl}")


im.show()
im_ciede2000.show()
im_eucl.show()