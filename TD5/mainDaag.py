from PIL import Image
import math
from IPython.display import display,HTML
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


im = Image.open("TD5/IMG_3135.png")
#im = Image.open("TD5/IMG_1964.png")
im = im.convert("RGB")
px = im.load()

im2 = Image.new('RGB', (im.width, im.height)) 
px2 = im2.load()

w,h = im.size

def list_couleur(px,w,h):
    lst = {}
    for x in range (w):
        for y in range (h):
            if px[x,y] in lst:
                lst[px[x,y]] += 1
            else:
                lst[px[x,y]] = 1
    return lst
            
def draw_rectangle(rgb_color,w,h):
    color = f'rgb({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]})'
    html = f'<svg width="{w}" height="{h}"><rect width="100" height="100" fill="{color}" /></svg>'

    display(HTML(html))

def palette_k(dict , k) :
    pal = []
    for i in range (k) :
        max = 0
        key_max = 0
        for key in dict :
            if dict[key] > max :
                max = dict[key]
                key_max = key
        dict[key_max] = -1
        pal.append(key_max)
    return pal

pal = palette_k(list_couleur(px,w,h),200)



def euclidean_distance(color1, color2):
    return math.sqrt(sum((c1 - c2)**2 for c1, c2 in zip(color1, color2)))

def sort_colors_by_distance(color_dict):
    filtered_colors = {color: freq for color, freq in color_dict.items() if freq >= 5}
    colors = list(filtered_colors.keys())
    sorted_colors = sorted(colors, key=lambda c: filtered_colors[c])

    # Triez les couleurs en fonction de leur distance euclidienne
    sorted_colors = sorted(sorted_colors, key=lambda c1: min(euclidean_distance(c1, c2) for c2 in sorted_colors))

    return sorted_colors

# Exemple d'utilisation :
couleurs = {
    (255, 0, 0): 10,
    (0, 255, 0): 5,
    (0, 0, 255): 8,
    (128, 128, 0): 3,
    # Ajoutez d'autres couleurs au besoin
}

couleurs_triees = sort_colors_by_distance(list_couleur(px,w,h))
print(couleurs_triees)


def plot_color_palette(rgb_values):

    rgb_values = np.array(rgb_values) / 255.0

    custom_cmap = ListedColormap(rgb_values)

    data = np.arange(len(rgb_values)).reshape(1, -1)

    fig, ax = plt.subplots(figsize=(len(rgb_values), 1))

    cax = ax.pcolormesh(data, cmap=custom_cmap, edgecolors='w', linewidth=0.5)

    ax.set_yticks([])
    ax.set_xticks(np.arange(0.5, len(rgb_values), 1))
    ax.set_xticklabels([])

    plt.show(block=True)

def split_colors_into_segments(sorted_colors, k):
    n = len(sorted_colors)
    segment_size = n // k

    color_segments = [sorted_colors[i:i+segment_size] for i in range(0, n, segment_size)]

    # Si le nombre de couleurs n'est pas divisible par k, ajouter les couleurs restantes au dernier segment
    if n % k != 0:
        color_segments[-1].extend(sorted_colors[-(n % k):])

    pal = []
    for segment in color_segments:
        avg_color = np.mean(segment, axis=0)
        # Convertir les valeurs de couleur moyenne en entiers
        avg_color = tuple(int(val) for val in avg_color)
        pal.append(avg_color)

    return pal


def palette_2(dict,k):
    return(split_colors_into_segments(sort_colors_by_distance(dict),k))
pal2 = palette_2(list_couleur(px,w,h),30)

plot_color_palette(pal2)

def dist_color(c1,c2):
    return(((c1[0]-c2[0])**2+(c1[1]-c2[1])**2+(c1[2]-c2[2])**2)**(1/2))

def recolorier(pal,px,px2,w,h):
    erreur = 0
    for x in range(w):
        for y in range(h):
            color = px[x,y]
            dist = dist_color(pal[0],color)
            col_plus_proche = pal[0]
            for i in range(1,len(pal)):
                dist_col = dist_color(pal[i],color)
                if dist > dist_col:
                    dist = dist_col
                    col_plus_proche = pal[i]
            px2[x,y] = (col_plus_proche[0],col_plus_proche[1],col_plus_proche[2])
            erreur += dist
    print(erreur/(w*h))

recolorier(pal2,px,px2,w,h)
im2.show()
im.show()
    
                     
             

