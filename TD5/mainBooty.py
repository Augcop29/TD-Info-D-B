from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

im = Image.open("TD5/IMG_1964.png")
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

def palette_k2(dict , eps , k) :
    lst_key = []
    for key in dict :
        if dict[key] <= eps :
            lst_key.append(key)
    n = len(lst_key)
    for i in range (n) :
        del dict[lst_key[i]]
    lst = [(0 , 0 , 0)]
    while len(dict) != 0 :
        min = (255**2)*3
        key_min = (0,0,0)
        for key in dict :
            if dist_eu2(key , lst[-1]) < min :
                key_min = key
        del dict[key_min]
        lst.append(key_min)
    
    n = len(lst)
    h = n//k
    lst_c = []
    k_i = h//2
    while k_i < n :
        lst_c.append(lst[k_i])
        k_i += h
    return lst_c



pal = palette_k2(list_couleur(px , w , h) , 10 , 20)

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

    plt.show(block=True)

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

recolorier(im , 20)





