from PIL import Image
from IPython.display import display

im = Image.open("IMG_3135.png")
im = im.convert("RGB")  # important pour bien avoir 3 couleurs
px = im.load()
w , h = im.size

def list_couleurs(px , w , h) :
    lst = {}
    for x in range (w) :
        for y in range (h) :
            if px[x,y] in lst :
                lst[px[x,y]] += 1
            else :
                lst[px[x,y]] = 1
    return lst

print(list_couleurs(px , w , h))

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
        pal.append(max)
    return pal
        