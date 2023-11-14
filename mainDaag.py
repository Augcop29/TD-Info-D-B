from PIL import Image
from IPython.display import display

im = Image.open("IMG_3135.png")
im = im.convert("RGB")
px = im.load()

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
            

print(list_couleur(px,w,h))