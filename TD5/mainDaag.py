from PIL import Image
from IPython.display import display,HTML

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

print(palette_k(list_couleur(px,w,h),4))














