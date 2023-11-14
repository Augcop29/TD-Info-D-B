################

im2 = Image.new('RGB', (im.width, im.height)) 
px2 = im2.load()
W, H = im.size
setRegion(W//3, H//3, W//3, H//3, (255, 255, 255), px2)
display(im2)

#################