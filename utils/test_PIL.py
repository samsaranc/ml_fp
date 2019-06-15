from PIL import Image
import io

img = '../data_split.ml_fp.3way/tra/S/S_imagenet_358.jpg'

im = Image.open(img)

'''
with open(img) as f:
   io = io.BytesIO(f.read())
im = Image.open(io)
'''

#img = '../data_split.ml_fp.3way/tra/S/S_imagenet_358.jpg'
