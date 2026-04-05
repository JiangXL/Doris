from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img_width, img_height = 400, 300

def resize_img_to_array(img, img_shape=(244, 244)):
    img_array = np.array(
        img.resize(
            img_shape, 
            Image.LANCZOS
        )
    )
    
    return img_array

def image_grid(fn_images : list, 
               text : list =[], 
               top : int = 8, 
               per_row : int =4):
    """
    fn_images is a list of image paths.
    text is a list of annotations.
    top is how many images you want to display
    per_row is the number of images to show per row.
    """
    for i in range(len(fn_images[:top])):
        if i % 4 == 0:
             _ , ax = plt.subplots(1, per_row, 
                                   sharex='col', 
                                   sharey='row', 
                                   figsize=(24, 6))
        j = i % 4
        image = Image.open(fn_images[i])
        image = resize_img_to_array(image, 
                                    img_shape=(img_width, 
                                               img_height))
        ax[j].imshow(image)
        ax[j].axis('off')
        if text:
            ax[j].annotate(text[i],
                          (0,0), (0, -32), 
                           xycoords='axes fraction', 
                           textcoords='offset points', 
                           va='top')
