from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from os import listdir
datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.10,
        height_shift_range=0.10,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='constant'
        ,cval=255)
root='/home/kawsar/aA-WORKING/archive/Potato/'
imgs=listdir(root)
for img in imgs:
    img = load_img(root+img)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir='/home/kawsar/aA-WORKING/archive/PotatoM/', save_prefix='Cercospora_leaf_spot', save_format='jpeg'):
        i += 1
        if i > 10:
            break