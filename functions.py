from keras.preprocessing import image
import numpy as np
from tqdm import tqdm



def path_to_tensor(img_path, size):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=size)
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(df, size):
    list_of_tensors = [path_to_tensor(row.filename, size) for row in tqdm(df)]
    return np.vstack(list_of_tensors)

train_image_list = []
for filename in tqdm(glob.glob('data/train/*.jpg')):
  train_image_list.append(load_and_resize(filename, (image_size, image_size)))

print(train_image_list)

tqdm


test_image_list = []
for filename in glob.glob('data/test/*.jpg'):  # assuming gif
    im = Image.open(filename)
    test_image_list.append(im)
test_df = pd.DataFrame.from_records(test_image_list)

train_generator = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True) \
    .flow_from_directory(
    'data/train',
    target_size=(image_size, image_size),
    batch_size=17)

# this is a similar generator, for validation data
validation_generator = ImageDataGenerator(rescale=1. / 255) \
    .flow_from_directory(
    'data/test',
    target_size=(image_size, image_size),
    batch_size=17)