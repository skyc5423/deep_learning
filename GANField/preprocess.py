from PIL import Image
import os
import numpy as np

size = 256

for img_file_name in os.listdir('./downloads/bing3'):
    save_name = os.path.join('./downloads/bing3/preprocessed_256', img_file_name)
    if not img_file_name.endswith('jpg'):
        continue
    if os.path.exists(save_name):
        continue
    img = Image.open('./downloads/bing2/%s' % img_file_name)
    if np.array(img).shape.__len__() < 3 or np.array(img).shape[2] > 3:
        continue
    if np.argmin(img.size) == 0:
        wide = True
    else:
        wide = False
    crop_img_size = int(np.min(img.size))

    if crop_img_size < size:
        continue

    crop_offset = int((np.max(img.size) - crop_img_size) / 2)

    if wide:
        bbox = (0, crop_offset, crop_img_size, crop_img_size + crop_offset)
    else:
        bbox = (crop_offset, 0, crop_offset + crop_img_size, crop_img_size)

    crop_img = img.crop(bbox)
    resize_img = crop_img.resize((size, size))

    resize_img.save(save_name)
    print('save file ' + save_name + '....')
