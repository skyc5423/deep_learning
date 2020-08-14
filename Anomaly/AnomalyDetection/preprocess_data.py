import matplotlib

matplotlib.use('Agg')
from PIL import Image
import os

size_list = []

PATH_DB = '/home/ybrain/sangmin/food_dataset/images'
preprocessed_db_path = '/home/ybrain/sangmin/food_dataset/preprocessed_224_224'

if not os.path.isdir(preprocessed_db_path):
    os.mkdir(preprocessed_db_path)

total_num = 0

for category in os.listdir(PATH_DB):
    category_path = os.path.join(PATH_DB, category)
    new_category_path = os.path.join(preprocessed_db_path, category)

    if not os.path.isdir(category_path):
        continue
    if not os.path.isdir(new_category_path):
        os.mkdir(new_category_path)

    for file_name in os.listdir(category_path):
        file_path = os.path.join(category_path, file_name)
        new_file_path = os.path.join(new_category_path, file_name.replace('jpg', 'png'))

        original_img = Image.open(file_path)
        original_size = original_img.size
        if original_size[0] > original_size[1]:
            bias = int((original_size[0] - original_size[1]) / 2)
            cropped_img = original_img.crop((bias, 0, bias + original_size[1], original_size[1]))
        elif original_size[0] < original_size[1]:
            bias = int((original_size[1] - original_size[0]) / 2)
            cropped_img = original_img.crop((0, bias, original_size[0], bias + original_size[0]))
        else:
            cropped_img = original_img

        resized_img = cropped_img.resize((224, 224))
        resized_img.save(new_file_path)

        original_img.close()
        cropped_img.close()
        resized_img.close()
        total_num += 1

print(total_num)
