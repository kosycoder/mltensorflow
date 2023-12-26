import pathlib
imgdir_path = pathlib.Path('img_align_celeba')
file_list = sorted([str(path) for path in imgdir_path.glob('*.jpg')])
print(file_list[:5])

i=0
label_list = []
with open("img_align_celeba/list_attr_celeba.txt","r") as f:
    lines = f.readlines()
    for line in lines:
        if i > 1:
            line = line.split()
            #1列目のインデックスを除いてint化、-1を0に変換
            line = [int(i) if i == '1' else int(0) for i in line[1:]]
            label_list.append(line)
        i += 1
print(label_list[:1])

import tensorflow as tf
ds_files_labels = tf.data.Dataset.from_tensor_slices((file_list,label_list))
def load_and_preprocess(path, label):
    """画像パスから画像を読み込み、ラベルと共に返す関数"""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image,tf.float32) / 255.0
    return image,label

ds_images_labels = ds_files_labels.map(load_and_preprocess)

import matplotlib.pyplot as plt

fig = plt.figure(figsize = (10,5))
for i,example in enumerate(ds_images_labels.take(6)):
    ax = fig.add_subplot(2,3,i+1)
    ax.set_xticks([]);ax.set_yticks([])
    ax.imshow(example[0])
    if example[1][20].numpy() == 1:
        label='Male'
    else:
        label='Female'
    ax.set_title('{}'.format(label),size=15)

plt.show()
