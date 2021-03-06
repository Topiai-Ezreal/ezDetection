import numpy as np
import cv2
import random
import os
import glob

# calculate means and std
train_txt_path = '/home/tupeng/DL/Classifier/temporal-segment-network-pytorch/list/tube_rgb_train_split.txt'

means = [0, 0, 0]
stdevs = [0, 0, 0]

index = 1
num_imgs = 0
with open(train_txt_path, 'r') as f:
    lines = f.readlines()
    random.shuffle(lines)
    # lines = lines[:2]

    for line in lines:
        eles = line.strip().split(' ')
        print('{}/{}'.format(index, len(lines)))
        index += 1

        datas = glob.glob(os.path.join(eles[0], 'diff_nor*.jpg'))
        for data in datas:
            num_imgs += 1
            img = cv2.imread(data)
            img = img.astype(np.float32) / 255.
            for i in range(3):
                means[i] += img[:, :, i].mean()
                stdevs[i] += img[:, :, i].std()

means.reverse()
stdevs.reverse()

means = np.asarray(means) / num_imgs
stdevs = np.asarray(stdevs) / num_imgs

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))