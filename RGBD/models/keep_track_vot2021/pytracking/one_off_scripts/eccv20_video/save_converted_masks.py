import os
import numpy as np
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
# Code to save response map
# import matplotlib
# score = scores_act.clamp(0.0).detach().cpu().numpy().squeeze()
# matplotlib.image.imsave('/home/goutam/test_im.png', score, cmap='inferno')
#
def generate_video():
    # image_path = '/home/goutam/projects/lwtl_eccv20_video/breakdance'
    anno_path = '/home/goutam/data/tracking_datasets/DAVIS-2017-trainval-480p/DAVIS/Annotations/480p/gold-fish/'
    out_path = '/home/goutam/projects/lwtl_eccv20_video/gold-fish-anno/'
    num_images = None

    im_list = sorted(os.listdir(anno_path))

    if num_images is not None:
        im_list = im_list[:num_images]

    for i_ in im_list:
        im = cv.imread(anno_path + '/' + i_)
        mask = (im[:, :, 2] == 128) * (im[:, :, 1] == 0) * (im[:, :, 0] == 0)
        im = im * mask.reshape(480,854,1)
        #im = im.mean(-1)
        #im = (im > 0.0).astype(np.float)
        #plt.imsave(out_path + '/' + i_, im, cmap='inferno')
        cv.imwrite(out_path + '/' + i_, im)

if __name__ == '__main__':
    generate_video()