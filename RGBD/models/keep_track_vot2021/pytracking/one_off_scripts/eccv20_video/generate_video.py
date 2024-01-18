import os

# Code to save response map
# import matplotlib
# score = scores_act.clamp(0.0).detach().cpu().numpy().squeeze()
# matplotlib.image.imsave('/home/goutam/test_im.png', score, cmap='inferno')
#
def generate_video():
    # image_path = '/home/goutam/projects/lwtl_eccv20_video/breakdance'
    anno_path = '/home/goutam/projects/lwtl_eccv20_video/breakdance'
    num_images = None

    im_list = sorted(os.listdir(anno_path))

    if num_images is not None:
        im_list = im_list[:num_images]


    a = 1


if __name__ == '__main__':
    generate_video()