import matplotlib.pyplot as plt
import cv2
import torch

from pytracking.utils.plotting import show_tensor, draw_figure, overlay_mask, torch_to_numpy



def draw_proposals(im, proposals):
    for i in range(proposals.shape[0]):
        bb = proposals[i, :]
        cv2.rectangle(im, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), (255, 0, 0), 1)

    return im


def draw_rect(im, bb):
    cv2.rectangle(im, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), (0, 255, 0), 3)

    return im


def show_batch(train_frames=None, train_anno=None, train_mask=None, test_frames=None, test_anno=None, test_mask=None,
               object_class_name=None, normalize=False, proposals=None, fig_num=0, timeout=0.01):
    # Assumes that data has been sampled in sequence mode, i.e. shape is (seq_length, batch, c, h, w)
    assert train_frames is not None or test_frames is not None

    batch_size = train_frames.shape[0] if train_frames is not None else test_frames.shape[0]

    num_train_im = train_frames.shape[1] if train_frames is not None else 0
    num_test_im = test_frames.shape[1] if test_frames is not None else 0

    num_cols = num_train_im + num_test_im
    num_rows = batch_size

    # Handle cases when we dont have unknown class
    if object_class_name is None:
        object_class_name_disp = ['Unknown' for _ in range(batch_size)]
    else:
        object_class_name_disp = [x if x is not None else 'Unknown' for x in object_class_name]

    fig = plt.figure(fig_num)
    plt.tight_layout()
    plt.cla()

    for i in range(0, batch_size):
        for j in range(0, num_train_im):
            frame = train_frames[i, j, :, :, :].clone().squeeze().cpu()
            frame = torch_to_numpy(frame, normalize)

            if train_mask is not None:
                frame = overlay_mask(frame, train_mask[i, j, ...], alpha=0.5, colors=None, contour_thickness=None)

            if train_anno is not None:
                frame = draw_rect(frame, train_anno[i, j, :])

            plt.subplot(num_rows, num_cols, i * num_cols + j + 1)

            if i == 0:
                plt.gca().set_title('Train {}'.format(j+1))

            plt.imshow(frame)
            plt.axis('off')
            plt.axis('equal')

            # plt.title('Class: ' + object_class_name_disp[i])

        for j in range(0, num_test_im):
            frame = test_frames[i, j, :, :, :].clone().squeeze().cpu()
            frame = torch_to_numpy(frame, normalize)

            if proposals is not None:
                frame = draw_proposals(frame, proposals[i, j, :, :])

            if test_mask is not None:
                frame = overlay_mask(frame, test_mask[i, j, ...], alpha=0.5, colors=None, contour_thickness=None)

            if test_anno is not None:
                frame = draw_rect(frame, test_anno[i, j, :])

            plt.subplot(num_rows, num_cols, i*num_cols + num_train_im + j + 1)
            if i == 0:
                plt.gca().set_title('Test {}'.format(j + 1))

            plt.imshow(frame)
            plt.axis('off')
            plt.axis('equal')

            # plt.title('Class: ' + object_class_name_disp[i])

    # draw_figure(fig)
    plt.draw()
    plt.pause(0.1)
    plt.waitforbuttonpress(timeout=timeout)

    return fig_num


def show_batch_response(data, labels, image=None, fig_num=0, normalize=True):
    if not isinstance(data, list):
        data = [data]
        labels = [labels]

    data = [d.clone().detach().cpu().view(-1, d.shape[-2], d.shape[-1]) for d in data]
    data = [d.permute(1, 0, 2).contiguous().view(d.shape[1], -1) for d in data]

    if image is not None:
        image = image.clone().cpu().view(-1, 3, image.shape[-2], image.shape[-1])
        image = image.permute(1, 2, 0, 3).contiguous().view(3, image.shape[-2], -1)

    showing_im = int(image is not None)
    num_rows = len(data) + showing_im

    plt.figure(fig_num)
    plt.tight_layout()
    plt.cla()

    # Show image
    if image is not None:
        plt.subplot(num_rows, 1, 1)

        im_disp = torch_to_numpy(image, normalize)

        plt.imshow(im_disp)

        plt.title('Image')
        plt.ylabel('Image')
        plt.axis('off')
        plt.axis('equal')

    for i in range(showing_im, num_rows):
        plt.subplot(num_rows, 1, i + 1)

        im = data[i-showing_im][:, :].numpy()

        if normalize:
            plt.imshow(im, vmin=0, vmax=1)
        else:
            plt.imshow(im)

        plt.title(labels[i-showing_im])
        plt.ylabel(labels[i-showing_im])
        plt.axis('off')
        plt.axis('equal')

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()
