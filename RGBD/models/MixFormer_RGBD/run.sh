cd external/vot2022rgbd/mixformer_large
vot evaluate --workspace . mixformerrgbd_large_rgbd
vot analysis --workspace . mixformerrgbd_large_rgbd  --nocache --format html
vot pack --workspace . mixformerrgbd_large_rgbd  # for submission
cd ../../../