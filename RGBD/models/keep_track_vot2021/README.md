# README KEEPTRACK

The tracker is implemented in python using pytorch within a linux environment and requires a GPU with CUDA.

Please install all requirements listed in `pytracking/install.sh`
The tracker has the same dependencies as super_dimp please check the pytracking github repo for more information if there any problems setting up the tracker.
Make sure `ltr/external` contains a working setting for PrRoiPooling (it should already be contained here otherwise load it via git submodules.)

Please dowload the checkpoint files of the tracker. You can find them here https://polybox.ethz.ch/index.php/s/WrpeaU3kZ1ZgjKU

Please modify the file `pytracking/evaluation/local.py` accordingly.

In order to test the tracker you can run `python test_tracker.py` with a visdom server running.

To run the VOTpy toolkit for 2021 please navigate to `pytracking/VOTpy` and run

- `python vot_evaluate.py votlt2021 keep_track_release_0 --workspace_path  path to votlt2021 workspace` to run the long term tracker.
- `python vot_evaluate.py vot2021 ar_keep_track_seg_release_0 --workspace_path  path to vot2021 workspace` to run the short term tracker.

this runs the tracker using the vottoolkit via the cli. It also creates the trackers.ini files.


Thanks and reach out to me chmayer@vision.ee.ethz.ch if there are any problems/questions.