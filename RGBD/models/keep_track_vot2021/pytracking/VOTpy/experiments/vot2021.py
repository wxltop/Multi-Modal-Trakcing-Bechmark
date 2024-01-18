from pytracking.evaluation import Tracker, trackerlist

def ar_keep_track_seg_release_0():
    trackers = trackerlist('alpha_refine', 'keep_track_seg', 0)
    return trackers
