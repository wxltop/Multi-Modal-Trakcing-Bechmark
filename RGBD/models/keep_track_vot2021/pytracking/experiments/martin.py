from pytracking.evaluation import Tracker, get_dataset, trackerlist

def supra1():
    trackers = trackerlist('dimp', 'super_dimp_ml_l05', range(0,2)) + \
               trackerlist('dimp', 'super_dimp_ml_l10', range(0,2)) + \
               trackerlist('dimp', 'super_dimp_ml_l015', range(0,2))

    dataset = get_dataset('got10k_val')

    return trackers, dataset

def supra2():
    trackers = trackerlist('dimp', 'super_dimp_ml_l05', range(0,2)) + \
               trackerlist('dimp', 'super_dimp_ml_l10', range(0,2)) + \
               trackerlist('dimp', 'super_dimp_ml_l015', range(0,2))

    dataset = get_dataset('got10k_val')[::-1]
    return trackers, dataset

def supra3():
    trackers = trackerlist('dimp', 'dimp50_yawei_heteronet', range(0, 5))

    dataset = get_dataset('nfs', 'uav', 'otb')
    return trackers, dataset

def supra4():
    trackers = trackerlist('dimp', 'dimp50_yawei_heteronet', range(0, 5))

    dataset = get_dataset('nfs', 'uav', 'otb')[::-1]
    return trackers, dataset

def supra5():
    trackers = trackerlist('dimp', 'dimp50_yawei_baseline', range(0, 5))

    dataset = get_dataset('nfs', 'uav', 'otb')
    return trackers, dataset

def supra6():
    trackers = trackerlist('dimp', 'dimp50_yawei_baseline', range(0, 5))

    dataset = get_dataset('nfs', 'uav', 'otb')[::-1]
    return trackers, dataset

def supra7():
    trackers = trackerlist('dimp', 'dimp50_yawei_baseline', range(0, 5))

    dataset = get_dataset('lasot')
    return trackers, dataset

def supra8():
    trackers = trackerlist('dimp', 'dimp50_yawei_baseline', range(0, 5))

    dataset = get_dataset('lasot')[::-1]
    return trackers, dataset


def trackingnet():
    trackers = trackerlist('dimp', 'dimp50_yawei_heteronet', range(1)) + \
               trackerlist('dimp', 'dimp50_yawei_baseline', range(1))

    dataset = get_dataset('trackingnet')
    return trackers, dataset


def trackingnet2():
    trackers = trackerlist('dimp', 'super_dimp_nce_l05', range(1))

    dataset = get_dataset('trackingnet')
    return trackers, dataset


def lasot():
    trackers = trackerlist('dimp', 'dimp50_yawei_heteronet', range(0,5))

    dataset = get_dataset('lasot')
    return trackers, dataset


def lasot2():
    trackers = trackerlist('dimp', 'dimp50_yawei_heteronet', range(0, 5))

    dataset = get_dataset('lasot')[::-1]
    return trackers, dataset

def lasot3():
    trackers = trackerlist('dimp', 'dimp50_bbkl005_mg200_frozen_l1l2_im15sa6_doubleflip_cascade', range(0,2))

    dataset = get_dataset('lasot')
    return trackers, dataset

def lasot4():
    trackers = trackerlist('dimp', 'dimp50_bbkl005_mg200_frozen_l1l2_im15sa6_doubleflip_cascade', range(0,2))

    dataset = get_dataset('lasot')[::-1]
    return trackers, dataset

def got10k():
    trackers = trackerlist('dimp', 'super_dimp', range(0,2)) + \
               trackerlist('dimp', 'super_dimp_ncep075_l25', range(0,2)) + \
               trackerlist('dimp', 'super_dimp_ncep075_l10', range(0,2))

    dataset = get_dataset('got10k_val')
    return trackers, dataset

def got10k2():
    trackers = trackerlist('dimp', 'super_dimp_ncep075_l05', range(0,2)) + \
               trackerlist('dimp', 'super_dimp_ncep075_l015', range(0,2))

    dataset = get_dataset('got10k_val')
    return trackers, dataset

def got10k3():
    trackers = trackerlist('dimp', 'dimp50_bbkl005_mg200_highres_im15_bbw01_ksz3', range(0,3))

    dataset = get_dataset('got10k_val')
    return trackers, dataset


def atomml():
    # trackers = trackerlist('atom', 'ml_mb10_atom_p05_005_prop128', range(3,5)) + \
    #            trackerlist('atom', 'atom_default_clone', range(3, 5))
    trackers = trackerlist('atom', 'atom_ml_sampling_tracking', range(1))

    # dataset = get_dataset('otb', 'nfs', 'uav')
    dataset = get_dataset('trackingnet')
    return trackers, dataset

def xps():
    trackers = trackerlist('etcom', 'sdlearn_300_onlytestloss_lr_causal_mg30', range(1))

    dataset = get_dataset('otb', 'nfs', 'uav')

    return trackers, dataset


def testing():
    trackers = trackerlist('escoiou', 'ksz4_sa5nw_loc17_none_mlu005_attl23_fix_im025_lr01', range(1))

    dataset = get_dataset('otb')

    return trackers, [dataset[11]]
