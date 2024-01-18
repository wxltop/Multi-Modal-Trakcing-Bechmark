from pytracking import run_tracker


def main():
    tracker_name = 'escoiou'
    tracker_param = 'ablation_net_attl23'
    runid = None
    dataset = 'nfs'
    sequence = 0
    debug = 1

    visdom_info = {'server': '127.0.0.1', 'port': 8098}

    run_tracker(tracker_name, tracker_param, runid, dataset, sequence, debug, visdom_info=visdom_info)


if __name__ == '__main__':
    main()