import pytracking

pytracking.run_tracker("keep_track", "release",
                       dataset_name="vot2021", sequence="animal", debug=3,
                       visdom_info={'use_visdom': True, 'server': "129.132.67.103", 'port': 8097})
