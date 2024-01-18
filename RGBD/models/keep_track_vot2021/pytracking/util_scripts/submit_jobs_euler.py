import os
import subprocess


TITANRTX = 'TITANRTX'
RTX2080 = 'GeForceRTX2080Ti'
V100 = 'TeslaV100_SXM2_32GB'


def run_script(script_name, executable, job_name=None, num_jobs=None, start_job_idx=None, stop_job_idx=None,
               verbose=False, after=None, mem=None, num_cpus=None, time=None, output_path=None,
               num_gpus=None, gpu_model=None, gpu_mem=None, scratch_space=None, args=None):
    cmd = ['bsub']

    if after is not None:
        # Run after given job:
        cmd += ['-w', 'done({})'.format(after)]

    if job_name is not None:
        # Define job name:
        if (num_jobs == 1 or num_jobs is None) and (start_job_idx is None and stop_job_idx is None):
            cmd += ['-J', '{}'.format(job_name)]
        else:
            if start_job_idx is not None and stop_job_idx is not None:
                cmd += ['-J', '{}[{}-{}]'.format(job_name, start_job_idx, stop_job_idx)]
            else:
                cmd += ['-J', '{}[1-{}]'.format(job_name, num_jobs)]

    # Time for your computations
    if time is not None:
        cmd += ['-W', time]
    # Number of processors
    if num_cpus is not None:
        cmd += ['-n', '{}'.format(num_cpus)]

    # Advanced resources
    res = []

    if mem is not None:
        res += ['mem={}'.format(mem)]
    if num_gpus is not None and num_gpus > 0:
        res += ['ngpus_excl_p={}'.format(num_gpus)]
    if scratch_space is not None:
        res += ['scratch={}'.format(scratch_space)]

    if len(res) > 0:
        res = ','.join(res)
        res = 'rusage[' + res + ']'
        cmd += ['-R', res]

    # Advance GPU resources
    res = []

    if gpu_mem is not None:
        res += ['gpu_mtotal0>={}'.format(gpu_mem)]
    if gpu_model is not None:
        res += ['gpu_model0=={}'.format(gpu_model)]

    if len(res) > 0:
        res = ','.join(res)
        res = 'select[' + res + ']'
        cmd += ['-R', res]

    if output_path is not None and job_name is not None:
        cmd += ['-o', os.path.join(output_path, job_name)]

    if verbose:
        print('\n=== Running: {} ===\n'.format(script_name))

    if executable != '<' and executable is not None:
        cmd += [executable]
        cmd += [script_name]

        if args is not None:
            for arg in args:
                cmd += [arg]

        cmd = ' '.join(cmd).strip()

        print(cmd)
        subprocess.check_call(cmd.split(' '))

    else:
        cmd = ' '.join(cmd).strip()

        myenv = os.environ.copy()
        for i, arg in enumerate(args):
            myenv['PARAM{}'.format(i+1)] = arg

        print(cmd + ' < {}'.format(script_name))
        with open(script_name, 'r') as f:
            p = subprocess.Popen(cmd.split(' '), env=myenv, stdin=f)
            p.wait()


if __name__ == '__main__':

    tracker_name = 'dimp_memory_learning'

    # exps = [
    #
    #
    #
    #
    #     # ('super_dimp_memory_learning_max_score_no_ths_sa_scale_8_fsize_30_rescale_speedup', 'lasot'),
    #     # ('super_dimp_memory_learning_max_score_no_ths_peak_matching_v1_mixed_wo_cor_logic_sa_scale_8_fsize_30_rescale_speedup_id0_weight_increase', 'lasot'),
    #     # ('super_dimp_memory_learning_max_score_no_ths_peak_matching_v1_mixed_wo_cor_logic_sa_scale_8_fsize_30_rescale_speedup_id0_weight_increase_sab_coords', 'lasot'),
    #     # ('super_dimp_memory_learning_max_score_no_ths_peak_matching_v1_mixed_wo_cor_logic_sa_scale_8_fsize_30_rescale_speedup_sab_coords', 'lasot'),
    #     # ('super_dimp_memory_learning_peak_matching_v1_mixed_wo_cor_logic_sa_scale_8_fsize_30_rescale_speedup_id0_weight_increase_sab_coords', 'lasot'),
    #     # ('super_dimp_memory_learning_peak_matching_v1_mixed_wo_cor_logic_sa_scale_8_fsize_30_rescale_speedup_sab_coords', 'lasot'),
    #
    #     # ('super_dimp_memory_learning_max_score_no_ths', 'lasot_extension_subset'),
    #     # ('super_dimp_memory_learning_max_score_no_ths_sa_scale_8_fsize_30_rescale_speedup', 'lasot_extension_subset'),
    #     # ('super_dimp_memory_learning_max_score_no_ths_peak_matching_v1_mixed_wo_cor_logic_sa_scale_8_fsize_30_rescale_speedup_id0_weight_increase', 'lasot_extension_subset'),
    #     # ('super_dimp_memory_learning_max_score_no_ths_peak_matching_v1_mixed_wo_cor_logic_sa_scale_8_fsize_30_rescale_speedup_id0_weight_increase_sab_coords', 'lasot_extension_subset'),
    #     # ('super_dimp_memory_learning_max_score_no_ths_peak_matching_v1_mixed_wo_cor_logic_sa_scale_8_fsize_30_rescale_speedup_sab_coords', 'lasot_extension_subset'),
    #     # ('super_dimp_memory_learning_peak_matching_v1_mixed_wo_cor_logic_sa_scale_8_fsize_30_rescale_speedup_id0_weight_increase_sab_coords', 'lasot_extension_subset'),
    #     # ('super_dimp_memory_learning_peak_matching_v1_mixed_wo_cor_logic_sa_scale_8_fsize_30_rescale_speedup_sab_coords', 'lasot_extension_subset')
    # ]

    datasets = [
        'lasot',
        'nfs',
        'uav',
    ]
    #
    parameters = [
        'super_dimp_memory_learning_max_score_no_ths_rescale',
        'super_dimp_memory_learning_max_score_no_ths_sa_scale_8_fsize_30'
        # 'super_dimp_memory_learning_max_score_no_ths_peak_matching_v1_mixed_wo_cor_logic_sa_scale_8_fsize_30_rescale_speedup_id0_weight_increase_no_data_mining'
    ]

    # exps = [
    #     ('dimp', 'super_dimp', 'lasot_extension_subset'),
    #     ('dimp_weight_learning', 'super_dimp_hinge', 'lasot_extension_subset'),
    #
    # ]


    # for (tracker_name, params, dataset) in exps:
    for dataset in datasets:
        for params in parameters:
            job_name = '{}_{}_{}'.format(tracker_name, params, dataset)
            run_script(script_name='job_euler.sh', executable=None, start_job_idx=1, stop_job_idx=5, verbose=True, mem=8192,
                       num_cpus=8, time='24:00', output_path='./euler_logs', num_gpus=1, scratch_space=125000, job_name=job_name,
                       args=[tracker_name, params, dataset])