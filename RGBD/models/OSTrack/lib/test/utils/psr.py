import numpy as np
from scipy.ndimage.measurements import label


def psr_fix_region(score_map, main_lobe_size=5):
    """
    Compute the B{peak-to-sidelobe ratio} (PSR) of the correlation output.  This is
    a good measure of quality. PSRs typically range from 4.0 to 70.0.  A
    when the PSR drops to 7.0-9.0 typically indicates that the tracker is
    having difficulty due to occlusion or fast appearance changes. It is
    rare to see a tracking failure for a PSR above 10.0.  Note that the
    quality method remaps PSR to a more intuitive range of [0.0,1.0].

    @return: The Peak-to-Sidelobe Ratio
    @rtype:  float
    """

    _, cols = score_map.shape

    # Find the peak
    i = score_map.argmax()
    x, y = i // cols, i % cols
    pk = score_map[x, y]
    assert pk == score_map.max()

    # Mask out the sidelobe
    mask = np.ones(score_map.shape, dtype=np.bool)
    mask[x - main_lobe_size:x + main_lobe_size, y - main_lobe_size:y + main_lobe_size] = False
    corr = score_map.flatten()
    mask = mask.flatten()
    sidelobe = corr[mask]

    # compute the psr
    mn = sidelobe.mean()
    sd = sidelobe.std()
    psr = (pk - mn) / sd
    return psr


def psr_dynamic(score_map, main_lobe_score_ratio_thresh=0.1):
    # Find the peak
    _, cols = score_map.shape
    i = score_map.argmax()
    x, y = i // cols, i % cols
    peak = score_map[x, y]
    assert peak == score_map.max()

    # find all components
    # mean = score_map.mean()
    # std = score_map.std()
    threshold = main_lobe_score_ratio_thresh * peak
    peak_regions = (score_map > threshold).astype(np.int)

    structure = np.ones((3, 3), dtype=np.int)  # this defines the connection filter
    labeled, num_components = label(peak_regions, structure)

    # Find main lobe
    peak_labeled = labeled[x, y]
    main_lobe_mask = (labeled == peak_labeled)

    # Mask out the sidelobe
    mask = np.ones(score_map.shape, dtype=np.bool)
    mask[main_lobe_mask] = False
    corr = score_map.flatten()
    mask = mask.flatten()
    side_lobe = corr[mask]

    # compute the psr
    side_mean = side_lobe.mean()
    side_std = side_lobe.std()
    psr = (peak - side_mean) / side_std

    # compute area of main lobe
    main_lobe_area = main_lobe_mask.astype(np.int).sum()

    return psr, main_lobe_area, peak
