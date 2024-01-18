from collections import namedtuple
import importlib
from lib.test.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = "lib.test.evaluation.%sdataset"  # Useful abbreviations to reduce the clutter

dataset_dict = dict(
    trackingnet=DatasetInfo(module=pt % "trackingnet", class_name="TrackingNetDataset", kwargs=dict()),
    got10k_test=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='test')),
    got10k_val=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='val')),
    got10k_ltrval=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='ltrval')),
    lasot=DatasetInfo(module=pt % "lasot", class_name="LaSOTDataset", kwargs=dict()),
    lasot_lmdb=DatasetInfo(module=pt % "lasot_lmdb", class_name="LaSOTlmdbDataset", kwargs=dict()),

    vot=DatasetInfo(module=pt % "vot", class_name="VOTDataset", kwargs=dict()),
    otb=DatasetInfo(module=pt % "otb", class_name="OTBDataset", kwargs=dict()),
    uav=DatasetInfo(module=pt % "uav", class_name="UAVDataset", kwargs=dict()),
    tc128=DatasetInfo(module=pt % "tc128", class_name="TC128Dataset", kwargs=dict()),
    tc128ce=DatasetInfo(module=pt % "tc128ce", class_name="TC128CEDataset", kwargs=dict()),
    nfs=DatasetInfo(module=pt % "nfs", class_name="NFSDataset", kwargs=dict()),
    vot20=DatasetInfo(module=pt % "vot", class_name="VOTDataset", kwargs=dict(year=20)),
    vot22=DatasetInfo(module=pt % "vot", class_name="VOTDataset", kwargs=dict(year=22)),
    vot22_rgbd_color=DatasetInfo(module=pt % "votrgbd", class_name="VOTRGBDDataset", kwargs=dict(dtype='color')),
    vot22_rgbd_depth=DatasetInfo(module=pt % "votrgbd", class_name="VOTRGBDDataset", kwargs=dict(dtype='colormap')),
    lasot_train=DatasetInfo(module=pt % "lasot_train", class_name="LaSOTTrainDataset", kwargs=dict()),
    lasot_extension_subset=DatasetInfo(module=pt % "lasotextensionsubset", class_name="LaSOTExtensionSubsetDataset", kwargs=dict()),
    itb=DatasetInfo(module=pt % "itb", class_name="ITBDataset", kwargs=dict()),
    tnl2k=DatasetInfo(module=pt % "tnl2k", class_name="TNL2kDataset", kwargs=dict()),

    lasot_p1=DatasetInfo(module=pt % "lasot", class_name="LaSOTDataset", kwargs=dict(part='p1')),
    lasot_p2=DatasetInfo(module=pt % "lasot", class_name="LaSOTDataset", kwargs=dict(part='p2')),
    lasot_p3=DatasetInfo(module=pt % "lasot", class_name="LaSOTDataset", kwargs=dict(part='p3')),
)


def load_dataset(name: str):
    """ Import and load a single dataset."""
    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs)  # Call the constructor
    return dataset.get_sequence_list()


def get_dataset(*args):
    """ Get a single or set of datasets."""
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name))
    return dset