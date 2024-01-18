import torch
from torch.utils.data.distributed import DistributedSampler
# datasets related
from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet, TNL2K
from lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.train.dataset.coco_seq_depth import MSCOCOSeq_depth
from lib.train.dataset.depthtrack import DepthTrack
from lib.train.dataset.got10k_depth import Got10k_depth
from lib.train.dataset.lasot_depth import Lasot_depth
from lib.train.dataset.saliency import Saliency
from lib.train.dataset.youtube_vos import Youtube_VOS


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE
    settings.use_cross_attn = getattr(cfg.MODEL, 'USE_CROSS_ATTN', False)
    settings.use_tar_mem = getattr(cfg.MODEL, 'USE_TAR_MEM', False)
    settings.use_dis_mem = getattr(cfg.MODEL, 'USE_DIS_MEM', False)
    settings.use_motion = getattr(cfg.MODEL, 'USE_MOTION', False)
    settings.train_cls = getattr(cfg.TRAIN, 'TRAIN_CLS', False)

    settings.merge_feat = getattr(cfg.MODEL, 'MERGE_FEAT', False)
    settings.load_prev = getattr(cfg.MODEL, 'LOAD_PREV', False)

    settings.template_align = getattr(cfg.MODEL, 'TEMPLATE_ALIGN', False)

    vl_model = getattr(cfg.MODEL, 'VL_MODEL', False)
    settings.text_model = None
    if vl_model:
        settings.text_model = getattr(cfg.MODEL.LANGUAGE, 'TYPE')


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "COCO17",
                        "VID", "TRACKINGNET", "GOT10K_official_val", "REFCOCO+", "TNL2K", "youtube_vos", "saliency",
                        "depthtrack", "depthtrack_val", "lasot_depth", "coco_depth", "got10k_depth",
                        "depthtrack_rgbd", "depthtrack_rgbd_val", "depthtrack_rgb", "depthtrack_rgb_val"]
        # dtype = '3xD'
        dtype = 'colormap'
        if name == "LASOT":
            if settings.use_lmdb:
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader, bert_model=settings.text_model))
        if name == "GOT10K_vottrain":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader))
        if name == "GOT10K_train_full":
            if settings.use_lmdb:
                print("Building got10k_train_full from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader))
        if name == "GOT10K_votval":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader))
        if name == "GOT10K_official_val":
            if settings.use_lmdb:
                raise ValueError("Not implement")
            else:
                datasets.append(Got10k(settings.env.got10k_val_dir, split=None, image_loader=image_loader))
        if name == "COCO17":
            if settings.use_lmdb:
                print("Building COCO2017 from lmdb")
                datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir, version="2017", image_loader=image_loader))
            else:
                datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader))
        if name == "REFCOCO+":
            if settings.use_lmdb:
                print("Building COCO2017 from lmdb")
                datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir, split='train_refcoco+', version="2014", image_loader=image_loader))
            else:
                datasets.append(MSCOCOSeq(settings.env.coco_dir, split='train_refcoco+', version="2014", image_loader=image_loader, bert_model=settings.text_model))
        if name == "VID":
            if settings.use_lmdb:
                print("Building VID from lmdb")
                datasets.append(ImagenetVID_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader))
        if name == "TRACKINGNET":
            if settings.use_lmdb:
                print("Building TrackingNet from lmdb")
                datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader))
            else:
                # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
                datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader))
        if name == "TNL2K":
            if settings.use_lmdb:
                raise ValueError("NOT SUPPORTED")
                # print("Building TrackingNet from lmdb")
                # datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader))
            else:
                # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
                datasets.append(TNL2K(settings.env.tnl2k_dir, image_loader=image_loader, text_model=settings.text_model))
        if name == "youtube_vos":
            if settings.use_lmdb:
                raise ValueError("NOT SUPPORTED")
                # print("Building TrackingNet from lmdb")
                # datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader))
            else:
                # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
                datasets.append(Youtube_VOS(settings.env.youtube_vos_dir, image_loader=image_loader))
        if name == "saliency":
            datasets.append(Saliency(settings.env.saliency_dir, image_loader=image_loader))
        if name == "depthtrack":
            datasets.append(DepthTrack(settings.env.depthtrack_dir, split='train', image_loader=image_loader, dtype=dtype))
        if name == "depthtrack_val":
            datasets.append(DepthTrack(settings.env.depthtrack_dir, split='val', image_loader=image_loader, dtype=dtype))
        if name == "lasot_depth":
            datasets.append(Lasot_depth(settings.env.lasot_dir, split='train', image_loader=image_loader, dtype=dtype))
        if name == "coco_depth":
            datasets.append(MSCOCOSeq_depth(settings.env.coco_dir, version="2017", image_loader=image_loader, dtype=dtype))
        if name == "got10k_depth":
            datasets.append(Got10k_depth(settings.env.got10k_dir, split='votval', image_loader=image_loader, dtype=dtype))
        if name == "depthtrack_rgbd":
            datasets.append(DepthTrack(settings.env.depthtrack_dir, split='train', image_loader=image_loader, dtype='rgbcolormap'))
        if name == "depthtrack_rgbd_val":
            datasets.append(DepthTrack(settings.env.depthtrack_dir, split='val', image_loader=image_loader, dtype='rgbcolormap'))
        if name == "depthtrack_rgb":
            datasets.append(DepthTrack(settings.env.depthtrack_dir, split='val', image_loader=image_loader, dtype='color'))
        if name == "depthtrack_rgb_val":
            datasets.append(DepthTrack(settings.env.depthtrack_dir, split='val', image_loader=image_loader, dtype='color'))

    return datasets


def build_dataloaders(cfg, settings):
    # Data transform
    # transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
    #                                 tfm.RandomHorizontalFlip(probability=0.5))
    #
    # transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
    #                                 tfm.RandomHorizontalFlip_Norm(probability=0.5),
    #                                 tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
    #
    # transform_val = tfm.Transform(tfm.ToTensor(),
    #                               tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),)

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))


    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                       output_sz=output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='sequence',
                                                       transform=transform_train,
                                                       joint_transform=transform_joint,
                                                       settings=settings)

    data_processing_val = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                     output_sz=output_sz,
                                                     center_jitter_factor=settings.center_jitter_factor,
                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                     mode='sequence',
                                                     transform=transform_val,
                                                     joint_transform=transform_joint,
                                                     settings=settings)

    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    print("sampler_mode", sampler_mode)
    use_seg = getattr(cfg.DATA, "USE_SEG", False) or getattr(cfg.TRAIN, "TRAIN_SEG", False)
    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode, train_cls=train_cls, use_seg=use_seg, merge_feat=settings.merge_feat)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    # Validation samplers and loaders
    dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
                                          p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                          samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                          max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                          num_template_frames=settings.num_template, processing=data_processing_val,
                                          frame_sample_mode=sampler_mode, train_cls=train_cls, use_seg=use_seg, merge_feat=settings.merge_feat)
    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg):
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    train_seg = getattr(cfg.TRAIN, "TRAIN_SEG", False)
    merge_feat = getattr(cfg.MODEL, 'MERGE_FEAT', False)
    load_prev = getattr(cfg.MODEL, 'LOAD_PREV', False)
    vl_model = getattr(cfg.MODEL, 'VL_MODEL', False)
    if train_cls:
        print("Only training classification head. Learnable parameters are shown below.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "cls_head" in n and p.requires_grad]}
        ]

        for n, p in net.named_parameters():
            if "cls_head" not in n:
                p.requires_grad = False
            else:
                print(n)

    elif train_seg:
        print("Only training segmentation head. Learnable parameters are shown below.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "seg_head" in n and p.requires_grad]}
        ]

        for n, p in net.named_parameters():
            if "seg_head" not in n:
                p.requires_grad = False
            else:
                print(n)

    elif merge_feat:
        print("Only training merge_module head. Learnable parameters are shown below.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if (("merge_module"  in n) or ("box_head"  in n)) and p.requires_grad]}
        ]

        for n, p in net.named_parameters():
            if ("merge_module"  in n) or ("box_head"  in n):
                print(n)
            else:
                p.requires_grad = False
    else:
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    elif cfg.TRAIN.SCHEDULER.TYPE == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.EPOCH, eta_min=0)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
