BATCH_SIZE = 64
LEARNING_RATE = 1e-3 # convnext-tiny에서는 4e-3
MAX_EPOCHS = 300
VAL_INTERVAL = 1
N_CLASSES = 40

model = dict(
    type='ImageClassifier',
    backbone=dict(type='ConvNeXt', arch='tiny', drop_path_rate=0.1),
    head=dict(
        type='LinearClsHead',
        num_classes=N_CLASSES,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        init_cfg=None),
    init_cfg=dict(
        type='TruncNormal', layer=['Conv2d', 'Linear'], std=0.02, bias=0.0),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]))


data_preprocessor = dict(
    num_classes=N_CLASSES,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1/3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='PackInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=5,
    dataset=dict(
        type='ImageNet',
        data_root='data/imagenet',
        ann_file='meta/train_40.txt',
        data_prefix='train_40',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True))
val_dataloader = dict(
    batch_size=BATCH_SIZE*4,
    num_workers=5,
    dataset=dict(
        type='ImageNet',
        data_root='data/imagenet',
        ann_file='meta/val_40.txt',
        data_prefix='val_40',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False))
val_evaluator = dict(type='Accuracy', topk=(1, 5))
test_dataloader = val_dataloader
test_evaluator = val_evaluator


optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=LEARNING_RATE,
        weight_decay=0.05,
        eps=1e-08,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys=dict({
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        })),
    clip_grad=dict(max_norm=10e1000))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        end=20,
        convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', eta_min=1e-05, by_epoch=True, begin=20)
]

train_cfg = dict(by_epoch=True, max_epochs=MAX_EPOCHS, val_interval=VAL_INTERVAL)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=4096) # 원래값 convnextT config에서는 64

default_scope = 'mmpretrain'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=500), # for not save
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', 
                  vis_backends=[
                      dict(
                          type='WandbVisBackend', 
                          init_kwargs=dict(entity='brotherhoon88',
                                           project='AUG_IN40_2', # check
                                           name='config_carrot-cifar100'))])
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
custom_hooks = [dict(type='EMAHook', momentum=0.0001, priority='ABOVE_NORMAL')]

