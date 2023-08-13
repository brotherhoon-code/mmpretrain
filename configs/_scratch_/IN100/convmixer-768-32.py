BATCH_SIZE = 32 # 원래 64이나 학습시 OOM발생으로 32로 축소
LEARNING_RATE = 0.01
MAX_EPOCHS = 100
N_CLASSES = 100

model = dict(
    type='ImageClassifier',
    backbone=dict(type='ConvMixer', arch='768/32', act_cfg=dict(type='ReLU')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=N_CLASSES,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))
dataset_type = 'ImageNet'
data_preprocessor = dict(
    num_classes=N_CLASSES,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
bgr_mean = [103.53, 116.28, 123.675]
bgr_std = [57.375, 57.12, 58.395]
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
        hparams=dict(pad_val=[104, 116, 124], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]),
    dict(type='PackInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=233,
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
        ann_file='meta/train_100.txt',
        data_prefix='train_100',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True))
val_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=5,
    dataset=dict(
        type='ImageNet',
        data_root='data/imagenet',
        ann_file='meta/val_100.txt',
        data_prefix='val_100',
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
    clip_grad=dict(max_norm=5.0))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        end=20,
        convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', eta_min=1e-05, by_epoch=True, begin=20)
]
train_cfg = dict(by_epoch=True, max_epochs=MAX_EPOCHS, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=640)
default_scope = 'mmpretrain'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2, save_best="auto"),
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
                          init_kwargs=dict(entity='draph-khh',
                                           project='classification',
                                           name='convmixer-768-32'))])
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=42, deterministic=False)
custom_hooks = [dict(type='EMAHook', momentum=0.0001, priority='ABOVE_NORMAL')]

