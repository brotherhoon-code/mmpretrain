dataset_type = 'CIFAR100'

data_preprocessor = dict(
    num_classes=100,
    # RGB format normalization parameters
    mean=[129.304, 124.07, 112.434],
    std=[68.17, 65.392, 70.418],
    # loaded images are already RGB format
    to_rgb=False)

train_pipeline = [
    dict(type='RandomCrop', crop_size=32, padding=4),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs')
]

test_pipeline = [dict(type='PackInputs')]

train_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=128,
    num_workers=4,
    dataset=dict(
        type='CIFAR100',
        data_prefix='data/cifar100',
        test_mode=False,
        pipeline=[
            dict(type='RandomCrop', crop_size=32, padding=4),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PackInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=True))

val_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=128,
    num_workers=4,
    dataset=dict(
        type='CIFAR100',
        data_prefix='data/cifar100/',
        test_mode=True,
        pipeline=[dict(type='PackInputs')]),
    sampler=dict(type='DefaultSampler', shuffle=False))

val_evaluator = dict(type='Accuracy', topk=(1, 5))

test_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=128,
    num_workers=4,
    dataset=dict(
        type='CIFAR100',
        data_prefix='data/cifar100/',
        test_mode=True,
        pipeline=[dict(type='PackInputs')]),
    sampler=dict(type='DefaultSampler', shuffle=False))

test_evaluator = dict(type='Accuracy', topk=(1, 5))

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005))

param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[60, 120, 160], gamma=0.2)

train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=10)

val_cfg = dict()

test_cfg = dict()

auto_scale_lr = dict(base_batch_size=128*8)

default_scope = 'mmpretrain'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=500),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False)
    )

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]

visualizer = dict(type='UniversalVisualizer', 
                  vis_backends=[
                      dict(
                          type='WandbVisBackend', 
                          init_kwargs=dict(project='cls-model-exp', 
                                           name='config_carrot-cifar100'))])

log_level = 'INFO'
load_from = None
resume = False

randomness = dict(seed=None, deterministic=True)

model = dict(
    type='ImageClassifier',
    backbone=dict(type='CustomNonLocalResNet',
                  deep_stem=False,
                  n_local_cls="GaussianNonLocalBlock",
                  n_local_stage_idx=[1]
                  ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))

launcher = 'none'

work_dir = './work_dir/config_nonlocalnet-cifar100-s2s3-gau'
