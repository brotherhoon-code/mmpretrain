# dataset settings
BATCH_SIZE = 64
LEARNING_RATE = 5e-4 # 5e-4*BATCH_SIZE*1/512, lr = 5e-4 * 128(batch_size) * 8(n_gpu) / 512 = 0.001
MAX_EPOCHS = 100
VAL_INTERVAL = 5
N_CLASSES = 20

dataset_type = 'ImageNet'

data_preprocessor = dict(
    num_classes=N_CLASSES,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/train_20.txt',
        data_prefix='train_20',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/val_20.txt',
        data_prefix='val_20',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# lr = 5e-4 * 128(batch_size) * 8(n_gpu) / 512 = 0.001
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr= LEARNING_RATE, # 0.045789 
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=5.0)
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=20,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=20)
]

train_cfg = dict(by_epoch=True, max_epochs=MAX_EPOCHS, val_interval=VAL_INTERVAL)
val_cfg = dict()
test_cfg = dict()

auto_scale_lr = dict(base_batch_size=BATCH_SIZE)
default_scope = 'mmpretrain'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
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
                          init_kwargs=dict(entity='brotherhoon88',
                                           project='!AUG_IN20', # check
                                           name='config_carrot-cifar100'))])
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=None, deterministic=True)

model = dict(
    type='ImageClassifier',
    backbone=dict(type='A4',
                  stage_channels=[96, 192, 384, 768],
                  stage_blocks=[3, 3, 9, 3],
                  patch_size=[4, 2, 2, 2],
                  kernel_size=7),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=N_CLASSES,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))


launcher = 'none'

work_dir = './work_dir/config_carrot'
