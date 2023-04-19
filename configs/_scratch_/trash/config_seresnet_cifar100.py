## SEResNet50 대상으로 컨피그 생성
model = dict(
    type='ImageClassifier',
    backbone=dict(type='CustomSEResNet', 
                  block_num=[3,4,6,3]),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=2048,
        # topk=(1, 5),
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))

dataset_type = 'CIFAR100'
img_norm_cfg = dict(
    mean=[129.304, 124.07, 112.434], std=[68.17, 65.392, 70.418], to_rgb=False)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[129.304, 124.07, 112.434],
        std=[68.17, 65.392, 70.418],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(
        type='Normalize',
        mean=[129.304, 124.07, 112.434],
        std=[68.17, 65.392, 70.418],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=4,
    train=dict(
        type='CIFAR100',
        data_prefix='data/cifar100',
        pipeline=[
            dict(type='RandomCrop', size=32, padding=4),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[129.304, 124.07, 112.434],
                std=[68.17, 65.392, 70.418],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='CIFAR100',
        data_prefix='data/cifar100',
        pipeline=[
            dict(
                type='Normalize',
                mean=[129.304, 124.07, 112.434],
                std=[68.17, 65.392, 70.418],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True),
    test=dict(
        type='CIFAR100',
        data_prefix='data/cifar100',
        pipeline=[
            dict(
                type='Normalize',
                mean=[129.304, 124.07, 112.434],
                std=[68.17, 65.392, 70.418],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True))

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[30,60,90], gamma=0.5)
runner = dict(type='EpochBasedRunner', max_epochs=100)

checkpoint_config = dict(interval=20)
log_config = dict(
            interval=10, # iter 기준으로 로깅을 남깁니다
            hooks=[
                dict(type='MMClsWandbHook',
                     init_kwargs={
                         'entity': 'paper-brotherhoon',
                         'project': 'cls-model-exp',
                         'name': 'senet50',
                         'notes': 'vanila'},
                     log_checkpoint=True,
                     log_checkpoint_metadata=False,
                     num_eval_images=10),
                dict(type='TextLoggerHook')
            ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dir/seresnet_cifar100'
gpu_ids = [0]
visualizer = dict(
    type='ClsVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend'),
    ]
)
randomness = dict(seed=42, deterministic=True)