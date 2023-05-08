_base_ = [
    '../_base_/datasets/cifar100_bs16.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(type='Carrot'), 
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)