import datetime 

config = {
    'teacher': dict(model=dict(type='resnet18',
                               load_from=None,
                               # TODO: We want two options: either load from pretrained models or train *num_teacher* new models
                               params=dict(
                                    dropout=True,
                                    dropout_prob=0.2)
                                ),
                    optimizer=dict(type='SGD',
                                   param=dict(
                                        lr=0.01,
                                        momentum=0.9,
                                        # betas=(0.9, 0.999),
                                        # eps=1e-8,
                                    )),
                    # training config for teacher models
                    train=dict(num_epochs=150,
                               batch_size=64)
                    ),
    'assign_label': dict(batch_size=64, # The batch size of labeling should be interger times of K
                         teacher_file='/home/stu6/2024PRML_Project2/teachers',
                         T=5.0
                        ),
    'student': dict(model=dict(type='resnet50',
                                     params=dict(dropout=True,
                                                 dropout_prob=0.2)
                                     ),
                    optimizer=dict(type='SGD',
                                   param=dict(
                                   lr=0.01,
                                   momentum=0.9,
                                   # betas=(0.9, 0.999),
                                   # eps=1e-8,
                                )),
                    train=dict(num_epochs=80,
                               batch_size=64, 
                    )),
    'labeled_dataset': dict(transform=dict(crop=True,
                                    input_size=64,
                                    h_flip=True,
                                    h_flip_p=0.5,
                                    v_flip=True,
                                    v_flip_p=0.5,
                                    gaussian_blur=False,
                                    kernel_size=11,
                                    sigma=5,
                                    gaussian_blur_p=0.1
                                    )),
    
    'unlabeled_dataset': dict(K=2, # number of augmentation
                              transform=dict(crop=True,
                                             input_size=64,
                                             h_flip=True,
                                             h_flip_p=0.5,
                                             v_flip=True,
                                             v_flip_p=0.5,
                                             gaussian_blur=False,
                                             kernel_size=11,
                                             sigma=5,
                                             gaussian_blur_p=0.1)),
    'alpha': 0.25,
    # 'criterion': TODO: maybe we want to use some other criteria...
    
    'val': dict(input_size=64,
                batch_size=64,
                transform=dict(crop=True,
                               input_size=64,
                               h_flip=True,
                               h_flip_p=0.5,
                               v_flip=True,
                               v_flip_p=0.5,
                               gaussian_blur=False,
                               kernel_size=11,
                               sigma=5,
                               gaussian_blur_p=0.1)),
    'data_dir': '/home/stu6/EuroSAT_PRML24/Task_B',
    'work_dir': '/home/stu6/2024PRML_Project2/TaskB'
}