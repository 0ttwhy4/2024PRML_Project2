import datetime 

config = {
    'model': dict(type='resnet18',
                  params=dict(
                      dropout=True,
                      dropout_prob=0.2
                    )),
    'optimizer': dict(type='AdamW',
                      param=dict(
                      lr=0.001,
                      # momentum=0.9,
                      betas=(0.9, 0.999),
                      eps=1e-8,
                    )),
    'train': dict(num_epochs=100,
                  input_size=64,
                  batch_size=128,
                  transform=dict(crop=True,
                                 h_flip=True,
                                 h_flip_p=0.5,
                                 v_flip=True,
                                 v_flip_p=0.5,
                                 rot=False,
                                 rot_degrees=(180, 180),
                                 gaussian_blur=False,
                                 kernel_size=11,
                                 sigma=5,
                                 gaussian_blur_p=0.2)
                  ),
    'val': dict(input_size=64,
                batch_size=64),
    'data_dir': '/home/stu6/EuroSAT_PRML24/Task_B',
    'work_dir': '/home/stu6/2024PRML_Project2/TaskB/{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
}

ensemble_config = {
    'num': 3,
    'paths': ['/home/stu6/2024PRML_Project2/TaskB/2024-06-05-10:21:04/best_model.pt',
              '/home/stu6/2024PRML_Project2/TaskB/2024-06-05-10:39:25/best_model.pt',
              '/home/stu6/2024PRML_Project2/TaskB/2024-06-05-10:47:48/best_model.pt'],
    'weights': [1, 1, 1]
}