import datetime 

config = {
    'model': dict(type='resnet18',
                  params=dict(
                      dropout=True,
                      dropout_prob=0.5
                    )),
    'optimizer': dict(type='AdamW',
                      param=dict(
                      lr=0.001,
                      betas=(0.9, 0.999),
                      eps=1e-8
                    )),
    'train': dict(num_epochs=50,
                  input_size=64,
                  batch_size=64,
                  transform=dict(crop=True,
                                 flip=True,
                                 flip_p=0.5)
                  ),
    'val': dict(input_size=64,
                batch_size=64),
    'data_dir': '/home/stu6/EuroSAT_PRML24/Task_A',
    'work_dir': '/home/stu6/2024PRML_Project2/TaskA/{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
}