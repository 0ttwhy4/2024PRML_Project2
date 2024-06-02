import datetime 

config = {
    'optimizer': dict(type='Adam',
                      param=dict(
                      lr=0.001,
                      betas=(0.9, 0.999)
                    )),
    'data_dir': '/home/stu6/EuroSAT_PRML24/Task_A',
    'work_dir': '../TaskA/{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')),
    'train': dict(num_epochs=50,
                     input_size=64,
                     batch_size=64,
                     ),
    'val': dict(input_size=64,
                batch_size=64)
}