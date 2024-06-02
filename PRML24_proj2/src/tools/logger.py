import logging

class MyFormatter(logging.Formatter):

    def __init__(self):
        super().__init__()
        self.formats = {
            logging.DEBUG: logging.Formatter('%(message)s'),
            logging.INFO: logging.Formatter('%(asctime)s - INFO - %(message)s'),
            logging.WARNING: logging.Formatter('%(asctime)s - WARNING - %(message)s'),
            logging.ERROR: logging.Formatter('%(asctime)s - ERROR - %(message)s'),
            logging.CRITICAL: logging.Formatter('%(asctime)s - CRITICAL - %(message)s')
        }

    def format(self, record):
        # 根据记录的级别选择合适的格式
        log_fmt = self.formats.get(record.levelno)
        return log_fmt.format(record)
    
def get_logger(log_dir):
    logging.basicConfig(level=logging.DEBUG, filemode='w')
    logger = logging.getLogger(__name__)
    logger.removeHandler(logging.StreamHandler())
    
    handler = logging.FileHandler(log_dir)
    handler.setLevel(logging.DEBUG)
    formatter = MyFormatter()
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    
    return logger