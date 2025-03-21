import logging
import sys
import os

class StdLogger():
    def __init__(self, file_path = "", stream = False, level=logging.INFO):
        formatter = logging.Formatter(fmt='%(asctime)s %(message)s',
                                      datefmt="%H:%M:%S")
        self.logger = logging.getLogger(os.path.basename(__file__))
        self.logger.setLevel(level)
        self.file_path = file_path

        if file_path:
            file_hander = logging.FileHandler(file_path)
            file_hander.setFormatter(formatter)
            self.logger.addHandler(file_hander)

        if stream:
            stream_handler = logging.StreamHandler(stream=sys.stderr)
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

# cfg = OmegaConf.load(
#     os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_conf.yaml'))
# log_dir = os.path.join(cfg.logger.log_dir, str(int(time())))
# os.makedirs(log_dir)
# file_path = os.path.join(log_dir, "train.log")

Logger = StdLogger("", level=logging.INFO).logger