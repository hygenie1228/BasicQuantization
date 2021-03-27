import time
import datetime
import logging

from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self):
        self.log_interval = 0
        self.start_time = time.time()
        
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        # to stream
        formatter = logging.Formatter('\x1b[2;32m[%(asctime)s]\x1b[0m\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        
        # to file
        timestamp = str(datetime.datetime.now())[:-7]
        file_handler = logging.FileHandler('./outputs/' + timestamp + '.log')

        formatter = logging.Formatter('[%(asctime)s]\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(stream_handler)
        self.logger.addHandler(file_handler)

        # tensorboard
        self.writer = SummaryWriter('../runs')

        self.precision_file = './outputs/' + timestamp + '_precision.txt'
        self.quan_precision_file = './outputs/' + timestamp + '_quan_precision.txt'
    
    def set_log_interval(self, log_interval):
        self.log_interval = log_interval

    def info(self, message):
        self.logger.info(message)

    def train_log(self, iters, loss, batch_size):
        if iters % self.log_interval == 0:
            duration = time.time() - self.start_time
            self.writer.add_scalar('Train_Loss/cls_loss', loss.item(), iters)
            self.writer.add_scalar('Train_Loss/samples_per_sec', batch_size*self.log_interval/duration, iters)
            self.start_time = time.time()

    def test_log(self, epoch, acc, best_acc):
        self.writer.add_scalar('Test_Acc/accuracy', acc, epoch)
        self.writer.add_scalar('Test_Acc/best_accuracy', best_acc, epoch)

        self.logger.info("Epoch %d" % epoch)
        self.logger.info("Precision@1      : %.3f%%" % acc)
        self.logger.info("Best Precision@1 : %.3f%%" % best_acc)

    def quantize_log(self, epoch, acc1, acc2, best_acc, weight_ratio):
        self.logger.info("Epoch %d" % epoch)
        self.logger.info("(Weight ratio        : " + "%.6f"%weight_ratio + ")")
        self.logger.info("Prec : %.2f%%  |  Quan Prec : %.2f%%  |  Best Quan Prec : %.2f%%"%(acc1, acc2, best_acc))

logger = Logger()