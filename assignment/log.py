
import logging

class Logger(object):

    def __init__(self, log_file_name, log_level, logger_name):

        #创建一个logger
        self.__logger = logging.getLogger(logger_name)

        #指定日志的最低输出级别，默认为WARN级别
        self.__logger.setLevel(log_level)

        #创建一个handler用于写入日志文件
        file_handler = logging.FileHandler(log_file_name)

        #创建一个handler用于输出控制台
        console_handler = logging.StreamHandler()

        #定义handler的输出格式
        formatter = logging.Formatter('[%(asctime)s] - [logger name :%(name)s] - [%(filename)s file line:%(lineno)d] - %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 给logger添加handler
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)


    def get_log(self):
        return self.__logger


 
if __name__ == '__main__':
    logger = Logger(log_file_name='log.txt', log_level=logging.DEBUG, logger_name="test").get_log()
    logger.debug('testing ... ')
