import logging
import coloredlogs
from utils.args import args
#import args
import sys
import os

def setup_logger(name, logfile=None):

    logger_instance = logging.getLogger(name)
    i_handler = logging.FileHandler(logfile)
    i_handler.setLevel(logging.INFO)
    logger_instance.addHandler(i_handler)
    coloredlogs.install(
        level='DEBUG', logger=logger_instance,
        fmt='%(asctime)s %(name)s %(levelname)s %(message)s')
    return logger_instance


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception
try:
    logger = setup_logger("LOG", args.logfile)
except:
    log_file = os.path.join(args.log_dir, 'log.txt')
    args.logfile = log_file
    logger = setup_logger("LOG", args.logfile)
