from datetime import datetime
import logging
import os
import sys

PID = str(os.getpid())
ROTATE_LOGGER = None
ROTATE_LOGGER_PATH = os.path.dirname(os.path.realpath(__file__)) + "/log/" + \
                     sys.argv[0].split("/")[-1].replace(".py", ".log")
ROTATE_LOGGER_WHEN = "midnight"
ROTATE_LOGGER_BACKUP_COUNT = 30

class Bcolor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[35m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    LIGHTBLUE = '\x1b[36m'
    MUDDY = '\x1b[33m'
    CYAN = '\033[96m'
    RED = FAIL
    MAGENTA = HEADER


class DebugLevel:
    LOG = 1
    DEBUG = 2
    WARNING = 3
    ERROR = 4


debug_saving = False
debug_string = ""


def debug_msg(msg, level=DebugLevel.LOG):
    color1 = Bcolor.ENDC
    color2 = Bcolor.OKBLUE
    level_label = ""
    if level == DebugLevel.DEBUG:
        color1 = Bcolor.ENDC
        color2 = Bcolor.MUDDY
        level_label = "DEBUG: "
    elif level == DebugLevel.WARNING:
        color1 = Bcolor.ENDC
        color2 = Bcolor.WARNING
        level_label = "WARNING: "
    elif level == DebugLevel.ERROR:
        color1 = Bcolor.ENDC
        color2 = Bcolor.FAIL
        level_label = "ERROR: "
    print_string = u'[' + color1 + '{}' + Bcolor.ENDC + '] [' + PID + '] ' + color2 + level_label + '{}' + Bcolor.ENDC # with pid

    global ROTATE_LOGGER
    if not ROTATE_LOGGER:
        ROTATE_LOGGER = logging.getLogger("Rotating Log")
        ROTATE_LOGGER.setLevel(logging.INFO)
        log_dir = os.path.dirname(ROTATE_LOGGER_PATH)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        handler = logging.FileHandler(ROTATE_LOGGER_PATH)
        ROTATE_LOGGER.addHandler(handler)
    ROTATE_LOGGER.info(print_string.format(datetime.now().strftime('%y-%m-%d %H:%M:%S'), msg))

    global debug_saving
    if debug_saving:
        global debug_string
        debug_string += print_string.format(datetime.now().strftime('%y-%m-%d %H:%M:%S'), msg) + "\n"
    return


def colored_msg(msg, color=Bcolor.ENDC):
    print_string = color + u'{}' + Bcolor.ENDC
    print(print_string.format(msg))
    return
