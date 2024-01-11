import logging
import inspect
import os

class Logger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Handler for console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('\r\033[K\033[1;36m%(asctime)s\033[0m - %(message)s', "%H:%M"))
        self.logger.addHandler(console_handler)

        # Handler for log file will be added in main.py
        self.file_handler = None

    def set_log_file(self, log_file_path):
        if self.file_handler:
            self.logger.removeHandler(self.file_handler)

        self.file_handler = logging.FileHandler(log_file_path)
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M"))
        self.logger.addHandler(self.file_handler)

    def info(self, message):
        self._log(message, logging.INFO)

    def error(self, message):
        self._log(message, logging.ERROR, '\033[91m')  # Red color for errors

    def warning(self, message):
        self._log(message, logging.WARNING, '\033[93m')  # Yellow color for warnings

    def _log(self, message, level, color='\033[0m'):
        # Get the file path of the caller
        frame = inspect.stack()[2]
        module = inspect.getmodule(frame[0])
        filepath = os.path.relpath(module.__file__)

        # Clear the current line and print the message
        message = f'{color}{filepath}: {message}\033[0m'
        if level == logging.INFO:
            self.logger.info(message)
        elif level == logging.ERROR:
            self.logger.error(message)
        elif level == logging.WARNING:
            self.logger.warning(message)

# Create a global logger instance
logger = Logger()