import inspect
import os
import logging

class Logger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance.logger = logging.getLogger(__name__)
            cls._instance.logger.setLevel(logging.INFO)

            # 移除默认的 console handler
            for handler in cls._instance.logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    cls._instance.logger.removeHandler(handler)
            # Handler for console
            cls._instance.console_handler = logging.StreamHandler()
            cls._instance.console_handler.setFormatter(logging.Formatter('\r\033[K\033[34m%(asctime)s\033[0m - %(message)s', "%H:%M:%S"))
            cls._instance.logger.addHandler(cls._instance.console_handler)

            # Handler for log file will be added in main.py
            cls._instance.file_handler = None

        return cls._instance

    def set_log_file(self, log_file_path):
        if self.file_handler:
            self.logger.removeHandler(self.file_handler)
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        self.file_handler = logging.FileHandler(log_file_path)
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S"))
        self.logger.addHandler(self.file_handler)

        # Remove console handler to avoid duplicate output
        self.logger.removeHandler(self.console_handler)

    def info(self, message):
        self._log(message, logging.INFO)

    def error(self, message):
        self._log(message, logging.ERROR)  # Red color for errors

    def warning(self, message):
        self._log(message, logging.WARNING)  # Yellow color for warnings

    def _log(self, message, level, color='\033[0m'):
        # Get the file path of the caller
        frame = inspect.stack()[2]
        frame_info = inspect.getframeinfo(frame[0])
        module = inspect.getmodule(frame[0])
        filepath = os.path.relpath(module.__file__)
        lineno = frame_info.lineno

        # Clear the current line and print the message
        message = f'{filepath}:{lineno:0>4d} - {message}'
        if level == logging.INFO:
            self.logger.info(message)
        elif level == logging.ERROR:
            self.logger.error(message)
        elif level == logging.WARNING:
            self.logger.warning(message)
