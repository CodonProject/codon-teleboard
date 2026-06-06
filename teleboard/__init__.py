import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from .endpoint import TeleWriter, TeleBoard

__version__ = '0.0.1a2'