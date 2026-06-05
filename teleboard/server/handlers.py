from abc import ABC, abstractmethod
import io
from typing import Any, Dict, Optional
import zipfile
from pydantic import BaseModel
import numpy as np
from PIL import Image

from teleboard.server.backends import BaseBackend

class LogPayload(BaseModel):
    exp_id: str
    type: str
    tag: str
    value: Any
    step: int
    timestamp: float

class LogHandler(ABC):
    '''
    Abstract base class for log handlers.

    Handlers must implement the handle method to write payloads to BaseBackend.
    '''

    @abstractmethod
    def handle(self, backend: BaseBackend, payload: LogPayload, zip_file: Optional[zipfile.ZipFile] = None) -> None:
        '''
        Handles the log payload and writes it to BaseBackend.

        Args:
            backend (BaseBackend): The backend instance.
            payload (LogPayload): The log payload.
            zip_file (Optional[zipfile.ZipFile]): In-memory ZIP file containing media files.
        '''
        pass

class ScalarHandler(LogHandler):
    '''
    Handler for scalar log payloads.
    '''

    def handle(self, backend: BaseBackend, payload: LogPayload, zip_file: Optional[zipfile.ZipFile] = None) -> None:
        '''
        Writes scalar log payloads to BaseBackend.

        Args:
            backend (BaseBackend): The backend instance.
            payload (LogPayload): The log payload.
            zip_file (Optional[zipfile.ZipFile]): In-memory ZIP file.
        '''
        backend.add_scalar(payload.tag, float(payload.value), payload.step)

class TextHandler(LogHandler):
    '''
    Handler for text log payloads.
    '''

    def handle(self, backend: BaseBackend, payload: LogPayload, zip_file: Optional[zipfile.ZipFile] = None) -> None:
        '''
        Writes text log payloads to BaseBackend.

        Args:
            backend (BaseBackend): The backend instance.
            payload (LogPayload): The log payload.
            zip_file (Optional[zipfile.ZipFile]): In-memory ZIP file.
        '''
        backend.add_text(payload.tag, str(payload.value), payload.step)

class ImageHandler(LogHandler):
    '''
    Handler for image log payloads.

    Expects the value to be a dictionary or a string.
    If it's a file ID referencing an item in zip_file, reads and decodes the image.
    '''

    def handle(self, backend: BaseBackend, payload: LogPayload, zip_file: Optional[zipfile.ZipFile] = None) -> None:
        '''
        Decodes image bytes from the ZIP file and logs to BaseBackend.

        Args:
            backend (BaseBackend): The backend instance.
            payload (LogPayload): The log payload.
            zip_file (Optional[zipfile.ZipFile]): In-memory ZIP file.
        '''
        if zip_file is None:
            return

        file_id = str(payload.value)
        try:
            # Read from zip
            img_data = zip_file.read(file_id)
            # Load with PIL
            img = Image.open(io.BytesIO(img_data))
            # Convert to numpy array
            img_array = np.array(img)
            # TensorBoard expects CHW or HWC or HW. If HWC, check dimensions.
            # Normal PIL to numpy results in (H, W, C) or (H, W).
            dataformats = 'HWC' if len(img_array.shape) == 3 else 'HW'
            backend.add_image(payload.tag, img_array, payload.step, dataformats=dataformats)
        except Exception as e:
            print(f'[TeleBoard] Error handling image log for tag {payload.tag}: {e}')

class AudioHandler(LogHandler):
    '''
    Handler for audio log payloads.

    Expects value to be a dictionary containing 'file_id' and 'sample_rate'.
    The raw bytes of the audio are encoded as float32 in the zip file.
    '''

    def handle(self, backend: BaseBackend, payload: LogPayload, zip_file: Optional[zipfile.ZipFile] = None) -> None:
        '''
        Decodes audio bytes from the ZIP file and logs to BaseBackend.

        Args:
            backend (BaseBackend): The backend instance.
            payload (LogPayload): The log payload.
            zip_file (Optional[zipfile.ZipFile]): In-memory ZIP file.
        '''
        if zip_file is None:
            return

        meta = payload.value
        if not isinstance(meta, dict):
            return

        file_id = meta.get('file_id')
        sample_rate = meta.get('sample_rate', 44100)

        if not file_id:
            return

        try:
            audio_bytes = zip_file.read(file_id)
            # Reconstruct float32 1D numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
            backend.add_audio(payload.tag, audio_array, payload.step, sample_rate=sample_rate)
        except Exception as e:
            print(f'[TeleBoard] Error handling audio log for tag {payload.tag}: {e}')

class HistogramHandler(LogHandler):
    '''
    Handler for histogram log payloads.

    Expects the value to be a list of floats directly.
    '''

    def handle(self, backend: BaseBackend, payload: LogPayload, zip_file: Optional[zipfile.ZipFile] = None) -> None:
        '''
        Writes histogram data to BaseBackend.

        Args:
            backend (BaseBackend): The backend instance.
            payload (LogPayload): The log payload.
            zip_file (Optional[zipfile.ZipFile]): In-memory ZIP file.
        '''
        try:
            values = np.array(payload.value, dtype=np.float32)
            backend.add_histogram(payload.tag, values, payload.step)
        except Exception as e:
            print(f'[TeleBoard] Error handling histogram log for tag {payload.tag}: {e}')

class HandlerRegistry:
    '''
    Registry for managing and retrieving log handlers.

    Attributes:
        _handlers (Dict[str, LogHandler]): Mapping from log types to handlers.
    '''

    def __init__(self) -> None:
        '''
        Initializes the HandlerRegistry and registers default handlers.
        '''
        self._handlers: Dict[str, LogHandler] = {}
        self.register('scalar', ScalarHandler())
        self.register('text', TextHandler())
        self.register('image', ImageHandler())
        self.register('audio', AudioHandler())
        self.register('histogram', HistogramHandler())

    def register(self, log_type: str, handler: LogHandler) -> None:
        '''
        Registers a new log handler.

        Args:
            log_type (str): The type of log.
            handler (LogHandler): The handler instance.
        '''
        self._handlers[log_type] = handler

    def get_handler(self, log_type: str) -> Optional[LogHandler]:
        '''
        Retrieves a log handler for a given log type.

        Args:
            log_type (str): The type of log.

        Returns:
            Optional[LogHandler]: The handler instance if found, else None.
        '''
        return self._handlers.get(log_type)

registry = HandlerRegistry()