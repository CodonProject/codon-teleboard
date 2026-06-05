from abc import ABC, abstractmethod
import json
import os
import threading
import time
from typing import Any, Dict, List
import wave
import numpy as np
from PIL import Image

class BaseBackend(ABC):
    '''
    Abstract base class for storage/logging backends.
    '''

    @abstractmethod
    def add_scalar(self, tag: str, value: float, step: int) -> None:
        '''
        Logs a scalar value.

        Args:
            tag (str): Data identifier/tag.
            value (float): Value to plot.
            step (int): Global step value to record.
        '''
        pass

    @abstractmethod
    def add_text(self, tag: str, value: str, step: int) -> None:
        '''
        Logs a text value.

        Args:
            tag (str): Data identifier/tag.
            value (str): Text string to log.
            step (int): Global step value to record.
        '''
        pass

    @abstractmethod
    def add_image(self, tag: str, image_data: Any, step: int, dataformats: str = 'CHW') -> None:
        '''
        Logs an image.

        Args:
            tag (str): Data identifier/tag.
            image_data (Any): Image data (e.g., numpy.ndarray or torch.Tensor).
            step (int): Global step value to record.
            dataformats (str): Image data format (e.g., 'CHW', 'HWC', 'HW').
        '''
        pass

    @abstractmethod
    def add_audio(self, tag: str, audio_data: Any, step: int, sample_rate: int = 44100) -> None:
        '''
        Logs an audio clip.

        Args:
            tag (str): Data identifier/tag.
            audio_data (Any): Audio data (e.g., numpy.ndarray or torch.Tensor).
            step (int): Global step value to record.
            sample_rate (int): Sample rate of the audio clip.
        '''
        pass

    @abstractmethod
    def add_histogram(self, tag: str, values: Any, step: int) -> None:
        '''
        Logs a histogram.

        Args:
            tag (str): Data identifier/tag.
            values (Any): Values to build histogram (e.g., numpy.ndarray or torch.Tensor).
            step (int): Global step value to record.
        '''
        pass

    @abstractmethod
    def flush(self) -> None:
        '''
        Flushes the pending logs to the backend.
        '''
        pass

    @abstractmethod
    def close(self) -> None:
        '''
        Closes the backend and releases resources.
        '''
        pass


class FileBackend(BaseBackend):
    '''
    A file-based storage backend that logs data to local disk in structured JSONL files.

    It also saves images and audio clips into separate files for web access.

    Attributes:
        log_dir (str): Directory where the experiment log files will be written.
        lock (threading.Lock): Mutex lock to ensure thread safety.
    '''

    def __init__(self, log_dir: str) -> None:
        '''
        Initializes the FileBackend.

        Args:
            log_dir (str): Directory where the experiment logs will be written.
        '''
        self.log_dir = log_dir
        self.lock = threading.Lock()
        os.makedirs(self.log_dir, exist_ok=True)

    def _append_jsonl(self, filename: str, data: Dict[str, Any]) -> None:
        '''
        Appends a dictionary as a JSON line into a file.

        Args:
            filename (str): The name of the file to append to.
            data (Dict[str, Any]): The dict containing the log payload.
        '''
        filepath = os.path.join(self.log_dir, filename)
        with self.lock:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data) + '\n')

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        '''
        Logs a scalar value by appending it to scalars.jsonl.

        Args:
            tag (str): Data identifier/tag.
            value (float): Value to plot.
            step (int): Global step value to record.
        '''
        payload = {
            'tag': tag,
            'value': float(value),
            'step': int(step),
            'timestamp': time.time()
        }
        self._append_jsonl('scalars.jsonl', payload)

    def add_text(self, tag: str, value: str, step: int) -> None:
        '''
        Logs a text value by appending it to texts.jsonl.

        Args:
            tag (str): Data identifier/tag.
            value (str): Text string to log.
            step (int): Global step value to record.
        '''
        payload = {
            'tag': tag,
            'value': str(value),
            'step': int(step),
            'timestamp': time.time()
        }
        self._append_jsonl('texts.jsonl', payload)

    def add_image(self, tag: str, image_data: Any, step: int, dataformats: str = 'CHW') -> None:
        '''
        Logs an image. Saves the image file and records meta in images.jsonl.

        Args:
            tag (str): Data identifier/tag.
            image_data (Any): Image data (e.g., numpy.ndarray or torch.Tensor).
            step (int): Global step value to record.
            dataformats (str): Image data format (e.g., 'CHW', 'HWC', 'HW').
        '''
        if hasattr(image_data, 'detach'):
            img_array = image_data.detach().cpu().numpy()
        elif isinstance(image_data, np.ndarray):
            img_array = image_data
        else:
            img_array = np.array(image_data)

        if dataformats == 'CHW':
            if len(img_array.shape) == 3:
                img_array = np.transpose(img_array, (1, 2, 0))
        elif dataformats == 'HW':
            pass

        if np.issubdtype(img_array.dtype, np.floating):
            if img_array.max() <= 1.01:
                img_array = (img_array * 255.0).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)

        tag_safe = tag.replace('/', '_').replace('\\', '_').replace(' ', '_')
        image_dir = os.path.join(self.log_dir, 'images', tag_safe)
        os.makedirs(image_dir, exist_ok=True)

        filename = f'{step}.png'
        filepath = os.path.join(image_dir, filename)

        with self.lock:
            img = Image.fromarray(img_array)
            img.save(filepath, 'PNG')

        web_path = f'images/{tag_safe}/{filename}'
        payload = {
            'tag': tag,
            'step': int(step),
            'url': web_path,
            'timestamp': time.time()
        }
        self._append_jsonl('images.jsonl', payload)

    def add_audio(self, tag: str, audio_data: Any, step: int, sample_rate: int = 44100) -> None:
        '''
        Logs an audio clip as a WAV file and appends metadata to audios.jsonl.

        Args:
            tag (str): Data identifier/tag.
            audio_data (Any): Audio data (1D numpy array of float32).
            step (int): Global step value to record.
            sample_rate (int): Sample rate of the audio clip.
        '''
        if hasattr(audio_data, 'detach'):
            snd_array = audio_data.detach().cpu().numpy()
        elif isinstance(audio_data, np.ndarray):
            snd_array = audio_data
        else:
            snd_array = np.array(audio_data)

        snd_array = snd_array.flatten()

        if np.issubdtype(snd_array.dtype, np.floating):
            max_val = np.abs(snd_array).max()
            if max_val > 0.0:
                snd_array = snd_array / max_val
            snd_array = (snd_array * 32767.0).astype(np.int16)

        tag_safe = tag.replace('/', '_').replace('\\', '_').replace(' ', '_')
        audio_dir = os.path.join(self.log_dir, 'audios', tag_safe)
        os.makedirs(audio_dir, exist_ok=True)

        filename = f'{step}.wav'
        filepath = os.path.join(audio_dir, filename)

        with self.lock:
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(snd_array.tobytes())

        web_path = f'audios/{tag_safe}/{filename}'
        payload = {
            'tag': tag,
            'step': int(step),
            'url': web_path,
            'timestamp': time.time()
        }
        self._append_jsonl('audios.jsonl', payload)

    def add_histogram(self, tag: str, values: Any, step: int) -> None:
        '''
        Logs a histogram of values.

        Args:
            tag (str): Data identifier/tag.
            values (Any): Values to build histogram (numpy.ndarray or torch.Tensor).
            step (int): Global step value to record.
        '''
        if hasattr(values, 'detach'):
            val_array = values.detach().cpu().numpy()
        elif isinstance(values, np.ndarray):
            val_array = values
        else:
            val_array = np.array(values)

        val_list = val_array.flatten().tolist()

        payload = {
            'tag': tag,
            'values': val_list,
            'step': int(step),
            'timestamp': time.time()
        }
        self._append_jsonl('histograms.jsonl', payload)

    def flush(self) -> None:
        '''
        Flushes pending logs. This is a no-op as logs are written synchronously.
        '''
        pass

    def close(self) -> None:
        '''
        Closes the backend.
        '''
        pass