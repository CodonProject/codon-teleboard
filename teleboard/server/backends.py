from abc import ABC, abstractmethod
from typing import Any
from torch.utils.tensorboard import SummaryWriter

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

class TensorBoardBackend(BaseBackend):
    '''
    TensorBoard storage backend wrapping PyTorch's SummaryWriter.

    Attributes:
        writer (SummaryWriter): The SummaryWriter instance.
    '''

    def __init__(self, log_dir: str) -> None:
        '''
        Initializes the TensorBoardBackend.

        Args:
            log_dir (str): Directory where TensorBoard event files will be written.
        '''
        self.writer = SummaryWriter(log_dir=log_dir)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        '''
        Logs a scalar value using SummaryWriter.

        Args:
            tag (str): Data identifier/tag.
            value (float): Value to plot.
            step (int): Global step value to record.
        '''
        self.writer.add_scalar(tag, value, step)

    def add_text(self, tag: str, value: str, step: int) -> None:
        '''
        Logs a text value using SummaryWriter.

        Args:
            tag (str): Data identifier/tag.
            value (str): Text string to log.
            step (int): Global step value to record.
        '''
        self.writer.add_text(tag, value, step)

    def add_image(self, tag: str, image_data: Any, step: int, dataformats: str = 'CHW') -> None:
        '''
        Logs an image using SummaryWriter.

        Args:
            tag (str): Data identifier/tag.
            image_data (Any): Image data (e.g., numpy.ndarray or torch.Tensor).
            step (int): Global step value to record.
            dataformats (str): Image data format (e.g., 'CHW', 'HWC', 'HW').
        '''
        self.writer.add_image(tag, image_data, step, dataformats=dataformats)

    def add_audio(self, tag: str, audio_data: Any, step: int, sample_rate: int = 44100) -> None:
        '''
        Logs an audio clip using SummaryWriter.

        Args:
            tag (str): Data identifier/tag.
            audio_data (Any): Audio data (e.g., numpy.ndarray or torch.Tensor).
            step (int): Global step value to record.
            sample_rate (int): Sample rate of the audio clip.
        '''
        self.writer.add_audio(tag, audio_data, step, sample_rate=sample_rate)

    def add_histogram(self, tag: str, values: Any, step: int) -> None:
        '''
        Logs a histogram using SummaryWriter.

        Args:
            tag (str): Data identifier/tag.
            values (Any): Values to build histogram (e.g., numpy.ndarray or torch.Tensor).
            step (int): Global step value to record.
        '''
        self.writer.add_histogram(tag, values, step)

    def flush(self) -> None:
        '''
        Flushes the pending logs to TensorBoard.
        '''
        self.writer.flush()

    def close(self) -> None:
        '''
        Closes the SummaryWriter.
        '''
        self.writer.close()