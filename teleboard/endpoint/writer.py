import io
import json
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid
import zipfile
import numpy as np
from PIL import Image
import requests

class TeleWriter:
    '''
    A client for logging experimental results to the TeleBoard server.

    This client runs a background thread to send log payloads in batches,
    supporting customizable send intervals, API key authentication, and
    automatic reconnection with exponential backoff.

    Attributes:
        exp_id (str): The unique experiment identifier.
        server_url (str): The base URL of the TeleBoard server.
        api_key (str): The API key used for authentication.
        send_interval (float): The minimum interval in seconds between sending batches.
        queue (queue.Queue): Thread-safe queue storing payloads to be sent.
        running (bool): Flag indicating whether the background thread is running.
        session (requests.Session): Persistent HTTP session.
        send_thread (threading.Thread): Background thread running the sender loop.
    '''

    def __init__(
        self,
        exp_id: str,
        server_url: str = 'http://127.0.0.1:8000',
        api_key: str = 'teleboard_secret',
        send_interval: float = 0.0
    ) -> None:
        '''
        Initializes the TeleWriter.

        Args:
            exp_id (str): The unique experiment identifier.
            server_url (str): The base URL of the TeleBoard server.
            api_key (str): The API key used for authentication.
            send_interval (float): The minimum interval in seconds between sending batches.
        '''
        self.exp_id = exp_id
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.send_interval = send_interval
        self.queue: queue.Queue = queue.Queue()
        self.running = True

        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({'X-API-Key': self.api_key})

        self.send_thread = threading.Thread(target=self._backend_sender, daemon=True)
        self.send_thread.start()

    def add_log(self, log_type: str, tag: str, value: Any, step: int, raw_bytes: Optional[bytes] = None) -> None:
        '''
        Generic method to add a log payload of any type to the sending queue.

        Args:
            log_type (str): The type of log (e.g., 'scalar', 'text', 'image', 'audio', 'histogram').
            tag (str): The tag associated with the log.
            value (Any): The log value.
            step (int): The current global step.
            raw_bytes (Optional[bytes]): Raw bytes associated with media if needed.
        '''
        self.queue.put({
            'exp_id': self.exp_id,
            'type': log_type,
            'tag': tag,
            'value': value,
            'step': int(step),
            'timestamp': time.time(),
            '_raw_bytes': raw_bytes
        })

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        '''
        Adds a scalar value to the logging queue.

        Args:
            tag (str): Data identifier/tag.
            scalar_value (float): Value to plot.
            global_step (int): Global step value to record.
        '''
        self.add_log('scalar', tag, float(scalar_value), global_step)

    def add_text(self, tag: str, text_value: str, global_step: int) -> None:
        '''
        Adds a text log to the logging queue.

        Args:
            tag (str): Data identifier/tag.
            text_value (str): Text string to log.
            global_step (int): Global step value to record.
        '''
        self.add_log('text', tag, str(text_value), global_step)

    def add_histogram(self, tag: str, values: Any, global_step: int) -> None:
        '''
        Adds a histogram to the logging queue.

        Args:
            tag (str): Data identifier/tag.
            values (Any): List, numpy.ndarray or torch.Tensor containing data.
            global_step (int): Global step value to record.
        '''
        if hasattr(values, 'tolist'):
            values_list = values.tolist()
        elif isinstance(values, np.ndarray):
            values_list = values.tolist()
        else:
            values_list = list(values)

        self.add_log('histogram', tag, values_list, global_step)

    def add_image(self, tag: str, img_tensor: Any, global_step: int) -> None:
        '''
        Adds an image to the logging queue.

        Args:
            tag (str): Data identifier/tag.
            img_tensor (Any): numpy.ndarray, PIL Image, torch.Tensor, or matplotlib.figure.Figure representing the image.
            global_step (int): Global step value to record.
        '''
        is_matplotlib_fig = False
        try:
            import matplotlib.figure as mfig
            if isinstance(img_tensor, mfig.Figure):
                is_matplotlib_fig = True
        except ImportError:
            pass

        if not is_matplotlib_fig and hasattr(img_tensor, 'savefig'):
            is_matplotlib_fig = True

        if is_matplotlib_fig:
            buf = io.BytesIO()
            img_tensor.savefig(buf, format='PNG', bbox_inches='tight')
            png_bytes = buf.getvalue()
            file_id = f'{uuid.uuid4()}.png'
            self.add_log('image', tag, file_id, global_step, raw_bytes=png_bytes)
            return

        if hasattr(img_tensor, 'detach'):
            img_tensor = img_tensor.detach().cpu().numpy()

        if isinstance(img_tensor, Image.Image):
            img = img_tensor
        elif isinstance(img_tensor, np.ndarray):
            if np.issubdtype(img_tensor.dtype, np.floating):
                if img_tensor.max() <= 1.01:
                    img_tensor = (img_tensor * 255.0).astype(np.uint8)
                else:
                    img_tensor = img_tensor.astype(np.uint8)

            shape = img_tensor.shape
            if len(shape) == 3:
                if shape[0] in (1, 3, 4) and shape[2] not in (1, 3, 4):
                    img_tensor = np.transpose(img_tensor, (1, 2, 0))

            if len(img_tensor.shape) == 3 and img_tensor.shape[2] == 1:
                img_tensor = np.squeeze(img_tensor, axis=2)

            img = Image.fromarray(img_tensor)
        else:
            raise TypeError('Unsupported image type. Use numpy.ndarray, PIL.Image, torch.Tensor, or matplotlib.figure.Figure.')

        buf = io.BytesIO()
        img.save(buf, format='PNG')
        png_bytes = buf.getvalue()

        file_id = f'{uuid.uuid4()}.png'
        self.add_log('image', tag, file_id, global_step, raw_bytes=png_bytes)

    def add_audio(self, tag: str, snd_tensor: Any, global_step: int, sample_rate: int = 44100) -> None:
        '''
        Adds an audio clip to the logging queue.

        Args:
            tag (str): Data identifier/tag.
            snd_tensor (Any): numpy.ndarray or torch.Tensor containing audio data (1D array of float32).
            global_step (int): Global step value to record.
            sample_rate (int): Sample rate of the audio clip.
        '''
        if hasattr(snd_tensor, 'detach'):
            snd_tensor = snd_tensor.detach().cpu().numpy()

        if isinstance(snd_tensor, np.ndarray):
            snd_array = snd_tensor.astype(np.float32).flatten()
        else:
            snd_array = np.array(snd_tensor, dtype=np.float32).flatten()

        audio_bytes = snd_array.tobytes()

        file_id = f'{uuid.uuid4()}.raw'
        meta = {
            'file_id': file_id,
            'sample_rate': sample_rate
        }
        self.add_log('audio', tag, meta, global_step, raw_bytes=audio_bytes)

    def _backend_sender(self) -> None:
        '''
        Background loop that polls for queued logs and sends them to the server.

        Supports throttling via send_interval and handles final flush during shutdown.
        '''
        last_send_time = time.time()

        while self.running or not self.queue.empty():
            elapsed = time.time() - last_send_time

            if self.running and self.send_interval > 0.0 and elapsed < self.send_interval:
                time_to_sleep = min(self.send_interval - elapsed, 0.1)
                if time_to_sleep > 0.0:
                    time.sleep(time_to_sleep)
                    continue

            batch: List[Dict[str, Any]] = []

            while not self.queue.empty():
                try:
                    item = self.queue.get_nowait()
                    batch.append(item)
                except queue.Empty: break

            if not batch:
                if not self.running: break
                try:
                    item = self.queue.get(timeout=0.1)
                    batch.append(item)
                except queue.Empty: continue

            if batch:
                self._send_batch_with_retry(batch)
                for _ in range(len(batch)):
                    self.queue.task_done()
                last_send_time = time.time()

    def _send_batch_with_retry(self, batch: List[Dict[str, Any]]) -> None:
        '''
        Sends a batch of logs to the server, retrying with exponential backoff on failure.

        Args:
            batch (List[Dict[str, Any]]): The list of log payloads to send.
        '''
        backoff = 1.0
        max_backoff = 60.0

        payloads_for_json: List[Dict[str, Any]] = []
        files_to_zip: List[Tuple[str, bytes]] = []

        for item in batch:
            cleaned_item = {
                'exp_id': item['exp_id'],
                'type': item['type'],
                'tag': item['tag'],
                'value': item['value'],
                'step': item['step'],
                'timestamp': item['timestamp']
            }
            payloads_for_json.append(cleaned_item)

            if item.get('_raw_bytes') is not None:
                if item['type'] == 'image':
                    file_id = item['value']
                elif item['type'] == 'audio':
                    file_id = item['value']['file_id']
                else:
                    file_id = str(uuid.uuid4())
                
                files_to_zip.append((file_id, item['_raw_bytes']))

        form_data = {
            'payloads': json.dumps(payloads_for_json)
        }

        while True:
            try:
                files = None
                zip_stream = None
                if files_to_zip:
                    zip_stream = io.BytesIO()
                    with zipfile.ZipFile(zip_stream, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for filename, data in files_to_zip:
                            zip_file.writestr(filename, data)
                    zip_stream.seek(0)
                    files = {
                        'file': ('media.zip', zip_stream, 'application/zip')
                    }

                try:
                    response = self.session.post(
                        f'{self.server_url}/api/logs',
                        data=form_data,
                        files=files,
                        timeout=10
                    )
                finally:
                    if zip_stream is not None:
                        zip_stream.close()

                if response.status_code == 200:
                    break
                
                if response.status_code == 403:
                    print(f'[TeleBoard] Authentication error (403): {response.text}')
                    break

                print(f'[TeleBoard] Server returned error {response.status_code}: {response.text}. Retrying...')
            except requests.RequestException as e:
                print(f'[TeleBoard] Network error: {e}. Reconnecting in {backoff} seconds...')

            sleep_remaining = backoff
            while sleep_remaining > 0.0:
                if not self.running:
                    print('[TeleBoard] Aborting failed log delivery due to client closing.')
                    return
                sleep_interval = min(sleep_remaining, 0.2)
                time.sleep(sleep_interval)
                sleep_remaining -= sleep_interval

            backoff = min(backoff * 2.0, max_backoff)

    def close(self) -> None:
        '''
        Closes the writer, ensuring all remaining logs are flushed to the server.
        '''
        self.running = False
        self.queue.join()
        self.session.close()