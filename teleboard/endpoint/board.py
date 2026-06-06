from typing import Dict
from teleboard.endpoint.writer import TeleWriter

class TeleBoard:
    '''
    A manager class for holding global connection settings and opening individual experiment writers.

    Attributes:
        server_url (str): The base URL of the TeleBoard server.
        api_key (str): The API key used for authentication.
        send_interval (float): The minimum interval in seconds between sending batches.
        writers (Dict[str, TeleWriter]): Cache of opened TeleWriter instances.
    '''

    def __init__(
        self,
        server_url: str = 'http://localhost:8000',
        api_key: str = 'teleboard_secret',
        send_interval: float = 0.0
    ) -> None:
        '''
        Initializes the TeleBoard manager.

        Args:
            server_url (str): The base URL of the TeleBoard server.
            api_key (str): The API key used for authentication.
            send_interval (float): The minimum interval in seconds between sending batches.
        '''
        self.server_url = server_url
        self.api_key = api_key
        self.send_interval = send_interval
        self.writers: Dict[str, TeleWriter] = {}

    def open_exp(self, exp_id: str) -> TeleWriter:
        '''
        Opens an experiment writer for the given experiment ID, using a cached instance if available.

        Args:
            exp_id (str): The unique experiment identifier.

        Returns:
            TeleWriter: The corresponding TeleWriter client instance.
        '''
        if exp_id not in self.writers:
            self.writers[exp_id] = TeleWriter(
                exp_id=exp_id,
                server_url=self.server_url,
                api_key=self.api_key,
                send_interval=self.send_interval
            )
        return self.writers[exp_id]

    def close(self) -> None:
        '''
        Closes all opened experiment writers, flushing remaining logs.
        '''
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()