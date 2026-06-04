from contextlib import asynccontextmanager
import io
import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Set
import zipfile
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
import uvicorn

from .backends import BaseBackend, TensorBoardBackend
from .handlers import LogPayload, registry

backends: Dict[str, BaseBackend] = {}

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    '''
    Lifespan event handler for the FastAPI application.

    Args:
        app (FastAPI): The FastAPI application instance.
    '''
    yield
    for backend in backends.values():
        backend.close()

app = FastAPI(title='TeleBoard Server', lifespan=lifespan)

async def verify_api_key(x_api_key: Optional[str] = Header(None, alias='X-API-Key')) -> None:
    '''
    Verifies the API key provided in the request headers.

    Args:
        x_api_key (Optional[str]): The API key passed in the header.

    Raises:
        HTTPException: If the API key is invalid or missing.
    '''
    expected_key = os.getenv('TELEBOARD_API_KEY', 'teleboard_secret')
    if x_api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Invalid or missing API Key'
        )

def get_backend(exp_id: str) -> BaseBackend:
    '''
    Retrieves or creates a BaseBackend for the given experiment ID.

    Args:
        exp_id (str): The unique experiment identifier.

    Returns:
        BaseBackend: The BaseBackend instance.
    '''
    if exp_id not in backends:
        full_path = os.path.join('./runs', exp_id)
        backends[exp_id] = TensorBoardBackend(log_dir=full_path)
    return backends[exp_id]

def process_batch(payloads: List[LogPayload], zip_bytes: Optional[bytes] = None) -> None:
    '''
    Processes a batch of payloads and writes them to backends.

    Args:
        payloads (List[LogPayload]): A list of log payloads.
        zip_bytes (Optional[bytes]): Optional raw bytes of the zip file containing media assets.
    '''
    affected_backends: Set[BaseBackend] = set()

    zip_file = None
    if zip_bytes is not None:
        zip_file = zipfile.ZipFile(io.BytesIO(zip_bytes))

    try:
        for payload in payloads:
            handler = registry.get_handler(payload.type)
            if handler is not None:
                backend = get_backend(payload.exp_id)
                handler.handle(backend, payload, zip_file)
                affected_backends.add(backend)

        for backend in affected_backends:
            backend.flush()
    finally:
        if zip_file is not None:
            zip_file.close()

@app.post('/api/logs', dependencies=[Depends(verify_api_key)])
async def receive_logs(
    payloads: str = Form(...),
    file: Optional[UploadFile] = File(None),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> Dict[str, Any]:
    '''
    Endpoint to receive logs (JSON + media ZIP) and queue them for processing.

    Args:
        payloads (str): List of log payloads as a JSON string.
        file (Optional[UploadFile]): ZIP file containing the assets.
        background_tasks (BackgroundTasks): Background tasks manager.

    Returns:
        Dict[str, Any]: A dict containing status and count of received logs.
    '''
    try:
        parsed_payloads = [LogPayload(**p) for p in json.loads(payloads)]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Invalid JSON or log payloads: {e}'
        )

    zip_bytes = None
    if file is not None:
        zip_bytes = await file.read()

    background_tasks.add_task(process_batch, parsed_payloads, zip_bytes)
    return {'status': 'success', 'received_count': len(parsed_payloads)}

def run(host:str='0.0.0.0', port:int=8000):
    uvicorn.run(app, host=host, port=port)