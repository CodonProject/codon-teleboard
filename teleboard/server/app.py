from contextlib import asynccontextmanager
import io
import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Set
import zipfile
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from teleboard.server.backends import BaseBackend, FileBackend
from teleboard.server.handlers import LogPayload, registry

backends: Dict[str, BaseBackend] = {}

class ConnectionManager:
    '''
    Manager for maintaining active WebSocket connections and broadcasting messages.
    '''

    def __init__(self) -> None:
        '''
        Initializes the ConnectionManager.
        '''
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        '''
        Accepts a WebSocket connection and registers it.

        Args:
            websocket (WebSocket): The WebSocket connection instance.
        '''
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        '''
        Deregisters a WebSocket connection.

        Args:
            websocket (WebSocket): The WebSocket connection instance.
        '''
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        '''
        Broadcasts a JSON message to all active WebSocket connections.

        Args:
            message (Dict[str, Any]): The message payload to broadcast.
        '''
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

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

# Mount the static run directory to serve saved image and audio assets
os.makedirs('./runs', exist_ok=True)
app.mount('/runs', StaticFiles(directory='runs'), name='runs')

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
        backends[exp_id] = FileBackend(log_dir=full_path)
    return backends[exp_id]

async def process_batch(payloads: List[LogPayload], zip_bytes: Optional[bytes] = None) -> None:
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
        
        # Broadcast real-time update to all active web clients
        await manager.broadcast({'type': 'update_notice'})
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

@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket) -> None:
    '''
    WebSocket endpoint to stream real-time dashboard events and logs.

    Args:
        websocket (WebSocket): WebSocket connection instance.
    '''
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, listen for client requests if any
            data = await websocket.receive_text()
            try:
                request = json.loads(data)
                action = request.get('action')
                if action == 'get_runs':
                    runs = await get_runs()
                    await websocket.send_json({'type': 'runs', 'data': runs})
                elif action == 'get_scalars':
                    exp_id = request.get('exp_id')
                    scalars = await get_scalars(exp_id)
                    await websocket.send_json({'type': 'scalars', 'exp_id': exp_id, 'data': scalars})
                elif action == 'get_texts':
                    exp_id = request.get('exp_id')
                    texts = await get_texts(exp_id)
                    await websocket.send_json({'type': 'texts', 'exp_id': exp_id, 'data': texts})
                elif action == 'get_images':
                    exp_id = request.get('exp_id')
                    images = await get_images(exp_id)
                    await websocket.send_json({'type': 'images', 'exp_id': exp_id, 'data': images})
                elif action == 'get_audios':
                    exp_id = request.get('exp_id')
                    audios = await get_audios(exp_id)
                    await websocket.send_json({'type': 'audios', 'exp_id': exp_id, 'data': audios})
            except Exception:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

@app.get('/api/runs')
async def get_runs() -> List[Dict[str, Any]]:
    '''
    Recursively scans the './runs' directory and returns detailed info of each experiment run.

    An experiment run folder is recognized if it contains any of the .jsonl files,
    or directly contains images or audios directories.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing run statistics and active tags.
    '''
    runs_dir = './runs'
    if not os.path.isdir(runs_dir):
        return []

    results = []

    # Helper generator to find all subdirectories that acts as run roots
    for root, dirs, files in os.walk(runs_dir):
        # Determine if this folder is an experiment run directory
        is_run = False
        target_files = ['scalars.jsonl', 'texts.jsonl', 'images.jsonl', 'audios.jsonl', 'histograms.jsonl']
        
        if any(f in files for f in target_files):
            is_run = True
        elif 'images' in dirs or 'audios' in dirs:
            is_run = True

        if is_run:
            # The experiment ID is the path relative to './runs' with forward slashes
            exp_id = os.path.relpath(root, runs_dir).replace('\\', '/')
            
            types = []
            scalar_tags = set()
            text_tags = set()
            image_tags = set()
            audio_tags = set()
            mtime = os.path.getmtime(root)

            # Check scalars
            scalars_path = os.path.join(root, 'scalars.jsonl')
            if os.path.exists(scalars_path):
                types.append('scalar')
                mtime = max(mtime, os.path.getmtime(scalars_path))
                try:
                    with open(scalars_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                scalar_tags.add(data.get('tag'))
                except Exception:
                    pass

            # Check texts
            texts_path = os.path.join(root, 'texts.jsonl')
            if os.path.exists(texts_path):
                types.append('text')
                mtime = max(mtime, os.path.getmtime(texts_path))
                try:
                    with open(texts_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                text_tags.add(data.get('tag'))
                except Exception:
                    pass

            # Check images
            images_path = os.path.join(root, 'images.jsonl')
            if os.path.exists(images_path):
                types.append('image')
                mtime = max(mtime, os.path.getmtime(images_path))
                try:
                    with open(images_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                image_tags.add(data.get('tag'))
                except Exception:
                    pass

            # Check audios
            audios_path = os.path.join(root, 'audios.jsonl')
            if os.path.exists(audios_path):
                types.append('audio')
                mtime = max(mtime, os.path.getmtime(audios_path))
                try:
                    with open(audios_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                audio_tags.add(data.get('tag'))
                except Exception:
                    pass

            results.append({
                'exp_id': exp_id,
                'types': types,
                'tags': list(scalar_tags | text_tags | image_tags | audio_tags),
                'scalar_tags': list(scalar_tags),
                'text_tags': list(text_tags),
                'image_tags': list(image_tags),
                'audio_tags': list(audio_tags),
                'mtime': mtime
            })
    
    # Sort by modification time descending
    results.sort(key=lambda x: x['mtime'], reverse=True)
    return results

@app.get('/api/runs/{exp_id:path}/scalars')
async def get_scalars(exp_id: str) -> Dict[str, List[Dict[str, Any]]]:
    '''
    Retrieves scalar logs for a given experiment ID grouped by tag.

    Args:
        exp_id (str): The unique experiment identifier (can contain subpaths).

    Returns:
        Dict[str, List[Dict[str, Any]]]: Grouped scalar records.
    '''
    scalars_path = os.path.join('./runs', exp_id, 'scalars.jsonl')
    if not os.path.exists(scalars_path):
        return {}

    grouped_data: Dict[str, List[Dict[str, Any]]] = {}
    try:
        with open(scalars_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    tag = item.get('tag')
                    if tag not in grouped_data:
                        grouped_data[tag] = []
                    grouped_data[tag].append({
                        'value': item.get('value'),
                        'step': item.get('step'),
                        'timestamp': item.get('timestamp')
                    })
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to read scalars for {exp_id}: {e}'
        )

    # Sort each tag's items by step
    for tag in grouped_data:
        grouped_data[tag].sort(key=lambda x: x['step'])

    return grouped_data

@app.get('/api/runs/{exp_id:path}/texts')
async def get_texts(exp_id: str) -> Dict[str, List[Dict[str, Any]]]:
    '''
    Retrieves text logs for a given experiment ID grouped by tag.

    Args:
        exp_id (str): The unique experiment identifier.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Grouped text records.
    '''
    texts_path = os.path.join('./runs', exp_id, 'texts.jsonl')
    if not os.path.exists(texts_path):
        return {}

    grouped_data: Dict[str, List[Dict[str, Any]]] = {}
    try:
        with open(texts_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    tag = item.get('tag')
                    if tag not in grouped_data:
                        grouped_data[tag] = []
                    grouped_data[tag].append({
                        'value': item.get('value'),
                        'step': item.get('step'),
                        'timestamp': item.get('timestamp')
                    })
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to read texts for {exp_id}: {e}'
        )

    # Sort each tag's items by step
    for tag in grouped_data:
        grouped_data[tag].sort(key=lambda x: x['step'])

    return grouped_data

@app.get('/api/runs/{exp_id:path}/images')
async def get_images(exp_id: str) -> Dict[str, List[Dict[str, Any]]]:
    '''
    Retrieves image log metadata for a given experiment ID grouped by tag.

    Args:
        exp_id (str): The unique experiment identifier.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Grouped image records with web URL path.
    '''
    images_path = os.path.join('./runs', exp_id, 'images.jsonl')
    if not os.path.exists(images_path):
        return {}

    grouped_data: Dict[str, List[Dict[str, Any]]] = {}
    try:
        with open(images_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    tag = item.get('tag')
                    if tag not in grouped_data:
                        grouped_data[tag] = []
                    grouped_data[tag].append({
                        'step': item.get('step'),
                        'url': f'/runs/{exp_id}/{item.get("url")}',
                        'timestamp': item.get('timestamp')
                    })
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to read images for {exp_id}: {e}'
        )

    # Sort each tag's items by step
    for tag in grouped_data:
        grouped_data[tag].sort(key=lambda x: x['step'])

    return grouped_data

@app.get('/api/runs/{exp_id:path}/audios')
async def get_audios(exp_id: str) -> Dict[str, List[Dict[str, Any]]]:
    '''
    Retrieves audio log metadata for a given experiment ID grouped by tag.

    Args:
        exp_id (str): The unique experiment identifier.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Grouped audio records with web URL path.
    '''
    audios_path = os.path.join('./runs', exp_id, 'audios.jsonl')
    if not os.path.exists(audios_path):
        return {}

    grouped_data: Dict[str, List[Dict[str, Any]]] = {}
    try:
        with open(audios_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    tag = item.get('tag')
                    if tag not in grouped_data:
                        grouped_data[tag] = []
                    grouped_data[tag].append({
                        'step': item.get('step'),
                        'url': f'/runs/{exp_id}/{item.get("url")}',
                        'timestamp': item.get('timestamp')
                    })
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to read audios for {exp_id}: {e}'
        )

    # Sort each tag's items by step
    for tag in grouped_data:
        grouped_data[tag].sort(key=lambda x: x['step'])

    return grouped_data

@app.get('/', response_class=HTMLResponse)
async def get_dashboard() -> str:
    '''
    Serves the single-page Dashboard HTML.
    '''
    index_path = os.path.join(os.path.dirname(__file__), 'index.html')
    if not os.path.exists(index_path):
        return '<html><body><h1>Dashboard file (index.html) not found in server package folder!</h1></body></html>'
    
    with open(index_path, 'r', encoding='utf-8') as f:
        return f.read()

def run(host:str='0.0.0.0', port:int=8000):
    uvicorn.run(app, host=host, port=port)