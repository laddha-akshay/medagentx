from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
from .ingest import load_notes
from .embeddings import Embedder
from .indexer import FaissIndexer
from .query_agent import answer_clinical
import numpy as np
app = FastAPI()
static_dir = os.path.join(os.path.dirname(__file__), 'static')
app.mount('/static', StaticFiles(directory=static_dir), name='static')
embedder = Embedder()
dim = embedder.model.get_sentence_embedding_dimension()
indexer = FaissIndexer(dim=dim)
indexer.load()
@app.post('/upload-notes')
async def upload_notes(file: UploadFile = File(...)):
    path = f"/tmp/temp_{file.filename}"
    with open(path, 'wb') as f:
        f.write(await file.read())
    df = load_notes(path)
    if df.empty:
        return JSONResponse({'status':'ok', 'count':0, 'message':'no notes found'})
    texts = df['text'].tolist()
    vecs = embedder.embed_texts(texts).astype('float32')
    metas = df.to_dict(orient='records')
    indexer.add(vecs, metas)
    indexer.save()
    return {'status':'ok', 'count': len(metas)}
@app.get('/query')
def query(q: str = Query(...)):
    qv = embedder.embed_texts([q]).astype('float32')
    hits = indexer.search(qv, k=5)
    answer = answer_clinical(q, hits)
    return JSONResponse({'answer': answer.dict(), 'hits': hits})
@app.get('/')
def root():
    return HTMLResponse(open(os.path.join(static_dir, 'index.html')).read())
