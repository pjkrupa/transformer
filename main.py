from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("BAAI/bge-m3", device="cpu")

class EmbeddingRequest(BaseModel):
    text: str

@app.post("/embeddings")
def embeddings(req: EmbeddingRequest):
    vec = model.encode([req.text])[0]
    return {"embedding": vec.tolist()}

# launch this with:
# uvicorn main:app --host 0.0.0.0 --port 8001 --reload

