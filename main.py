from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder

# better re-ranker: BAAI/bge-reranker-v2-m3
# faster re-ranker: cross-encoder/ms-marco-MiniLM-L-6-v2
app = FastAPI()
embedder = SentenceTransformer("BAAI/bge-m3", device="cpu")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

class EmbeddingRequest(BaseModel):
    text: str

class RerankItem(BaseModel):
    id: str
    text: str

class RerankingRequest(BaseModel):
    query: str
    items: list[RerankItem]
    top_n: int = 5

@app.post("/embeddings")
def embeddings(req: EmbeddingRequest):
    vec = embedder.encode([req.text])[0]
    return {"embedding": vec.tolist()}

@app.post("/reranking")
def reranking(req: RerankingRequest):
    pairs = [[req.query, item.text] for item in req.items]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(req.items, scores),
        key=lambda x: x[1],
        reverse=True
    )

    top = ranked[:req.top_n]

    return {"query": req.query,
            "results": [{"id": item.id, "score": float(score)} for item, score in top]
            }

# launch this with:
# uvicorn main:app --host 0.0.0.0 --port 8001 --reload