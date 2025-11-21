import faiss
import pickle
from pathlib import Path
INDEX_PATH = Path("med_faiss_index.bin")
META_PATH = Path("med_meta.pkl")
class FaissIndexer:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []
    def add(self, vectors, metas):
        self.index.add(vectors)
        self.metadata.extend(metas)
    def save(self):
        faiss.write_index(self.index, str(INDEX_PATH))
        with open(META_PATH, "wb") as f:
            pickle.dump(self.metadata, f)
    def load(self):
        if INDEX_PATH.exists():
            self.index = faiss.read_index(str(INDEX_PATH))
        if META_PATH.exists():
            with open(META_PATH, 'rb') as f:
                self.metadata = pickle.load(f)
    def search(self, qvec, k=5):
        if getattr(self.index, 'ntotal', 0) == 0:
            return []
        k = min(k, getattr(self.index, 'ntotal', 0))
        D, I = self.index.search(qvec, k)
        results = []
        for row in I:
            seen = set()
            for idx in row:
                if idx < 0 or idx >= len(self.metadata):
                    continue
                if idx in seen:
                    continue
                seen.add(idx)
                results.append(self.metadata[idx])
        return results
