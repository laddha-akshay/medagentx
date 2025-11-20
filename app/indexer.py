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
        D, I = self.index.search(qvec, k)
        results = []
        for row in I:
            for idx in row:
                results.append(self.metadata[idx])
        return results
