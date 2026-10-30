import time, uuid
import numpy as np


class SemanticCache:
    def __init__(self, redis_client, encoder, threshold=0.85, ttl=3600, namespace="chat:v1"):
        self.r, self.encoder, self.threshold, self.ttl, self.ns = redis_client, encoder, threshold, ttl, namespace

    def embed(self, text):
        return self.encoder.encode([text], normalize_embeddings=True)[0].astype("float32")

    def lookup(self, prompt, vec=None):
        # returns (response or None, best_score, entry_id or None)
        vec = self.embed(prompt) if vec is None else vec
        ids = [i.decode() for i in self.r.smembers(f"{self.ns}:ids")]
        if not ids:
            self.r.incr(f"{self.ns}:stat:misses")
            return None, 0.0, None
        pipe = self.r.pipeline()
        for i in ids:
            pipe.hget(f"{self.ns}:e:{i}", "vec")
        vecs = pipe.execute()
        alive = [(i, np.frombuffer(v, dtype="float32")) for i, v in zip(ids, vecs) if v is not None]
        dead = [i for i, v in zip(ids, vecs) if v is None]            # entries whose TTL has run out
        if dead:
            self.r.srem(f"{self.ns}:ids", *dead)
        best, best_id = 0.0, None
        if alive:
            scores = np.stack([v for _, v in alive]) @ vec
            j = int(scores.argmax()); best, best_id = float(scores[j]), alive[j][0]
        if best_id is not None and best >= self.threshold:
            resp = self.r.hget(f"{self.ns}:e:{best_id}", "response")
            if resp is not None:
                self.r.incr(f"{self.ns}:stat:hits")
                return resp.decode(), best, best_id
        self.r.incr(f"{self.ns}:stat:misses")
        return None, best, None

    def store(self, prompt, response, vec=None, ttl=None):
        vec = self.embed(prompt) if vec is None else vec
        i = uuid.uuid4().hex[:12]
        key = f"{self.ns}:e:{i}"
        self.r.hset(key, mapping={"prompt": prompt, "response": response, "vec": vec.tobytes(), "ts": time.time()})
        self.r.expire(key, ttl or self.ttl)
        self.r.sadd(f"{self.ns}:ids", i)
        return i

    def stats(self):
        h = int(self.r.get(f"{self.ns}:stat:hits") or 0); m = int(self.r.get(f"{self.ns}:stat:misses") or 0)
        return {"hits": h, "misses": m, "hit_rate": round(h / (h + m), 3) if h + m else 0.0, "entries": self.r.scard(f"{self.ns}:ids")}

    def clear(self):
        for k in self.r.scan_iter(f"{self.ns}:*"):
            self.r.delete(k)
