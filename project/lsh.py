import numpy as np
from collections import defaultdict
from typing import List, Iterable, Dict, Tuple

class MinHashLSH:
    """
    MinHash LSH για Jaccard similarity πάνω σε σύνολα tokens (π.χ. genre_names).
    """

    def __init__(self, num_perm: int = 80, num_bands: int = 20, seed: int = 42):
        assert num_perm % num_bands == 0, "num_perm must be divisible by num_bands"
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands

        rng = np.random.default_rng(seed)
        # large prime for hashing
        self.p = 4294967311  # > 2^32
        self.a = rng.integers(1, self.p - 1, size=num_perm, dtype=np.int64)
        self.b = rng.integers(0, self.p - 1, size=num_perm, dtype=np.int64)

        self.buckets: Dict[Tuple[int, ...], List[int]] = defaultdict(list)

    @staticmethod
    def _token_hash(token: str) -> int:
        # stable hash -> uint32-like
        return (hash(token) & 0xFFFFFFFF)

    def _minhash_signature(self, tokens: Iterable[str]) -> np.ndarray:
        tokens = list(tokens)
        if len(tokens) == 0:
            # empty set -> signature all max
            return np.full(self.num_perm, np.iinfo(np.int64).max, dtype=np.int64)

        x = np.array([self._token_hash(t) for t in tokens], dtype=np.int64)
        # compute (a*x + b) % p for each permutation, take min over tokens
        sig = np.min((self.a[:, None] * x[None, :] + self.b[:, None]) % self.p, axis=1)
        return sig

    def _band_key(self, sig: np.ndarray, band: int) -> Tuple[int, ...]:
        start = band * self.rows_per_band
        end = start + self.rows_per_band
        return (band, *sig[start:end].tolist())

    def add(self, tokens: List[str], index: int):
        sig = self._minhash_signature(tokens)
        for band in range(self.num_bands):
            key = self._band_key(sig, band)
            self.buckets[key].append(index)

    def query(self, tokens: List[str]) -> List[int]:
        sig = self._minhash_signature(tokens)
        candidates = set()
        for band in range(self.num_bands):
            key = self._band_key(sig, band)
            candidates.update(self.buckets.get(key, []))
        return list(candidates)
