"""Fixed-coordinate preprocessing and replace-one private structure release.

The schema and constants are protocol inputs, never fitted to protected records.
Dataset construction, classifier fitting and diagnostic releases are out of scope.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StructurePrivacyProtocol:
    epsilon: float = 1.0
    delta: float = 1e-5
    clip_norm: float = 7.0
    categorical_buckets: int = 4

    def __post_init__(self) -> None:
        if not 0 < self.epsilon <= 1:
            raise ValueError("This classical Gaussian calibration requires 0 < epsilon <= 1")
        if not 0 < self.delta < 1 or self.clip_norm <= 0:
            raise ValueError("Invalid delta or clipping radius")
        if self.categorical_buckets < 2:
            raise ValueError("At least two fixed categorical buckets are required")

    def sigma(self, public_n: int) -> float:
        if public_n < 1:
            raise ValueError("The public client size must be positive")
        return float(2 * self.clip_norm**2 / public_n / self.epsilon
                     * np.sqrt(2 * np.log(1.25 / self.delta)))

    def manifest(self) -> dict:
        return {
            **asdict(self),
            "adjacency": "replace one raw record within a fixed-size, fixed client partition",
            "public_inputs": ["feature schema", "numeric transforms", "hash algorithm",
                              "hash buckets", "client sizes and weights", "hyperparameters"],
            "numeric_missing": "zero value plus fixed missing indicator",
            "categorical_encoding": "unsigned BLAKE2b-64 modulo public bucket count per field",
            "fitted_preprocessing_parameters": False,
            "protected_outputs": "one noisy second-moment structure per disjoint client and its postprocessing",
            "excluded": ["benchmark cleaning and partition construction", "classifier training or messages",
                         "unprotected empirical diagnostics", "repeated releases without composition"],
            "rng_scope": "reproducible seeds are for trusted simulations only; not deployment releases",
        }


def numeric_rule(column: str) -> tuple[str, float]:
    if column in {"src_port", "dst_port", "tcp.srcport", "tcp.dstport", "udp.port"}:
        return "linear", 65535.0
    if column.startswith("tcp.connection.") or column in {
        "http.response", "mqtt.conflag.cleansess", "dns.qry.qu",
        "dns.retransmission", "dns.retransmit_request", "dns.retransmit_request_in",
    }:
        return "linear", 1.0
    if column in {"duration", "udp.time_delta"}:
        return "signed_log1p", 86400.0
    if column in {"arp.opcode", "arp.hw.size", "mqtt.conflags", "mqtt.hdrflags",
                  "mqtt.msgtype", "mqtt.ver", "dns_rcode"}:
        return "linear", 255.0
    if column in {"dns_qclass", "dns_qtype", "dns.qry.type", "icmp.seq_le"}:
        return "linear", 65535.0
    if column == "http_status_code":
        return "linear", 999.0
    return "signed_log1p", float(2**32 - 1)


class FixedFeatureMap:
    def __init__(self, numeric: list[str], categorical: list[str], buckets: int = 4,
                 *, canonicalize_numeric_zero: bool = True):
        self.numeric = tuple(numeric)
        self.categorical = tuple(categorical)
        self.buckets = buckets
        self.canonicalize_numeric_zero = canonicalize_numeric_zero
        if buckets < 2 or set(numeric) & set(categorical):
            raise ValueError("Invalid fixed schema")
        self.feature_names = tuple(
            [name for col in numeric for name in (f"num::{col}", f"missing::{col}")]
            + [f"hash::{col}::{bucket}" for col in categorical for bucket in range(buckets)]
        )

    def manifest(self) -> dict:
        return {
            "numeric": {col: {"transform": numeric_rule(col)[0], "scale": numeric_rule(col)[1]}
                        for col in self.numeric},
            "categorical": list(self.categorical),
            "buckets_per_field": self.buckets,
            "feature_names": list(self.feature_names),
            "dimension": len(self.feature_names),
            "constant_columns_removed": False,
            "categorical_numeric_zero_canonicalization": self.canonicalize_numeric_zero,
            "schema_sha256": hashlib.sha256(json.dumps(
                [self.numeric, self.categorical, self.buckets,
                 [numeric_rule(c) for c in self.numeric], self.canonicalize_numeric_zero], sort_keys=True
            ).encode()).hexdigest(),
        }

    def bucket(self, column: str, value: str) -> int:
        payload = json.dumps([column, value], ensure_ascii=True, separators=(",", ":"))
        return int.from_bytes(hashlib.blake2b(payload.encode(), digest_size=8).digest(), "big") % self.buckets

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        missing_columns = set(self.numeric + self.categorical) - set(frame.columns)
        if missing_columns:
            raise ValueError(f"Missing public schema columns: {sorted(missing_columns)}")
        out = np.zeros((len(frame), len(self.feature_names)), dtype=np.float64)
        for index, column in enumerate(self.numeric):
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            missing = ~np.isfinite(values)
            values = np.where(missing, 0.0, values)
            transform, scale = numeric_rule(column)
            if transform == "signed_log1p":
                values = np.sign(values) * np.log1p(np.abs(values)) / np.log1p(scale)
            else:
                values = values / scale
            out[:, 2 * index] = np.clip(values, -1, 1)
            out[:, 2 * index + 1] = missing
        offset = 2 * len(self.numeric)
        for index, column in enumerate(self.categorical):
            values = frame[column].fillna("__missing__").astype(str).str.strip().str.lower()
            if self.canonicalize_numeric_zero:
                # Public, rowwise correction for numeric-zero serialization artifacts.
                # Opaque nonzero category tokens (e.g. topic identifiers) are unchanged.
                values = values.str.replace(r"^[+-]?(?:0+(?:\.0*)?|\.0+)(?:e[+-]?\d+)?$",
                                            "0", regex=True)
            # Cache deterministic evaluations, not a learned vocabulary or category order.
            lookup = {v: self.bucket(column, v) for v in values.unique()}
            codes = values.map(lookup).to_numpy(dtype=int)
            out[np.arange(len(frame)), offset + index * self.buckets + codes] = 1.0
        if not np.isfinite(out).all():
            raise ValueError("Non-finite fixed feature map output")
        return out


def clipped_augmented(x: np.ndarray, y: np.ndarray, clip_norm: float) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Binary labels must use the public 0/1 encoding")
    z = np.column_stack([x, 2.0 * y - 1.0])
    return z * np.minimum(1.0, clip_norm / np.maximum(np.linalg.norm(z, axis=1), 1e-12))[:, None]


def empirical_structure(x: np.ndarray, y: np.ndarray, clip_norm: float) -> np.ndarray:
    z = clipped_augmented(x, y, clip_norm)
    if not len(z):
        raise ValueError("Empty client")
    return z.T @ z / len(z)


def symmetric_noise(rng: np.random.Generator, p: int, sigma: float) -> np.ndarray:
    out = np.diag(rng.normal(0, sigma, p))
    upper = np.triu_indices(p, 1)
    out[upper] = rng.normal(0, sigma / np.sqrt(2), len(upper[0]))
    out[(upper[1], upper[0])] = out[upper]
    return out
