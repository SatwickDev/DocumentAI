#!/usr/bin/env python3
"""
Ultra-Fast Contrast Score Calculator (3+2 Ensemble)
Refactored to run sequentially for speed.
"""

import numpy as np
import cv2
import time
import hashlib
from numba import jit, prange
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class ContrastConfig:
    primary_methods: List[str] = field(default_factory=lambda: [
        "rms", "michelson", "local"
    ])
    fallback_methods: List[str] = field(default_factory=lambda: [
        "weber", "entropy"
    ])
    use_downsampling: bool = True
    downsample_threshold: int = 1000
    downsample_factor: float = 0.5
    consistency_threshold: float = 0.7
    min_methods_required: int = 2
    cache_enabled: bool = True
    max_workers: int = 3
    method_weights: Dict[str, float] = field(default_factory=lambda: {
        "rms": 0.35,
        "michelson": 0.30,
        "local": 0.25,
        "weber": 0.05,
        "entropy": 0.05
    })
    min_contrast: float = 0.05
    good_contrast: float = 0.25
    excellent_contrast: float = 0.4

@dataclass
class MethodResult:
    method_name: str
    score: float
    confidence: float
    processing_time_ms: float
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)

@dataclass
class ContrastResult:
    is_valid: bool
    value: float
    confidence: float
    warnings: List[str]
    fallback_used: bool
    processing_time_ms: float
    method_results: Dict[str, MethodResult]
    methods_used: List[str]
    consistency_score: float

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def fast_rms(image: np.ndarray) -> float:
    mean = np.mean(image)
    return np.sqrt(np.mean((image - mean) ** 2)) / 255.0

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def fast_local_std(image: np.ndarray, w: int) -> float:
    h, ww = image.shape
    vals = []
    for i in prange(0, h-w+1, w):
        for j in prange(0, ww-w+1, w):
            win = image[i:i+w, j:j+w]
            vals.append(np.std(win))
    return np.mean(vals) / 255.0 if vals else 0.0

def shared_preprocessing(image: np.ndarray, config: ContrastConfig) -> np.ndarray:
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            gray = image[:,:,0]
    else:
        gray = image.copy()
    if config.use_downsampling and max(gray.shape) > config.downsample_threshold:
        scale = config.downsample_factor
        gray = cv2.resize(gray, (int(gray.shape[1]*scale), int(gray.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32)

def method_worker(method, image, config: ContrastConfig) -> MethodResult:
    t0 = time.perf_counter()
    try:
        if method == "rms":
            score = fast_rms(image)
            conf = min(1.0, 2*score)
        elif method == "michelson":
            mx, mn = np.max(image), np.min(image)
            denom = mx + mn + 1e-10
            if denom == 0: return MethodResult(method, 0.0, 0.0, (time.perf_counter()-t0)*1000, False, ["Zero denom"])
            score = min((mx - mn) / denom, 2.0)
            conf = min(1.0, 2*score)
        elif method == "local":
            score = fast_local_std(image, 15)
            conf = min(1.0, 2*score)
        elif method == "weber":
            bg = np.median(image)
            if bg == 0: return MethodResult(method, 0.0, 0.0, (time.perf_counter()-t0)*1000, False, ["Zero bg"])
            score = np.sqrt(np.mean(((image-bg)/(bg+1e-10))**2))
            score = min(score, 2.0) / 2.0
            conf = min(1.0, 2*score)
        elif method == "entropy":
            hist, _ = np.histogram(image.flatten(), bins=256, range=(0,256), density=True)
            hist = hist[hist > 0]
            if len(hist) == 0: return MethodResult(method, 0.0, 0.0, (time.perf_counter()-t0)*1000, False, ["Uniform image"])
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            score = entropy / 7.0
            conf = min(1.0, score)
        else:
            return MethodResult(method, 0.0, 0.0, (time.perf_counter()-t0)*1000, False, ["Unknown method"])
        return MethodResult(method, float(score), float(conf), (time.perf_counter()-t0)*1000, True)
    except Exception as e:
        return MethodResult(method, 0.0, 0.0, (time.perf_counter()-t0)*1000, False, [str(e)])

def detect_outliers(results: List[tuple]) -> List[str]:
    if len(results) < 3: return []
    scores = np.array([score for _, score, _ in results])
    median_score = np.median(scores)
    mad = np.median(np.abs(scores - median_score))
    if mad == 0: return []
    threshold = 5.0 * mad
    return [method for method, score, _ in results if abs(score - median_score) > threshold]

class ContrastScoreCalculator:
    def __init__(self, config: Optional[ContrastConfig] = None):
        self.config = config or ContrastConfig()
        self._cache = {} if self.config.cache_enabled else None

    def calculate_contrast_score(self, image: np.ndarray) -> ContrastResult:
        config = self.config
        t0 = time.perf_counter()
        if not isinstance(image, np.ndarray) or image.size < 100 or min(image.shape[:2]) < 10:
            return ContrastResult(False, 0.0, 0.0, ["Invalid image"], False, 0, {}, [], 0.0)
        cache_key = None
        if config.cache_enabled:
            h = hashlib.md5(image.tobytes() if image.size<200000 else image[::5,::5].tobytes()).hexdigest()
            cache_key = f"{h}_{image.shape}_{str(config.primary_methods)}"
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                cached.processing_time_ms = (time.perf_counter()-t0)*1000
                return cached
        gray = shared_preprocessing(image, config)
        method_results = {}
        valid = []
        for m in config.primary_methods:
            res = method_worker(m, gray, config)
            method_results[res.method_name] = res
            if res.is_valid:
                valid.append((res.method_name, res.score, res.confidence))
        scores = np.array([v[1] for v in valid])
        if len(valid) >= config.min_methods_required:
            std = np.std(scores)
            mean = np.mean(scores)
            consistency = 1.0 - std / max(abs(mean), 1.0)
            if consistency >= config.consistency_threshold:
                ensemble = np.average(scores, weights=[config.method_weights.get(m,1.0) for m,_,_ in valid])
                conf = min(1.0, np.mean([v[2] for v in valid]))
                warnings = []
                if ensemble >= config.excellent_contrast:
                    warnings.append("Excellent contrast detected")
                elif ensemble >= config.good_contrast:
                    warnings.append("Good contrast detected")
                elif ensemble >= config.min_contrast:
                    warnings.append("Adequate contrast detected")
                else:
                    warnings.append("Low contrast detected - may need enhancement")
                result = ContrastResult(True, float(ensemble), float(conf), warnings, False, (time.perf_counter()-t0)*1000, method_results, [v[0] for v in valid], consistency)
                if cache_key:
                    self._cache[cache_key] = result
                return result
        fallback = []
        for m in config.fallback_methods:
            res = method_worker(m, gray, config)
            method_results[m] = res
            if res.is_valid:
                fallback.append((res.method_name, res.score, res.confidence))
        all_valid = valid + fallback
        scores = np.array([v[1] for v in all_valid]) if all_valid else np.array([0.0])
        std = np.std(scores)
        mean = np.mean(scores)
        consistency = 1.0 - std / max(abs(mean), 1.0)
        ensemble = np.average(scores, weights=[config.method_weights.get(m,1.0) for m,_,_ in all_valid]) if all_valid else 0.0
        conf = min(1.0, np.mean([v[2] for v in all_valid])) if all_valid else 0.0
        warnings = []
        if ensemble >= config.excellent_contrast:
            warnings.append("Excellent contrast detected")
        elif ensemble >= config.good_contrast:
            warnings.append("Good contrast detected")
        elif ensemble >= config.min_contrast:
            warnings.append("Adequate contrast detected")
        else:
            warnings.append("Low contrast detected - may need enhancement")
        result = ContrastResult(True, float(ensemble), float(conf), warnings, True, (time.perf_counter()-t0)*1000, method_results, [v[0] for v in all_valid], consistency)
        if cache_key:
            self._cache[cache_key] = result
        return result

    def clear_cache(self):
        if self._cache:
            self._cache.clear()

def calculate_contrast_score(image: np.ndarray, config: Optional[ContrastConfig]=None) -> ContrastResult:
    return ContrastScoreCalculator(config).calculate_contrast_score(image)