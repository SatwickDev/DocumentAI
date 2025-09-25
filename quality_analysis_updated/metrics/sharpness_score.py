#!/usr/bin/env python3
"""
Ultra-Fast Sharpness Score Calculator (3+2 Ensemble)
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
class SharpnessConfig:
    primary_methods: List[str] = field(default_factory=lambda: [
        "laplacian", "sobel", "localvar"
    ])
    fallback_methods: List[str] = field(default_factory=lambda: [
        "fft", "roberts"
    ])
    use_downsampling: bool = True
    downsample_threshold: int = 1000
    downsample_factor: float = 0.5
    consistency_threshold: float = 0.7
    min_methods_required: int = 2
    cache_enabled: bool = True
    max_workers: int = 3
    method_weights: Dict[str, float] = field(default_factory=lambda: {
        "laplacian": 0.35,
        "sobel": 0.3,
        "localvar": 0.25,
        "fft": 0.05,
        "roberts": 0.05
    })
    blurry_threshold: float = 0.12
    acceptable_threshold: float = 0.2
    sharp_threshold: float = 0.5
    very_sharp_threshold: float = 0.8

@dataclass
class MethodResult:
    method_name: str
    score: float
    confidence: float
    processing_time_ms: float
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)

@dataclass
class SharpnessResult:
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
def fast_laplacian_var(image: np.ndarray) -> float:
    lap = np.zeros_like(image)
    h, w = image.shape
    for y in prange(1, h-1):
        for x in prange(1, w-1):
            lap[y,x] = -image[y-1,x] - image[y+1,x] - image[y,x-1] - image[y,x+1] + 4*image[y,x]
    return np.var(lap) / 1000.0

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def fast_sobel(image: np.ndarray) -> float:
    h, w = image.shape
    total = 0.0
    count = 0
    for y in prange(1,h-1):
        for x in prange(1,w-1):
            gx = (-image[y-1,x-1]+image[y-1,x+1]-2*image[y,x-1]+2*image[y,x+1]-image[y+1,x-1]+image[y+1,x+1])
            gy = (-image[y-1,x-1]-2*image[y-1,x]-image[y-1,x+1]+image[y+1,x-1]+2*image[y+1,x]+image[y+1,x+1])
            mag = np.sqrt(gx*gx+gy*gy)
            total += mag
            count += 1
    return total/count/255.0 if count > 0 else 0.0

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def fast_local_var(image: np.ndarray, ws: int = 9) -> float:
    h,w = image.shape
    hw = ws//2
    if h < ws or w < ws: return 0.0
    total = 0.0
    count = 0
    for y in prange(hw, h-hw):
        for x in prange(hw, w-hw):
            local = image[y-hw:y+hw+1,x-hw:x+hw+1]
            total += np.var(local)
            count += 1
    return total/count/1000.0 if count > 0 else 0.0

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def fast_roberts(image: np.ndarray) -> float:
    h,w = image.shape
    total = 0.0
    count = 0
    for y in prange(h-1):
        for x in prange(w-1):
            gx = image[y,x] - image[y+1,x+1]
            gy = image[y,x+1] - image[y+1,x]
            mag = np.sqrt(gx*gx+gy*gy)
            total += mag
            count += 1
    return total/count/255.0 if count > 0 else 0.0

def shared_preprocessing(image: np.ndarray, config: SharpnessConfig) -> np.ndarray:
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

def method_worker(method, image, config: SharpnessConfig) -> MethodResult:
    t0 = time.perf_counter()
    try:
        if method == "laplacian":
            score = fast_laplacian_var(image)
            conf = min(1.0, 2*score)
        elif method == "sobel":
            score = fast_sobel(image)
            conf = min(1.0, 2*score)
        elif method == "localvar":
            score = fast_local_var(image)
            conf = min(1.0, 2*score)
        elif method == "fft":
            fft = np.fft.fft2(image)
            mag = np.abs(fft)
            h,w = image.shape
            cy,cx = h//2, w//2
            y,x = np.ogrid[:h,:w]
            dists = np.sqrt((y-cy)**2+(x-cx)**2)
            maxd = min(cy,cx)
            high = mag[dists>maxd*0.7]
            score = float(np.mean(high))/255.0 if high.size else 0.0
            conf = min(1.0, 2*score)
        elif method == "roberts":
            score = fast_roberts(image)
            conf = min(1.0, 2*score)
        else:
            return MethodResult(method, 0.0, 0.0, (time.perf_counter()-t0)*1000, False, ["Unknown method"])
        return MethodResult(method, float(score), float(conf), (time.perf_counter()-t0)*1000, True)
    except Exception as e:
        return MethodResult(method, 0.0, 0.0, (time.perf_counter()-t0)*1000, False, [str(e)])

def detect_outliers(results: List[tuple]) -> List[str]:
    if len(results) < 3: return []
    scores = np.array([score for _, score, _ in results])
    med = np.median(scores)
    mad = np.median(np.abs(scores-med))
    if mad == 0: return []
    return [method for method, score, _ in results if abs(score-med) > 3*mad]

class SharpnessScoreCalculator:
    def __init__(self, config: Optional[SharpnessConfig] = None):
        self.config = config or SharpnessConfig()
        self._cache = {} if self.config.cache_enabled else None

    def calculate_sharpness_score(self, image: np.ndarray) -> SharpnessResult:
        config = self.config
        t0 = time.perf_counter()
        if not isinstance(image, np.ndarray) or image.size < 100 or min(image.shape[:2]) < 10:
            return SharpnessResult(False, 0.0, 0.0, ["Invalid image"], False, 0, {}, [], 0.0)
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
                if ensemble >= config.very_sharp_threshold:
                    warnings.append("Very sharp image - excellent quality")
                elif ensemble >= config.sharp_threshold:
                    warnings.append("Sharp image - good quality")
                elif ensemble >= config.acceptable_threshold:
                    warnings.append("Acceptable sharpness")
                elif ensemble >= config.blurry_threshold:
                    warnings.append("Slightly blurry image")
                else:
                    warnings.append("Blurry image - may need enhancement")
                result = SharpnessResult(True, float(ensemble), float(conf), warnings, False, (time.perf_counter()-t0)*1000, method_results, [v[0] for v in valid], consistency)
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
        if ensemble >= config.very_sharp_threshold:
            warnings.append("Very sharp image - excellent quality")
        elif ensemble >= config.sharp_threshold:
            warnings.append("Sharp image - good quality")
        elif ensemble >= config.acceptable_threshold:
            warnings.append("Acceptable sharpness")
        elif ensemble >= config.blurry_threshold:
            warnings.append("Slightly blurry image")
        else:
            warnings.append("Blurry image - may need enhancement")
        result = SharpnessResult(True, float(ensemble), float(conf), warnings, True, (time.perf_counter()-t0)*1000, method_results, [v[0] for v in all_valid], consistency)
        if cache_key:
            self._cache[cache_key] = result
        return result

    def clear_cache(self):
        if self._cache:
            self._cache.clear()

def calculate_sharpness_score(image: np.ndarray, config: Optional[SharpnessConfig]=None) -> SharpnessResult:
    return SharpnessScoreCalculator(config).calculate_sharpness_score(image)