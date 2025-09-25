#!/usr/bin/env python3
"""
Ultra-Fast Noise Level Calculator (3+2 Ensemble)
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
class NoiseConfig:
    primary_methods: List[str] = field(default_factory=lambda: [
        "wavelet", "mad", "laplacian"
    ])
    fallback_methods: List[str] = field(default_factory=lambda: [
        "spectral", "localvar"
    ])
    use_downsampling: bool = True
    downsample_threshold: int = 1000
    downsample_factor: float = 0.5
    consistency_threshold: float = 0.7
    min_methods_required: int = 2
    cache_enabled: bool = True
    max_workers: int = 3
    method_weights: Dict[str, float] = field(default_factory=lambda: {
        "wavelet": 0.35,
        "mad": 0.3,
        "laplacian": 0.25,
        "spectral": 0.05,
        "localvar": 0.05
    })
    low_noise: float = 0.1
    moderate_noise: float = 0.25
    high_noise: float = 0.5

@dataclass
class MethodResult:
    method_name: str
    score: float
    confidence: float
    processing_time_ms: float
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)

@dataclass
class NoiseResult:
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
def fast_wavelet_noise(detail: np.ndarray) -> float:
    arr = np.abs(detail.flatten())
    n = len(arr)
    if n == 0: return 0.0
    median = np.median(arr)
    return median / 0.6745

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def fast_mad(arr: np.ndarray) -> float:
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def shared_preprocessing(image: np.ndarray, config: NoiseConfig) -> np.ndarray:
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

def method_worker(method, image, config: NoiseConfig) -> MethodResult:
    t0 = time.perf_counter()
    try:
        if method == "wavelet":
            h_detail = cv2.filter2D(image, -1, np.array([[1, -1]], dtype=np.float32))
            v_detail = cv2.filter2D(image, -1, np.array([[1], [-1]], dtype=np.float32))
            d_detail = cv2.filter2D(image, -1, np.array([[1, -1], [-1, 1]], dtype=np.float32) / 2.0)
            h = fast_wavelet_noise(h_detail)
            v = fast_wavelet_noise(v_detail)
            d = fast_wavelet_noise(d_detail)
            score = float(np.sqrt((h**2 + v**2 + d**2) / 3.0)) / 255.0
            conf = 1.0 - score
        elif method == "mad":
            lap = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
            med = np.median(np.abs(lap.flatten()))
            mad = fast_mad(np.abs(lap.flatten()))
            score = (mad / 0.6745) / 255.0
            conf = 1.0 - score
        elif method == "laplacian":
            kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32)
            filt = cv2.filter2D(image, -1, kernel)
            score = float(np.std(filt)) / 255.0
            conf = 1.0 - score
        elif method == "spectral":
            fft = np.fft.fft2(image)
            mag = np.abs(fft)
            h, w = image.shape
            cy, cx = h//2, w//2
            y, x = np.ogrid[:h, :w]
            dists = np.sqrt((y-cy)**2 + (x-cx)**2)
            maxd = min(cy,cx)
            high = mag[dists > maxd*0.7]
            score = min(1.0, np.std(high)/10000.0) if high.size else 0.0
            conf = 1.0 - score
        elif method == "localvar":
            wsize = 7
            kernel = np.ones((wsize,wsize), np.float32)/(wsize*wsize)
            mean = cv2.filter2D(image, -1, kernel)
            mean_sq = cv2.filter2D(image**2, -1, kernel)
            var = np.maximum(mean_sq - mean**2, 0)
            stds = np.sqrt(var)
            score = float(np.median(stds[stds>0])) / 255.0
            conf = 1.0 - score
        else:
            return MethodResult(method, 0.0, 0.0, (time.perf_counter()-t0)*1000, False, ["Unknown method"])
        return MethodResult(method, float(score), float(conf), (time.perf_counter()-t0)*1000, True)
    except Exception as e:
        return MethodResult(method, 0.0, 0.0, (time.perf_counter()-t0)*1000, False, [str(e)])

class NoiseLevelCalculator:
    def __init__(self, config: Optional[NoiseConfig] = None):
        self.config = config or NoiseConfig()
        self._cache = {} if self.config.cache_enabled else None

    def calculate_noise_level(self, image: np.ndarray) -> NoiseResult:
        config = self.config
        t0 = time.perf_counter()
        if not isinstance(image, np.ndarray) or image.size < 100 or min(image.shape[:2]) < 10:
            return NoiseResult(False, 0.0, 0.0, ["Invalid image"], False, 0, {}, [], 0.0)
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
                if ensemble <= config.low_noise:
                    warnings.append("Low noise level - excellent image quality")
                elif ensemble <= config.moderate_noise:
                    warnings.append("Moderate noise level - acceptable quality")
                elif ensemble <= config.high_noise:
                    warnings.append("High noise level - may need denoising")
                else:
                    warnings.append("Very high noise level - significant quality degradation")
                result = NoiseResult(True, float(ensemble), float(conf), warnings, False, (time.perf_counter()-t0)*1000, method_results, [v[0] for v in valid], consistency)
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
        if ensemble <= config.low_noise:
            warnings.append("Low noise level - excellent image quality")
        elif ensemble <= config.moderate_noise:
            warnings.append("Moderate noise level - acceptable quality")
        elif ensemble <= config.high_noise:
            warnings.append("High noise level - may need denoising")
        else:
            warnings.append("Very high noise level - significant quality degradation")
        result = NoiseResult(True, float(ensemble), float(conf), warnings, True, (time.perf_counter()-t0)*1000, method_results, [v[0] for v in all_valid], consistency)
        if cache_key:
            self._cache[cache_key] = result
        return result

    def clear_cache(self):
        if self._cache:
            self._cache.clear()

def calculate_noise_level(image: np.ndarray, config: Optional[NoiseConfig]=None) -> NoiseResult:
    return NoiseLevelCalculator(config).calculate_noise_level(image)