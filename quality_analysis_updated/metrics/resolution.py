#!/usr/bin/env python3
"""
Ultra-Fast Resolution Calculator (3+2 Ensemble)
Refactored to run sequentially for speed.
"""

import numpy as np
import cv2
import time
import hashlib
from numba import jit, prange
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class ResolutionConfig:
    primary_methods: List[str] = field(default_factory=lambda: [
        "dimension", "feature_density", "dpi"
    ])
    fallback_methods: List[str] = field(default_factory=lambda: [
        "metadata", "comparative"
    ])
    use_downsampling: bool = True
    downsample_threshold: int = 1000
    downsample_factor: float = 0.5
    consistency_threshold: float = 0.7
    min_methods_required: int = 2
    cache_enabled: bool = True
    max_workers: int = 3
    method_weights: Dict[str, float] = field(default_factory=lambda: {
        "dimension": 0.35,
        "feature_density": 0.30,
        "dpi": 0.25,
        "metadata": 0.05,
        "comparative": 0.05
    })
    min_readable_dpi: int = 150
    optimal_ocr_dpi: int = 300
    high_quality_dpi: int = 600

@dataclass
class MethodResult:
    method_name: str
    score: Tuple[int, int]  # (width, height)
    estimated_dpi: Optional[float]
    confidence: float
    processing_time_ms: float
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)

@dataclass
class ResolutionResult:
    is_valid: bool
    value: Tuple[int, int]
    confidence: float
    warnings: List[str]
    fallback_used: bool
    processing_time_ms: float
    method_results: Dict[str, MethodResult]
    methods_used: List[str]
    consistency_score: float
    estimated_dpi: Optional[float]

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def fast_feature_density(image: np.ndarray, threshold: int = 30) -> float:
    h, w = image.shape
    if h < 9 or w < 9:
        return 0.0
    feature_count = 0
    total_pixels = ((h - 8) // 8) * ((w - 8) // 8)
    for y in prange(4, h - 4, 8):
        for x in prange(4, w - 4, 8):
            gx = (
                -image[y - 4, x - 4] + image[y - 4, x + 4]
                - 2 * image[y, x - 4] + 2 * image[y, x + 4]
                - image[y + 4, x - 4] + image[y + 4, x + 4]
            )
            gy = (
                -image[y - 4, x - 4] - 2 * image[y - 4, x] - image[y - 4, x + 4]
                + image[y + 4, x - 4] + 2 * image[y + 4, x] + image[y + 4, x + 4]
            )
            gradient_mag = np.sqrt(gx * gx + gy * gy)
            if gradient_mag > threshold:
                feature_count += 1
    return float(feature_count) / float(total_pixels) if total_pixels > 0 else 0.0

def shared_preprocessing(image: np.ndarray, config: ResolutionConfig) -> np.ndarray:
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
    return gray

def method_worker(method, image, config: ResolutionConfig) -> MethodResult:
    t0 = time.perf_counter()
    try:
        px = (image.shape[1], image.shape[0])
        est_dpi = None
        if method == "dimension":
            score = px
            est_dpi = None
            conf = 0.8 if min(px) > 800 else 0.5
        elif method == "feature_density":
            density = fast_feature_density(image.astype(np.float32))
            score = px
            est_dpi = 300 if density > 0.4 else 150 if density > 0.2 else 72
            conf = min(1.0, 0.6 + density)
        elif method == "dpi":
            width, height = px
            possible_dpis = [72, 96, 150, 200, 300, 600]
            best_dpi = min(possible_dpis, key=lambda dpi: abs(width - int(8.27*dpi)) + abs(height - int(11.69*dpi)))
            score = px
            est_dpi = best_dpi
            conf = 0.9 if abs(width - int(8.27*best_dpi)) < 50 and abs(height - int(11.69*best_dpi)) < 50 else 0.7
        elif method == "metadata":
            score = px
            est_dpi = None
            conf = 0.6
        elif method == "comparative":
            score = px
            est_dpi = None
            conf = 0.5
        else:
            return MethodResult(method, px, None, 0.0, (time.perf_counter()-t0)*1000, False, ["Unknown method"])
        return MethodResult(method, score, est_dpi, conf, (time.perf_counter()-t0)*1000, True)
    except Exception as e:
        return MethodResult(method, (0,0), None, 0.0, (time.perf_counter()-t0)*1000, False, [str(e)])

def detect_outliers(results: List[tuple]) -> List[str]:
    if len(results) < 3: return []
    ws = np.array([score[0] for _, score, _, _ in results])
    hs = np.array([score[1] for _, score, _, _ in results])
    w_med, h_med = np.median(ws), np.median(hs)
    w_mad, h_mad = np.median(np.abs(ws-w_med)), np.median(np.abs(hs-h_med))
    if w_mad == 0 or h_mad == 0: return []
    return [method for method, (w,h), _, _ in results if abs(w-w_med)>3*w_mad or abs(h-h_med)>3*h_mad]

class ResolutionCalculator:
    def __init__(self, config: Optional[ResolutionConfig] = None):
        self.config = config or ResolutionConfig()
        self._cache = {} if self.config.cache_enabled else None

    def calculate_resolution(self, image: np.ndarray) -> ResolutionResult:
        config = self.config
        t0 = time.perf_counter()
        if not isinstance(image, np.ndarray) or image.size < 100 or min(image.shape[:2]) < 10:
            return ResolutionResult(False, (0,0), 0.0, ["Invalid image"], False, 0, {}, [], 0.0, None)
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
                valid.append((res.method_name, res.score, res.confidence, res.estimated_dpi))
        ws = np.array([v[1][0] for v in valid])
        hs = np.array([v[1][1] for v in valid])
        if len(valid) >= config.min_methods_required:
            std = np.std(np.concatenate([ws, hs]))
            mean = np.mean(np.concatenate([ws, hs]))
            consistency = 1.0 - std / max(abs(mean), 1.0)
            if consistency >= config.consistency_threshold:
                w = np.average(ws, weights=[config.method_weights.get(m,1.0) for m,_,_,_ in valid])
                h = np.average(hs, weights=[config.method_weights.get(m,1.0) for m,_,_,_ in valid])
                conf = min(1.0, np.mean([v[2] for v in valid]))
                dpis = [v[3] for v in valid if v[3] is not None]
                dpi = float(np.mean(dpis)) if dpis else None
                warnings = []
                if dpi and dpi < config.min_readable_dpi:
                    warnings.append(f"Estimated DPI ({dpi:.0f}) below minimum readable threshold")
                elif dpi and dpi < config.optimal_ocr_dpi:
                    warnings.append(f"Estimated DPI ({dpi:.0f}) below optimal OCR threshold")
                elif dpi and dpi >= config.high_quality_dpi:
                    warnings.append(f"High quality DPI ({dpi:.0f}) - excellent for all applications")
                result = ResolutionResult(True, (int(round(w)), int(round(h))), float(conf), warnings, False, (time.perf_counter()-t0)*1000, method_results, [v[0] for v in valid], consistency, dpi)
                if cache_key:
                    self._cache[cache_key] = result
                return result
        fallback = []
        for m in config.fallback_methods:
            res = method_worker(m, gray, config)
            method_results[m] = res
            if res.is_valid:
                fallback.append((res.method_name, res.score, res.confidence, res.estimated_dpi))
        all_valid = valid + fallback
        ws = np.array([v[1][0] for v in all_valid]) if all_valid else np.array([0.0])
        hs = np.array([v[1][1] for v in all_valid]) if all_valid else np.array([0.0])
        std = np.std(np.concatenate([ws, hs]))
        mean = np.mean(np.concatenate([ws, hs]))
        consistency = 1.0 - std / max(abs(mean), 1.0)
        w = np.average(ws, weights=[config.method_weights.get(m,1.0) for m,_,_,_ in all_valid]) if all_valid else 0.0
        h = np.average(hs, weights=[config.method_weights.get(m,1.0) for m,_,_,_ in all_valid]) if all_valid else 0.0
        conf = min(1.0, np.mean([v[2] for v in all_valid])) if all_valid else 0.0
        dpis = [v[3] for v in all_valid if v[3] is not None]
        dpi = float(np.mean(dpis)) if dpis else None
        warnings = []
        if dpi and dpi < config.min_readable_dpi:
            warnings.append(f"Estimated DPI ({dpi:.0f}) below minimum readable threshold")
        elif dpi and dpi < config.optimal_ocr_dpi:
            warnings.append(f"Estimated DPI ({dpi:.0f}) below optimal OCR threshold")
        elif dpi and dpi >= config.high_quality_dpi:
            warnings.append(f"High quality DPI ({dpi:.0f}) - excellent for all applications")
        result = ResolutionResult(True, (int(round(w)), int(round(h))), float(conf), warnings, True, (time.perf_counter()-t0)*1000, method_results, [v[0] for v in all_valid], consistency, dpi)
        if cache_key:
            self._cache[cache_key] = result
        return result

    def clear_cache(self):
        if self._cache:
            self._cache.clear()

def calculate_resolution(image: np.ndarray, config: Optional[ResolutionConfig]=None) -> ResolutionResult:
    return ResolutionCalculator(config).calculate_resolution(image)