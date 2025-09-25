#!/usr/bin/env python3
"""
Ultra-Fast Brightness Score Calculator (3+2 Ensemble)
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
class BrightnessConfig:
    primary_methods: List[str] = field(default_factory=lambda: [
        "histogram", "regional", "adaptive"
    ])
    fallback_methods: List[str] = field(default_factory=lambda: [
        "statistical", "robust"
    ])
    use_downsampling: bool = True
    downsample_threshold: int = 1000
    downsample_factor: float = 0.5
    consistency_threshold: float = 0.7
    min_methods_required: int = 2
    cache_enabled: bool = True
    max_workers: int = 3
    method_weights: Dict[str, float] = field(default_factory=lambda: {
        "histogram": 0.35,
        "regional": 0.30,
        "adaptive": 0.25,
        "statistical": 0.05,
        "robust": 0.05
    })
    too_dark: float = 0.2
    optimal_min: float = 0.3
    optimal_max: float = 0.7
    too_bright: float = 0.85

@dataclass
class MethodResult:
    method_name: str
    score: float
    confidence: float
    processing_time_ms: float
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)

@dataclass
class BrightnessResult:
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
def fast_mean(image: np.ndarray) -> float:
    return np.mean(image)

def shared_preprocessing(image: np.ndarray, config: BrightnessConfig) -> np.ndarray:
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

def method_worker(method, image, config: BrightnessConfig) -> MethodResult:
    t0 = time.perf_counter()
    try:
        if method == "histogram":
            hist = cv2.calcHist([image],[0],None,[256],[0,256]).flatten()
            total = np.sum(hist)
            norm_hist = hist/total if total>0 else hist
            mean = np.dot(np.arange(256), norm_hist)
            score = mean/255.0
            conf = 1.0 - abs(score-0.5)
        elif method == "regional":
            h,w = image.shape
            regions_y = min(4, h // 32) or 1
            regions_x = min(4, w // 32) or 1
            vals = []
            for i in range(regions_y):
                for j in range(regions_x):
                    y1 = (i*h)//regions_y
                    y2 = ((i+1)*h)//regions_y
                    x1 = (j*w)//regions_x
                    x2 = ((j+1)*w)//regions_x
                    r = image[y1:y2,x1:x2]
                    vals.append(np.mean(r)/255.0)
            score = float(np.mean(vals))
            conf = 1.0 - np.std(vals)
        elif method == "adaptive":
            h,w = image.shape
            block = min(15, max(7, min(image.shape)//20*2+1))
            local = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,block,0)
            mean = fast_mean(image)
            score = mean/255.0
            conf = 1.0 - abs(score-0.5)
        elif method == "statistical":
            arr = image.flatten()
            mean = fast_mean(image)
            median = float(np.median(arr))
            score = median/255.0
            conf = 1.0 - abs(score-0.5)
        elif method == "robust":
            arr = image.flatten()
            q25, q75 = np.percentile(arr, [25,75])
            trimmed = arr[(arr>=q25)&(arr<=q75)]
            score = float(np.mean(trimmed))/255.0 if trimmed.size>0 else float(np.mean(arr))/255.0
            conf = 1.0 - abs(score-0.5)
        else:
            return MethodResult(method, 0.0, 0.0, (time.perf_counter()-t0)*1000, False, ["Unknown method"])
        return MethodResult(method, float(score), float(conf), (time.perf_counter()-t0)*1000, True)
    except Exception as e:
        return MethodResult(method, 0.0, 0.0, (time.perf_counter()-t0)*1000, False, [str(e)])

class BrightnessScoreCalculator:
    def __init__(self, config: Optional[BrightnessConfig] = None):
        self.config = config or BrightnessConfig()
        self._cache = {} if self.config.cache_enabled else None

    def calculate_brightness_score(self, image: np.ndarray) -> BrightnessResult:
        config = self.config
        t0 = time.perf_counter()
        if not isinstance(image, np.ndarray) or image.size < 100 or min(image.shape[:2]) < 10:
            return BrightnessResult(False, 0.0, 0.0, ["Invalid image"], False, 0, {}, [], 0.0)
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
                if ensemble < config.too_dark:
                    warnings.append("Very dark image - may need brightening")
                elif ensemble < config.optimal_min:
                    warnings.append("Slightly underexposed")
                elif ensemble > config.too_bright:
                    warnings.append("Very bright image - may be overexposed")
                elif ensemble > config.optimal_max:
                    warnings.append("Slightly overexposed")
                else:
                    warnings.append("Optimal brightness level")
                result = BrightnessResult(True, float(ensemble), float(conf), warnings, False, (time.perf_counter()-t0)*1000, method_results, [v[0] for v in valid], consistency)
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
        if ensemble < config.too_dark:
            warnings.append("Very dark image - may need brightening")
        elif ensemble < config.optimal_min:
            warnings.append("Slightly underexposed")
        elif ensemble > config.too_bright:
            warnings.append("Very bright image - may be overexposed")
        elif ensemble > config.optimal_max:
            warnings.append("Slightly overexposed")
        else:
            warnings.append("Optimal brightness level")
        result = BrightnessResult(True, float(ensemble), float(conf), warnings, True, (time.perf_counter()-t0)*1000, method_results, [v[0] for v in all_valid], consistency)
        if cache_key:
            self._cache[cache_key] = result
        return result

    def clear_cache(self):
        if self._cache:
            self._cache.clear()

def calculate_brightness_score(image: np.ndarray, config: Optional[BrightnessConfig]=None) -> BrightnessResult:
    return BrightnessScoreCalculator(config).calculate_brightness_score(image)