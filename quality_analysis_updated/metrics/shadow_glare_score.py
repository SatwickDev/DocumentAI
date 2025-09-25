#!/usr/bin/env python3
"""
Ultra-Fast Shadow/Glare Score Calculator (3+2 Ensemble)
Detects abnormal shadow and glare on documents robustly.
"""

import numpy as np
import cv2
import time
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class ShadowGlareConfig:
    primary_methods: List[str] = field(default_factory=lambda: [
        "global_shadow", "local_brightness_var", "glare_area"
    ])
    fallback_methods: List[str] = field(default_factory=lambda: [
        "entropy_check", "histogram_tail"
    ])
    shadow_threshold: float = 0.08
    glare_threshold: float = 0.06
    consistency_threshold: float = 0.7
    min_methods_required: int = 2
    cache_enabled: bool = True
    method_weights: Dict[str, float] = field(default_factory=lambda: {
        "global_shadow": 0.35,
        "local_brightness_var": 0.30,
        "glare_area": 0.20,
        "entropy_check": 0.10,
        "histogram_tail": 0.05
    })

@dataclass
class MethodResult:
    method_name: str
    score: float # 0.0 = clean, 1.0 = severe shadow/glare
    confidence: float
    processing_time_ms: float
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ShadowGlareResult:
    is_valid: bool
    value: float
    confidence: float
    warnings: List[str]
    fallback_used: bool
    processing_time_ms: float
    method_results: Dict[str, MethodResult]
    methods_used: List[str]
    consistency_score: float

def shared_preprocessing(image: np.ndarray, config: ShadowGlareConfig) -> np.ndarray:
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        gray = image
    return gray

def method_worker(method, image, config: ShadowGlareConfig) -> MethodResult:
    t0 = time.perf_counter()
    H, W = image.shape[:2]
    try:
        if method == "global_shadow":
            shadow_mask = (image < 64).astype(np.uint8)
            shadow_area = np.sum(shadow_mask) / (H * W)
            score = min(1.0, shadow_area / config.shadow_threshold)
            conf = 1.0 - score
            warn = ["Significant shadow area"] if score > 0.1 else []
            return MethodResult(method, score, conf, (time.perf_counter()-t0)*1000, True, warn, {"shadow_area": shadow_area})
        elif method == "local_brightness_var":
            block = 32
            local_var = []
            for y in range(0, H-block, block):
                for x in range(0, W-block, block):
                    roi = image[y:y+block, x:x+block]
                    local_var.append(np.std(roi))
            var = np.mean(local_var) if local_var else 0.0
            score = min(1.0, var / 32.0)
            conf = 1.0 - score
            warn = ["High local brightness variance"] if score > 0.2 else []
            return MethodResult(method, score, conf, (time.perf_counter()-t0)*1000, True, warn, {"local_var": var})
        elif method == "glare_area":
            glare_mask = (image > 242).astype(np.uint8)
            glare_area = np.sum(glare_mask) / (H * W)
            score = min(1.0, glare_area / config.glare_threshold)
            conf = 1.0 - score
            warn = ["Significant glare area"] if score > 0.1 else []
            return MethodResult(method, score, conf, (time.perf_counter()-t0)*1000, True, warn, {"glare_area": glare_area})
        elif method == "entropy_check":
            hist, _ = np.histogram(image.flatten(), bins=256, range=(0,256), density=True)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            score = 1.0 - min(entropy/7.0, 1.0)
            conf = entropy/7.0
            warn = ["Low entropy; possible overbright or shadow"] if score > 0.3 else []
            return MethodResult(method, score, conf, (time.perf_counter()-t0)*1000, True, warn, {"entropy": entropy})
        elif method == "histogram_tail":
            hist = cv2.calcHist([image],[0],None,[256],[0,256]).flatten()
            shadow_tail = np.sum(hist[:15]) / np.sum(hist)
            glare_tail = np.sum(hist[240:]) / np.sum(hist)
            score = max(shadow_tail, glare_tail)
            conf = 1.0 - score
            warn = ["Histogram has heavy tail; possible shadow/glare"] if score > 0.07 else []
            return MethodResult(method, score, conf, (time.perf_counter()-t0)*1000, True, warn, {"shadow_tail": shadow_tail, "glare_tail": glare_tail})
        else:
            return MethodResult(method, 1.0, 0.0, (time.perf_counter()-t0)*1000, False, ["Unknown method"], {})
    except Exception as e:
        return MethodResult(method, 1.0, 0.0, (time.perf_counter()-t0)*1000, False, [str(e)], {})

def detect_outliers(results: List[Tuple[str, float, float]]) -> List[str]:
    if len(results) < 3:
        return []
    scores = np.array([score for _, score, _ in results])
    q1, q3 = np.percentile(scores, 25), np.percentile(scores, 75)
    iqr = q3 - q1
    return [method for method, score, _ in results if score < q1-1.5*iqr or score > q3+1.5*iqr]

class ShadowGlareScoreCalculator:
    def __init__(self, config: Optional[ShadowGlareConfig] = None):
        self.config = config or ShadowGlareConfig()
        self._cache = {} if self.config.cache_enabled else None

    def calculate_shadow_glare_score(self, image: np.ndarray) -> ShadowGlareResult:
        config = self.config
        t0 = time.perf_counter()
        if not isinstance(image, np.ndarray) or image.size < 100 or min(image.shape[:2]) < 10:
            return ShadowGlareResult(False, 1.0, 0.0, ["Invalid image"], False, 0, {}, [], 0.0)

        cache_key = None
        if config.cache_enabled:
            image_bytes = image.tobytes() if image.size < 200000 else image[::5,::5].tobytes()
            hashid = hashlib.md5(image_bytes).hexdigest()[:12]
            cache_key = f"{hashid}_{image.shape}_{str(config.primary_methods)}"
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                cached.processing_time_ms = (time.perf_counter()-t0)*1000
                return cached

        preproc = shared_preprocessing(image, config)

        method_results = {}
        valid = []
        for m in config.primary_methods:
            res = method_worker(m, preproc, config)
            method_results[res.method_name] = res
            if res.is_valid:
                valid.append((res.method_name, res.score, res.confidence))

        scores = np.array([v[1] for v in valid])
        if len(valid) >= config.min_methods_required:
            std = np.std(scores)
            mean = np.mean(scores)
            consistency = 1.0 - std / max(mean, 1.0)
            if consistency >= config.consistency_threshold:
                ensemble = np.average(scores, weights=[config.method_weights.get(m,1.0) for m,_,_ in valid])
                conf = min(1.0, np.mean([v[2] for v in valid]))
                result = ShadowGlareResult(True, float(ensemble), float(conf), [], False, (time.perf_counter()-t0)*1000, method_results, [v[0] for v in valid], consistency)
                if cache_key:
                    self._cache[cache_key] = result
                return result

        fallback = []
        for m in config.fallback_methods:
            res = method_worker(m, preproc, config)
            method_results[res.method_name] = res
            if res.is_valid:
                fallback.append((res.method_name, res.score, res.confidence))
        all_valid = valid + fallback
        scores = np.array([v[1] for v in all_valid]) if all_valid else np.array([1.0])
        std = np.std(scores)
        mean = np.mean(scores)
        consistency = 1.0 - std / max(mean, 1.0)
        ensemble = np.average(scores, weights=[config.method_weights.get(m,1.0) for m,_,_ in all_valid]) if all_valid else 1.0
        conf = min(1.0, np.mean([v[2] for v in all_valid])) if all_valid else 0.0
        outliers = detect_outliers(all_valid)
        result = ShadowGlareResult(True, float(ensemble), float(conf), [], True, (time.perf_counter()-t0)*1000, method_results, [v[0] for v in all_valid], consistency)
        if cache_key:
            self._cache[cache_key] = result
        return result

    def clear_cache(self):
        if self._cache:
            self._cache.clear()

def calculate_shadow_glare_score(image: np.ndarray, config: Optional[ShadowGlareConfig]=None) -> ShadowGlareResult:
    return ShadowGlareScoreCalculator(config).calculate_shadow_glare_score(image)