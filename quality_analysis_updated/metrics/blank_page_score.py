#!/usr/bin/env python3
"""
Ultra-Fast Blank Page Score Calculator (3+2 Ensemble)
Detects blank or near-blank pages robustly.
"""

import numpy as np
import cv2
import time
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class BlankPageConfig:
    primary_methods: List[str] = field(default_factory=lambda: [
        "white_pixel_ratio", "text_blob_count", "mean_std"
    ])
    fallback_methods: List[str] = field(default_factory=lambda: [
        "ocr_text", "edge_density"
    ])
    blankness_threshold: float = 0.98
    consistency_threshold: float = 0.7
    min_methods_required: int = 2
    cache_enabled: bool = True
    method_weights: Dict[str, float] = field(default_factory=lambda: {
        "white_pixel_ratio": 0.4,
        "text_blob_count": 0.3,
        "mean_std": 0.2,
        "ocr_text": 0.05,
        "edge_density": 0.05
    })

@dataclass
class MethodResult:
    method_name: str
    score: float # 1.0 = blank, 0.0 = not blank
    confidence: float
    processing_time_ms: float
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BlankPageResult:
    is_valid: bool
    value: float
    confidence: float
    warnings: List[str]
    fallback_used: bool
    processing_time_ms: float
    method_results: Dict[str, MethodResult]
    methods_used: List[str]
    consistency_score: float

def shared_preprocessing(image: np.ndarray, config: BlankPageConfig) -> np.ndarray:
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

def method_worker(method, image, config: BlankPageConfig) -> MethodResult:
    t0 = time.perf_counter()
    H, W = image.shape[:2]
    try:
        if method == "white_pixel_ratio":
            blank_pixels = np.sum(image > 245)
            blank_ratio = blank_pixels / (H * W)
            score = min(1.0, blank_ratio / config.blankness_threshold)
            conf = blank_ratio
            warn = ["Mostly white page"] if score > 0.95 else []
            return MethodResult(method, score, conf, (time.perf_counter()-t0)*1000, True, warn, {"blank_ratio": blank_ratio})
        elif method == "text_blob_count":
            _, bw = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            text_like = [c for c in contours if cv2.contourArea(c) > 30]
            count = len(text_like)
            score = 1.0 if count < 2 else 0.0
            conf = 1.0 if score == 1.0 else max(0.2, 1.0 - count/10.0)
            warn = ["No text blobs detected"] if score == 1.0 else []
            return MethodResult(method, score, conf, (time.perf_counter()-t0)*1000, True, warn, {"text_blob_count": count})
        elif method == "mean_std":
            mean = np.mean(image)
            std = np.std(image)
            score = 1.0 if mean > 240 and std < 10 else 0.0
            conf = 1.0 if score == 1.0 else max(0.2, 1.0 - std/50.0)
            warn = ["Low-variance, high-mean: likely blank"] if score == 1.0 else []
            return MethodResult(method, score, conf, (time.perf_counter()-t0)*1000, True, warn, {"mean": mean, "std": std})
        elif method == "ocr_text":
            # Placeholder for OCR: here, just simulate "no text found"
            # Real: use pytesseract.image_to_string and check length
            score = 1.0
            conf = 0.6
            warn = ["No text detected by OCR"]
            return MethodResult(method, score, conf, (time.perf_counter()-t0)*1000, True, warn, {})
        elif method == "edge_density":
            edges = cv2.Canny(image, 50, 150)
            edge_pixels = np.sum(edges > 0) / (H * W)
            score = 1.0 if edge_pixels < 0.005 else 0.0
            conf = 1.0 if score == 1.0 else max(0.2, 1.0-edge_pixels*20)
            warn = ["Very few edges detected"] if score == 1.0 else []
            return MethodResult(method, score, conf, (time.perf_counter()-t0)*1000, True, warn, {"edge_density": edge_pixels})
        else:
            return MethodResult(method, 0.0, 0.0, (time.perf_counter()-t0)*1000, False, ["Unknown method"], {})
    except Exception as e:
        return MethodResult(method, 0.0, 0.0, (time.perf_counter()-t0)*1000, False, [str(e)], {})

def detect_outliers(results: List[Tuple[str, float, float]]) -> List[str]:
    if len(results) < 3:
        return []
    scores = np.array([score for _, score, _ in results])
    q1, q3 = np.percentile(scores, 25), np.percentile(scores, 75)
    iqr = q3 - q1
    return [method for method, score, _ in results if score < q1-1.5*iqr or score > q3+1.5*iqr]

class BlankPageScoreCalculator:
    def __init__(self, config: Optional[BlankPageConfig] = None):
        self.config = config or BlankPageConfig()
        self._cache = {} if self.config.cache_enabled else None

    def calculate_blank_page_score(self, image: np.ndarray) -> BlankPageResult:
        config = self.config
        t0 = time.perf_counter()
        if not isinstance(image, np.ndarray) or image.size < 100 or min(image.shape[:2]) < 10:
            return BlankPageResult(False, 1.0, 0.0, ["Invalid image"], False, 0, {}, [], 0.0)

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
                result = BlankPageResult(True, float(ensemble), float(conf), [], False, (time.perf_counter()-t0)*1000, method_results, [v[0] for v in valid], consistency)
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
        scores = np.array([v[1] for v in all_valid]) if all_valid else np.array([0.0])
        std = np.std(scores)
        mean = np.mean(scores)
        consistency = 1.0 - std / max(mean, 1.0)
        ensemble = np.average(scores, weights=[config.method_weights.get(m,1.0) for m,_,_ in all_valid]) if all_valid else 0.0
        conf = min(1.0, np.mean([v[2] for v in all_valid])) if all_valid else 0.0
        outliers = detect_outliers(all_valid)
        result = BlankPageResult(True, float(ensemble), float(conf), [], True, (time.perf_counter()-t0)*1000, method_results, [v[0] for v in all_valid], consistency)
        if cache_key:
            self._cache[cache_key] = result
        return result

    def clear_cache(self):
        if self._cache:
            self._cache.clear()

def calculate_blank_page_score(image: np.ndarray, config: Optional[BlankPageConfig]=None) -> BlankPageResult:
    return BlankPageScoreCalculator(config).calculate_blank_page_score(image)