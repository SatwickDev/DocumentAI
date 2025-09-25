#!/usr/bin/env python3
"""
Ultra-Fast Edge/Crop Score Calculator (3+2 Ensemble)
Detects cropped/mis-scanned documents robustly.
"""

import numpy as np
import cv2
import time
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class EdgeCropConfig:
    primary_methods: List[str] = field(default_factory=lambda: [
        "contour_bbox", "projection_profile", "hough_lines"
    ])
    fallback_methods: List[str] = field(default_factory=lambda: [
        "aspect_ratio", "area_ratio"
    ])
    area_ratio_threshold: float = 0.7
    margin_threshold: float = 0.05
    consistency_threshold: float = 0.7
    min_methods_required: int = 2
    cache_enabled: bool = True
    method_weights: Dict[str, float] = field(default_factory=lambda: {
        "contour_bbox": 0.35,
        "projection_profile": 0.30,
        "hough_lines": 0.20,
        "aspect_ratio": 0.10,
        "area_ratio": 0.05
    })

@dataclass
class MethodResult:
    method_name: str
    score: float  # 0.0 = good, 1.0 = severe crop
    confidence: float
    processing_time_ms: float
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EdgeCropResult:
    is_valid: bool
    value: float
    confidence: float
    warnings: List[str]
    fallback_used: bool
    processing_time_ms: float
    method_results: Dict[str, MethodResult]
    methods_used: List[str]
    consistency_score: float

def shared_preprocessing(image: np.ndarray, config: EdgeCropConfig) -> np.ndarray:
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

def method_worker(method, image, config: EdgeCropConfig) -> MethodResult:
    t0 = time.perf_counter()
    H, W = image.shape[:2]
    try:
        if method == "contour_bbox":
            edges = cv2.Canny(image, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return MethodResult(method, 1.0, 0.0, (time.perf_counter()-t0)*1000, False, ["No contours"], {})
            page_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(page_contour)
            area_ratio = (w * h) / (W * H)
            margins = [x / W, y / H, (W - (x + w)) / W, (H - (y + h)) / H]
            bad_margin = any(m > config.margin_threshold for m in margins)
            crop_ratio = 1.0 - area_ratio if (area_ratio < config.area_ratio_threshold or bad_margin) else 0.0
            conf = 1.0 - crop_ratio
            meta = {"area_ratio": area_ratio, "margins": margins, "bounding_box": [int(x), int(y), int(w), int(h)]}
            warn = ["Possible crop/margin issue"] if crop_ratio > 0 else []
            return MethodResult(method, crop_ratio, conf, (time.perf_counter()-t0)*1000, True, warn, meta)
        elif method == "projection_profile":
            vertical_proj = np.sum(image < 240, axis=0) / H
            horizontal_proj = np.sum(image < 240, axis=1) / W
            v_margin = np.argmax(vertical_proj > 0.05)
            v_margin_r = W - np.argmax(vertical_proj[::-1] > 0.05)
            h_margin = np.argmax(horizontal_proj > 0.05)
            h_margin_b = H - np.argmax(horizontal_proj[::-1] > 0.05)
            margins = [v_margin/W, h_margin/H, (W-v_margin_r)/W, (H-h_margin_b)/H]
            crop_detected = any(m > config.margin_threshold for m in margins)
            crop_ratio = 1.0 if crop_detected else 0.0
            conf = 1.0 - crop_ratio
            meta = {"proj_margins": margins}
            warn = ["Projection profile suggests cropping"] if crop_detected else []
            return MethodResult(method, crop_ratio, conf, (time.perf_counter()-t0)*1000, True, warn, meta)
        elif method == "hough_lines":
            edges = cv2.Canny(image, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=int(0.6*min(H,W)), maxLineGap=20)
            lines_img = np.zeros_like(image)
            if lines is not None:
                for l in lines:
                    x1, y1, x2, y2 = l[0]
                    cv2.line(lines_img, (x1,y1), (x2,y2), 255, 2)
            line_count = 0 if lines is None else len(lines)
            is_good = line_count >= 2
            crop_ratio = 0.0 if is_good else 1.0
            conf = 1.0 - crop_ratio
            meta = {"hough_line_count": line_count}
            warn = ["Not enough strong lines detected"] if not is_good else []
            return MethodResult(method, crop_ratio, conf, (time.perf_counter()-t0)*1000, True, warn, meta)
        elif method == "aspect_ratio":
            aspect = W / H
            expected = 8.5 / 11.0
            diff = abs(aspect - expected)
            crop_ratio = min(diff/0.5, 1.0)
            conf = 1.0 - crop_ratio
            meta = {"aspect_ratio": aspect, "expected": expected}
            warn = ["Aspect ratio deviates"] if crop_ratio > 0.2 else []
            return MethodResult(method, crop_ratio, conf, (time.perf_counter()-t0)*1000, True, warn, meta)
        elif method == "area_ratio":
            nonwhite = np.sum(image < 240) / (W * H)
            crop_ratio = 1.0 if nonwhite < config.area_ratio_threshold else 0.0
            conf = 1.0 - crop_ratio
            meta = {"nonwhite_ratio": nonwhite}
            warn = ["Very little content; possible crop"] if crop_ratio > 0 else []
            return MethodResult(method, crop_ratio, conf, (time.perf_counter()-t0)*1000, True, warn, meta)
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

class EdgeCropScoreCalculator:
    def __init__(self, config: Optional[EdgeCropConfig] = None):
        self.config = config or EdgeCropConfig()
        self._cache = {} if self.config.cache_enabled else None

    def calculate_edge_crop_score(self, image: np.ndarray) -> EdgeCropResult:
        config = self.config
        t0 = time.perf_counter()
        if not isinstance(image, np.ndarray) or image.size < 100 or min(image.shape[:2]) < 10:
            return EdgeCropResult(False, 1.0, 0.0, ["Invalid image"], False, 0, {}, [], 0.0)

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
                result = EdgeCropResult(True, float(ensemble), float(conf), [], False, (time.perf_counter()-t0)*1000, method_results, [v[0] for v in valid], consistency)
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
        result = EdgeCropResult(True, float(ensemble), float(conf), [], True, (time.perf_counter()-t0)*1000, method_results, [v[0] for v in all_valid], consistency)
        if cache_key:
            self._cache[cache_key] = result
        return result

    def clear_cache(self):
        if self._cache:
            self._cache.clear()

def calculate_edge_crop_score(image: np.ndarray, config: Optional[EdgeCropConfig]=None) -> EdgeCropResult:
    return EdgeCropScoreCalculator(config).calculate_edge_crop_score(image)