#!/usr/bin/env python3
"""
Ultra-Fast Blur Score Calculator (3+2 Ensemble) - Refactored
Now sequential (no ProcessPoolExecutor) for speed.
"""

import numpy as np
import cv2
import time
import hashlib
import warnings
from numba import jit, prange
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=UserWarning, module="numba")


@dataclass
class FastBlurConfig:
    primary_methods: List[str] = field(default_factory=lambda: [
        "laplacian_variance", "tenengrad", "sobel_gradient"
    ])
    fallback_methods: List[str] = field(default_factory=lambda: [
        "fft_analysis", "modified_laplacian"
    ])
    use_multiscale: bool = True
    use_downsampling: bool = True
    downsample_threshold: int = 1000  # px
    downsample_factor: float = 0.5
    consistency_threshold: float = 0.7
    min_methods_required: int = 2
    cache_enabled: bool = True
    max_workers: int = 3
    method_weights: Dict[str, float] = field(default_factory=lambda: {
        "laplacian_variance": 0.35,
        "tenengrad": 0.30,
        "sobel_gradient": 0.20,
        "fft_analysis": 0.10,
        "modified_laplacian": 0.05
    })
    quality_threshold: float = 40.0


@dataclass
class MethodResult:
    method_name: str
    score: float
    confidence: float
    processing_time_ms: float
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FastBlurResult:
    is_valid: bool
    value: float
    confidence: float
    warnings: List[str]
    fallback_used: bool
    processing_time_ms: float
    method_results: Dict[str, MethodResult]
    methods_used: List[str]
    outliers_detected: List[str]
    consistency_score: float


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def fast_laplacian_var(image: np.ndarray) -> float:
    h, w = image.shape
    var_sum = 0.0
    count = 0
    for i in prange(1, h-1):
        for j in prange(1, w-1):
            lap = 4.0 * image[i, j] - image[i-1, j] - image[i+1, j] - image[i, j-1] - image[i, j+1]
            var_sum += lap * lap
            count += 1
    return var_sum / max(count, 1)


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def fast_tenengrad(image: np.ndarray) -> float:
    h, w = image.shape
    t_sum = 0.0
    count = 0
    for i in prange(1, h-1):
        for j in prange(1, w-1):
            gx = float(image[i, j+1]) - float(image[i, j-1])
            gy = float(image[i+1, j]) - float(image[i-1, j])
            t_sum += gx*gx + gy*gy
            count += 1
    return t_sum / max(count, 1)


def shared_preprocessing(image: np.ndarray, config: FastBlurConfig) -> np.ndarray:
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            image = image[:, :, 0]
    image = image.astype(np.float32)
    if config.use_downsampling and max(image.shape) > config.downsample_threshold:
        scale = config.downsample_factor
        image = cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image.astype(np.uint8)).astype(np.float32)
    except Exception:
        pass
    try:
        image = cv2.bilateralFilter(image.astype(np.uint8), 5, 10, 10).astype(np.float32)
    except Exception:
        pass
    return image


def method_worker(method, image, config: FastBlurConfig) -> MethodResult:
    t0 = time.perf_counter()
    try:
        if method == "laplacian_variance":
            score = fast_laplacian_var(image)
        elif method == "tenengrad":
            score = fast_tenengrad(image)
        elif method == "sobel_gradient":
            gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            score = np.mean(np.sqrt(gx**2 + gy**2))
        elif method == "fft_analysis":
            f = np.fft.fft2(image)
            fshift = np.fft.fftshift(f)
            mag = np.abs(fshift)
            h, w = image.shape
            cy, cx = h//2, w//2
            mask = np.zeros_like(mag)
            cv2.circle(mask, (cx, cy), min(h, w)//8, 1, -1)
            high_freq = np.sum(mag[mask == 0])
            total = np.sum(mag)
            score = float((high_freq / max(total, 1e-10))*100)
        elif method == "modified_laplacian":
            kern = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=np.float32)
            lap = cv2.filter2D(image, -1, kern)
            score = float(np.var(lap))
        else:
            raise Exception("Unknown method")
        conf = min(1.0, max(0.1, score / (config.quality_threshold*2)))
        return MethodResult(method, score, conf, (time.perf_counter()-t0)*1000, True, [], {})
    except Exception as e:
        return MethodResult(method, 0.0, 0.0, (time.perf_counter()-t0)*1000, False, [str(e)], {})


def detect_outliers(results: List[Tuple[str, float, float]]) -> List[str]:
    if len(results) < 3:
        return []
    scores = np.array([score for _, score, _ in results])
    q1, q3 = np.percentile(scores, 25), np.percentile(scores, 75)
    iqr = q3 - q1
    return [method for method, score, _ in results if score < q1-1.5*iqr or score > q3+1.5*iqr]


class BlurScoreCalculator:
    def __init__(self, config: Optional[FastBlurConfig] = None):
        self.config = config or FastBlurConfig()
        self._cache = {} if self.config.cache_enabled else None

    def calculate_blur_score(self, image: np.ndarray) -> FastBlurResult:
        config = self.config
        t0 = time.perf_counter()

        if not isinstance(image, np.ndarray) or image.size < 100 or min(image.shape[:2]) < 10:
            return FastBlurResult(False, 0.0, 0.0, ["Invalid image"], False, 0, {}, [], [], 0.0)

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
                result = FastBlurResult(True, float(ensemble), float(conf), [], False, (time.perf_counter()-t0)*1000, method_results, [v[0] for v in valid], [], consistency)
                if cache_key:
                    self._cache[cache_key] = result
                return result

        fallback = []
        for m in config.fallback_methods:
            res = method_worker(m, preproc, config)
            method_results[m] = res
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
        result = FastBlurResult(True, float(ensemble), float(conf), [], True, (time.perf_counter()-t0)*1000, method_results, [v[0] for v in all_valid], outliers, consistency)
        if cache_key:
            self._cache[cache_key] = result
        return result

    def clear_cache(self):
        if self._cache:
            self._cache.clear()


def calculate_blur_score(image: np.ndarray, config: Optional[FastBlurConfig] = None) -> FastBlurResult:
    return BlurScoreCalculator(config).calculate_blur_score(image)