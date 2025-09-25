#!/usr/bin/env python3
"""
Ultra-Fast Skew Angle Calculator (3+2 Ensemble)
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
class SkewConfig:
    primary_methods: List[str] = field(default_factory=lambda: [
        "projection_profile", "hough_transform", "text_line_detection"
    ])
    fallback_methods: List[str] = field(default_factory=lambda: [
        "autocorrelation", "fourier_transform"
    ])
    use_downsampling: bool = True
    downsample_threshold: int = 1000
    downsample_factor: float = 0.5
    consistency_threshold: float = 0.7
    min_methods_required: int = 2
    cache_enabled: bool = True
    max_workers: int = 3
    method_weights: Dict[str, float] = field(default_factory=lambda: {
        "projection_profile": 0.35,
        "hough_transform": 0.30,
        "text_line_detection": 0.25,
        "autocorrelation": 0.05,
        "fourier_transform": 0.05
    })

@dataclass
class MethodResult:
    method_name: str
    angle: float
    confidence: float
    processing_time_ms: float
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)

@dataclass
class SkewResult:
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
def fast_projection_variance_batch(binary_image: np.ndarray, angles: np.ndarray) -> np.ndarray:
    h, w = binary_image.shape
    center_x, center_y = w // 2, h // 2
    variances = np.zeros(len(angles), dtype=np.float64)
    for angle_idx in prange(len(angles)):
        angle_rad = np.radians(angles[angle_idx])
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        projection = np.zeros(h, dtype=np.float64)
        for y in range(h):
            for x in range(w):
                if binary_image[y, x] > 0:
                    rx = cos_a * (x - center_x) - sin_a * (y - center_y)
                    ry = sin_a * (x - center_x) + cos_a * (y - center_y)
                    proj_y = int(ry + center_y)
                    if 0 <= proj_y < h:
                        projection[proj_y] += 1.0
        mean_val = np.mean(projection)
        variance = np.mean((projection - mean_val) ** 2)
        variances[angle_idx] = variance
    return variances

def shared_preprocessing(image: np.ndarray, config: SkewConfig) -> Dict[str, np.ndarray]:
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        gray = image.copy()
    if config.use_downsampling and max(gray.shape) > config.downsample_threshold:
        scale = config.downsample_factor
        gray = cv2.resize(gray, (int(gray.shape[1]*scale), int(gray.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    return {"gray": gray, "binary": binary, "edges": edges}

def method_worker(method, imgs, config: SkewConfig) -> MethodResult:
    t0 = time.perf_counter()
    try:
        if method == "projection_profile":
            angles = np.arange(-10, 10.2, 0.2)
            variances = fast_projection_variance_batch(imgs["binary"], angles)
            idx = np.argmax(variances)
            angle = angles[idx]
            conf = min(1.0, variances[idx]/(np.mean(variances)+1e-4))
        elif method == "hough_transform":
            lines = cv2.HoughLines(imgs["edges"], 1, np.pi / 180, threshold=max(20, min(imgs["edges"].shape)//20))
            if lines is None or len(lines) == 0:
                return MethodResult(method, 0.0, 0.0, (time.perf_counter()-t0)*1000, False, ["No lines"])
            arr = []
            for l in lines:
                a = np.degrees(l[0][1])-90
                if a > 45: a -= 90
                if a < -45: a += 90
                if -10 <= a <= 10: arr.append(a)
            if not arr:
                return MethodResult(method, 0.0, 0.0, (time.perf_counter()-t0)*1000, False, ["No valid angles"])
            angle = float(np.median(arr))
            conf = max(0.1, 1.0-np.std(arr)/5.0)
        elif method == "text_line_detection":
            bin = cv2.adaptiveThreshold(imgs["gray"],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
            lines_img = cv2.morphologyEx(bin, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) < 3:
                return MethodResult(method, 0.0, 0.0, (time.perf_counter()-t0)*1000, False, ["Few lines"])
            arr = []
            for c in contours:
                if cv2.contourArea(c) < 100: continue
                [vx, vy, _, _] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
                a = np.degrees(np.arctan2(vy, vx))
                if a > 45: a -= 90
                if a < -45: a += 90
                if -10 <= a <= 10: arr.append(a)
            if not arr:
                return MethodResult(method, 0.0, 0.0, (time.perf_counter()-t0)*1000, False, ["No angles"])
            angle = float(np.median(arr))
            conf = (max(0.1, 1.0-np.std(arr)/3.0)+min(1.0,len(arr)/10.0))/2
        elif method == "autocorrelation":
            bin = imgs["binary"]
            angles = np.arange(-10, 10.2, 0.2)
            corrs = []
            h,w = bin.shape
            center = (w//2,h//2)
            for a in angles:
                mat = cv2.getRotationMatrix2D(center, a, 1.0)
                rot = cv2.warpAffine(bin, mat, (w,h))
                proj = np.sum(rot, axis=1)
                if len(proj)>1:
                    shifted = np.roll(proj,1)
                    corr = np.corrcoef(proj, shifted)[0,1]
                    corrs.append(corr if not np.isnan(corr) else 0.0)
                else:
                    corrs.append(0.0)
            idx = np.argmax(corrs)
            angle = angles[idx]
            conf = min(1.0, (corrs[idx]-np.mean(corrs))/max(abs(corrs[idx]),abs(np.mean(corrs)),0.1))
        elif method == "fourier_transform":
            f = np.fft.fft2(imgs["gray"])
            fshift = np.fft.fftshift(f)
            logmag = np.log(np.abs(fshift)+1e-10)
            h,w = logmag.shape
            cy,cx = h//2,w//2
            angles = np.arange(-10,10.2,0.2)
            energies = []
            for a in angles:
                rad = np.radians(a+90)
                maxd = min(cy,cx)*0.8
                dists = np.linspace(-maxd,maxd,int(maxd*2))
                e=0;ct=0
                for d in dists:
                    x=int(cx+d*np.cos(rad))
                    y=int(cy+d*np.sin(rad))
                    if 0<=x<w and 0<=y<h: e+=logmag[y,x]; ct+=1
                energies.append(e/max(ct,1))
            idx = np.argmax(energies)
            angle = angles[idx]
            mean_e = np.mean(energies)
            conf = min(1.0, (energies[idx]-mean_e)/(mean_e+1e-10))
        else:
            return MethodResult(method, 0.0, 0.0, (time.perf_counter()-t0)*1000, False, [f"Unknown method"])
        return MethodResult(method, float(angle), float(conf), (time.perf_counter()-t0)*1000, True)
    except Exception as e:
        return MethodResult(method, 0.0, 0.0, (time.perf_counter()-t0)*1000, False, [str(e)])

class SkewAngleCalculator:
    def __init__(self, config: Optional[SkewConfig] = None):
        self.config = config or SkewConfig()
        self._cache = {} if self.config.cache_enabled else None

    def calculate_skew_angle(self, image: np.ndarray) -> SkewResult:
        config = self.config
        t0 = time.perf_counter()
        if not isinstance(image, np.ndarray) or image.size < 100 or min(image.shape[:2]) < 10:
            return SkewResult(False, 0.0, 0.0, ["Invalid image"], False, 0, {}, [], 0.0)
        cache_key = None
        if config.cache_enabled:
            h = hashlib.md5(image.tobytes() if image.size<200000 else image[::5,::5].tobytes()).hexdigest()
            cache_key = f"{h}_{image.shape}_{str(config.primary_methods)}"
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                cached.processing_time_ms = (time.perf_counter()-t0)*1000
                return cached
        imgs = shared_preprocessing(image, config)
        method_results = {}
        valid = []
        for m in config.primary_methods:
            res = method_worker(m, imgs, config)
            method_results[res.method_name] = res
            if res.is_valid:
                valid.append((res.method_name, res.angle, res.confidence))
        angles = np.array([v[1] for v in valid])
        if len(valid) >= config.min_methods_required:
            std = np.std(angles)
            mean = np.mean(angles)
            consistency = 1.0 - std / max(abs(mean), 1.0)
            if consistency >= config.consistency_threshold:
                ensemble = np.average(angles, weights=[config.method_weights.get(m,1.0) for m,_,_ in valid])
                conf = min(1.0, np.mean([v[2] for v in valid]))
                result = SkewResult(True, float(ensemble), float(conf), [], False, (time.perf_counter()-t0)*1000, method_results, [v[0] for v in valid], consistency)
                if cache_key:
                    self._cache[cache_key] = result
                return result
        fallback = []
        for m in config.fallback_methods:
            res = method_worker(m, imgs, config)
            method_results[m] = res
            if res.is_valid:
                fallback.append((res.method_name, res.angle, res.confidence))
        all_valid = valid + fallback
        angles = np.array([v[1] for v in all_valid]) if all_valid else np.array([0.0])
        std = np.std(angles)
        mean = np.mean(angles)
        consistency = 1.0 - std / max(abs(mean), 1.0)
        ensemble = np.average(angles, weights=[config.method_weights.get(m,1.0) for m,_,_ in all_valid]) if all_valid else 0.0
        conf = min(1.0, np.mean([v[2] for v in all_valid])) if all_valid else 0.0
        result = SkewResult(True, float(ensemble), float(conf), [], True, (time.perf_counter()-t0)*1000, method_results, [v[0] for v in all_valid], consistency)
        if cache_key:
            self._cache[cache_key] = result
        return result

    def clear_cache(self):
        if self._cache:
            self._cache.clear()
            
    # --- Convenience function ---

def calculate_skew_angle(image: np.ndarray, config: Optional[SkewConfig]=None) -> SkewResult:
    return SkewAngleCalculator(config).calculate_skew_angle(image)