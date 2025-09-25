"""
Enhanced Quality Analyzer merging existing and temp components
"""
import os
import yaml
import numpy as np
from PIL import Image
import cv2
import fitz  # PyMuPDF
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from numba import jit
import gc

logger = logging.getLogger(__name__)

# Import the preprocessing operations from temp
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'temp'))
from preprocessing_ops import preprocess_adaptive

@dataclass
class EnhancedQualityMetrics:
    """Enhanced quality metrics combining existing and new metrics"""
    # Existing metrics
    blur_score: float
    contrast_score: float
    noise_level: float
    sharpness_score: float
    brightness_score: float
    skew_angle: float
    text_coverage: float
    ocr_confidence: float
    margin_safety: float
    duplicate_blank_score: float
    compression_artifacts: float
    page_consistency: float
    
    # New metrics from temp
    edge_crop_score: float = 1.0
    shadow_glare_score: float = 1.0
    blank_page_score: float = 1.0
    resolution_score: float = 1.0
    
    def __post_init__(self):
        """Validate all metrics are between 0 and 1"""
        for field_name, value in self.__dict__.items():
            if isinstance(value, (int, float)) and not 0 <= value <= 1.1:  # 1.1 for small float errors
                raise ValueError(f"{field_name} must be between 0 and 1, got {value}")

@dataclass
class PageAnalysisResult:
    """Result of analyzing a single page"""
    page_number: int
    metrics: EnhancedQualityMetrics
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    needs_preprocessing: bool = False
    preprocessing_applied: List[str] = field(default_factory=list)

@dataclass
class DocumentQualityResult:
    """Overall document quality analysis result"""
    total_pages: int
    overall_score: float
    verdict: str  # 'excellent', 'good', 'needs_preprocessing', 'poor'
    page_analyses: List[PageAnalysisResult]
    critical_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    preprocessed_pages: Optional[List[int]] = None

class EnhancedQualityAnalyzer:
    """Enhanced quality analyzer with preprocessing capabilities"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.thresholds = self.config.get('quality', {}).get('thresholds', {})
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'quality': {
                'thresholds': {
                    'excellent': 0.8,
                    'good': 0.6,
                    'needs_preprocessing': 0.4,
                    'poor': 0.0
                },
                'critical_metrics': [
                    'blur_score', 'contrast_score', 'ocr_confidence',
                    'blank_page_score', 'edge_crop_score'
                ],
                'weights': {
                    'blur_score': 1.5,
                    'contrast_score': 1.5,
                    'ocr_confidence': 2.0,
                    'blank_page_score': 2.0,
                    'edge_crop_score': 1.5,
                    'shadow_glare_score': 1.2,
                    'resolution_score': 1.0
                }
            },
            'preprocessing': {
                'auto_apply_threshold': 0.5,
                'max_preprocessing_attempts': 2
            }
        }
    
    async def analyze_document(self, file_path: str, 
                             apply_preprocessing: bool = True,
                             save_preprocessed: bool = False) -> DocumentQualityResult:
        """Analyze document quality with optional preprocessing"""
        start_time = time.time()
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load document
        if file_path.lower().endswith('.pdf'):
            doc = fitz.open(file_path)
            total_pages = len(doc)
        else:
            # Handle image files
            total_pages = 1
        
        # Analyze pages in parallel
        page_analyses = []
        with ThreadPoolExecutor(max_workers=min(8, total_pages)) as executor:
            future_to_page = {}
            
            for page_num in range(total_pages):
                future = executor.submit(self._analyze_page, file_path, page_num)
                future_to_page[future] = page_num
            
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    result = future.result()
                    page_analyses.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing page {page_num}: {e}")
                    page_analyses.append(self._create_error_page_result(page_num))
        
        # Sort by page number
        page_analyses.sort(key=lambda x: x.page_number)
        
        # Apply preprocessing if needed and requested
        preprocessed_pages = []
        if apply_preprocessing:
            for analysis in page_analyses:
                if analysis.needs_preprocessing:
                    success = await self._apply_preprocessing(
                        file_path, analysis.page_number, save_preprocessed
                    )
                    if success:
                        preprocessed_pages.append(analysis.page_number)
                        # Re-analyze the preprocessed page
                        new_analysis = self._analyze_page(file_path, analysis.page_number)
                        analysis.metrics = new_analysis.metrics
                        analysis.preprocessing_applied = ['adaptive_preprocessing']
        
        # Calculate overall results
        overall_score = self._calculate_overall_score(page_analyses)
        verdict = self._determine_verdict(overall_score, page_analyses)
        critical_issues = self._identify_critical_issues(page_analyses)
        recommendations = self._generate_recommendations(page_analyses, verdict)
        
        return DocumentQualityResult(
            total_pages=total_pages,
            overall_score=overall_score,
            verdict=verdict,
            page_analyses=page_analyses,
            critical_issues=critical_issues,
            recommendations=recommendations,
            processing_time=time.time() - start_time,
            preprocessed_pages=preprocessed_pages if preprocessed_pages else None
        )
    
    def _analyze_page(self, file_path: str, page_num: int) -> PageAnalysisResult:
        """Analyze a single page"""
        start_time = time.time()
        
        # Extract page image
        if file_path.lower().endswith('.pdf'):
            doc = fitz.open(file_path)
            page = doc[page_num]
            pix = page.get_pixmap(dpi=200)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            img_array = np.array(img)
            doc.close()
        else:
            img = Image.open(file_path)
            img_array = np.array(img)
        
        # Calculate all metrics in parallel
        metrics = self._calculate_all_metrics(img_array)
        
        # Identify issues
        issues = self._identify_page_issues(metrics)
        
        # Generate recommendations
        recommendations = self._generate_page_recommendations(metrics)
        
        # Determine if preprocessing is needed
        needs_preprocessing = self._needs_preprocessing(metrics)
        
        return PageAnalysisResult(
            page_number=page_num,
            metrics=metrics,
            issues=issues,
            recommendations=recommendations,
            processing_time=time.time() - start_time,
            needs_preprocessing=needs_preprocessing
        )
    
    def _calculate_all_metrics(self, img_array: np.ndarray) -> EnhancedQualityMetrics:
        """Calculate all quality metrics"""
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Use ThreadPoolExecutor for parallel metric calculation
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all metric calculations
            futures = {
                'blur': executor.submit(self._calculate_blur_score, gray),
                'contrast': executor.submit(self._calculate_contrast_score, gray),
                'noise': executor.submit(self._calculate_noise_level, gray),
                'sharpness': executor.submit(self._calculate_sharpness_score, gray),
                'brightness': executor.submit(self._calculate_brightness_score, gray),
                'skew': executor.submit(self._calculate_skew_angle, gray),
                'edge_crop': executor.submit(self._calculate_edge_crop_score, gray),
                'shadow_glare': executor.submit(self._calculate_shadow_glare_score, gray),
                'blank_page': executor.submit(self._calculate_blank_page_score, gray),
                'resolution': executor.submit(self._calculate_resolution_score, img_array)
            }
            
            # Get results
            results = {name: future.result() for name, future in futures.items()}
        
        # Additional metrics that depend on other processing
        text_coverage = self._calculate_text_coverage(gray)
        ocr_confidence = self._calculate_ocr_confidence(gray)
        margin_safety = self._calculate_margin_safety(gray)
        
        return EnhancedQualityMetrics(
            blur_score=results['blur'],
            contrast_score=results['contrast'],
            noise_level=results['noise'],
            sharpness_score=results['sharpness'],
            brightness_score=results['brightness'],
            skew_angle=results['skew'],
            text_coverage=text_coverage,
            ocr_confidence=ocr_confidence,
            margin_safety=margin_safety,
            duplicate_blank_score=1.0,  # Placeholder
            compression_artifacts=1.0,   # Placeholder
            page_consistency=1.0,        # Placeholder
            edge_crop_score=results['edge_crop'],
            shadow_glare_score=results['shadow_glare'],
            blank_page_score=results['blank_page'],
            resolution_score=results['resolution']
        )
    
    @jit(nopython=True)
    def _calculate_blur_score(self, gray: np.ndarray) -> float:
        """Calculate blur score using Laplacian variance"""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        # Normalize to 0-1 (higher is better)
        return min(1.0, variance / 1000.0)
    
    def _calculate_contrast_score(self, gray: np.ndarray) -> float:
        """Calculate contrast score"""
        std_dev = np.std(gray)
        return min(1.0, std_dev / 128.0)
    
    def _calculate_noise_level(self, gray: np.ndarray) -> float:
        """Calculate noise level (inverted for score)"""
        # Apply median filter and calculate difference
        denoised = cv2.medianBlur(gray, 3)
        noise = np.mean(np.abs(gray.astype(float) - denoised.astype(float)))
        # Invert and normalize (lower noise = higher score)
        return max(0.0, 1.0 - noise / 50.0)
    
    def _calculate_sharpness_score(self, gray: np.ndarray) -> float:
        """Calculate sharpness using gradient magnitude"""
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        sharpness = np.mean(gradient_magnitude)
        return min(1.0, sharpness / 100.0)
    
    def _calculate_brightness_score(self, gray: np.ndarray) -> float:
        """Calculate brightness score"""
        mean_brightness = np.mean(gray)
        # Optimal brightness around 128
        score = 1.0 - abs(mean_brightness - 128) / 128
        return max(0.0, score)
    
    def _calculate_skew_angle(self, gray: np.ndarray) -> float:
        """Calculate skew angle score"""
        # This is a simplified version - real implementation would detect actual skew
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is None:
            return 1.0
        
        # Analyze line angles to detect skew
        angles = []
        for rho, theta in lines[:, 0]:
            angle = theta * 180 / np.pi
            angles.append(angle)
        
        if angles:
            median_angle = np.median(angles)
            skew = abs(median_angle - 90) if median_angle > 45 else abs(median_angle)
            # Convert to score (0 degrees = 1.0, 45 degrees = 0.0)
            return max(0.0, 1.0 - skew / 45.0)
        
        return 1.0
    
    def _calculate_edge_crop_score(self, gray: np.ndarray) -> float:
        """Calculate if important content is cropped at edges"""
        h, w = gray.shape
        edge_size = min(h, w) // 20  # Check 5% of edges
        
        # Check edges for content
        top_edge = gray[:edge_size, :]
        bottom_edge = gray[-edge_size:, :]
        left_edge = gray[:, :edge_size]
        right_edge = gray[:, -edge_size:]
        
        # Calculate edge content
        edge_content = (
            np.mean(top_edge < 240) +
            np.mean(bottom_edge < 240) +
            np.mean(left_edge < 240) +
            np.mean(right_edge < 240)
        ) / 4
        
        # Less content at edges = better score
        return max(0.0, 1.0 - edge_content)
    
    def _calculate_shadow_glare_score(self, gray: np.ndarray) -> float:
        """Calculate shadow and glare score"""
        # Detect very dark and very bright regions
        very_dark = np.sum(gray < 30) / gray.size
        very_bright = np.sum(gray > 225) / gray.size
        
        # Calculate score (less extreme values = better)
        shadow_glare = very_dark + very_bright
        return max(0.0, 1.0 - shadow_glare * 2)
    
    def _calculate_blank_page_score(self, gray: np.ndarray) -> float:
        """Calculate if page is blank"""
        # Check variance - blank pages have low variance
        variance = np.var(gray)
        if variance < 100:
            return 0.0  # Likely blank
        return 1.0
    
    def _calculate_resolution_score(self, img_array: np.ndarray) -> float:
        """Calculate resolution quality score"""
        h, w = img_array.shape[:2]
        # Assume 200 DPI is good, 300+ is excellent
        pixels = h * w
        if pixels < 1000000:  # Less than 1MP
            return 0.3
        elif pixels < 4000000:  # Less than 4MP
            return 0.6
        else:
            return 1.0
    
    def _calculate_text_coverage(self, gray: np.ndarray) -> float:
        """Calculate text coverage on page"""
        # Simple text detection using morphological operations
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Count text pixels
        text_pixels = np.sum(closed == 0)
        total_pixels = gray.size
        coverage = text_pixels / total_pixels
        
        # Normalize to reasonable range (5-50% coverage is normal)
        return min(1.0, coverage * 2)
    
    def _calculate_ocr_confidence(self, gray: np.ndarray) -> float:
        """Estimate OCR confidence based on image quality"""
        # This is a simplified estimation
        # Real implementation would run actual OCR
        blur = self._calculate_blur_score(gray)
        contrast = self._calculate_contrast_score(gray)
        noise = self._calculate_noise_level(gray)
        
        # Weighted average
        confidence = (blur * 0.3 + contrast * 0.4 + noise * 0.3)
        return confidence
    
    def _calculate_margin_safety(self, gray: np.ndarray) -> float:
        """Calculate if text has safe margins"""
        h, w = gray.shape
        margin = min(h, w) // 20  # 5% margin
        
        # Check if content is within margins
        center_region = gray[margin:-margin, margin:-margin]
        if center_region.size == 0:
            return 0.0
        
        # Calculate how much content is in center vs edges
        center_content = np.mean(center_region < 240)
        total_content = np.mean(gray < 240)
        
        if total_content > 0:
            return min(1.0, center_content / total_content)
        return 1.0
    
    def _needs_preprocessing(self, metrics: EnhancedQualityMetrics) -> bool:
        """Determine if page needs preprocessing"""
        critical_metrics = self.config['quality']['critical_metrics']
        threshold = self.config['preprocessing']['auto_apply_threshold']
        
        for metric_name in critical_metrics:
            if hasattr(metrics, metric_name):
                value = getattr(metrics, metric_name)
                if value < threshold:
                    return True
        
        return False
    
    async def _apply_preprocessing(self, file_path: str, page_num: int, 
                                   save: bool = False) -> bool:
        """Apply preprocessing to a page"""
        try:
            # Extract page image
            if file_path.lower().endswith('.pdf'):
                doc = fitz.open(file_path)
                page = doc[page_num]
                pix = page.get_pixmap(dpi=200)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                doc.close()
            else:
                img = Image.open(file_path)
                img_array = np.array(img)
            
            # Apply adaptive preprocessing
            preprocessed = preprocess_adaptive(img_array)
            
            if save:
                # Save preprocessed image
                output_path = file_path.replace('.pdf', f'_preprocessed_page_{page_num}.png')
                cv2.imwrite(output_path, preprocessed)
            
            return True
            
        except Exception as e:
            logger.error(f"Error preprocessing page {page_num}: {e}")
            return False
    
    def _calculate_overall_score(self, page_analyses: List[PageAnalysisResult]) -> float:
        """Calculate overall document score"""
        if not page_analyses:
            return 0.0
        
        weights = self.config['quality']['weights']
        total_score = 0.0
        total_weight = 0.0
        
        for analysis in page_analyses:
            metrics = analysis.metrics
            page_score = 0.0
            page_weight = 0.0
            
            for metric_name, weight in weights.items():
                if hasattr(metrics, metric_name):
                    value = getattr(metrics, metric_name)
                    page_score += value * weight
                    page_weight += weight
            
            if page_weight > 0:
                total_score += page_score / page_weight
                total_weight += 1
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_verdict(self, overall_score: float, 
                          page_analyses: List[PageAnalysisResult]) -> str:
        """Determine overall verdict"""
        thresholds = self.config['quality']['thresholds']
        
        # Check for critical issues
        critical_page_count = sum(1 for p in page_analyses if p.needs_preprocessing)
        if critical_page_count > len(page_analyses) * 0.3:
            return 'needs_preprocessing'
        
        if overall_score >= thresholds['excellent']:
            return 'excellent'
        elif overall_score >= thresholds['good']:
            return 'good'
        elif overall_score >= thresholds['needs_preprocessing']:
            return 'needs_preprocessing'
        else:
            return 'poor'
    
    def _identify_critical_issues(self, page_analyses: List[PageAnalysisResult]) -> List[str]:
        """Identify critical issues across document"""
        issue_counts = {}
        
        for analysis in page_analyses:
            for issue in analysis.issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Return issues affecting more than 20% of pages
        threshold = len(page_analyses) * 0.2
        return [issue for issue, count in issue_counts.items() if count > threshold]
    
    def _identify_page_issues(self, metrics: EnhancedQualityMetrics) -> List[str]:
        """Identify issues for a specific page"""
        issues = []
        
        if metrics.blur_score < 0.5:
            issues.append("Page is blurry")
        if metrics.contrast_score < 0.4:
            issues.append("Low contrast")
        if metrics.noise_level < 0.6:
            issues.append("High noise level")
        if metrics.skew_angle < 0.8:
            issues.append("Page is skewed")
        if metrics.blank_page_score < 0.5:
            issues.append("Page appears blank")
        if metrics.edge_crop_score < 0.7:
            issues.append("Content may be cropped")
        if metrics.shadow_glare_score < 0.6:
            issues.append("Shadows or glare detected")
        
        return issues
    
    def _generate_page_recommendations(self, metrics: EnhancedQualityMetrics) -> List[str]:
        """Generate recommendations for a specific page"""
        recommendations = []
        
        if metrics.blur_score < 0.5:
            recommendations.append("Apply sharpening filter")
        if metrics.contrast_score < 0.4:
            recommendations.append("Enhance contrast using CLAHE")
        if metrics.noise_level < 0.6:
            recommendations.append("Apply denoising filter")
        if metrics.skew_angle < 0.8:
            recommendations.append("Deskew the page")
        if metrics.brightness_score < 0.3 or metrics.brightness_score > 0.7:
            recommendations.append("Normalize brightness")
        if metrics.resolution_score < 0.6:
            recommendations.append("Scan at higher resolution (300+ DPI)")
        
        return recommendations
    
    def _generate_recommendations(self, page_analyses: List[PageAnalysisResult], 
                                verdict: str) -> List[str]:
        """Generate overall recommendations"""
        recommendations = []
        
        if verdict == 'poor':
            recommendations.append("Document quality is too poor for reliable processing. Please re-scan.")
        elif verdict == 'needs_preprocessing':
            recommendations.append("Document requires preprocessing for optimal results.")
            recommendations.append("Apply adaptive preprocessing to improve quality.")
        elif verdict == 'good':
            recommendations.append("Document quality is good. Minor preprocessing may improve results.")
        else:
            recommendations.append("Document quality is excellent. Ready for processing.")
        
        # Add specific recommendations based on common issues
        all_page_recs = []
        for analysis in page_analyses:
            all_page_recs.extend(analysis.recommendations)
        
        # Get most common recommendations
        rec_counts = {}
        for rec in all_page_recs:
            rec_counts[rec] = rec_counts.get(rec, 0) + 1
        
        # Add top 3 most common recommendations
        sorted_recs = sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)
        for rec, count in sorted_recs[:3]:
            if count > len(page_analyses) * 0.2:  # Affects >20% of pages
                recommendations.append(f"{rec} (affects {count} pages)")
        
        return recommendations
    
    def _create_error_page_result(self, page_num: int) -> PageAnalysisResult:
        """Create error result for failed page analysis"""
        return PageAnalysisResult(
            page_number=page_num,
            metrics=EnhancedQualityMetrics(
                blur_score=0, contrast_score=0, noise_level=0,
                sharpness_score=0, brightness_score=0, skew_angle=0,
                text_coverage=0, ocr_confidence=0, margin_safety=0,
                duplicate_blank_score=0, compression_artifacts=0,
                page_consistency=0
            ),
            issues=["Failed to analyze page"],
            recommendations=["Re-scan this page"],
            needs_preprocessing=True
        )

# Make it compatible with existing code
UniversalDocumentAnalyzer = EnhancedQualityAnalyzer