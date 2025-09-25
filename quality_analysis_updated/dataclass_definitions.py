#!/usr/bin/env python3
"""
Data class definitions for Universal Document Analyzer
Contains all the data structures used throughout the analysis system
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum
from datetime import datetime


class DocumentType(Enum):
    """Document quality classification"""
    DIGITAL = "digital"
    GOOD_SCAN = "good_scan"
    POOR_SCAN = "poor_scan"
    CORRUPTED = "corrupted"
    UNPROCESSABLE = "unprocessable"


class SourceType(Enum):
    """Document source type classification"""
    NATIVE_PDF = "native_pdf"
    IMAGE_SCAN = "image_scan"
    HYBRID = "hybrid"
    TEXT_DOCUMENT = "text_document"
    SPREADSHEET = "spreadsheet"
    IMAGE_FILE = "image_file"
    UNKNOWN = "unknown"


class TextLayerType(Enum):
    """Text layer quality classification"""
    NATIVE = "native"
    OCR = "ocr"
    NONE = "none"
    HYBRID = "hybrid"


@dataclass
class MetricValidationResult:
    """Result of metric validation with error handling"""
    is_valid: bool
    value: Union[float, int, Tuple]
    confidence: float = 1.0
    warnings: List[str] = field(default_factory=list)
    fallback_used: bool = False
    processing_time_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class MetricCalculationError(Exception):
    """Exception for metric calculation failures"""
    pass


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for document analysis with validation"""
    blur_score: MetricValidationResult
    resolution: MetricValidationResult
    skew_angle: MetricValidationResult
    contrast_score: MetricValidationResult
    noise_level: MetricValidationResult
    sharpness_score: MetricValidationResult
    brightness_score: MetricValidationResult
    edge_crop_score: MetricValidationResult
    shadow_glare_score: MetricValidationResult
    blank_page_score: MetricValidationResult

    # Computed aggregate metrics
    overall_quality_score: Optional[float] = None
    processing_confidence: Optional[float] = None
    recommended_actions: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Compute aggregate metrics after initialization"""
        self._compute_aggregate_metrics()

    def _compute_aggregate_metrics(self):
        """Compute overall quality score and confidence including all 10 metrics"""

        # All 10 metrics included
        valid_metrics = [m for m in [
            self.blur_score, self.resolution, self.skew_angle,
            self.contrast_score, self.noise_level,
            self.sharpness_score, self.brightness_score,
            self.edge_crop_score, self.shadow_glare_score, self.blank_page_score
        ] if m.is_valid]

        if valid_metrics:
            self.processing_confidence = (
                sum(m.confidence for m in valid_metrics) / len(valid_metrics)
            )

            quality_components = []

            # Example weights (adjust as needed)
            # Blur: 20%, Contrast: 15%, Noise: 10%, Sharpness: 10%, Skew: 10%, Brightness: 10%
            # EdgeCrop: 5%, ShadowGlare: 5%, BlankPage: 5%, Resolution: 10%
            # (weights add to 100%)
            if self.blur_score.is_valid:
                quality_components.append(
                    (min(self.blur_score.value / 100, 1.0), 0.20)
                )
            if self.contrast_score.is_valid:
                quality_components.append(
                    (min(self.contrast_score.value * 2, 1.0), 0.15)
                )
            if self.noise_level.is_valid:
                quality_components.append(
                    (1.0 - min(self.noise_level.value, 1.0), 0.10)
                )
            if self.sharpness_score.is_valid:
                quality_components.append(
                    (min(self.sharpness_score.value / 50, 1.0), 0.10)
                )
            if self.skew_angle.is_valid:
                skew_quality = max(0, 1.0 - abs(self.skew_angle.value) / 10)
                quality_components.append((skew_quality, 0.10))
            if self.brightness_score.is_valid:
                brightness_diff = abs(self.brightness_score.value - 0.5)
                brightness_quality = 1.0 - brightness_diff * 2
                quality_components.append((max(0, brightness_quality), 0.10))
            if self.resolution.is_valid:
                # Assume value is a tuple (width, height)
                width, height = (self.resolution.value if isinstance(self.resolution.value, (tuple, list)) else (self.resolution.value, self.resolution.value))
                dpi_quality = min(width, height) / 300.0  # 300dpi = good
                dpi_quality = min(dpi_quality, 1.0)
                quality_components.append((dpi_quality, 0.10))
            if self.edge_crop_score.is_valid:
                edge_quality = 1.0 - min(self.edge_crop_score.value, 1.0)
                quality_components.append((edge_quality, 0.05))
            if self.shadow_glare_score.is_valid:
                glare_quality = 1.0 - min(self.shadow_glare_score.value, 1.0)
                quality_components.append((glare_quality, 0.05))
            if self.blank_page_score.is_valid:
                blank_quality = 1.0 - min(self.blank_page_score.value, 1.0)
                quality_components.append((blank_quality, 0.05))

            if quality_components:
                total_weight = sum(weight for _, weight in quality_components)
                self.overall_quality_score = (
                    sum(score * weight for score, weight in quality_components)
                    / total_weight
                )
        else:
            self.processing_confidence = 0.0
            self.overall_quality_score = 0.0


@dataclass
class PageSourceInfo:
    """Information about the source and nature of a document page"""
    source_type: SourceType
    has_images: bool
    text_layer_type: TextLayerType
    compression_detected: bool
    image_count: int

    # Additional metadata
    file_format: Optional[str] = None
    color_space: Optional[str] = None
    bit_depth: Optional[int] = None
    compression_ratio: Optional[float] = None
    metadata_quality: float = 1.0


@dataclass
class ProcessingMetadata:
    """Metadata about the processing pipeline"""
    processing_start_time: datetime
    processing_end_time: Optional[datetime] = None
    total_processing_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_cores_used: int = 1
    algorithms_used: Dict[str, str] = field(default_factory=dict)
    fallbacks_triggered: List[str] = field(default_factory=list)
    warnings_generated: List[str] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass
class ContentAnalysisInfo:
    """Detailed content analysis information"""
    word_count: int = 0
    char_count: int = 0
    paragraph_count: int = 0
    line_count: int = 0
    sentence_count: int = 0

    # Language and encoding
    detected_language: Optional[str] = None
    encoding_confidence: float = 1.0

    # Structure analysis
    has_tables: bool = False
    table_count: int = 0
    has_headers: bool = False
    has_footnotes: bool = False

    # Content quality
    text_coherence_score: float = 1.0
    ocr_confidence: Optional[float] = None
    character_recognition_errors: int = 0


@dataclass
class PageAnalysis:
    """Complete analysis result for a single page/sheet/item"""
    page_num: int
    doc_type: DocumentType
    confidence: float
    metrics: QualityMetrics
    source: PageSourceInfo
    issues: List[str]

    # Extended analysis data
    content_info: Optional[ContentAnalysisInfo] = None
    processing_metadata: Optional[ProcessingMetadata] = None
    recommendations: List[str] = field(default_factory=list)

    # Quality flags
    needs_ocr: bool = False
    needs_preprocessing: bool = False
    is_machine_readable: bool = True
    has_security_issues: bool = False

    def __post_init__(self):
        """Post-initialization processing"""
        self._validate_analysis()
        self._generate_recommendations()

    def _validate_analysis(self):
        """Validate the analysis results"""
        # Ensure confidence is in valid range
        self.confidence = max(0.0, min(1.0, self.confidence))

        # Validate page number
        if self.page_num <= 0:
            self.issues.append("invalid_page_number")

    def _generate_recommendations(self):
        """Generate processing recommendations based on analysis"""
        if not self.recommendations:  # Only generate if not already provided
            recs = []

            # Image quality recommendations
            if (
                self.metrics.blur_score.is_valid
                and self.metrics.blur_score.value < 50
            ):
                recs.append("image_enhancement")
                self.needs_preprocessing = True

            if (
                self.metrics.skew_angle.is_valid
                and abs(self.metrics.skew_angle.value) > 2.0
            ):
                recs.append("deskew_correction")
                self.needs_preprocessing = True

            if (
                self.metrics.contrast_score.is_valid
                and self.metrics.contrast_score.value < 0.15
            ):
                recs.append("contrast_enhancement")
                self.needs_preprocessing = True

            if (
                self.metrics.noise_level.is_valid
                and self.metrics.noise_level.value > 0.3
            ):
                recs.append("noise_reduction")
                self.needs_preprocessing = True

            # Content recommendations
            if self.content_info and self.content_info.word_count == 0:
                recs.append("content_extraction_failed")
                self.is_machine_readable = False

            self.recommendations = recs if recs else ["no_action_needed"]


@dataclass
class DocumentAnalysisResult:
    """Complete document analysis result"""
    file_path: str
    file_type: str
    total_pages: int
    pages: List[PageAnalysis]

    # Aggregate statistics
    overall_quality: float = 0.0
    processing_success_rate: float = 1.0
    total_processing_time_ms: float = 0.0

    # Summary statistics
    digital_pages: int = 0
    good_scan_pages: int = 0
    poor_scan_pages: int = 0
    failed_pages: int = 0

    # Content summary
    total_words: int = 0
    total_images: int = 0
    pages_needing_ocr: int = 0
    pages_needing_preprocessing: int = 0

    def __post_init__(self):
        """Compute summary statistics"""
        self._compute_summary_statistics()

    def _compute_summary_statistics(self):
        """Compute aggregate statistics from page analyses"""
        if not self.pages:
            return

        # Count document types
        for page in self.pages:
            if page.doc_type == DocumentType.DIGITAL:
                self.digital_pages += 1
            elif page.doc_type == DocumentType.GOOD_SCAN:
                self.good_scan_pages += 1
            elif page.doc_type == DocumentType.POOR_SCAN:
                self.poor_scan_pages += 1
            else:
                self.failed_pages += 1

            # Aggregate content
            if page.content_info:
                self.total_words += page.content_info.word_count

            self.total_images += page.source.image_count

            if page.needs_ocr:
                self.pages_needing_ocr += 1

            if page.needs_preprocessing:
                self.pages_needing_preprocessing += 1

        # Compute overall quality
        valid_qualities = [p.metrics.overall_quality_score for p in self.pages
                           if p.metrics.overall_quality_score is not None]
        if valid_qualities:
            self.overall_quality = sum(valid_qualities) / len(valid_qualities)

        # Compute processing success rate
        successful_pages = len([
            p for p in self.pages
            if p.doc_type != DocumentType.UNPROCESSABLE
        ])
        self.processing_success_rate = successful_pages / len(self.pages)

        # Sum processing times
        for page in self.pages:
            if page.processing_metadata:
                self.total_processing_time_ms += (
                    page.processing_metadata.total_processing_time_ms
                )


@dataclass
class AnalyzerConfiguration:
    """Configuration settings for the document analyzer"""
    # Performance settings
    max_workers: int = 4
    memory_limit_mb: int = 2048
    processing_timeout_seconds: int = 300

    # Quality thresholds
    minimum_blur_score: float = 30.0
    maximum_noise_level: float = 0.5
    maximum_skew_angle: float = 10.0
    minimum_contrast: float = 0.1

    # Processing options
    enable_caching: bool = True
    enable_fallbacks: bool = True
    strict_validation: bool = False
    enable_preprocessing: bool = True

    # Output options
    include_debug_info: bool = False
    save_intermediate_results: bool = False
    compression_level: int = 6

    # Security settings
    validate_file_integrity: bool = True
    scan_for_malicious_content: bool = False
    maximum_file_size_mb: int = 100


# Exception classes for better error handling
class DocumentAnalysisError(Exception):
    """Base exception for document analysis errors"""
    pass


class FileProcessingError(DocumentAnalysisError):
    """Error during file processing"""
    pass


class ValidationError(DocumentAnalysisError):
    """Error during validation"""
    pass


class ConfigurationError(DocumentAnalysisError):
    """Error in configuration"""
    pass
