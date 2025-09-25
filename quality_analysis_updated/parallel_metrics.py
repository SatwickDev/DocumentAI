from dataclass_definitions import QualityMetrics


def analyze_page_metrics_parallel(gray_img):
    """
    Analyze all quality metrics for a page sequentially to ensure deterministic results.

    Each metric function returns a result object (with .value, .confidence, .warnings, etc).
    All results are combined into a single QualityMetrics dataclass instance for downstream use.
    Processing is done sequentially to prevent race conditions and ensure consistent scores.
    """
    # Import all metric calculation functions
    from metrics.blur_score import calculate_blur_score
    from metrics.resolution import calculate_resolution
    from metrics.skew_angle import calculate_skew_angle
    from metrics.contrast_score import calculate_contrast_score
    from metrics.noise_level import calculate_noise_level
    from metrics.sharpness_score import calculate_sharpness_score
    from metrics.brightness_score import calculate_brightness_score
    from metrics.edge_crop_score import calculate_edge_crop_score
    from metrics.shadow_glare_score import calculate_shadow_glare_score
    from metrics.blank_page_score import calculate_blank_page_score

    metrics_to_run = {
        "blur_score": lambda: calculate_blur_score(gray_img),
        "resolution": lambda: calculate_resolution(gray_img),
        "skew_angle": lambda: calculate_skew_angle(gray_img),
        "contrast_score": lambda: calculate_contrast_score(gray_img),
        "noise_level": lambda: calculate_noise_level(gray_img),
        "sharpness_score": lambda: calculate_sharpness_score(gray_img),
        "brightness_score": lambda: calculate_brightness_score(gray_img),
        "edge_crop_score": lambda: calculate_edge_crop_score(gray_img),
        "shadow_glare_score": lambda: calculate_shadow_glare_score(gray_img),
        "blank_page_score": lambda: calculate_blank_page_score(gray_img),
    }

    results = {}
    # Process metrics sequentially to ensure deterministic behavior and consistent results
    metric_names = list(metrics_to_run.keys())  
    for name in metric_names:
        try:
            results[name] = metrics_to_run[name]()
        except Exception as e:
            print(f"Error computing {name}: {e}")
            results[name] = None

    return QualityMetrics(
        blur_score=results["blur_score"],
        resolution=results["resolution"],
        skew_angle=results["skew_angle"],
        contrast_score=results["contrast_score"],
        noise_level=results["noise_level"],
        sharpness_score=results["sharpness_score"],
        brightness_score=results["brightness_score"],
        edge_crop_score=results["edge_crop_score"],
        shadow_glare_score=results["shadow_glare_score"],
        blank_page_score=results["blank_page_score"]
    )