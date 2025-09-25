import yaml

def load_quality_config(config_path="quality_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_metric_category(metric_value, thresholds, reverse=False):
    if reverse:
        if metric_value <= thresholds["excellent"]:
            return "excellent"
        elif metric_value <= thresholds["good"]:
            return "good"
        elif metric_value <= thresholds["medium"]:
            return "medium"
        else:
            return "low"
    else:
        if metric_value >= thresholds["excellent"]:
            return "excellent"
        elif metric_value >= thresholds["good"]:
            return "good"
        elif metric_value >= thresholds["medium"]:
            return "medium"
        else:
            return "low"

def normalize_metric(metric_value, thresholds, reverse=False):
    keys = ["excellent", "good", "medium", "low"]
    vals = [thresholds[k] for k in keys]
    if reverse:
        vals = vals[::-1]
    min_val, max_val = min(vals), max(vals)
    if max_val == min_val:
        return 1.0
    norm = (metric_value - min_val) / (max_val - min_val)
    norm = max(0, min(norm, 1))
    return 1.0 - norm if reverse else norm

def calculate_confidence_from_metrics(metrics, config, weights=None, critical_metrics=None):
    # Only treat very low confidence if a CRITICAL metric is low
    if critical_metrics is None:
        critical_metrics = [
            "blur_score", "sharpness_score", "skew_angle", "noise_level"
        ]  # contrast/brightness left out!
    if weights is None:
        weights = {
            "blur_score": 3,
            "sharpness_score": 3,
            "skew_angle": 2,
            "noise_level": 2,
            "contrast_score": 1,
            "brightness_score": 1,
        }
    reverse_metrics = {"noise_level", "skew_angle"}
    for metric in critical_metrics:
        if metric in metrics and metric in config:
            cat = get_metric_category(metrics[metric], config[metric], reverse=(metric in reverse_metrics))
            if cat == "low":
                confidence_cats = config.get("confidence_category", config.get("confidence_score"))
                if confidence_cats and "medium" in confidence_cats:
                    return confidence_cats["medium"] - 0.01
                else:
                    return 0.55
    # Otherwise, weighted mean
    total = 0
    total_weight = 0
    for metric, value in metrics.items():
        if metric in weights and metric in config:
            reverse = metric in reverse_metrics
            norm = normalize_metric(value, config[metric], reverse)
            weight = weights.get(metric, 1)
            total += norm * weight
            total_weight += weight
    return total / total_weight if total_weight > 0 else 0.0

def get_confidence_category(confidence, thresholds):
    if confidence >= thresholds["excellent"]:
        return "excellent"
    elif confidence >= thresholds["good"]:
        return "good"
    elif confidence >= thresholds["medium"]:
        return "medium"
    else:
        return "low"

def get_confidence_verdict(confidence, thresholds):
    if confidence >= thresholds["excellent"]:
        return "direct analysis"
    elif confidence >= thresholds["good"]:
        return "pre-processing"
    elif confidence >= thresholds["medium"]:
        return "azure document analysis"
    else:
        return "reupload"

def generate_recommendations(per_metric_category, config):
    recs = []
    rec_section = config.get("recommendations", {})
    for metric, category in per_metric_category.items():
        advices = rec_section.get(metric, {}).get(category, [])
        for advice in advices:
            recs.append(f"{metric.replace('_', ' ').capitalize()}: {advice}")
    return recs

def verdict_for_page(metrics, config):
    reverse_metrics = {"noise_level", "skew_angle"}
    per_metric_category = {}
    # First, check blank_page_score as a hard blocker
    if "blank_page_score" in metrics and "blank_page_score" in config:
        blank_cat = get_metric_category(metrics["blank_page_score"], config["blank_page_score"], reverse=True)
        per_metric_category["blank_page_score"] = blank_cat
        if blank_cat == "low":
            recommendations = [
                "Quality issue detected: Blank page score. This cannot be fixed by pre-processing. Please re-scan or recapture the page."
            ]
            return "re-scan", per_metric_category, 0.0, "low", recommendations

    # Calculate confidence only with "main" metrics, NOT edge_crop, shadow_glare, blank_page
    for metric, value in metrics.items():
        if metric in config and metric not in {"confidence_score", "confidence_category", "recommendations", "edge_crop_score", "shadow_glare_score", "blank_page_score"}:
            cat = get_metric_category(value, config[metric], reverse=(metric in reverse_metrics))
            per_metric_category[metric] = cat

    confidence = calculate_confidence_from_metrics(metrics, config)
    confidence_cats = config.get("confidence_category", config.get("confidence_score"))
    confidence_category = get_confidence_category(confidence, confidence_cats)
    verdict = get_confidence_verdict(confidence, config["confidence_score"])
    recommendations = generate_recommendations(per_metric_category, config)
    return verdict, per_metric_category, confidence, confidence_category, recommendations