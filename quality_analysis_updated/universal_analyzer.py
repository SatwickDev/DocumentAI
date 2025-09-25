import fitz
import numpy as np
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from parallel_metrics import analyze_page_metrics_parallel
from quality_config import load_quality_config, verdict_for_page
from verdict_icons import verdict_icon, verdict_color, metric_icon
import cv2

class PageSourceInfo:
    def __init__(self, source_type, has_images, text_layer_type, compression_detected, image_count):
        self.source_type = source_type
        self.has_images = has_images
        self.text_layer_type = text_layer_type
        self.compression_detected = compression_detected
        self.image_count = image_count

class PageAnalysis:
    def __init__(self, page_num, metrics, source, processing_time, error=None):
        self.page_num = page_num
        self.metrics = metrics
        self.source = source
        self.processing_time = processing_time
        self.error = error

def get_metric_value(metric):
    # Helper to extract .value or just return the metric if already scalar
    return getattr(metric, 'value', metric)

def analyze_single_page(args):
    page_num, file_path = args
    try:
        doc = fitz.open(file_path)
        page = doc[page_num]
        t0 = time.time()
        pix = page.get_pixmap(matrix=fitz.Matrix(0.33, 0.33), colorspace=fitz.csGRAY)
        gray_img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        metric_img = cv2.resize(gray_img, (gray_img.shape[1] // 4, gray_img.shape[0] // 4), interpolation=cv2.INTER_AREA)
        metrics = analyze_page_metrics_parallel(metric_img)
        source_info = PageSourceInfo(
            source_type="pdf", has_images=True, text_layer_type="unknown", compression_detected=True, image_count=1,
        )
        processing_time = time.time() - t0
        doc.close()
        # page_num is 0-based; +1 for human-readable
        return PageAnalysis(page_num + 1, metrics, source_info, processing_time)
    except Exception as e:
        return PageAnalysis(page_num + 1, None, None, 0.0, error=str(e))

def analyze_pdf_fast_parallel(file_path: str, max_workers=2):
    doc = fitz.open(file_path)
    total_pages = len(doc)
    doc.close()
    t0 = time.time()
    results = [None] * total_pages
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and collect futures in order
        futures = [executor.submit(analyze_single_page, (i, file_path)) for i in range(total_pages)]
        # Process results in order to ensure deterministic behavior
        for i, fut in enumerate(futures):
            result = fut.result()
            if result:
                results[i] = result
    elapsed = time.time() - t0
    print(f"PDF analysis completed in {elapsed:.2f}s for {total_pages} pages")
    return results

def load_image_for_page(file_path, page_num):
    # page_num is 1-based
    doc = fitz.open(file_path)
    page = doc[page_num-1]
    pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), colorspace=fitz.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    doc.close()
    return img

# Preprocessing functions removed - will be handled by separate preprocessing service

def print_results(results, file_path, config, title="UNIVERSAL DOCUMENT ANALYSIS", force_direct_analysis=False):
    from quality_config import get_metric_category

    results = [r for r in results if r]
    print(f"\n==== {title} ====")
    print(f"Document: {Path(file_path).name}")
    print(f"Total Pages: {len(results)}")
    total_time = sum((r.processing_time or 0) for r in results)
    print(f"Total Processing Time (sum of page workers): {total_time:.2f}s")
    print("\nPage-by-page summary:")
    all_direct = True

    for r in results:
        if r.error:
            print(f"Page {r.page_num:3d}: ERROR: {r.error}")
            all_direct = False
        else:
            m = r.metrics
            metric_dict = {
                "blur_score": get_metric_value(m.blur_score),
                "contrast_score": get_metric_value(m.contrast_score),
                "noise_level": get_metric_value(m.noise_level),
                "sharpness_score": get_metric_value(m.sharpness_score),
                "brightness_score": get_metric_value(m.brightness_score),
                "skew_angle": get_metric_value(m.skew_angle),
                "edge_crop_score": get_metric_value(m.edge_crop_score),
                "shadow_glare_score": get_metric_value(m.shadow_glare_score),
                "blank_page_score": get_metric_value(m.blank_page_score),
            }
            # Force verdict if override is set
            if force_direct_analysis:
                verdict = "direct analysis"
                icon = verdict_icon(verdict)
                colored_verdict = verdict_color(verdict, verdict)
                recommendations = []
            else:
                verdict, per_metric, confidence, confidence_category, recommendations = verdict_for_page(metric_dict, config)
                if verdict == "direect analysis":
                    verdict = "direct analysis"
                icon = verdict_icon(verdict)
                colored_verdict = verdict_color(verdict, verdict)
                if verdict != "direct analysis":
                    all_direct = False

            metric_strs = []
            metric_keys = [
                ("blur_score",    m.blur_score),
                ("contrast_score", m.contrast_score),
                ("noise_level",   m.noise_level),
                ("sharpness_score", m.sharpness_score),
                ("brightness_score", m.brightness_score),
                ("skew_angle",    m.skew_angle),
                ("edge_crop_score", m.edge_crop_score),
                ("shadow_glare_score", m.shadow_glare_score),
                ("blank_page_score", m.blank_page_score),
            ]
            for metric_name, metric_value in metric_keys:
                scalar_val = get_metric_value(metric_value)
                if metric_name in config:
                    reverse = metric_name in {"noise_level", "skew_angle", "edge_crop_score", "shadow_glare_score", "blank_page_score"}
                    category = get_metric_category(scalar_val, config[metric_name], reverse=reverse)
                    icon_metric = metric_icon(category)
                    if metric_name == "skew_angle":
                        metric_strs.append(f"{scalar_val:.2f}° {icon_metric}")
                    elif metric_name in {"contrast_score", "noise_level", "brightness_score", "edge_crop_score", "shadow_glare_score", "blank_page_score"}:
                        metric_strs.append(f"{scalar_val:.3f} {icon_metric}")
                    else:
                        metric_strs.append(f"{scalar_val:.1f} {icon_metric}")

            print((
                f"Page {r.page_num:3d}: "
                f"Blur={metric_strs[0]} "
                f"Contrast={metric_strs[1]} "
                f"Noise={metric_strs[2]} "
                f"Sharpness={metric_strs[3]} "
                f"Brightness={metric_strs[4]} "
                f"Skew={metric_strs[5]} "
                f"EdgeCrop={metric_strs[6]} "
                f"ShadowGlare={metric_strs[7]} "
                f"BlankPage={metric_strs[8]} "
                f"Verdict={icon} {colored_verdict} Time={r.processing_time:.2f}s"
            ))
            # Show recommendations only for pre-processing pages or re-scan verdict
            if (verdict == "pre-processing" or verdict == "re-scan") and recommendations:
                print("    Recommendations:")
                for rec in recommendations:
                    print(f"      - {rec}")

    if all_direct or force_direct_analysis:
        print("\n✅ All pages are ready for direct analysis (🟢). No pre-processing required.")

# PDF preprocessing functions moved to separate preprocessing service
# def save_preprocessed_pdf(...) - removed

# Preprocessing functions moved to separate preprocessing service
# def auto_preprocess_pages(...) - removed

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Robust PDF Quality Analyzer (with adaptive pre-processing)")
    parser.add_argument("file_path", help="PDF file to analyze")
    parser.add_argument("--workers", "-w", type=int, default=2, help="Number of parallel workers (default: 2)")
    parser.add_argument("--config", "-c", type=str, default="quality_config.yaml", help="Path to quality config file")
    args = parser.parse_args()

    config = load_quality_config(args.config)
    results = analyze_pdf_fast_parallel(args.file_path, max_workers=args.workers)
    print_results(results, args.file_path, config)

    preproc_candidates = []
    preproc_recommendations = dict()  # page_num -> recommendations
    for r in results:
        if r.error:
            continue
        m = r.metrics
        metric_dict = {
            "blur_score": get_metric_value(m.blur_score),
            "contrast_score": get_metric_value(m.contrast_score),
            "noise_level": get_metric_value(m.noise_level),
            "sharpness_score": get_metric_value(m.sharpness_score),
            "brightness_score": get_metric_value(m.brightness_score),
            "skew_angle": get_metric_value(m.skew_angle),
            "edge_crop_score": get_metric_value(m.edge_crop_score),
            "shadow_glare_score": get_metric_value(m.shadow_glare_score),
            "blank_page_score": get_metric_value(m.blank_page_score),
        }
        verdict, per_metric, confidence, confidence_category, recommendations = verdict_for_page(metric_dict, config)
        if verdict == "pre-processing":
            preproc_recommendations[r.page_num] = recommendations
            if recommendations:
                preproc_candidates.append(r.page_num)
    print(f"[DEBUG] Pre-processing candidates (final): {preproc_candidates}")

    # Quality analysis complete - preprocessing handled by separate service
    if preproc_candidates:
        print(f"Found {len(preproc_candidates)} pages that may benefit from preprocessing")
        print("Preprocessing recommendations will be handled by the preprocessing service")
    
    print("Quality Analysis Complete!")

if __name__ == "__main__":
    main()