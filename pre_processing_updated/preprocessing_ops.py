import cv2
import numpy as np

def adaptive_preprocess(image, orig_skew=0.0, skew_threshold=2.0):
    contrast = image.std() / 255.0
    brightness = np.mean(image) / 255.0
    processed = image.copy()
    deskewed = False

    # 1. Deskew ONLY if skew is significant
    if abs(orig_skew) > skew_threshold:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        max_allowed_rotation = 5.0
        angle = max(-max_allowed_rotation, min(max_allowed_rotation, -orig_skew))
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        processed = cv2.warpAffine(processed, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        deskewed = True

    # 2. Gentle CLAHE if contrast is low
    if contrast < 0.20:
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
        processed = clahe.apply(processed.astype(np.uint8))

    # 3. Contrast stretch only if very faint
    if contrast < 0.13:
        min_val = np.percentile(processed, 2)
        max_val = np.percentile(processed, 98)
        processed = np.clip((processed - min_val) * 255.0 / (max_val - min_val + 1e-5), 0, 255).astype(np.uint8)

    # 4. Mild denoise
    processed = cv2.fastNlMeansDenoising(processed.astype(np.uint8), None, 5, 7, 21)

    # 5. Slight black point boost for all pages (NEW)
    processed[processed < 70] = 0

    # 6. Gentle brightness normalization if needed
    post_brightness = np.mean(processed) / 255.0
    if post_brightness < 0.8:
        processed = np.clip(processed * 1.05, 0, 255).astype(np.uint8)

    # 7. If output is much darker than input, revert
    if (np.mean(processed) < np.mean(image) - 0.1*255) or (processed.std() > 1.7*image.std()):
        return image.copy(), deskewed

    return processed, deskewed