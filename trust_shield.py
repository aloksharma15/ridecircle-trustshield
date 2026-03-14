import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ==========================
# DEMO MODE — No ML models needed
# Pure CV pipeline, highly improved
# ==========================
DEMO_MODE = True


# ==========================
# IMAGE ALIGNMENT
# Uses ORB + Homography to align post image to pre image
# ==========================
def align_images(img_pre, img_post):
    gray_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2GRAY)
    gray_post = cv2.cvtColor(img_post, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(3000)
    kp1, des1 = orb.detectAndCompute(gray_pre, None)
    kp2, des2 = orb.detectAndCompute(gray_post, None)

    if des1 is None or des2 is None:
        return img_post, False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.72 * n.distance]

    if len(good) < 8:
        return img_post, False

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    if M is None:
        return img_post, False

    aligned = cv2.warpPerspective(img_post, M, (img_pre.shape[1], img_pre.shape[0]))
    return aligned, True


# ==========================
# LIGHTING NORMALIZATION
# Reduces false positives from lighting/shadow changes
# ==========================
def normalize_lighting(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ==========================
# EDGE CHANGE DETECTION
# Detects structural changes like dents, cracks
# ==========================
def detect_edge_changes(img_pre, img_post):
    gray_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2GRAY)
    gray_post = cv2.cvtColor(img_post, cv2.COLOR_BGR2GRAY)

    edges_pre = cv2.Canny(gray_pre, 50, 150)
    edges_post = cv2.Canny(gray_post, 50, 150)

    edge_diff = cv2.absdiff(edges_pre, edges_post)
    edge_diff = cv2.GaussianBlur(edge_diff, (7, 7), 0)
    _, edge_thresh = cv2.threshold(edge_diff, 30, 255, cv2.THRESH_BINARY)

    edge_change_ratio = np.sum(edge_thresh > 0) / (edge_thresh.shape[0] * edge_thresh.shape[1])
    return edge_change_ratio


# ==========================
# COLOR CHANGE DETECTION
# Detects paint damage, scratches, rust
# ==========================
def detect_color_changes(img_pre, img_post):
    hsv_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2HSV)
    hsv_post = cv2.cvtColor(img_post, cv2.COLOR_BGR2HSV)

    diff_h = cv2.absdiff(hsv_pre[:, :, 0], hsv_post[:, :, 0])
    diff_s = cv2.absdiff(hsv_pre[:, :, 1], hsv_post[:, :, 1])

    color_diff = cv2.addWeighted(diff_h, 0.5, diff_s, 0.5, 0)
    color_diff = cv2.GaussianBlur(color_diff, (7, 7), 0)
    _, color_thresh = cv2.threshold(color_diff, 25, 255, cv2.THRESH_BINARY)

    color_change_ratio = np.sum(color_thresh > 0) / (color_thresh.shape[0] * color_thresh.shape[1])
    return color_change_ratio


# ==========================
# DAMAGE CONTOUR ANALYSIS
# Main structural difference detection
# ==========================
def analyze_damage_contours(img_pre, img_post):
    gray_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2GRAY)
    gray_post = cv2.cvtColor(img_post, cv2.COLOR_BGR2GRAY)

    ssim_score, diff = ssim(gray_pre, gray_post, full=True)

    diff_map = (1 - diff) * 255
    diff_map = diff_map.astype("uint8")
    diff_map = cv2.GaussianBlur(diff_map, (9, 9), 0)

    # Adaptive threshold — much better than fixed 100
    diff_thresh = cv2.adaptiveThreshold(
        diff_map, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, -3
    )

    # Morphological cleanup — removes noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_CLOSE, kernel)
    diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = 0
    largest_area = 0
    total_intensity = 0
    damage_count = 0
    total_pixels = gray_pre.shape[0] * gray_pre.shape[1]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 800:  # Increased from 600 to reduce noise
            total_area += area
            largest_area = max(largest_area, area)
            damage_count += 1

            mask = np.zeros_like(diff_map)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_intensity = cv2.mean(diff_map, mask=mask)[0]
            total_intensity += mean_intensity * area

    damage_ratio = total_area / total_pixels
    largest_ratio = largest_area / total_pixels
    avg_intensity = (total_intensity / total_area) if total_area > 0 else 0
    normalized_intensity = avg_intensity / 255

    return ssim_score, damage_ratio, largest_ratio, normalized_intensity, damage_count


# ==========================
# SINGLE SIDE PIPELINE
# ==========================
def trust_shield_check(pre_path, post_path):

    img_pre = cv2.imread(pre_path)
    img_post = cv2.imread(post_path)

    if img_pre is None or img_post is None:
        return {"success": False, "error": "IMAGE LOAD FAILED"}

    # Resize for consistent processing
    img_pre = cv2.resize(img_pre, (900, 675))
    img_post = cv2.resize(img_post, (900, 675))

    # Step 1 — Normalize lighting to reduce false positives
    img_pre = normalize_lighting(img_pre)
    img_post = normalize_lighting(img_post)

    # Step 2 — Align images
    aligned, aligned_ok = align_images(img_pre, img_post)
    if not aligned_ok:
        aligned = img_post  # Fall back to original if alignment fails

    # Step 3 — Edge change detection (catches dents, cracks)
    edge_change = detect_edge_changes(img_pre, aligned)

    # Step 4 — Color change detection (catches scratches, paint damage)
    color_change = detect_color_changes(img_pre, aligned)

    # Step 5 — SSIM + contour analysis
    ssim_score, damage_ratio, largest_ratio, normalized_intensity, damage_count = analyze_damage_contours(img_pre, aligned)

    # ==========================
    # IMPROVED SEVERITY FORMULA
    # Weighted combination of all 4 signals
    # ==========================
    severity_score = (
        0.35 * damage_ratio +
        0.20 * largest_ratio +
        0.25 * edge_change +
        0.20 * color_change
    )

    # SSIM penalty — if SSIM is very low, boost severity
    if ssim_score < 0.80:
        severity_score = min(severity_score + 0.15, 1.0)
    elif ssim_score < 0.88:
        severity_score = min(severity_score + 0.08, 1.0)

    # ==========================
    # IMPROVED THRESHOLDS
    # More precise damage classification
    # ==========================
    if ssim_score > 0.93 and severity_score < 0.06 and edge_change < 0.03:
        status = "CLEAR"
    elif severity_score > 0.30 or (ssim_score < 0.75 and damage_count > 3):
        status = "MAJOR DAMAGE"
    elif severity_score > 0.08 or damage_count >= 2:
        status = "MODERATE DAMAGE"
    elif severity_score > 0.03 or damage_count >= 1:
        status = "MINOR DAMAGE"
    else:
        status = "CLEAR"

    return {
        "success": True,
        "status": status,
        "ssim_score": round(ssim_score, 4),
        "severity_score": round(severity_score, 4),
        "damage_ratio": round(damage_ratio, 4),
        "largest_blob_ratio": round(largest_ratio, 4),
        "edge_change_ratio": round(edge_change, 4),
        "color_change_ratio": round(color_change, 4),
        "damage_count": damage_count
    }


# ==========================
# MULTI SIDE AGGREGATOR
# ==========================
def trust_shield_multi(pre_dict, post_dict):

    sides = {}
    total_severity = 0
    total_damage_ratio = 0
    total_ssim = 0
    max_severity = 0
    valid_pairs = 0
    total_damage_count = 0

    for side in ["front", "rear", "left", "right"]:

        pre_path = pre_dict.get(side)
        post_path = post_dict.get(side)

        if not pre_path or not post_path:
            continue

        result = trust_shield_check(pre_path, post_path)

        if not result["success"]:
            sides[side] = {"error": result.get("error")}
            continue

        sides[side] = result

        total_severity += result["severity_score"]
        total_damage_ratio += result["damage_ratio"]
        total_ssim += result["ssim_score"]
        max_severity = max(max_severity, result["severity_score"])
        total_damage_count += result["damage_count"]
        valid_pairs += 1

    if valid_pairs == 0:
        return {"success": False, "error": "NO VALID IMAGE PAIRS"}

    avg_severity = total_severity / valid_pairs
    avg_damage_ratio = total_damage_ratio / valid_pairs
    avg_ssim = total_ssim / valid_pairs

    # ==========================
    # OVERALL STATUS
    # Uses both average AND worst-case side
    # ==========================
    if avg_ssim > 0.93 and avg_severity < 0.06 and total_damage_count == 0:
        overall_status = "APPROVED"
    elif max_severity > 0.30 or avg_severity > 0.20:
        overall_status = "MAJOR DAMAGE"
    elif avg_severity > 0.07 or total_damage_count >= 3:
        overall_status = "MODERATE DAMAGE"
    elif avg_severity > 0.03 or total_damage_count >= 1:
        overall_status = "MINOR DAMAGE"
    else:
        overall_status = "APPROVED"

    return {
        "success": True,
        "overall_status": overall_status,
        "overall_severity_score": round(avg_severity, 4),
        "overall_damage_ratio": round(avg_damage_ratio, 4),
        "overall_ssim": round(avg_ssim, 4),
        "sides": sides
    }
