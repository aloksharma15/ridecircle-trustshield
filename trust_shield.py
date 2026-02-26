import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO

# ==========================
# DEVICE
# ==========================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ==========================
# LOAD YOLO
# ==========================
yolo_model = YOLO("yolov8n.pt")

# ==========================
# LOAD CNN (logging only)
# ==========================
classifier = models.mobilenet_v2(weights=None)
classifier.classifier[1] = nn.Linear(classifier.last_channel, 2)
classifier.load_state_dict(
    torch.load("models/damage_classifier.pth", map_location=device)
)
classifier = classifier.to(device)
classifier.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ==========================
# YOLO DETECTION
# ==========================
def detect_car_region(image):
    results = yolo_model(image, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = yolo_model.names[cls_id]
        if label == "car":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            return image[y1:y2, x1:x2], (x1, y1, x2, y2)

    return image, (0, 0, image.shape[1], image.shape[0])


# ==========================
# CNN LOGGER
# ==========================
def classify_damage(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = classifier(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    print("CNN Prediction:", predicted.item(),
          "Confidence:", round(confidence.item(), 3))

    return predicted.item(), confidence.item()


# ==========================
# SINGLE SIDE PIPELINE
# ==========================
def trust_shield_check(pre_path, post_path):

    img_pre = cv2.imread(pre_path)
    img_post = cv2.imread(post_path)

    if img_pre is None or img_post is None:
        return {"success": False, "error": "IMAGE LOAD FAILED"}

    gray_pre = cv2.cvtColor(img_pre, cv2.COLOR_BGR2GRAY)
    gray_post = cv2.cvtColor(img_post, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(gray_pre, None)
    kp2, des2 = orb.detectAndCompute(gray_post, None)

    if des1 is None or des2 is None:
        return {"success": False, "error": "INSUFFICIENT FEATURES"}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 8:
        return {"success": False, "error": "ALIGNMENT FAILED"}

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    if M is None:
        return {"success": False, "error": "HOMOGRAPHY FAILED"}

    aligned = cv2.warpPerspective(img_post, M,
                                  (img_pre.shape[1], img_pre.shape[0]))

    pre_car, _ = detect_car_region(img_pre)
    post_car, _ = detect_car_region(aligned)

    post_car = cv2.resize(post_car,
                          (pre_car.shape[1], pre_car.shape[0]))

    gray_pre_car = cv2.cvtColor(pre_car, cv2.COLOR_BGR2GRAY)
    gray_post_car = cv2.cvtColor(post_car, cv2.COLOR_BGR2GRAY)

    ssim_score, diff = ssim(gray_pre_car, gray_post_car, full=True)

    diff_map = (1 - diff) * 255
    diff_map = diff_map.astype("uint8")
    diff_map = cv2.GaussianBlur(diff_map, (5, 5), 0)

    _, diff_thresh = cv2.threshold(diff_map, 100, 255,
                                   cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(diff_thresh,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    total_area = 0
    largest_area = 0
    total_intensity = 0
    damage_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 600:
            total_area += area
            largest_area = max(largest_area, area)
            damage_count += 1

            mask = np.zeros_like(diff_map)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_intensity = cv2.mean(diff_map, mask=mask)[0]
            total_intensity += mean_intensity * area

    total_pixels = gray_pre_car.shape[0] * gray_pre_car.shape[1]
    damage_ratio = total_area / total_pixels
    largest_ratio = largest_area / total_pixels

    avg_intensity = (total_intensity / total_area) if total_area > 0 else 0
    normalized_intensity = avg_intensity / 255

    severity_score = (
        0.5 * damage_ratio +
        0.3 * largest_ratio +
        0.2 * normalized_intensity
    )

    if ssim_score > 0.96:
        status = "CLEAR"
    elif severity_score > 0.12:
        status = "MAJOR DAMAGE"
    elif severity_score > 0.04:
        status = "MODERATE DAMAGE"
    else:
        status = "MINOR DAMAGE"

    return {
        "success": True,
        "status": status,
        "ssim_score": round(ssim_score, 4),
        "severity_score": round(severity_score, 4),
        "damage_ratio": round(damage_ratio, 4),
        "largest_blob_ratio": round(largest_ratio, 4),
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
    valid_pairs = 0

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
        valid_pairs += 1

    if valid_pairs == 0:
        return {"success": False, "error": "NO VALID IMAGE PAIRS"}

    avg_severity = total_severity / valid_pairs
    avg_damage_ratio = total_damage_ratio / valid_pairs
    avg_ssim = total_ssim / valid_pairs

    if avg_ssim > 0.96:
        overall_status = "APPROVED"
    elif avg_severity > 0.12:
        overall_status = "MAJOR DAMAGE"
    elif avg_severity > 0.04:
        overall_status = "MODERATE DAMAGE"
    else:
        overall_status = "MINOR DAMAGE"

    return {
        "success": True,
        "overall_status": overall_status,
        "overall_severity_score": round(avg_severity, 4),
        "overall_damage_ratio": round(avg_damage_ratio, 4),
        "overall_ssim": round(avg_ssim, 4),
        "sides": sides
    }