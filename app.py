from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import cv2
import time
import requests
import numpy as np
from trust_shield import trust_shield_check, trust_shield_multi

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

# Ensure folders exist (important for Render)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# One-time result store (cleared after GET)
LAST_RESULT = None


@app.route("/", methods=["GET", "POST"])
def index():
    global LAST_RESULT

    if request.method == "POST":
        pre = request.files["pre"]
        post = request.files["post"]

        pre_path = os.path.join(UPLOAD_FOLDER, pre.filename)
        post_path = os.path.join(UPLOAD_FOLDER, post.filename)

        pre.save(pre_path)
        post.save(post_path)

        response = trust_shield_check(pre_path, post_path)

        if not response["success"]:
            LAST_RESULT = {
                "processed": True,
                "error": response["error"]
            }
            return redirect(url_for("index"))

        ts = str(int(time.time()))
        kp = f"keypoints_{ts}.jpg"
        al = f"aligned_{ts}.jpg"
        hm = f"heatmap_{ts}.jpg"
        bx = f"boxed_{ts}.jpg"

        if "keypoints" in response:
            cv2.imwrite(os.path.join(RESULT_FOLDER, kp), response["keypoints"])
        if "aligned" in response:
            cv2.imwrite(os.path.join(RESULT_FOLDER, al), response["aligned"])
        if "heatmap" in response:
            cv2.imwrite(os.path.join(RESULT_FOLDER, hm), response["heatmap"])

        LAST_RESULT = {
            "processed": True,
            "status": response["status"],
            "score": response["ssim_score"],
            "damage_count": response["damage_count"]
        }

        return redirect(url_for("index"))

    result = LAST_RESULT
    LAST_RESULT = None

    return render_template("index.html", result=result)


@app.route("/analyze", methods=["POST"])
def analyze_multi():

    data = request.get_json()

    required_fields = [
        "before_front", "before_rear", "before_left", "before_right",
        "after_front", "after_rear", "after_left", "after_right"
    ]

    for field in required_fields:
        if field not in data:
            return jsonify({"success": False, "error": f"{field} missing"}), 400

    pre_dict = {}
    post_dict = {}

    timestamp = str(int(time.time()))

    def download_image(url, save_path):
        response = requests.get(url)
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        cv2.imwrite(save_path, img)

    for side in ["front", "rear", "left", "right"]:

        before_url = data[f"before_{side}"]
        after_url = data[f"after_{side}"]

        before_path = os.path.join(
            UPLOAD_FOLDER, f"{side}_before_{timestamp}.jpg"
        )
        after_path = os.path.join(
            UPLOAD_FOLDER, f"{side}_after_{timestamp}.jpg"
        )

        download_image(before_url, before_path)
        download_image(after_url, after_path)

        pre_dict[side] = before_path
        post_dict[side] = after_path

    result = trust_shield_multi(pre_dict, post_dict)

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
