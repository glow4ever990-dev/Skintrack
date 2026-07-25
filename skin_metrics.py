# -*- coding: utf-8 -*-
"""
分析核心：把一张脸的照片压成一组数字（指标），存进表里。
之后"跟几十张历史照片对比"就变成了比较数字，不用把几十张图同时读进内存。

注意：这里全部是机械的像素统计，不做任何医学判断。
"""

import io
import re
from datetime import datetime

import cv2
import numpy as np

FACE_SIZE = 512          # 所有脸统一缩放到这个尺寸，等于做了粗略对齐
REGIONS = ["额头", "眼周", "鼻部", "左脸颊", "右脸颊", "嘴周", "下颌"]
METRICS = ["粗糙度", "肤色不匀", "泛红度", "斑点占比", "反光度"]

_cascade = None

def _get_cascade():
    global _cascade
    if _cascade is None:
        try:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        except AttributeError:
            # opencv-python-headless 某些版本没有 cv2.data
            import os
            for d in [
                "/usr/share/opencv4/haarcascades",
                "/usr/share/opencv/haarcascades",
                "/usr/local/share/opencv4/haarcascades",
                os.path.join(os.path.dirname(cv2.__file__), "data"),
            ]:
                p = os.path.join(d, "haarcascade_frontalface_default.xml")
                if os.path.isfile(p):
                    path = p
                    break
            else:
                path = "haarcascade_frontalface_default.xml"
        _cascade = cv2.CascadeClassifier(path)
    return _cascade


# ---------------- 日期解析 ----------------
def parse_date(drive_file):
    """优先 EXIF 拍摄时间 → 文件名里的日期 → Drive 创建时间"""
    meta = drive_file.get("imageMediaMetadata") or {}
    t = meta.get("time")
    if t:
        try:
            return datetime.strptime(t, "%Y:%m:%d %H:%M:%S")
        except ValueError:
            pass

    name = drive_file.get("name", "")
    m = re.search(r"(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})", name)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    ct = drive_file.get("createdTime")
    if ct:
        try:
            return datetime.strptime(ct[:19], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            pass
    return datetime.now()


# ---------------- 解码 + 裁脸 + 归一化 ----------------
def decode(image_bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("图片解码失败（HEIC 格式请先转成 JPG）")
    return img


def crop_face(img):
    """
    裁出人脸并统一到 FACE_SIZE。
    统一尺寸 = 不同日期的照片大致对齐，同一区域落在同一位置。
    返回 (脸图, 是否检测到人脸)
    """
    h, w = img.shape[:2]
    scale = min(1.0, 900 / max(h, w))
    small = cv2.resize(img, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_AREA) if scale < 1 else img

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = _get_cascade().detectMultiScale(gray, 1.15, 5, minSize=(60, 60))

    if len(faces):
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        # 稍微放大范围，把额头和下巴包进来
        cx, cy = x + fw / 2, y + fh / 2
        side = max(fw, fh) * 1.45
        x1, y1 = int(cx - side / 2), int(cy - side / 2 - side * 0.05)
        x2, y2 = int(x1 + side), int(y1 + side)
        detected = True
    else:
        # 没检测到脸就取中间的正方形
        side = min(small.shape[:2])
        x1 = (small.shape[1] - side) // 2
        y1 = (small.shape[0] - side) // 2
        x2, y2 = x1 + side, y1 + side
        detected = False

    pad = max(0, -x1, -y1, x2 - small.shape[1], y2 - small.shape[0])
    if pad:
        small = cv2.copyMakeBorder(small, pad, pad, pad, pad,
                                   cv2.BORDER_REPLICATE)
        x1, y1, x2, y2 = x1 + pad, y1 + pad, x2 + pad, y2 + pad

    face = small[y1:y2, x1:x2]
    face = cv2.resize(face, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_AREA)
    return face, detected


def normalize_light(face):
    """
    把亮度拉到统一基准，让不同灯光下拍的照片可比。
    只动 L 通道（亮度），保留色彩关系。
    """
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    lab[:, :, 0] = np.clip((L - L.mean()) / (L.std() + 1e-6) * 45 + 150, 0, 255)
    return lab


# ---------------- 区域划分 ----------------
def region_boxes(size=FACE_SIZE):
    s = size
    return {
        "额头":   (int(.20 * s), int(.10 * s), int(.80 * s), int(.28 * s)),
        "眼周":   (int(.14 * s), int(.30 * s), int(.86 * s), int(.46 * s)),
        "鼻部":   (int(.40 * s), int(.42 * s), int(.60 * s), int(.66 * s)),
        "左脸颊": (int(.13 * s), int(.46 * s), int(.36 * s), int(.72 * s)),
        "右脸颊": (int(.64 * s), int(.46 * s), int(.87 * s), int(.72 * s)),
        "嘴周":   (int(.33 * s), int(.66 * s), int(.67 * s), int(.82 * s)),
        "下颌":   (int(.22 * s), int(.80 * s), int(.78 * s), int(.94 * s)),
    }


# ---------------- 指标计算 ----------------
def analyze(image_bytes):
    """
    返回 {区域: {指标: 数值}}，外加整体信息。
    """
    img = decode(image_bytes)
    face, detected = crop_face(img)

    # 轻度降噪：手机传感器噪点会把"粗糙度"和"斑点数"顶得很高，
    # 双边滤波能压掉噪点但保留真实的纹理边缘。
    face_dn = cv2.bilateralFilter(face, 5, 25, 5)
    lab = normalize_light(face_dn)

    L = lab[:, :, 0]
    A = lab[:, :, 1]          # 越大越偏红
    a_mean_face = float(A.mean())

    # 带通滤波 = 只保留"毛孔/细纹"这个尺度的纹理。
    # 不用简单高通，是因为高通会把相机噪点也算进去，数值会失真。
    L_fine = cv2.GaussianBlur(L, (0, 0), 1.4)   # 砍掉像素级噪点
    L_base = cv2.GaussianBlur(L, (0, 0), 5.0)   # 砍掉大面积明暗
    highpass = L_fine - L_base

    # 估计这张照片的噪点水平（用中位绝对偏差，比标准差抗干扰）
    resid_all = L - L_fine
    noise_sigma = float(np.median(np.abs(resid_all - np.median(resid_all))) * 1.4826)

    out = {}
    for name, (x1, y1, x2, y2) in region_boxes().items():
        l = L[y1:y2, x1:x2]
        a = A[y1:y2, x1:x2]
        hp = highpass[y1:y2, x1:x2]
        if l.size == 0:
            continue

        # 斑点：比周围明显暗一截的小块。
        # 阈值跟着这张照片自己的噪点水平走，否则暗光照片会被判出一堆假斑。
        lf = L_fine[y1:y2, x1:x2]
        blur = cv2.GaussianBlur(lf, (0, 0), 9)
        resid = blur - lf
        thr = max(4.0, 3.0 * noise_sigma)
        dark = (resid > thr).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        spot_area = float(dark.mean() * 100)

        out[name] = {
            "粗糙度":    round(float(hp.std()), 3),
            "肤色不匀":  round(float(cv2.GaussianBlur(l, (0, 0), 5).std()), 3),
            "泛红度":    round(float(a.mean() - a_mean_face), 3),
            "斑点占比":  round(spot_area, 3),
            "反光度":    round(float((l > np.percentile(L, 97)).mean() * 100), 3),
        }

    return {
        "regions": out,
        "face_detected": detected,
        "face_image": face,
    }


# ---------------- 两张图的直接对比（找茬模式） ----------------
def pairwise_diff(bytes_a, bytes_b):
    """返回 (热力图, 叠加图, 并排图, 分区差异 dict)"""
    fa, _ = crop_face(decode(bytes_a))
    fb, _ = crop_face(decode(bytes_b))

    la = normalize_light(fa)
    lb = normalize_light(fb)

    diff = np.sqrt(np.sum((la - lb) ** 2, axis=2))
    diff = cv2.GaussianBlur(diff, (5, 5), 0)

    hi = np.percentile(diff, 99) + 1e-6
    norm = np.clip(diff / hi, 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(fb, 0.55, heat, 0.45, 0)

    # 差异最集中的区块画红框
    mask = (norm > 0.55).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxed = fb.copy()
    for c in cnts:
        if cv2.contourArea(c) < 150:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 0, 255), 2)

    per_region = {}
    for name, (x1, y1, x2, y2) in region_boxes().items():
        patch = diff[y1:y2, x1:x2]
        if patch.size:
            per_region[name] = round(float(patch.mean()), 2)

    side = np.hstack([fa, fb])
    return heat, overlay, boxed, side, per_region


def to_png_bytes(img_bgr):
    ok, buf = cv2.imencode(".png", img_bgr)
    return buf.tobytes() if ok else b""


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
