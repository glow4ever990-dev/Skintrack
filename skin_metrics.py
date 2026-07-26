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

_cascades = {}          # 文件名 -> 分类器 / None

def _cascade_dirs():
    import os
    dirs = [os.path.dirname(os.path.abspath(__file__))]   # 仓库里自带的副本优先
    try:
        dirs.append(cv2.data.haarcascades)
    except AttributeError:
        pass
    dirs += [
        os.path.join(os.path.dirname(cv2.__file__), "data"),
        "/usr/share/opencv4/haarcascades",
        "/usr/share/opencv/haarcascades",
        "/usr/local/share/opencv4/haarcascades",
    ]
    return dirs


def _get_cascade(name="haarcascade_frontalface_default.xml"):
    """
    取某个检测器。取不到返回 None，调用方自己降级，流程不中断。
    """
    if name in _cascades:
        return _cascades[name]

    _cascades[name] = None
    if not hasattr(cv2, "CascadeClassifier"):
        return None                      # 这个 opencv 构建里没有 objdetect

    import os
    for d in _cascade_dirs():
        p = os.path.join(d, name)
        if os.path.isfile(p):
            try:
                c = cv2.CascadeClassifier(p)
                if not c.empty():
                    _cascades[name] = c
                    break
            except Exception:
                continue
    return _cascades[name]


def face_detection_available():
    return _get_cascade() is not None


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
    """
    解码图片。先走 OpenCV；HEIC/HEIF 这类 OpenCV 不认的格式走 Pillow。
    """
    # 先走 Pillow：它能读 EXIF 的方向标记，手机竖拍的照片才不会躺着进来。
    # OpenCV 的 imdecode 完全忽略 EXIF，所以不能拿它当第一选择。
    try:
        from PIL import Image, ImageOps
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            pass

        import io as _io
        pil = Image.open(_io.BytesIO(image_bytes))
        pil = ImageOps.exif_transpose(pil)     # 按 EXIF 自动转正
        pil = pil.convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        pass

    # Pillow 读不了才退回 OpenCV
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img

    raise ValueError("无法解码这个格式")


# ---------------- Shot angle ----------------
# The same person is photographed from several angles (front, 45, 90).
# Comparing a front shot against a profile is meaningless, so every photo
# carries an angle tag and only same-tag photos are ever compared.
ANGLES = ["正面", "左45", "右45", "左90", "右90", "未标注"]

_NAME_HINTS = [
    ("左90", ("左90", "zuo90", "l90", "left90")),
    ("右90", ("右90", "you90", "r90", "right90")),
    ("左45", ("左45", "zuo45", "l45", "left45")),
    ("右45", ("右45", "you45", "r45", "right45")),
    ("正面", ("正面", "正脸", "zhengmian", "zhenglian", "front")),
]


def angle_from_name(filename):
    """Read the angle tag out of the file name. Most reliable source:
    the user controls it, and no detector can be fooled."""
    low = str(filename).lower()
    for tag, keys in _NAME_HINTS:
        if any(k in low for k in keys):
            return tag
    return None


def guess_angle(img):
    """Rough fallback when the file name says nothing.
    Only separates front from side - telling 45 from 90 reliably is not
    something these detectors can do, so those come from the name only."""
    h, w = img.shape[:2]
    scale = min(1.0, 800 / max(h, w))
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1 else img
    gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))

    front = _get_cascade()
    if front is not None:
        try:
            if len(front.detectMultiScale(gray, 1.15, 5, minSize=(60, 60))):
                return "正面"
        except Exception:
            pass

    prof = _get_cascade("haarcascade_profileface.xml")
    if prof is not None:
        for flip, tag in ((False, "右90"), (True, "左90")):
            g = cv2.flip(gray, 1) if flip else gray
            try:
                if len(prof.detectMultiScale(g, 1.15, 5, minSize=(60, 60))):
                    return tag
            except Exception:
                continue
    return "未标注"


def resolve_angle(img, filename):
    """File name wins; detector only fills the gap."""
    return angle_from_name(filename) or guess_angle(img)


# ---------------- Skin mask ----------------
def skin_mask(img, seed_box=None):
    """Keep skin, drop towel / hair / clothing / background.

    The reference colour is sampled from the photo itself rather than
    hard-coded, so it adapts to skin tone and lighting. A pale towel sits
    far enough from real skin in Cr/Cb and saturation to fall outside.
    """
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    Cr, Cb = ycc[:, :, 1].astype(np.float32), ycc[:, :, 2].astype(np.float32)
    S, V = hsv[:, :, 1].astype(np.float32), hsv[:, :, 2].astype(np.float32)
    h, w = img.shape[:2]

    # Seed: centre of the frame, where the face almost always is.
    seed = np.zeros((h, w), np.uint8)
    if seed_box:
        x1, y1, x2, y2 = seed_box
        seed[y1:y2, x1:x2] = 1
    else:
        cv2.ellipse(seed, (w // 2, int(h * 0.52)),
                    (int(w * 0.20), int(h * 0.24)), 0, 0, 360, 1, -1)
    sel = (seed > 0) & (S > 30) & (V > 40) & (V < 250)
    if sel.sum() < 200:
        sel = seed > 0
    if sel.sum() < 50:
        return face_mask(max(h, w))[:h, :w]

    cr0, cb0, s0 = np.median(Cr[sel]), np.median(Cb[sel]), np.median(S[sel])

    m = ((np.abs(Cr - cr0) < 14) & (np.abs(Cb - cb0) < 14)
         & (S > max(28.0, s0 * 0.45)) & (V > 35) & (V < 252))
    m = m.astype(np.uint8)

    k = np.ones((7, 7), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    # Keep only the biggest blob - stray hands or neck patches add noise.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == big).astype(np.uint8)

    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(h, w) * 0.012)


# ---------------- 对齐 ----------------
# 标准脸的几何：所有照片都被摆成两眼在这两个固定坐标上。
# 这样近景、远景、脸偏一点、头歪一点，出来都是同一个姿态。
COMPARE_SIZE = 1024      # 对比时按这个尺寸对齐，局部放大才不糊
EYE_Y   = 0.36      # 眼睛所在的高度（占画布比例）
EYE_LX  = 0.33      # 画面左边那只眼的横坐标
EYE_RX  = 0.67      # 画面右边那只眼的横坐标


def _find_face(gray):
    casc = _get_cascade()
    if casc is None:
        return None
    try:
        faces = casc.detectMultiScale(gray, 1.15, 5, minSize=(60, 60))
    except Exception:
        return None
    if not len(faces):
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def _find_eyes(gray, face_rect):
    """
    在人脸框的上半部分找两只眼睛，按左右分开各取一个。
    找不齐两只就返回 None。
    """
    x, y, w, h = face_rect
    # 眼睛只可能在脸的上 2/3，限定范围能少一大堆误检（鼻孔、嘴角很容易被当成眼）
    ty, by = y + int(h * 0.16), y + int(h * 0.62)
    roi = gray[max(0, ty):by, x:x + w]
    if roi.size == 0:
        return None

    found = []
    for name in ("haarcascade_eye_tree_eyeglasses.xml", "haarcascade_eye.xml"):
        casc = _get_cascade(name)
        if casc is None:
            continue
        try:
            eyes = casc.detectMultiScale(roi, 1.12, 6,
                                         minSize=(int(w * 0.10), int(w * 0.10)),
                                         maxSize=(int(w * 0.45), int(w * 0.45)))
        except Exception:
            eyes = []
        if len(eyes) >= 2:
            found = eyes
            break
        if len(eyes) and not len(found):
            found = eyes

    if len(found) < 2:
        return None

    mid = roi.shape[1] / 2.0
    cents = [(ex + ew / 2.0, ey + eh / 2.0, ew * eh) for ex, ey, ew, eh in found]
    left  = [c for c in cents if c[0] <  mid]
    right = [c for c in cents if c[0] >= mid]
    if not left or not right:
        return None

    # 每侧取面积最大的那个（通常就是真眼睛）
    lx, ly, _ = max(left,  key=lambda c: c[2])
    rx, ry, _ = max(right, key=lambda c: c[2])

    # 换回整图坐标
    off_y = max(0, ty)
    return (lx + x, ly + off_y), (rx + x, ry + off_y)


def _eyes_plausible(eyes, face_rect):
    """
    检测器经常把鼻孔、痣、阴影当成眼睛。这类误检的共同特征是
    两点靠得太近——一旦按它算缩放，整张脸会被放大好几倍，
    出来就是一片毛孔特写。这里按人脸框的比例做常识检查，
    不合理就宁可退回粗对齐，也不要一张废图。
    """
    (lx, ly), (rx, ry) = eyes
    x, y, w, h = face_rect
    dx, dy = rx - lx, ry - ly
    dist = float(np.hypot(dx, dy))

    # 1) 眼距应当占脸宽的三到六成
    if not (0.28 * w <= dist <= 0.62 * w):
        return False
    # 2) 两眼不该差太高（脸歪超过 25 度基本是误检）
    if abs(np.degrees(np.arctan2(dy, dx))) > 25:
        return False
    # 3) 眼睛该在脸的上半部
    rel = ((ly + ry) / 2.0 - y) / max(h, 1)
    if not (0.15 <= rel <= 0.58):
        return False
    return True


def align_face(img, size=FACE_SIZE):
    """
    把脸摆正到统一姿态，返回 (脸图, 对齐等级)。

    等级三档，越靠前越准：
      "eyes"   两眼定位成功——旋转/缩放/平移全部校正，不同远近的照片可以直接比
      "face"   只找到脸框——大小位置大致统一，但没校正角度
      "center" 什么都没找到——只能取画面中心，基本不可比
    """
    h, w = img.shape[:2]
    scale = min(1.0, 1100 / max(h, w))
    small = cv2.resize(img, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_AREA) if scale < 1 else img
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)          # 拉对比度，暗光照片也能检出

    face = _find_face(gray)

    # ---- 第一档：靠两眼做相似变换 ----
    if face is not None:
        eyes = _find_eyes(gray, face)
        if eyes is not None and _eyes_plausible(eyes, face):
            (lx, ly), (rx, ry) = eyes
            dx, dy = rx - lx, ry - ly
            dist = float(np.hypot(dx, dy))
            if dist > 1:
                target = (EYE_RX - EYE_LX) * size
                angle = float(np.degrees(np.arctan2(dy, dx)))
                M = cv2.getRotationMatrix2D(((lx + rx) / 2.0, (ly + ry) / 2.0),
                                            angle, target / dist)
                # 把两眼中点搬到画布上的固定位置
                M[0, 2] += (EYE_LX + EYE_RX) / 2.0 * size - (lx + rx) / 2.0
                M[1, 2] += EYE_Y * size - (ly + ry) / 2.0
                out = cv2.warpAffine(small, M, (size, size),
                                     flags=cv2.INTER_AREA,
                                     borderMode=cv2.BORDER_REPLICATE)
                return out, "eyes"

    # ---- 第二档：只有脸框 ----
    if face is not None:
        x, y, fw, fh = face
        cx, cy = x + fw / 2, y + fh / 2
        side = max(fw, fh) * 1.45
        x1, y1 = int(cx - side / 2), int(cy - side / 2 - side * 0.05)
        level = "face"
    else:
        # ---- 第三档：中心裁剪 ----
        side = min(small.shape[:2])
        x1 = (small.shape[1] - side) // 2
        y1 = (small.shape[0] - side) // 2
        level = "center"

    x2, y2 = int(x1 + side), int(y1 + side)
    pad = max(0, -x1, -y1, x2 - small.shape[1], y2 - small.shape[0])
    if pad:
        small = cv2.copyMakeBorder(small, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x1, y1, x2, y2 = x1 + pad, y1 + pad, x2 + pad, y2 + pad

    out = cv2.resize(small[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)
    return out, level


def crop_face(img):
    """老接口，保留兼容：返回 (脸图, 是否检测到脸)。"""
    face, level = align_face(img)
    return face, level != "center"


def face_mask(size=FACE_SIZE):
    """
    椭圆脸罩。背景、头发、衣领的差异远大于皮肤，不挡掉的话
    差异图上全是它们，真正的皮肤变化会被淹没。
    """
    m = np.zeros((size, size), np.float32)
    cv2.ellipse(m, (size // 2, int(size * 0.55)),
                (int(size * 0.34), int(size * 0.45)),
                0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), size * 0.02)


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
    """
    区域坐标是按"两眼固定在 EYE_Y 高度"这个标准姿态标定的。
    对齐等级到了 eyes，这些框才真正落在对应的部位上。
    """
    s = size
    def box(x1, y1, x2, y2):
        return (int(x1 * s), int(y1 * s), int(x2 * s), int(y2 * s))
    return {
        "额头":   box(.28, .07, .72, .25),
        "眼周":   box(.20, .28, .80, .44),
        "鼻部":   box(.42, .40, .58, .67),
        "左脸颊": box(.19, .45, .38, .70),
        "右脸颊": box(.62, .45, .81, .70),
        "嘴周":   box(.34, .70, .66, .84),
        "下颌":   box(.26, .84, .74, .96),
    }


# ---------------- 指标计算 ----------------
def analyze(image_bytes, filename=""):
    """
    返回 {区域: {指标: 数值}}，外加整体信息。
    filename is used to read the angle tag.
    """
    img = decode(image_bytes)
    angle = resolve_angle(img, filename)
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
        "angle": angle,
        "face_image": face,
    }


# ---------------- 两张图的直接对比（找茬模式） ----------------
def _match_illumination(a, b, mask):
    """
    把 b 的亮度分布拉到跟 a 一致。灯光不同造成的整体明暗差
    会盖过真实的皮肤变化，先扣掉这一层。
    只能修整体差异，修不了侧光、阴影方向不同这类问题。
    """
    la = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float32)
    lb = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = mask > 0.5
    if m.sum() < 100:
        return la, lb
    for c in range(3):
        ma, sa = la[:, :, c][m].mean(), la[:, :, c][m].std() + 1e-6
        mb, sb = lb[:, :, c][m].mean(), lb[:, :, c][m].std() + 1e-6
        lb[:, :, c] = (lb[:, :, c] - mb) / sb * sa + ma
    return la, lb


def _refine_align(ref, mov):
    """
    眼睛对齐之后还会剩残差，用 ECC 再磨。分两步：
      先欧氏变换（只转+平移）粗对，再仿射变换细对。
    第二步很重要——两张脸如果竖直比例不一样（比如一张闭眼没对上眼睛、
    只按脸框缩放），单纯平移旋转救不了，会出现"下巴对得上、
    眉毛对到眼睛"这种上下错位。仿射能把这个拉伸差修掉。
    对不上就退回上一步的结果，不会更糟。
    """
    try:
        ga = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gb = cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ga = cv2.GaussianBlur(ga, (0, 0), 2.5)
        gb = cv2.GaussianBlur(gb, (0, 0), 2.5)

        warp = np.eye(2, 3, dtype=np.float32)
        ok = False
        for mode, iters in ((cv2.MOTION_EUCLIDEAN, 60), (cv2.MOTION_AFFINE, 80)):
            try:
                crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-6)
                cv2.findTransformECC(ga, gb, warp, mode, crit, None, 5)
                ok = True
            except Exception:
                break          # 这一级没收敛就用上一级的结果

        if not ok:
            return mov, False
        return cv2.warpAffine(mov, warp, mov.shape[1::-1],
                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE), True
    except Exception:
        return mov, False


def diff_map(bytes_a, bytes_b, size=COMPARE_SIZE):
    """
    算两张照片的差异分布。返回一个 dict，后面画图和排名都用它。
    """
    fa, lvl_a = align_face(decode(bytes_a), size)
    fb, lvl_b = align_face(decode(bytes_b), size)

    fb, refined = _refine_align(fa, fb)

    # Intersect both photos' skin masks: only compare pixels that are skin
    # in BOTH shots, otherwise a towel that moved shows up as a huge change.
    mask = np.minimum(skin_mask(fa), skin_mask(fb))
    mask = np.minimum(mask, face_mask(size) * 0 + 1.0)
    if mask.mean() < 0.04:                 # masking went wrong - fall back
        mask = face_mask(size)
    la, lb = _match_illumination(fa, fb, mask)

    # 差异本身
    d = np.sqrt(np.sum((la - lb) ** 2, axis=2))

    # 边缘抑制：头发丝、睫毛、嘴唇轮廓这些地方，只要差半个像素
    # 差异值就爆表。按梯度强度给这些位置降权，剩下的才是皮肤上的变化。
    g = cv2.cvtColor(fa, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 2.0)
    w_edge = 1.0 / (1.0 + (grad / 18.0) ** 2)

    d = cv2.GaussianBlur(d, (0, 0), 3.0) * w_edge * mask

    return {
        "face_a": fa, "face_b": fb,
        "diff": d, "mask": mask,
        "level_a": lvl_a, "level_b": lvl_b,
        "refined": refined,
        "comparable": lvl_a == "eyes" and lvl_b == "eyes",
    }


def hot_blocks(d, mask, top=3, grid=16, min_gap=2):
    """
    把脸切成小格，找差异最集中的几块。
    比找轮廓稳得多——轮廓法会因为阈值一点点变化就框出完全不同的形状。
    """
    S = d.shape[0]
    cell = S // grid
    score = np.zeros((grid, grid), np.float32)
    for i in range(grid):
        for j in range(grid):
            sub = d[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell]
            sm  = mask[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell]
            score[i, j] = sub.mean() if sm.mean() > 0.6 else 0.0

    picked = []
    s = score.copy()
    for _ in range(top):
        i, j = np.unravel_index(int(s.argmax()), s.shape)
        if s[i, j] <= 0:
            break
        picked.append((i, j, float(s[i, j])))
        # 压掉周围，免得三个框全挤在一起
        s[max(0, i - min_gap):i + min_gap + 1,
          max(0, j - min_gap):j + min_gap + 1] = 0

    boxes = []
    for i, j, v in picked:
        pad = cell          # 上下左右各留一格环境，看得出是脸上哪儿
        x1 = max(0, j * cell - pad); y1 = max(0, i * cell - pad)
        x2 = min(S, (j + 1) * cell + pad); y2 = min(S, (i + 1) * cell + pad)
        boxes.append({"box": (x1, y1, x2, y2), "score": v,
                      "region": _which_region((x1 + x2) // 2, (y1 + y2) // 2, S)})
    return boxes


def _which_region(cx, cy, size=FACE_SIZE):
    for name, (x1, y1, x2, y2) in region_boxes(size).items():
        if x1 <= cx < x2 and y1 <= cy < y2:
            return name
    return "其他"


def comparison_image(bytes_a, bytes_b, top=3):
    """
    出一张合并好的对比图：
      上排  前 | 后 | 热力叠加（差异最大的几块用白框标 1 2 3）
      下排  每个标号处的局部放大，左边是前、右边是后
    图上只写数字，中文说明留给界面——图片里画中文要额外装字体，
    在 Streamlit Cloud 上很容易缺字变成方块。
    """
    r = diff_map(bytes_a, bytes_b)
    fa, fb, d, mask = r["face_a"], r["face_b"], r["diff"], r["mask"]
    S = fa.shape[0]

    hi = np.percentile(d[mask > 0.5], 99.0) + 1e-6
    norm = np.clip(d / hi, 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = (heat * mask[:, :, None]).astype(np.uint8)
    overlay = cv2.addWeighted(fb, 0.6, heat, 0.4, 0)

    blocks = hot_blocks(d, mask, top=top)

    marked = overlay.copy()
    for n, b in enumerate(blocks, 1):
        x1, y1, x2, y2 = b["box"]
        cv2.rectangle(marked, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(marked, str(n), (x1 + 4, max(16, y1 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    top_row = np.hstack([fa, fb, marked])

    # 下排：局部放大
    if blocks:
        # 放大倍数封顶 2 倍。超过就纯粹是把像素拉大，只会更糊，看不出更多东西。
        src_side = max(b["box"][2] - b["box"][0] for b in blocks)
        crop_h = int(min(S // 2, src_side * 2))
        tiles = []
        for n, b in enumerate(blocks, 1):
            x1, y1, x2, y2 = b["box"]
            ca = cv2.resize(fa[y1:y2, x1:x2], (crop_h, crop_h), interpolation=cv2.INTER_LANCZOS4)
            cb = cv2.resize(fb[y1:y2, x1:x2], (crop_h, crop_h), interpolation=cv2.INTER_LANCZOS4)
            b["crop_a"], b["crop_b"] = ca, cb
            pair = np.hstack([ca, np.full((crop_h, 4, 3), 255, np.uint8), cb])
            cv2.putText(pair, str(n), (6, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(pair)
            tiles.append(np.full((crop_h, 10, 3), 255, np.uint8))
        bottom = np.hstack(tiles[:-1])
        pad_w = top_row.shape[1] - bottom.shape[1]
        if pad_w > 0:
            bottom = np.hstack([bottom, np.full((bottom.shape[0], pad_w, 3), 255, np.uint8)])
        elif pad_w < 0:
            bottom = cv2.resize(bottom, (top_row.shape[1],
                                         int(bottom.shape[0] * top_row.shape[1] / bottom.shape[1])))
        gap = np.full((12, top_row.shape[1], 3), 255, np.uint8)
        merged = np.vstack([top_row, gap, bottom])
    else:
        merged = top_row

    per_region = {}
    for name, (x1, y1, x2, y2) in region_boxes(S).items():
        patch = d[y1:y2, x1:x2]
        if patch.size:
            per_region[name] = round(float(patch.mean()), 2)

    r.update({"heat": heat, "overlay": overlay, "marked": marked,
              "side": np.hstack([fa, fb]), "merged": merged,
              "blocks": blocks, "per_region": per_region})
    return r


def pairwise_diff(bytes_a, bytes_b):
    """老接口，保留兼容。"""
    r = comparison_image(bytes_a, bytes_b)
    return r["heat"], r["overlay"], r["marked"], r["side"], r["per_region"]


def to_png_bytes(img_bgr):
    ok, buf = cv2.imencode(".png", img_bgr)
    return buf.tobytes() if ok else b""


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
