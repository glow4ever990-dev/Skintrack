# -*- coding: utf-8 -*-
"""
皮肤变化追踪（自用版）
照片放 Google Drive，程序只读，不上传、不外发。
"""

import io
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import skin_metrics as sm
from drive_client import DriveClient

METRICS_FILE = "skin_metrics.csv"
COLS = ["file_id", "name", "date", "region", "metric", "value"]

# 方向：+1 表示数值变大 = 变差；-1 表示数值变大 = 变好
WORSE_IF_UP = {
    "粗糙度": 1,
    "肤色不匀": 1,
    "泛红度": 1,
    "斑点占比": 1,
    "反光度": 1,
}

st.set_page_config(page_title="皮肤变化追踪", page_icon="🪞", layout="wide")


# ---------------- 连接 ----------------
@st.cache_resource(show_spinner=False)
def get_client():
    sa = st.secrets.get("gcp_service_account")
    if sa is None:
        return None
    return DriveClient(dict(sa))


def folder_id():
    return st.secrets.get("drive_folder_id", "")


# ---------------- 数据读写 ----------------
def load_metrics(client, fid):
    file_id = client.find_file(fid, METRICS_FILE)
    if not file_id:
        return pd.DataFrame(columns=COLS), None
    try:
        txt = client.read_text_file(file_id)
        if not txt.strip():
            return pd.DataFrame(columns=COLS), file_id
        df = pd.read_csv(io.StringIO(txt))
        df["date"] = pd.to_datetime(df["date"])
        return df, file_id
    except Exception:
        return pd.DataFrame(columns=COLS), file_id


def save_metrics(client, df, file_id):
    if file_id is None:
        return False
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    client.update_text_file(file_id, buf.getvalue())
    return True


# ---------------- 同步 ----------------
def sync(client, fid, df, csv_id):
    files = client.list_images(fid)
    if not files:
        st.warning("文件夹里没找到图片。检查一下文件夹 ID，以及是否已经共享给服务账号。")
        return df

    done = set(df["file_id"].unique()) if len(df) else set()
    todo = [f for f in files if f["id"] not in done]

    st.info(f"云端共 {len(files)} 张，已分析 {len(done)} 张，本次需要处理 {len(todo)} 张。")
    if not todo:
        return df

    bar = st.progress(0.0)
    status = st.empty()
    rows, failed = [], []

    for i, f in enumerate(todo):
        status.text(f"处理中：{f['name']}  ({i + 1}/{len(todo)})")
        try:
            raw = client.fetch_image_bytes(f)
            res = sm.analyze(raw)
            d = sm.parse_date(f)
            for region, mets in res["regions"].items():
                for metric, val in mets.items():
                    rows.append([f["id"], f["name"], d, region, metric, val])
            del raw, res          # 立刻释放，内存里始终只有一张图
        except Exception as e:
            failed.append(f"{f['name']}：{e}")
        bar.progress((i + 1) / len(todo))

    status.empty()
    bar.empty()

    if rows:
        df = pd.concat([df, pd.DataFrame(rows, columns=COLS)], ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        if save_metrics(client, df, csv_id):
            st.success(f"完成，新增 {len(todo) - len(failed)} 张，结果已写回 Drive。")
        else:
            st.warning("分析完成，但没能写回 Drive（找不到 skin_metrics.csv）。"
                       "本次结果只存在当前会话里。")
    if failed:
        with st.expander(f"{len(failed)} 张处理失败"):
            for m in failed:
                st.text(m)
    return df


# ---------------- 页面 ----------------
st.title("🪞 皮肤变化追踪")

client = get_client()
if client is None or not folder_id():
    st.error("还没配置好。请按 README 把服务账号信息和文件夹 ID 填进 Streamlit Secrets。")
    st.stop()

if "df" not in st.session_state:
    with st.spinner("读取历史记录..."):
        st.session_state.df, st.session_state.csv_id = load_metrics(client, folder_id())

df = st.session_state.df
csv_id = st.session_state.csv_id

with st.sidebar:
    st.header("同步")
    if st.button("扫描 Drive 并分析新照片", type="primary", use_container_width=True):
        st.session_state.df = sync(client, folder_id(), df, csv_id)
        df = st.session_state.df
    st.caption(f"已收录 {df['file_id'].nunique() if len(df) else 0} 张照片")
    if not sm.face_detection_available():
        st.warning("人脸自动定位不可用，改用中心裁剪。"
                   "请尽量让脸在画面正中、大小一致，否则数据不可比。")
    if len(df):
        st.download_button("下载全部数据 (CSV)",
                           df.to_csv(index=False).encode("utf-8-sig"),
                           "skin_metrics.csv", use_container_width=True)

if not len(df):
    st.info("左边点「扫描 Drive 并分析新照片」开始。")
    st.stop()

photos = (df.groupby(["file_id", "name"])["date"].first()
            .reset_index().sort_values("date"))

tab1, tab2, tab3 = st.tabs(["最新 vs 历史", "趋势曲线", "两张找茬"])

# ===== Tab 1：一张新照片 对 全部历史 =====
with tab1:
    st.subheader("最新一张，跟历史所有照片比")

    target = st.selectbox(
        "要评估的照片",
        photos["file_id"].tolist()[::-1],
        format_func=lambda i: (f"{photos.set_index('file_id').loc[i, 'date']:%Y-%m-%d}"
                               f" — {photos.set_index('file_id').loc[i, 'name']}"),
    )
    t_date = photos.set_index("file_id").loc[target, "date"]

    hist = df[df["date"] < t_date]
    cur = df[df["file_id"] == target]

    if len(hist) == 0:
        st.info("这是最早的一张，前面没有历史可比。")
    else:
        n_hist = hist["file_id"].nunique()
        st.caption(f"对照组：这张之前的 {n_hist} 张照片")

        base = hist.groupby(["region", "metric"])["value"].agg(["mean", "std"]).reset_index()
        merged = cur.merge(base, on=["region", "metric"], how="left")
        merged["历史均值"] = merged["mean"].round(2)
        merged["本次"] = merged["value"].round(2)
        merged["变化"] = (merged["value"] - merged["mean"]).round(2)
        merged["z"] = ((merged["value"] - merged["mean"]) /
                       merged["std"].replace(0, np.nan)).round(2)

        def verdict(r):
            if pd.isna(r["z"]):
                return "数据不足"
            d = r["z"] * WORSE_IF_UP.get(r["metric"], 1)
            if d <= -1.0:
                return "🟢 明显改善"
            if d <= -0.4:
                return "🟢 轻微改善"
            if d < 0.4:
                return "⚪ 基本持平"
            if d < 1.0:
                return "🔴 轻微恶化"
            return "🔴 明显恶化"

        merged["判定"] = merged.apply(verdict, axis=1)

        c1, c2, c3 = st.columns(3)
        c1.metric("改善项", int(merged["判定"].str.contains("改善").sum()))
        c2.metric("持平项", int(merged["判定"].str.contains("持平").sum()))
        c3.metric("恶化项", int(merged["判定"].str.contains("恶化").sum()))

        show = merged[["region", "metric", "历史均值", "本次", "变化", "z", "判定"]]
        show = show.rename(columns={"region": "区域", "metric": "指标"})
        show = show.sort_values("z", key=lambda s: s.abs(), ascending=False)
        st.dataframe(show, use_container_width=True, hide_index=True)

        st.caption("z = 偏离历史平均多少个标准差。|z| 越大，说明这次跟你以往的常态差得越远。"
                   "这是纯统计比较，不构成任何医学判断。")

# ===== Tab 2：趋势 =====
with tab2:
    st.subheader("单项指标随时间变化")
    c1, c2 = st.columns(2)
    region = c1.selectbox("区域", sm.REGIONS)
    metric = c2.selectbox("指标", sm.METRICS)

    sub = df[(df["region"] == region) & (df["metric"] == metric)]
    sub = sub.sort_values("date")
    if len(sub) < 2:
        st.info("这个组合的数据点不够，至少需要 2 张照片。")
    else:
        series = sub.set_index("date")["value"]
        st.line_chart(series)
        c1, c2, c3 = st.columns(3)
        c1.metric("最早", f"{series.iloc[0]:.2f}")
        c2.metric("最新", f"{series.iloc[-1]:.2f}",
                  delta=f"{series.iloc[-1] - series.iloc[0]:+.2f}",
                  delta_color="inverse")
        c3.metric("历史均值", f"{series.mean():.2f}")

    st.divider()
    st.subheader("全区域总览（每格 = 相对历史均值的 z 值）")
    piv = df[df["metric"] == metric].copy()
    stats = piv.groupby("region")["value"].agg(["mean", "std"])
    piv = piv.merge(stats, on="region")
    piv["z"] = (piv["value"] - piv["mean"]) / piv["std"].replace(0, np.nan)
    heat = piv.pivot_table(index="region", columns=piv["date"].dt.strftime("%m-%d"),
                           values="z", aggfunc="mean")
    st.dataframe(heat.style.background_gradient(cmap="RdYlGn_r", axis=None)
                 .format("{:.1f}"), use_container_width=True)

# ===== Tab 3：找茬 =====
with tab3:
    st.subheader("挑两张，直接看差异")
    opts = photos["file_id"].tolist()
    fmt = lambda i: (f"{photos.set_index('file_id').loc[i, 'date']:%Y-%m-%d}"
                     f" — {photos.set_index('file_id').loc[i, 'name']}")
    c1, c2 = st.columns(2)
    a = c1.selectbox("前期", opts, index=0, format_func=fmt, key="pa")
    b = c2.selectbox("后期", opts, index=len(opts) - 1, format_func=fmt, key="pb")

    if st.button("开始对比", type="primary"):
        with st.spinner("下载并对比中..."):
            files = client.list_images(folder_id())
            fmap = {f["id"]: f for f in files}
            try:
                ba = client.fetch_image_bytes(fmap[a])
                bb = client.fetch_image_bytes(fmap[b])
                heat, overlay, boxed, side, per = sm.pairwise_diff(ba, bb)

                st.image(sm.bgr_to_rgb(side), caption="左：前期　右：后期（已对齐）",
                         use_container_width=True)
                c1, c2, c3 = st.columns(3)
                c1.image(sm.bgr_to_rgb(heat), caption="差异热力图（红=差异大）",
                         use_container_width=True)
                c2.image(sm.bgr_to_rgb(overlay), caption="热力图叠加",
                         use_container_width=True)
                c3.image(sm.bgr_to_rgb(boxed), caption="差异集中区块",
                         use_container_width=True)

                st.dataframe(
                    pd.DataFrame(sorted(per.items(), key=lambda x: -x[1]),
                                 columns=["区域", "平均差异"]),
                    use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"对比失败：{e}")

st.divider()
st.caption("本工具只做像素与区域的机械统计对比，不评估、不诊断任何皮肤或健康状况。"
           "光线、角度、表情、化妆、相机都会影响数值——固定拍摄条件，数据才有意义。")
