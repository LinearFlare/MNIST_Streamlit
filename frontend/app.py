import streamlit as st
import requests
import numpy as np
import pandas as pd
from PIL import Image
import io
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="✍️",
    layout="wide",
)

# ==========================================
# 1. 在侧边栏提供稳定的一键主题切换
# ==========================================
with st.sidebar:
    st.markdown("### 🎨 Theme Selector")
    theme_choice = st.selectbox(
        "Choose your style",
        options=["Catppuccin Mocha (Dark)", "Catppuccin Latte (Light)"],
        index=0  # 默认深色，如果需要默认浅色改成 1
    )
    st.markdown("---")

is_light = "Latte" in theme_choice

# 根据选择，硬编码锁定绝对配色，彻底摆脱 Streamlit 默认干扰
if is_light:
    # 🌿 Catppuccin Latte
    bg_color = "#eff1f5"          # base
    surface_color = "#e6e9ef"     # surface0
    overlay_color = "#ccd0da"     # overlay0
    text_color = "#4c4f69"        # text
    subtext_color = "#6c6f85"     # subtext0
    accent_color = "#7287fd"      # lavender
    accent_hover = "#1e66f5"      # blue
    btn_text = "#ffffff"
    ok_color = "#40a02b"          # green
    ok_bg = "rgba(64, 160, 43, 0.1)"
    err_color = "#d20f39"         # red
    err_bg = "rgba(210, 15, 57, 0.1)"
    canvas_bg = "#ffffff"
    canvas_stroke = "#000000"
else:
    # 🌌 Catppuccin Mocha
    bg_color = "#1e1e2e"          # base
    surface_color = "#313244"     # surface0
    overlay_color = "#45475a"     # overlay0
    text_color = "#cdd6f4"        # text
    subtext_color = "#a6adc8"     # subtext0
    accent_color = "#a6e3a1"      # green
    accent_hover = "#94e2d5"      # teal
    btn_text = "#111111"
    ok_color = "#a6e3a1"          # green
    ok_bg = "rgba(166, 227, 161, 0.15)"
    err_color = "#f38ba8"         # red
    err_bg = "rgba(243, 139, 168, 0.15)"
    canvas_bg = "#000000"
    canvas_stroke = "#ffffff"

# ==========================================
# 2. 强力注入 CSS 样式
# ==========================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

/* 全局主背景与文字颜色全面覆盖 */
html, body, .stApp, div[data-testid="stAppViewContainer"], div[data-testid="stMainBlockContainer"] {{
    font-family: 'Syne', sans-serif;
    background-color: {bg_color} !important;
    color: {text_color} !important;
}}

/* 针对所有标题与文本的强制着色 */
h1, h2, h3, h4, p, span, label, div {{
    font-family: 'Syne', sans-serif;
    color: {text_color} !important;
}}

/* 侧边栏全面覆盖 */
section[data-testid="stSidebar"], div[data-testid="stSidebarUserContent"], section[data-testid="stSidebar"] div {{
    background-color: {surface_color} !important;
    color: {text_color} !important;
}}
section[data-testid="stSidebar"] {{
    border-right: 1px solid {overlay_color} !important;
}}

/* 侧边栏内部的输入框与下拉框 */
div[data-testid="stTextInput"] input, div[data-baseweb="select"] div {{
    background-color: {bg_color} !important;
    color: {text_color} !important;
    border-color: {overlay_color} !important;
}}

/* 核心数字结果 */
.digit-result {{
    font-family: 'Space Mono', monospace;
    font-size: 112px;
    font-weight: 700;
    color: {accent_color} !important;
    text-align: center;
    line-height: 1;
    text-shadow: {"0 0 35px " + accent_color if not is_light else "none"};
    padding: 10px 0;
}}

.confidence-text {{
    font-family: 'Space Mono', monospace;
    font-size: 18px;
    color: {subtext_color} !important;
    text-align: center;
    margin-bottom: 25px;
}}

/* API 状态指示器 */
.api-status-ok {{
    background: {ok_bg};
    border: 1px solid {ok_color};
    border-radius: 8px;
    padding: 8px 16px;
    color: {ok_color} !important;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    display: inline-block;
}}

.api-status-err {{
    background: {err_bg};
    border: 1px solid {err_color};
    border-radius: 8px;
    padding: 8px 16px;
    color: {err_color} !important;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    display: inline-block;
}}

/* 预测大按钮 */
.stButton > button {{
    background: {accent_color} !important;
    color: {btn_text} !important;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    border: none !important;
    border-radius: 8px;
    padding: 12px 28px;
    font-size: 15px;
    width: 100%;
    transition: all 0.2s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}}

.stButton > button:hover {{
    background: {accent_hover} !important;
    transform: translateY(-1px);
}}

/* 修复 Canvas 组件周围出现的突兀灰色/白色色块 */
div.stBokeh {{{{
    background-color: {canvas_bg} !important;
}}}}
div[class*="st-canvas"] {{
    border: 2px solid {overlay_color} !important;
    border-radius: 12px !important;
    background-color: {canvas_bg} !important;
    padding: 10px;
    display: flex;
    justify-content: center;
}}

/* 指标 */
div[data-testid="stMetricValue"] {{
    font-family: 'Space Mono', monospace;
    color: {accent_color} !important;
}}

/* 空状态占位框 */
.placeholder-box {{
    border: 2px dashed {overlay_color};
    border-radius: 16px;
    padding: 60px 20px;
    text-align: center;
    color: {subtext_color} !important;
    background: {surface_color};
    font-family: 'Space Mono', monospace;
    font-size: 14px;
    margin-top: 10px;
}}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. 业务逻辑主体
# ==========================================
API_URL = st.sidebar.text_input(
    "Backend URL",
    value="http://localhost:8000"
)

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        info = r.json()
        if info.get("model_ready"):
            st.markdown('<div class="api-status-ok">● API online · model ready</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="api-status-err">⚠ API online · no weights</div>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<div class="api-status-err">✕ API offline</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
**How to use**
1. Draw a digit (0–9)
2. Click Predict
3. See the result
""")

st.markdown("# ✍️ MNIST Digit Recognizer")
st.markdown("Draw a handwritten digit and let the CNN classify it.")
st.markdown("---")

col_input, col_result = st.columns([1, 1], gap="large")
image_bytes = None

with col_input:
    st.markdown("### Draw your digit")

    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=24,             # 稍微加粗画笔，提高手写识别率
        stroke_color=canvas_stroke,
        background_color=canvas_bg,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=f"canvas_{theme_choice}", # 切换主题时强制重置画布底色
    )

    if canvas_result.image_data is not None:
        arr = canvas_result.image_data.astype(np.uint8)
        if arr.max() > 0:
            if is_light:
                # 浅色模式下反色，确保送进后端的永远是标准的黑底白字模型所需格式
                rgb_arr = arr[:, :, :3]
                arr_inverted = 255 - rgb_arr
                img = Image.fromarray(arr_inverted, "RGB")
            else:
                img = Image.fromarray(arr, "RGBA").convert("RGB")

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_bytes = buf.getvalue()

    predict_btn = st.button("🔍 Predict", use_container_width=True)

with col_result:
    st.markdown("### Prediction")

    if predict_btn:
        if image_bytes is None:
            st.warning("Draw a digit first.")
        else:
            try:
                with st.spinner("Running inference..."):
                    resp = requests.post(
                        f"{API_URL}/predict",
                        files={"file": ("digit.png", image_bytes, "image/png")},
                        timeout=10,
                    )

                if resp.status_code == 200:
                    data = resp.json()
                    pred = data["prediction"]
                    conf = data["confidence"]
                    probs = data["probabilities"]

                    st.markdown(f'<div class="digit-result">{pred}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="confidence-text">confidence: {conf:.1f}%</div>', unsafe_allow_html=True)

                    st.markdown("#### Probability Distribution")
                    prob_values = [probs[str(i)] for i in range(10)]
                    df = pd.DataFrame(
                        {"Probability (%)": prob_values},
                        index=[str(i) for i in range(10)]
                    )
                    st.bar_chart(df)

                    st.markdown("#### Top 3 Predictions")
                    top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    c1, c2, c3 = st.columns(3)

                    for col, (digit, prob) in zip([c1, c2, c3], top3):
                        col.metric(f"Digit {digit}", f"{prob:.1f}%")
                else:
                    st.error(f"API Error {resp.status_code}: {resp.text}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend API.")
            except Exception as e:
                st.error(str(e))
    else:
        st.markdown("""
<div class="placeholder-box">
Draw a digit<br>
then click Predict
</div>
""", unsafe_allow_html=True)