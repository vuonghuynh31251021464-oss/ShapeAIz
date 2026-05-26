import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time
from streamlit_drawable_canvas import st_canvas
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="ShapeAI",
    page_icon="🔷",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=Be+Vietnam+Pro:wght@300;400;500;600&family=Caveat:wght@600&display=swap');
html, body, [class*="css"] { font-family: 'Be Vietnam Pro', sans-serif; }
.stApp {
    background: #f7f3ee;
    background-image:
        radial-gradient(circle at 20% 20%, rgba(14,90,182,0.06) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(0,200,220,0.05) 0%, transparent 50%);
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.hero-wrapper { display:flex; flex-direction:column; align-items:center; padding:2.5rem 0 1rem; gap:0; }
.logo-svg { width:90px; height:90px; }
.app-title { font-family:'Playfair Display',serif; font-size:3rem; font-weight:700; color:#0d1b3e; letter-spacing:-1px; line-height:1; margin:0.4rem 0 0; }
.app-title span { color:#1a8cff; font-style:italic; }
.app-tagline { font-family:'Caveat',cursive; font-size:1.25rem; color:#5a6a8a; margin-top:0.3rem; letter-spacing:0.5px; }
.brush-divider { width:100%; height:3px; background:linear-gradient(90deg,transparent,#1a8cff 30%,#00c8dc 70%,transparent); border-radius:2px; margin:1.5rem 0; opacity:0.5; }
.section-label { font-family:'Playfair Display',serif; font-size:1.15rem; font-weight:700; color:#0d1b3e; margin-bottom:0.4rem; display:flex; align-items:center; gap:0.5rem; }
.section-hint { font-size:0.82rem; color:#8a94a6; font-weight:300; margin-bottom:0.8rem; font-style:italic; }
.canvas-frame { border:2px solid #d4dae8; border-radius:16px; overflow:hidden; box-shadow:0 8px 32px rgba(13,27,62,0.08),0 2px 8px rgba(13,27,62,0.04); background:#ffffff; position:relative; }
.canvas-frame::before { content:''; position:absolute; inset:0; border-radius:14px; background:linear-gradient(135deg,rgba(26,140,255,0.03),transparent); pointer-events:none; z-index:1; }
.stButton > button { width:100%; background:linear-gradient(135deg,#0d1b3e 0%,#1a4480 60%,#1a8cff 100%); color:#ffffff; font-family:'Be Vietnam Pro',sans-serif; font-weight:600; font-size:1.05rem; letter-spacing:0.5px; border:none; border-radius:12px; padding:0.75rem 2rem; cursor:pointer; transition:all 0.3s ease; box-shadow:0 4px 20px rgba(26,140,255,0.3); margin-top:0.5rem; }
.stButton > button:hover { transform:translateY(-2px); box-shadow:0 8px 30px rgba(26,140,255,0.4); }
.stButton > button:active { transform:translateY(0); }
.result-card { background:linear-gradient(135deg,#ffffff 0%,#f0f6ff 100%); border:1.5px solid #c8d8f0; border-radius:16px; padding:1.5rem 2rem; text-align:center; box-shadow:0 4px 24px rgba(13,27,62,0.08); margin-top:1rem; animation:slideUp 0.4s ease; }
@keyframes slideUp { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
.result-shape-name { font-family:'Playfair Display',serif; font-size:2.2rem; font-weight:700; color:#0d1b3e; letter-spacing:-0.5px; }
.result-shape-name span { color:#1a8cff; font-style:italic; }
.result-confidence { font-family:'Be Vietnam Pro',sans-serif; font-size:0.9rem; color:#5a6a8a; font-weight:400; margin-top:0.25rem; }
.result-emoji { font-size:2.8rem; line-height:1; margin-bottom:0.3rem; }
.confidence-bar-wrap { margin-top:0.8rem; background:#e4ecfa; border-radius:8px; height:8px; overflow:hidden; }
.confidence-bar-fill { height:100%; border-radius:8px; background:linear-gradient(90deg,#1a8cff,#00c8dc); transition:width 0.8s ease; }
.tips-card { background:#fff8f0; border:1.5px solid #f0dcc0; border-radius:12px; padding:1rem 1.25rem; margin-top:1.2rem; font-size:0.83rem; color:#7a5a3a; line-height:1.7; }
.tips-card b { font-family:'Playfair Display',serif; color:#5a3a1a; font-size:0.9rem; }
.stSpinner > div { border-color:#1a8cff transparent transparent transparent !important; }
.stSelectbox label, .stSlider label { font-family:'Be Vietnam Pro',sans-serif !important; font-size:0.83rem !important; color:#5a6a8a !important; font-weight:500 !important; }
.class-pills { display:flex; flex-wrap:wrap; gap:0.4rem; margin-top:0.5rem; }
.class-pill { background:#e8f0fe; color:#1a4480; border-radius:20px; padding:0.2rem 0.75rem; font-size:0.78rem; font-weight:500; font-family:'Be Vietnam Pro',sans-serif; border:1px solid #c8d8f0; }
.prob-row { display:flex; align-items:center; gap:0.6rem; margin-bottom:0.5rem; }
.prob-name { min-width:110px; font-size:0.85rem; color:#2a3a5a; }
.prob-bar-wrap { flex:1; background:#e4ecfa; border-radius:6px; height:10px; overflow:hidden; }
.prob-bar-fill { height:100%; border-radius:6px; background:linear-gradient(90deg,#1a8cff,#00c8dc); }
.prob-pct { width:38px; text-align:right; color:#5a6a8a; font-size:0.78rem; }
</style>
""", unsafe_allow_html=True)

LOGO_SVG = """
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" class="logo-svg">
  <defs><linearGradient id="g1" x1="0%" y1="0%" x2="100%" y2="100%">
    <stop offset="0%" style="stop-color:#00c8dc;stop-opacity:1"/>
    <stop offset="100%" style="stop-color:#0d1b3e;stop-opacity:1"/>
  </linearGradient></defs>
  <polygon points="100,18 172,80 100,172 28,80" fill="none" stroke="url(#g1)" stroke-width="8" stroke-linejoin="round"/>
  <polygon points="100,30 160,90 40,90" fill="none" stroke="url(#g1)" stroke-width="7" stroke-linejoin="round"/>
  <circle cx="100" cy="105" r="28" fill="none" stroke="url(#g1)" stroke-width="7"/>
  <circle cx="100" cy="18" r="6" fill="#00c8dc"/>
  <circle cx="172" cy="80" r="6" fill="#4a90d9"/>
  <circle cx="100" cy="172" r="6" fill="#0d1b3e"/>
  <circle cx="28" cy="80" r="6" fill="#1a8cff"/>
</svg>"""

st.markdown(f"""
<div class="hero-wrapper">
  {LOGO_SVG}
  <div class="app-title">Shape<span>AI</span></div>
  <div class="app-tagline">✦ Vẽ một hình — AI sẽ đoán ngay ✦</div>
</div>
<div class="brush-divider"></div>
""", unsafe_allow_html=True)

CLASS_NAMES = ['Hình tròn','Hình vuông','Hình tam giác','Hình chữ nhật','Hình elip','Ngôi sao']
CLASS_ICONS = ['⭕','🟦','🔺','▬','🫧','⭐']

def extract_features(gray):
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bw = cv2.dilate(bw, k3, iterations=2)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k5)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 80:
        return None
    perimeter = cv2.arcLength(cnt, True)
    M = cv2.moments(cnt)
    hu = cv2.HuMoments(M).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = w / (h + 1e-5)
    sq_diff = abs(aspect - 1.0)
    extent = area / ((w * h) + 1e-5)
    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-5)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    hull_perim = cv2.arcLength(hull, True)
    solidity = area / (hull_area + 1e-5)
    convexity = hull_perim / (perimeter + 1e-5)
    n_verts = []
    for eps_factor in [0.01, 0.02, 0.04, 0.06]:
        poly = cv2.approxPolyDP(cnt, eps_factor * perimeter, True)
        n_verts.append(len(poly))
    n_verts = [min(v, 20) for v in n_verts]
    if len(cnt) >= 5:
        (_, _), (ea, eb), _ = cv2.fitEllipse(cnt)
        ellipse_ratio = min(ea, eb) / (max(ea, eb) + 1e-5)
    else:
        ellipse_ratio = 1.0
    concavity_ratio = 1.0 - convexity
    resized = cv2.resize(gray, (64, 64))
    gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hog_hist, _ = np.histogram(ang.flatten(), bins=16, range=(0, 360), weights=mag.flatten())
    hog_hist = hog_hist / (hog_hist.sum() + 1e-5)
    if M['m00'] > 0:
        cx_m = M['m10'] / M['m00']
        cy_m = M['m01'] / M['m00']
    else:
        cx_m, cy_m = w / 2, h / 2
    angles_rad = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    radial = []
    for a in angles_rad:
        best = 0.0
        for pt in cnt[:, 0]:
            dx = pt[0] - cx_m
            dy = pt[1] - cy_m
            pt_angle = np.arctan2(dy, dx)
            if abs(pt_angle - a) < 0.2 or abs(pt_angle - a + 2*np.pi) < 0.2 or abs(pt_angle - a - 2*np.pi) < 0.2:
                dist = np.hypot(dx, dy)
                best = max(best, dist)
        radial.append(best)
    radial = np.array(radial)
    max_r = radial.max() + 1e-5
    radial_norm = radial / max_r
    radial_std = float(radial_norm.std())
    feat = np.concatenate([
        hu_log,
        [aspect, sq_diff, extent, circularity, solidity, convexity, concavity_ratio, ellipse_ratio, radial_std],
        n_verts,
        hog_hist,
        radial_norm,
    ])
    return feat

def draw_shape_stroke(img, label, cx, cy, sz=64):
    stroke_w = np.random.randint(2, 6)
    col = 0
    if label == 0:
        r = np.random.randint(16, 26)
        cv2.circle(img, (cx, cy), r, col, stroke_w)
    elif label == 1:
        s = np.random.randint(18, 32)
        cv2.rectangle(img, (cx-s, cy-s), (cx+s, cy+s), col, stroke_w)
    elif label == 2:
        jitter = lambda: np.random.randint(-4, 5)
        pts = np.array([
            [cx+jitter(), cy-20+jitter()],
            [cx-20+jitter(), cy+18+jitter()],
            [cx+20+jitter(), cy+18+jitter()]
        ], np.int32)
        cv2.polylines(img, [pts], True, col, stroke_w)
    elif label == 3:
        hw = np.random.randint(18, 28)
        hh = np.random.randint(9, 15)
        cv2.rectangle(img, (cx-hw, cy-hh), (cx+hw, cy+hh), col, stroke_w)
    elif label == 4:
        a = np.random.randint(18, 26)
        b = np.random.randint(8, 14)
        angle = np.random.randint(0, 90)
        cv2.ellipse(img, (cx, cy), (a, b), angle, 0, 360, col, stroke_w)
    elif label == 5:
        pts5 = []
        rot = np.random.uniform(0, 2*np.pi/5)
        outer_r = np.random.randint(18, 26)
        inner_r = int(outer_r * 0.42)
        for i in range(5):
            a = i * 2*np.pi/5 - np.pi/2 + rot
            pts5.append([int(cx + outer_r*np.cos(a)), int(cy + outer_r*np.sin(a))])
            a2 = (i+0.5)*2*np.pi/5 - np.pi/2 + rot
            pts5.append([int(cx + inner_r*np.cos(a2)), int(cy + inner_r*np.sin(a2))])
        cv2.polylines(img, [np.array(pts5, np.int32)], True, col, stroke_w)
        cv2.fillPoly(img, [np.array(pts5, np.int32)], col)

def augment(img):
    sz = img.shape[0]
    angle = np.random.uniform(-30, 30)
    M_rot = cv2.getRotationMatrix2D((sz/2, sz/2), angle, 1.0)
    img = cv2.warpAffine(img, M_rot, (sz, sz), borderValue=255)
    scale = np.random.uniform(0.75, 1.15)
    M_sc = cv2.getRotationMatrix2D((sz/2, sz/2), 0, scale)
    img = cv2.warpAffine(img, M_sc, (sz, sz), borderValue=255)
    tx, ty = np.random.randint(-8, 9), np.random.randint(-8, 9)
    M_tr = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M_tr, (sz, sz), borderValue=255)
    return img

def make_dataset(n=8000, sz=64):
    X, y = [], []
    for _ in range(n):
        img = np.full((sz, sz), 255, dtype=np.uint8)
        label = np.random.randint(0, 6)
        cx = sz//2 + np.random.randint(-6, 7)
        cy = sz//2 + np.random.randint(-6, 7)
        draw_shape_stroke(img, label, cx, cy, sz)
        img = augment(img)
        feat = extract_features(img)
        if feat is not None:
            X.append(feat)
            y.append(label)
    return np.array(X), np.array(y)

@st.cache_resource(show_spinner=False)
def load_model():
    X, y = make_dataset(8000)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    gb = GradientBoostingClassifier(n_estimators=400, learning_rate=0.08, max_depth=5, subsample=0.8, random_state=42)
    rf = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1)
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    gb.fit(X_s, y)
    rf.fit(X_s, y)
    svm.fit(X_s, y)
    ensemble = VotingClassifier(
        estimators=[('gb', gb), ('rf', rf), ('svm', svm)],
        voting='soft'
    )
    ensemble.fit(X_s, y)
    return ensemble, scaler

if "model_ready" not in st.session_state:
    st.session_state.model_ready = False

if not st.session_state.model_ready:
    with st.spinner("🎨 Đang khởi động ShapeAI — huấn luyện mô hình lần đầu (~20 giây)…"):
        model, scaler = load_model()
    st.session_state.model_ready = True
    st.rerun()
else:
    model, scaler = load_model()

st.markdown('<div class="section-label">🖊️ Bảng vẽ của bạn</div>', unsafe_allow_html=True)
st.markdown('<div class="section-hint">Vẽ tự do bất kỳ hình học nào — tròn, vuông, tam giác, chữ nhật, elip hay ngôi sao</div>', unsafe_allow_html=True)

col_tool, col_size, col_color = st.columns([2, 2, 1])
with col_tool:
    draw_mode = st.selectbox("Chế độ vẽ", ["freedraw","line"],
        format_func=lambda x: "✏️ Vẽ tự do" if x=="freedraw" else "📏 Đường thẳng",
        label_visibility="collapsed")
with col_size:
    stroke_w = st.slider("Độ dày nét", 4, 22, 10, label_visibility="collapsed")
with col_color:
    stroke_c = st.color_picker("Màu", "#1a4480", label_visibility="collapsed")

st.markdown('<div class="canvas-frame">', unsafe_allow_html=True)
canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=stroke_w,
    stroke_color=stroke_c,
    background_color="#ffffff",
    height=320, width=680,
    drawing_mode=draw_mode,
    key="canvas",
    display_toolbar=True,
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="class-pills">
  <span class="class-pill">⭕ Hình tròn</span>
  <span class="class-pill">🟦 Hình vuông</span>
  <span class="class-pill">🔺 Tam giác</span>
  <span class="class-pill">▬ Chữ nhật</span>
  <span class="class-pill">🫧 Elip</span>
  <span class="class-pill">⭐ Ngôi sao</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
btn_predict = st.button("🔍 Đoán hình ngay!", use_container_width=True)

if btn_predict:
    if canvas_result.image_data is None:
        st.warning("✏️ Hãy vẽ một hình trước nhé!")
    else:
        img_arr = canvas_result.image_data.astype(np.uint8)
        alpha = img_arr[:, :, 3] if img_arr.shape[2] == 4 else None
        if alpha is not None and alpha.sum() < 500:
            st.warning("✏️ Bảng vẽ trống! Hãy vẽ một hình trước.")
        else:
            with st.spinner("🤖 AI đang phân tích…"):
                rgba = Image.fromarray(img_arr, 'RGBA')
                bg = Image.new('RGB', rgba.size, (255, 255, 255))
                bg.paste(rgba, mask=rgba.split()[3])
                gray = np.array(bg.convert('L'))
                feat = extract_features(gray)
                if feat is None:
                    st.warning("⚠️ Không tìm thấy hình nào — hãy vẽ rõ hơn nhé!")
                else:
                    feat_s = scaler.transform(feat.reshape(1, -1))
                    proba = model.predict_proba(feat_s)[0]
                    idx = int(np.argmax(proba))
                    conf = float(proba[idx]) * 100
                    time.sleep(0.2)
            if feat is not None:
                icon = CLASS_ICONS[idx]
                name = CLASS_NAMES[idx]
                st.markdown(f"""
<div class="result-card">
  <div class="result-emoji">{icon}</div>
  <div class="result-shape-name">Đây là <span>{name}</span>!</div>
  <div class="result-confidence">Độ tự tin: {conf:.1f}%</div>
  <div class="confidence-bar-wrap">
    <div class="confidence-bar-fill" style="width:{int(conf)}%"></div>
  </div>
</div>""", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label" style="font-size:0.95rem;">📊 Phân tích chi tiết</div>', unsafe_allow_html=True)
                prob_rows = ""
                for i in np.argsort(proba)[::-1]:
                    p = float(proba[i]) * 100
                    bold = "font-weight:700;color:#0d1b3e;" if i == idx else ""
                    prob_rows += f"""
<div class="prob-row">
  <div class="prob-name" style="{bold}">{CLASS_ICONS[i]} {CLASS_NAMES[i].replace('Hình ','')}</div>
  <div class="prob-bar-wrap">
    <div class="prob-bar-fill" style="width:{int(p)}%;{'opacity:1' if i==idx else 'opacity:0.45'}"></div>
  </div>
  <div class="prob-pct" style="{bold}">{p:.1f}%</div>
</div>"""
                st.markdown(
                    f'<div style="background:#fff;border:1.5px solid #d4dae8;border-radius:12px;padding:1rem 1.25rem;">{prob_rows}</div>',
                    unsafe_allow_html=True)

st.markdown("""
<div class="tips-card">
  <b>💡 Mẹo vẽ tốt hơn</b><br>
  • Vẽ hình <b>lớn</b>, chiếm nhiều diện tích bảng vẽ<br>
  • Hình tròn &amp; elip: vẽ thành vòng <b>khép kín</b><br>
  • Tam giác: vẽ 3 cạnh rõ ràng, khép kín<br>
  • Ngôi sao: vẽ 5 cánh nhọn, khép kín<br>
  • Dùng nét bút <b>dày hơn</b> (8–14) để AI nhận dạng chính xác hơn<br>
  • Vuông vs Chữ nhật: vuông thì 4 cạnh <b>bằng nhau</b>, chữ nhật thì rộng hơn cao
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;margin-top:2.5rem;padding:1rem;
            font-family:'Be Vietnam Pro',sans-serif;font-size:0.78rem;
            color:#9aa3b4;border-top:1px solid #e0e6f0;">
  <span style="font-family:'Playfair Display',serif;font-weight:700;color:#0d1b3e;">Shape<span style="color:#1a8cff;font-style:italic">AI</span></span>
  &nbsp;·&nbsp; GBT + RF + SVM Ensemble · scikit-learn · Streamlit
  &nbsp;·&nbsp; Made with ♥ in Việt Nam
</div>
""", unsafe_allow_html=True)
