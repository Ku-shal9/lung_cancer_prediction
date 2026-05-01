import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve, confusion_matrix)
from sklearn.feature_selection import chi2
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LCS · Lung Cancer Study",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;1,300&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, .stApp { background:#040c18 !important; }
.main .block-container { padding: 2rem 2.5rem 4rem; max-width:1400px; }
/* Keep header visible so the sidebar collapse/expand control stays usable */
#MainMenu, footer { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#060f1e 0%,#04090f 100%) !important;
    border-right: 1px solid rgba(0,245,255,.12);
}
[data-testid="stSidebar"] > div:first-child { padding:0; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#04090f; }
::-webkit-scrollbar-thumb { background:rgba(0,245,255,.25); border-radius:3px; }

/* ── Streamlit button reset ── */
.stButton>button {
    width:100%;
    background:transparent;
    border:1px solid rgba(0,245,255,.12);
    color:rgba(160,210,240,.55);
    font-family:'IBM Plex Mono',monospace;
    font-size:.75rem;
    letter-spacing:.1em;
    text-transform:uppercase;
    padding:13px 18px;
    border-radius:8px;
    transition:all .25s;
    text-align:left;
}
.stButton>button:hover {
    background:rgba(0,245,255,.07);
    border-color:rgba(0,245,255,.35);
    color:#00f5ff;
    box-shadow:0 0 18px rgba(0,245,255,.12);
    transform:translateX(3px);
}
.stButton>button:focus { box-shadow:0 0 0 2px rgba(0,245,255,.3) !important; }

/* active-nav override injected via JS trick below */
.active-nav > .stButton > button {
    background:rgba(0,245,255,.09) !important;
    border-color:rgba(0,245,255,.45) !important;
    color:#00f5ff !important;
    box-shadow:0 0 22px rgba(0,245,255,.15) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] iframe { border-radius:10px; }

/* ── Slider ── */
.stSlider [data-baseweb="slider"] { padding-top:6px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HELPER: Plotly theme
# ─────────────────────────────────────────────
CYAN   = "#00f5ff"
RED    = "#ff3a5c"
GREEN  = "#00ff99"
ORANGE = "#ffb347"
PURPLE = "#b66dff"
PINK   = "#ff6eb4"

PALETTE = [CYAN, RED, GREEN, ORANGE, PURPLE, PINK,
           "#4df0ff","#ff8fa3","#6dffb4","#ffd06d"]

def apply_theme(fig, title="", height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Orbitron", size=14, color="#e8f4ff"),
                   x=0.03, y=0.97),
        plot_bgcolor="rgba(5,12,22,0)",
        paper_bgcolor="rgba(6,15,28,.7)",
        font=dict(family="IBM Plex Mono", color="#9abcd6", size=11),
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,245,255,.15)",
                    borderwidth=1, font=dict(size=10)),
        xaxis=dict(gridcolor="rgba(0,245,255,.07)", linecolor="rgba(0,245,255,.15)",
                   zerolinecolor="rgba(0,245,255,.1)"),
        yaxis=dict(gridcolor="rgba(0,245,255,.07)", linecolor="rgba(0,245,255,.15)",
                   zerolinecolor="rgba(0,245,255,.1)"),
        colorway=PALETTE,
    )
    return fig

# ─────────────────────────────────────────────
#  HELPER: UI components
# ─────────────────────────────────────────────
def page_header(tag, title, subtitle):
    st.markdown(f"""
    <div style="margin-bottom:2rem;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;
                  color:rgba(0,245,255,.5);letter-spacing:.25em;
                  text-transform:uppercase;margin-bottom:6px;">{tag}</div>
      <div style="font-family:'Orbitron',monospace;font-size:1.75rem;font-weight:700;
                  color:#e8f4ff;letter-spacing:.04em;line-height:1.2;">{title}</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:.92rem;
                  color:rgba(0,245,255,.65);margin-top:8px;letter-spacing:.04em;">{subtitle}</div>
      <div style="height:1px;background:linear-gradient(90deg,rgba(0,245,255,.4),transparent);
                  margin-top:20px;"></div>
    </div>
    """, unsafe_allow_html=True)

def metric_card(value, label, color=CYAN, icon=""):
    icon_html = f'<div style="font-size:1.05rem;margin-bottom:4px;">{icon}</div>' if icon else ""
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(0,245,255,.04) 0%,rgba(4,8,16,.85) 100%);
                border:1px solid rgba(0,245,255,.18);border-radius:12px;padding:22px 18px;
                text-align:center;position:relative;overflow:hidden;">
      <div style="position:absolute;top:0;left:0;right:0;height:2px;
                  background:linear-gradient(90deg,transparent,{color},transparent);"></div>
      {icon_html}
      <div style="font-family:'Orbitron',monospace;font-size:1.85rem;font-weight:700;
                  color:{color};text-shadow:0 0 22px {color}80;">{value}</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:.7rem;
                  color:rgba(160,210,240,.55);text-transform:uppercase;
                  letter-spacing:.12em;margin-top:6px;">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def info_box(text, kind="info"):
    colors = {"info":(CYAN,"rgba(0,245,255,.05)"),
              "warn":(ORANGE,"rgba(255,179,71,.05)"),
              "success":(GREEN,"rgba(0,255,153,.05)"),
              "danger":(RED,"rgba(255,58,92,.05)")}
    c, bg = colors.get(kind, colors["info"])
    st.markdown(f"""
    <div style="background:{bg};border:1px solid {c}33;border-left:3px solid {c};
                border-radius:0 8px 8px 0;padding:14px 18px;
                font-family:'IBM Plex Mono',monospace;font-size:.92rem;
                color:rgba(210,235,255,.88);margin:12px 0;line-height:1.7;">
      {text}
    </div>
    """, unsafe_allow_html=True)

def section_divider():
    st.markdown("""<div style="height:1px;background:linear-gradient(90deg,
        transparent,rgba(0,245,255,.2),transparent);margin:28px 0;"></div>""",
        unsafe_allow_html=True)

def badge(text, color=CYAN):
    st.markdown(f"""
    <span style="display:inline-block;padding:3px 11px;border-radius:20px;
                 background:{color}1a;border:1px solid {color}44;
                 font-family:'IBM Plex Mono',monospace;font-size:.7rem;
                 color:{color};letter-spacing:.07em;text-transform:uppercase;">
      {text}
    </span>&nbsp;
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    orig = pd.read_csv("lcs.csv")
    clean = pd.read_csv("cleaned_lcs.csv")
    return orig, clean

@st.cache_data
def train_model(df):
    features = ['smoking','anxiety','allergy','wheezing',
                'alcohol consuming','coughing','shortness of breath',
                'swallowing difficulty','chest pain']
    X = df[features]; y = df['lung_cancer']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=42)
    pipe = Pipeline([('scaler',StandardScaler()),('model',LogisticRegression(max_iter=1000))])
    pipe.fit(X_train,y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:,1]
    return pipe, X_test, y_test, y_pred, y_prob, features

@st.cache_resource
def fit_statsmodels_logit(df):
    X = df.drop(columns=['lung_cancer'])
    y = df['lung_cancer']
    X = sm.add_constant(X, has_constant='add')
    model = sm.Logit(y, X)
    return model.fit(disp=0)

df_orig, df_clean = load_data()
pipeline, X_test, y_test, y_pred, y_prob, MODEL_FEATURES = train_model(df_clean)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "original"

NAV = [
    ("overview", "Overview"),
    ("original", "Original Dataset"),
    ("cleaned", "Cleaned Dataset"),
    ("eda", "EDA"),
    ("feature", "Feature Analysis"),
    ("hypothesis", "Hypothesis"),
    ("model", "Model"),
]

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="text-align:center;padding:32px 20px 28px;
                border-bottom:1px solid rgba(0,245,255,.1);margin-bottom:20px;">
      <div style="font-size:2.2rem;margin-bottom:8px;">🫁</div>
      <div style="font-family:'Orbitron',monospace;font-size:1.15rem;font-weight:900;
                  color:#00f5ff;letter-spacing:.18em;
                  text-shadow:0 0 28px rgba(0,245,255,.55);">LCS</div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:.6rem;
                  color:rgba(160,210,240,.4);letter-spacing:.22em;
                  text-transform:uppercase;margin-top:5px;">Lung Cancer Study</div>
    </div>
    """, unsafe_allow_html=True)

    # Nav label
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:.6rem;
                color:rgba(0,245,255,.35);letter-spacing:.22em;text-transform:uppercase;
                padding:0 16px;margin-bottom:8px;">Navigation</div>
    """, unsafe_allow_html=True)

    for key, label in NAV:
        is_active = st.session_state.page == key
        btn_style = ""
        if is_active:
            st.markdown(f"""
            <style>
            div[data-testid="stVerticalBlock"] .element-container:has(button[kind]):last-child {{}}
            </style>
            """, unsafe_allow_html=True)
        col_btn = st.container()
        with col_btn:
            if is_active:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;padding:12px 18px;
                            background:rgba(0,245,255,.09);border:1px solid rgba(0,245,255,.42);
                            border-radius:8px;margin:3px 0;cursor:default;
                            box-shadow:0 0 20px rgba(0,245,255,.12);">
                  <span style="font-family:'IBM Plex Mono',monospace;font-size:.75rem;
                               letter-spacing:.08em;text-transform:uppercase;
                               color:#00f5ff;">{label}</span>
                  <span style="margin-left:auto;width:6px;height:6px;border-radius:50%;
                               background:#00f5ff;box-shadow:0 0 8px #00f5ff;"></span>
                </div>
                """, unsafe_allow_html=True)
            else:
                if st.button(label, key=f"nav_{key}"):
                    st.session_state.page = key
                    st.rerun()

    # Footer
    st.markdown("""
    <div style="position:fixed;bottom:0;left:0;width:260px;
                padding:16px 20px;border-top:1px solid rgba(0,245,255,.08);
                background:#04090f;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:.62rem;
                  color:rgba(160,210,240,.3);letter-spacing:.1em;text-align:center;">
        LUNG CANCER STUDY · v1.0
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PAGE 0 · OVERVIEW
# ─────────────────────────────────────────────
if st.session_state.page == "overview":
    page_header("00 / 07", "PROJECT OVERVIEW", "Concise summary of the study, context, and approach")

    st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:.72rem;
        color:rgba(0,245,255,.55);text-transform:uppercase;letter-spacing:.12em;
        margin-bottom:10px;">Overview</div>""", unsafe_allow_html=True)
    st.image("images/overview.png", use_container_width=True)

    section_divider()
    st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:.72rem;
        color:rgba(0,245,255,.55);text-transform:uppercase;letter-spacing:.12em;
        margin-bottom:10px;">Problem Statement</div>""", unsafe_allow_html=True)
    st.image("images/problem.png", use_container_width=True)

    section_divider()
    st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:.72rem;
        color:rgba(0,245,255,.55);text-transform:uppercase;letter-spacing:.12em;
        margin-bottom:10px;">Literature Review</div>""", unsafe_allow_html=True)
    lit_left, lit_right = st.columns(2)
    with lit_left:
        st.image("images/literature1.png", use_container_width=True)
    with lit_right:
        st.image("images/literature2.png", use_container_width=True)
    section_divider()
    st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:.72rem;
        color:rgba(0,245,255,.55);text-transform:uppercase;letter-spacing:.12em;
        margin-bottom:10px;">Solution</div>""", unsafe_allow_html=True)
    st.image("images/solution.png", use_container_width=True)

    section_divider()

# ─────────────────────────────────────────────
#  PAGE 1 · ORIGINAL DATASET
# ─────────────────────────────────────────────
elif st.session_state.page == "original":
    page_header("01 / 07", "ORIGINAL DATASET", "Raw data as collected — unprocessed, unfiltered, unaltered")

    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("1,157", "Total Records", CYAN)
    with c2: metric_card("16", "Features", PURPLE)
    with c3: metric_card("246", "Duplicates", ORANGE)
    with c4: metric_card("0", "Missing Values", GREEN)

    section_divider()

    # Charts row
    c_left, c_right = st.columns(2)

    with c_left:
        cancer_counts = df_orig['LUNG_CANCER'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=['Cancer Positive','Cancer Negative'],
            values=[cancer_counts.get('YES',0), cancer_counts.get('NO',0)],
            hole=.62,
            marker=dict(colors=[RED, CYAN],
                        line=dict(color='#04090f',width=3)),
            textinfo='percent+label',
            textfont=dict(family='IBM Plex Mono',size=11,color='#e8f4ff'),
            hovertemplate='%{label}<br>Count: %{value}<br>Pct: %{percent}<extra></extra>'
        )])
        fig.add_annotation(text=f"<b>1157</b><br><span style='font-size:10px'>Records</span>",
                           x=.5,y=.5,font=dict(family='Orbitron',size=16,color='#00f5ff'),
                           showarrow=False)
        fig = apply_theme(fig, "Target Distribution", 380)
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        gender_counts = df_orig['GENDER'].value_counts()
        fig = go.Figure(data=[go.Bar(
            x=['Male','Female'],
            y=[gender_counts.get('M',0), gender_counts.get('F',0)],
            marker=dict(
                color=[CYAN, PINK],
                line=dict(color='rgba(0,0,0,0)',width=0)
            ),
            text=[gender_counts.get('M',0), gender_counts.get('F',0)],
            textposition='outside',
            textfont=dict(family='IBM Plex Mono',size=12,color='#e8f4ff'),
            hovertemplate='%{x}: %{y}<extra></extra>'
        )])
        fig = apply_theme(fig, "Gender Distribution", 380)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    section_divider()

    # Age distribution
    df_orig_copy = df_orig.copy()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df_orig_copy['AGE'],
        nbinsx=30,
        marker=dict(color=CYAN, opacity=.75,
                    line=dict(color='rgba(0,245,255,.3)',width=1)),
        name='Age',
        hovertemplate='Age: %{x}<br>Count: %{y}<extra></extra>'
    ))
    fig.add_vline(x=df_orig_copy['AGE'].mean(), line_dash="dash",
                  line_color=ORANGE, annotation_text=f"Mean: {df_orig_copy['AGE'].mean():.1f}",
                  annotation_font=dict(family='IBM Plex Mono', color=ORANGE, size=11))
    fig = apply_theme(fig, "Age Distribution of Patients", 320)
    st.plotly_chart(fig, use_container_width=True)

    section_divider()

    info_box("The dataset contains <b>246 duplicate records</b> which create class imbalance bias. Before deduplication, cancer-positive cases accounted for ~44% of data. After cleanup this rises to ~52%, reflecting a more realistic distribution.", "warn")

    # Data table
    st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:.72rem;
        color:rgba(0,245,255,.6);text-transform:uppercase;letter-spacing:.12em;
        margin-bottom:10px;">Raw Data Preview</div>""", unsafe_allow_html=True)
    st.dataframe(
        df_orig.head(50).style
            .set_properties(**{'background-color':'rgba(5,12,25,.7)',
                               'color':'#cce8ff',
                               'font-family':'IBM Plex Mono',
                               'font-size':'12px'})
            .highlight_null(color='rgba(255,58,92,.2)'),
        use_container_width=True,
        height=360
    )

# ─────────────────────────────────────────────
#  PAGE 2 · CLEANED DATASET
# ─────────────────────────────────────────────
elif st.session_state.page == "cleaned":
    page_header("02 / 07", "CLEANED DATASET",
                "After deduplication, encoding & standardisation")

    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("911", "Records After Cleaning", GREEN)
    with c2: metric_card("246", "Duplicates Removed", RED)
    with c3: metric_card("474", "Cancer Positive", RED)
    with c4: metric_card("437", "Cancer Negative", CYAN)

    section_divider()

    # Before vs After
    c_left, c_right = st.columns(2)

    with c_left:
        fig = go.Figure()
        labels = ['Cancer +','Cancer -']
        before = [513/1157*100, 644/1157*100]
        after  = [474/911*100,  437/911*100]
        x = np.arange(len(labels))
        fig.add_trace(go.Bar(name='Before',x=labels,y=before,
                             marker_color=PURPLE,opacity=.75,
                             text=[f"{v:.1f}%" for v in before],textposition='outside',
                             textfont=dict(family='IBM Plex Mono',size=10,color='#e8f4ff')))
        fig.add_trace(go.Bar(name='After',x=labels,y=after,
                             marker_color=GREEN,opacity=.85,
                             text=[f"{v:.1f}%" for v in after],textposition='outside',
                             textfont=dict(family='IBM Plex Mono',size=10,color='#e8f4ff')))
        fig = apply_theme(fig, "Class Balance: Before vs After", 360)
        fig.update_layout(barmode='group', bargap=.15, bargroupgap=.05)
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        # Encoding showcase
        encoding_data = {
            'Original Value': ['M (Male)','F (Female)','YES','NO','2 (Yes)','1 (No)'],
            'Encoded Value':  [1, 0, 1, 0, 1, 0],
            'Column':         ['gender','gender','lung_cancer','lung_cancer','binary cols','binary cols']
        }
        enc_df = pd.DataFrame(encoding_data)
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Original</b>','<b>Encoded</b>','<b>Column</b>'],
                fill_color='rgba(0,245,255,.12)',
                align='left',
                font=dict(family='IBM Plex Mono',size=11,color='#00f5ff'),
                line_color='rgba(0,245,255,.2)',
                height=35
            ),
            cells=dict(
                values=[enc_df['Original Value'],enc_df['Encoded Value'],enc_df['Column']],
                fill_color=['rgba(5,12,25,.7)'],
                align='left',
                font=dict(family='IBM Plex Mono',size=10,color='#cce8ff'),
                line_color='rgba(0,245,255,.1)',
                height=30
            )
        )])
        fig = apply_theme(fig, "Binary Encoding Map", 360)
        st.plotly_chart(fig, use_container_width=True)

    section_divider()

    # Feature heatmap of cleaned data
    corr_matrix = df_clean.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=[[0,'#1a0a2e'],[0.5,'#04090f'],[1,'#00f5ff']],
        text=np.round(corr_matrix.values,2),
        texttemplate='%{text}',
        textfont=dict(size=8,family='IBM Plex Mono'),
        hovertemplate='%{x} vs %{y}<br>r = %{z:.3f}<extra></extra>',
        zmin=-1, zmax=1
    ))
    fig = apply_theme(fig, "Correlation Heatmap — Cleaned Features", 520)
    fig.update_layout(margin=dict(l=120,r=20,t=50,b=120))
    st.plotly_chart(fig, use_container_width=True)

    section_divider()

    info_box("All binary symptoms (1=No, 2=Yes) have been re-encoded to (0, 1). Gender encoded as M→1, F→0. Target variable YES→1, NO→0. Column names standardised to lowercase.", "success")

    st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:.72rem;
        color:rgba(0,245,255,.6);text-transform:uppercase;letter-spacing:.12em;
        margin-bottom:10px;">Cleaned Data Preview</div>""", unsafe_allow_html=True)
    st.dataframe(
        df_clean.head(50).style
            .set_properties(**{'background-color':'rgba(5,12,25,.7)',
                               'color':'#cce8ff',
                               'font-family':'IBM Plex Mono',
                               'font-size':'12px'})
            .background_gradient(cmap='Blues', subset=['age'], vmin=0),
        use_container_width=True,
        height=360
    )

# ─────────────────────────────────────────────
#  PAGE 3 · EDA
# ─────────────────────────────────────────────
elif st.session_state.page == "eda":
    page_header("03 / 07", "EXPLORATORY DATA ANALYSIS",
                "Uncovering patterns, distributions & relationships in the data")

    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("52.5", "Mean Age", CYAN)
    with c2: metric_card("16.7", "Age Std Dev", PURPLE)
    with c3: metric_card("87", "Max Age", ORANGE)
    with c4: metric_card("20", "Min Age", GREEN)

    section_divider()

    # Age distribution with KDE by cancer status
    c_left, c_right = st.columns(2)
    with c_left:
        fig = go.Figure()
        for val, label, color in [(1,'Cancer Positive',RED),(0,'Cancer Negative',CYAN)]:
            subset = df_clean[df_clean['lung_cancer']==val]['age']
            fig.add_trace(go.Histogram(
                x=subset, name=label, nbinsx=20,
                marker_color=color, opacity=.65,
                hovertemplate=f'{label}<br>Age: %{{x}}<br>Count: %{{y}}<extra></extra>'
            ))
        fig = apply_theme(fig, "Age Distribution by Cancer Status", 380)
        fig.update_layout(barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        age_groups = pd.cut(df_clean['age'],bins=[20,30,40,50,60,70,80,90],
                            labels=['20s','30s','40s','50s','60s','70s','80s'])
        group_df = pd.DataFrame({'age_group':age_groups,'lung_cancer':df_clean['lung_cancer']})
        pivot = group_df.groupby(['age_group','lung_cancer']).size().reset_index(name='count')
        pivot['status'] = pivot['lung_cancer'].map({1:'Positive',0:'Negative'})
        fig = px.bar(pivot, x='age_group', y='count', color='status',
                     color_discrete_map={'Positive':RED,'Negative':CYAN},
                     barmode='group',
                     labels={'age_group':'Age Group','count':'Patients'},
                     hover_data=['count'])
        fig = apply_theme(fig, "Cancer Cases by Age Group", 380)
        st.plotly_chart(fig, use_container_width=True)

    section_divider()

    # Symptom prevalence
    symptom_cols = [c for c in df_clean.columns
                    if c not in ['age','gender','lung_cancer']]
    prevalence = df_clean[symptom_cols].mean().sort_values(ascending=True)

    fig = go.Figure(go.Bar(
        x=prevalence.values * 100,
        y=[c.title() for c in prevalence.index],
        orientation='h',
        marker=dict(
            color=prevalence.values,
            colorscale=[[0,PURPLE],[0.5,CYAN],[1,GREEN]],
            line=dict(color='rgba(0,0,0,0)',width=0)
        ),
        text=[f"{v*100:.1f}%" for v in prevalence.values],
        textposition='outside',
        textfont=dict(family='IBM Plex Mono',size=10,color='#e8f4ff'),
        hovertemplate='%{y}: %{x:.1f}%<extra></extra>'
    ))
    fig = apply_theme(fig, "Symptom Prevalence in Dataset (%)", 440)
    fig.update_layout(margin=dict(l=160,r=60,t=50,b=20))
    st.plotly_chart(fig, use_container_width=True)

    section_divider()

    # Symptom vs cancer heatmap
    st.markdown("""<div style="font-family:'Orbitron',monospace;font-size:1rem;font-weight:600;
        color:#e8f4ff;letter-spacing:.06em;margin-bottom:16px;">
        Symptom Prevalence by Cancer Status</div>""", unsafe_allow_html=True)

    cols = [c for c in df_clean.columns if c != 'lung_cancer']
    heat_data = []
    for c in cols:
        pos_rate = df_clean[df_clean['lung_cancer']==1][c].mean()
        neg_rate = df_clean[df_clean['lung_cancer']==0][c].mean()
        heat_data.append({'Feature':c.title(),'Positive':pos_rate,'Negative':neg_rate,
                          'Difference':pos_rate-neg_rate})
    heat_df = pd.DataFrame(heat_data).sort_values('Difference',ascending=False)

    c1,c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(
            x=heat_df['Difference'],
            y=heat_df['Feature'],
            orientation='h',
            marker=dict(
                color=heat_df['Difference'],
                colorscale=[[0,PURPLE],[0.4,'#04090f'],[1,RED]],
                line=dict(color='rgba(0,0,0,0)',width=0)
            ),
            hovertemplate='%{y}<br>Δ Prevalence: %{x:.3f}<extra></extra>'
        ))
        fig = apply_theme(fig,"Feature: Positive - Negative Rate", 420)
        fig.update_layout(margin=dict(l=160,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Cancer +',x=heat_df['Feature'],y=heat_df['Positive'],
                             marker_color=RED,opacity=.8))
        fig.add_trace(go.Bar(name='Cancer -',x=heat_df['Feature'],y=heat_df['Negative'],
                             marker_color=CYAN,opacity=.8))
        fig = apply_theme(fig,"Feature Rate: Positive vs Negative Patients", 420)
        fig.update_layout(barmode='group',xaxis_tickangle=-40,
                          xaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(fig, use_container_width=True)

    section_divider()
    info_box("<b>Key EDA Insights:</b> Patients aged 50-70 show highest cancer incidence. Symptoms like chest pain, shortness of breath, and swallowing difficulty show dramatically higher rates in cancer-positive patients. Gender shows minimal differentiation across classes.", "info")

# ─────────────────────────────────────────────
#  PAGE 4 · FEATURE ANALYSIS
# ─────────────────────────────────────────────
elif st.session_state.page == "feature":
    page_header("04 / 07", "FEATURE ANALYSIS",
                "Chi-Square selection · Logistic regression · Backward elimination")

    # Chi-2 data
    K = df_clean.drop('lung_cancer', axis=1)
    m = df_clean['lung_cancer']
    chi_scores, p_values = chi2(K, m)
    chi_df = pd.DataFrame({
        'Feature': [f.title() for f in K.columns],
        'Chi2 Score': chi_scores,
        'P-Value': p_values
    }).sort_values('Chi2 Score', ascending=False)

    # Tier badges
    def get_tier(pv):
        if pv < 0.001: return ("🟢 Strong", GREEN)
        elif pv < 0.05: return ("🟡 Moderate", ORANGE)
        else: return ("🔴 Weak", RED)

    c1,c2,c3 = st.columns(3)
    strong = (chi_df['P-Value'] < 0.001).sum()
    moderate = ((chi_df['P-Value'] >= 0.001) & (chi_df['P-Value'] < 0.05)).sum()
    weak = (chi_df['P-Value'] >= 0.05).sum()
    with c1: metric_card(str(strong), "Strong Predictors (p<0.001)", GREEN, "🟢")
    with c2: metric_card(str(moderate), "Moderate Predictors", ORANGE, "🟡")
    with c3: metric_card(str(weak), "Weak/Dropped Features", RED, "🔴")

    section_divider()

    c_left, c_right = st.columns([3,2])

    with c_left:
        colors_chi = [GREEN if p<0.001 else ORANGE if p<0.05 else RED
                      for p in chi_df['P-Value']]
        fig = go.Figure(go.Bar(
            x=chi_df['Chi2 Score'],
            y=chi_df['Feature'],
            orientation='h',
            marker=dict(color=colors_chi,
                        line=dict(color='rgba(0,0,0,0)',width=0)),
            text=[f"χ²={v:.1f}" for v in chi_df['Chi2 Score']],
            textposition='outside',
            textfont=dict(family='IBM Plex Mono',size=9,color='#e8f4ff'),
            hovertemplate='%{y}<br>χ² = %{x:.2f}<extra></extra>'
        ))
        fig.add_vline(x=chi_df['Chi2 Score'].quantile(.5),line_dash='dash',
                      line_color=ORANGE, opacity=.5)
        fig = apply_theme(fig, "Chi-Square Feature Importance", 480)
        fig.update_layout(margin=dict(l=170,r=80,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        # p-value table
        chi_display = chi_df.copy()
        chi_display['Tier'] = chi_display['P-Value'].apply(lambda p: get_tier(p)[0])
        chi_display['P-Value'] = chi_display['P-Value'].apply(lambda x: f"{x:.2e}")
        chi_display['Chi2 Score'] = chi_display['Chi2 Score'].apply(lambda x: f"{x:.2f}")
        st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:.7rem;
            color:rgba(0,245,255,.6);text-transform:uppercase;letter-spacing:.12em;
            margin-bottom:10px;">Chi-Square Results</div>""", unsafe_allow_html=True)
        st.dataframe(
            chi_display[['Feature','Chi2 Score','P-Value','Tier']]
            .style.set_properties(**{
                'background-color':'rgba(5,12,25,.7)',
                'color':'#cce8ff',
                'font-family':'IBM Plex Mono',
                'font-size':'11px'
            }),
            use_container_width=True,
            height=460
        )

    section_divider()

    # Correlation with target
    corr = df_clean.corr()['lung_cancer'].drop('lung_cancer').sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=[c.title() for c in corr.index],
        y=corr.values,
        marker=dict(
            color=corr.values,
            colorscale=[[0,RED],[0.5,'#04090f'],[1,GREEN]],
            line=dict(color='rgba(0,0,0,0)',width=0)
        ),
        text=[f"{v:.3f}" for v in corr.values],
        textposition='outside',
        textfont=dict(family='IBM Plex Mono',size=9,color='#e8f4ff'),
        hovertemplate='%{x}<br>r = %{y:.4f}<extra></extra>'
    ))
    fig.add_hline(y=0, line_color='rgba(255,255,255,.2)')
    fig = apply_theme(fig, "Pearson Correlation with Lung Cancer Target", 360)
    fig.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(fig, use_container_width=True)

    section_divider()

    info_box("""<b>Backward Elimination Results (Logistic Regression, p > 0.05):</b><br>
    The following features were <b>dropped</b> due to statistical insignificance:
    <br>❌ peer_pressure &nbsp;|&nbsp; ❌ gender &nbsp;|&nbsp; ❌ age &nbsp;|&nbsp; ❌ yellow_fingers &nbsp;|&nbsp; ❌ chronic disease &nbsp;|&nbsp; ❌ fatigue<br><br>
    <b>Retained features for modelling:</b> smoking, anxiety, allergy, wheezing,
    alcohol consuming, coughing, shortness of breath, swallowing difficulty, chest pain""", "warn")

    info_box("""<b>Top Predictors by Chi-Square:</b><br>
    🥇 Swallowing Difficulty (χ²=271.5) &nbsp;|&nbsp;
    🥈 Chest Pain (χ²=269.0) &nbsp;|&nbsp;
    🥉 Shortness of Breath (χ²=256.1) &nbsp;|&nbsp;
    4th Smoking (χ²=241.0)""", "success")

# ─────────────────────────────────────────────
#  PAGE 5 · HYPOTHESIS
# ─────────────────────────────────────────────
elif st.session_state.page == "hypothesis":
    page_header("05 / 07", "HYPOTHESIS TESTING",
                "Statistical validation · Logistic regression · Class balance analysis")

    # Hypothesis box
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(0,245,255,.04),rgba(4,8,16,.9));
                border:1px solid rgba(0,245,255,.25);border-radius:12px;padding:28px;
                margin-bottom:28px;">
      <div style="display:flex;gap:32px;flex-wrap:wrap;">
        <div style="flex:1;min-width:240px;">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;
                      color:rgba(255,58,92,.7);letter-spacing:.2em;text-transform:uppercase;
                      margin-bottom:8px;">H₀ — Null Hypothesis</div>
          <div style="font-family:'DM Sans',sans-serif;font-size:.95rem;color:#cce8ff;
                      line-height:1.6;">
            Independent variables have <b>no significant effect</b> on lung cancer diagnosis.
          </div>
        </div>
        <div style="flex:1;min-width:240px;">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;
                      color:rgba(0,255,153,.7);letter-spacing:.2em;text-transform:uppercase;
                      margin-bottom:8px;">H₁ — Alternative Hypothesis</div>
          <div style="font-family:'DM Sans',sans-serif;font-size:.95rem;color:#cce8ff;
                      line-height:1.6;">
            Independent variables <b>significantly affect</b> lung cancer diagnosis.
          </div>
        </div>
        <div style="flex:0;min-width:180px;text-align:center;padding:16px;
                    background:rgba(0,255,153,.08);border:1px solid rgba(0,255,153,.3);
                    border-radius:10px;">
          <div style="font-family:'Orbitron',monospace;font-size:1rem;color:{GREEN};
                      text-shadow:0 0 18px {GREEN}88;font-weight:700;">H₀ REJECTED</div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;
                      color:rgba(0,255,153,.6);margin-top:6px;letter-spacing:.1em;">
            p-values &lt; 0.05<br>confirmed
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div style="font-family:'Orbitron',monospace;font-size:1rem;font-weight:600;
        color:#e8f4ff;letter-spacing:.06em;margin-bottom:14px;">
        Statsmodels Logistic Regression Output</div>""", unsafe_allow_html=True)

    try:
        sm_result = fit_statsmodels_logit(df_clean)
        st.markdown(f"""
        <div style="background:rgba(0,245,255,.08);border:1px solid rgba(0,245,255,.35);
                    border-radius:10px;padding:14px 16px;margin-bottom:14px;">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:.68rem;
                      color:rgba(0,245,255,.75);letter-spacing:.12em;text-transform:uppercase;
                      margin-bottom:8px;">Model Fit Statistics</div>
          <div style="display:flex;gap:18px;flex-wrap:wrap;
                      font-family:'IBM Plex Mono',monospace;font-size:.82rem;color:#cce8ff;">
            <div><b>Pseudo R²:</b> {sm_result.prsquared:.4f}</div>
            <div><b>Log-Likelihood:</b> {sm_result.llf:.4f}</div>
            <div><b>LL-Null:</b> {sm_result.llnull:.4f}</div>
            <div><b>LLR p-value:</b> {sm_result.llr_pvalue:.4e}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        coef_df = pd.DataFrame({
            'Feature': sm_result.params.index,
            'Coefficient': sm_result.params.values,
            'P-Value': sm_result.pvalues.values,
            'Odds Ratio': np.exp(sm_result.params.values)
        })
        coef_df['Feature'] = coef_df['Feature'].replace({'const': 'Intercept'})
        coef_df = coef_df.sort_values('P-Value')

        def color_predictor_rows(row):
            pv = row['P-Value']
            if pv < 0.001:
                bg = 'rgba(0,255,153,.18)'   # light green (strong)
            elif pv < 0.05:
                bg = 'rgba(255,235,120,.22)' # light yellow (medium)
            else:
                bg = 'rgba(255,58,92,.22)'   # light red (insignificant)
            return [f'background-color: {bg}; color: #e8f4ff'] * len(row)

        st.dataframe(
            coef_df.style.format({
                'Coefficient': '{:.4f}',
                'P-Value': '{:.4e}',
                'Odds Ratio': '{:.4f}',
            }).apply(color_predictor_rows, axis=1).set_properties(**{
                'color': '#cce8ff',
                'font-family': 'IBM Plex Mono',
                'font-size': '11px'
            }),
            use_container_width=True,
            height=320
        )

        st.markdown(f"""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:rgba(210,235,255,.82);
                    margin-top:10px;margin-bottom:12px;line-height:1.7;">
          <span style="display:inline-block;padding:2px 8px;border-radius:6px;
                       background:rgba(0,255,153,.18);border:1px solid rgba(0,255,153,.35);">
            Strong Predictor: p &lt; 0.001
          </span>
          &nbsp;
          <span style="display:inline-block;padding:2px 8px;border-radius:6px;
                       background:rgba(255,235,120,.22);border:1px solid rgba(255,235,120,.45);">
            Medium Predictor: 0.001 ≤ p &lt; 0.05
          </span>
          &nbsp;
          <span style="display:inline-block;padding:2px 8px;border-radius:6px;
                       background:rgba(255,58,92,.22);border:1px solid rgba(255,58,92,.45);">
            Insignificant: p ≥ 0.05
          </span>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("View full model summary", expanded=False):
            st.code(sm_result.summary().as_text(), language="text")
        st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)
    except Exception as e:
        info_box(f"Unable to display statsmodels output: {e}", "warn")

    section_divider()

    c_left, c_right = st.columns(2)

    with c_left:
        # Significant features
        sig_features = ['smoking','anxiety','allergy','wheezing',
                        'coughing','shortness of breath','swallowing difficulty','chest pain']
        insig_features = ['peer_pressure','gender','age','yellow_fingers',
                          'chronic disease','fatigue']

        fig = go.Figure()
        sig_vals = [df_clean[f].mean()*100 for f in sig_features]
        fig.add_trace(go.Bar(
            x=[f.title() for f in sig_features], y=sig_vals,
            marker_color=GREEN, opacity=.8,
            name='Significant (p < 0.05)',
            text=[f"{v:.1f}%" for v in sig_vals], textposition='outside',
            textfont=dict(family='IBM Plex Mono',size=9,color='#e8f4ff')
        ))
        fig = apply_theme(fig, "Significant Features (Retained)", 400)
        fig.update_layout(showlegend=False, xaxis_tickangle=-35,
                          xaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        insig_vals = []
        for f in insig_features:
            if f == 'age':
                insig_vals.append(df_clean[f].mean())
            else:
                insig_vals.append(df_clean[f].mean()*100)

        age_mean = df_clean['age'].mean()
        other_insig = ['peer_pressure','gender','yellow_fingers','chronic disease','fatigue']
        other_vals = [df_clean[f].mean()*100 for f in other_insig]

        fig = go.Figure(go.Bar(
            x=[f.title() for f in other_insig], y=other_vals,
            marker_color=RED, opacity=.65, name='Insignificant',
            text=[f"{v:.1f}%" for v in other_vals], textposition='outside',
            textfont=dict(family='IBM Plex Mono',size=9,color='#e8f4ff')
        ))
        fig = apply_theme(fig, "Insignificant Features (Dropped)", 400)
        fig.update_layout(showlegend=False, xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    section_divider()

    # Class balance
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(data=[go.Pie(
            labels=['Lung Cancer +','Lung Cancer -'],
            values=[474, 437],
            hole=.55,
            marker=dict(colors=[RED,CYAN],line=dict(color='#04090f',width=3)),
            textinfo='percent+value',
            textfont=dict(family='IBM Plex Mono',size=11,color='#e8f4ff'),
            pull=[.05,0],
            hovertemplate='%{label}<br>Count: %{value}<br>%{percent}<extra></extra>'
        )])
        fig.add_annotation(text="<b>52%/48%</b>",x=.5,y=.5,
                           font=dict(family='Orbitron',size=14,color=CYAN),showarrow=False)
        fig = apply_theme(fig, "Balanced Class Distribution (Post-Cleaning)", 380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Logistic significance summary (from statsmodels output)
        try:
            sm_result = fit_statsmodels_logit(df_clean)
            pvals = sm_result.pvalues.drop('const', errors='ignore').sort_values()
            pvals_for_plot = pvals.head(8)
            feat_names = [f.title() for f in pvals_for_plot.index]
            colors_p = [GREEN if p < 0.05 else RED for p in pvals_for_plot.values]

            fig = go.Figure(go.Bar(
                x=feat_names,
                y=[-np.log10(max(p, 1e-16)) for p in pvals_for_plot.values],
                marker=dict(color=colors_p, line=dict(color='rgba(0,0,0,0)', width=0)),
                text=[f"p={p:.4g}" for p in pvals_for_plot.values],
                textposition='outside',
                textfont=dict(family='IBM Plex Mono', size=9, color='#e8f4ff'),
                hovertemplate='%{x}<br>-log₁₀(p): %{y:.2f}<extra></extra>'
            ))
            fig.add_hline(y=-np.log10(0.05), line_dash='dash', line_color=ORANGE,
                          annotation_text="p = 0.05 threshold",
                          annotation_font=dict(family='IBM Plex Mono', color=ORANGE, size=10))
            fig = apply_theme(fig, "Statistical Significance (-log10 p-value)", 380)
            fig.update_layout(xaxis_tickangle=-35, xaxis=dict(tickfont=dict(size=9)))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            info_box(f"Statsmodels logistic regression could not be fit: {e}", "warn")

    section_divider()
    info_box("""<b>Conclusion:</b> We <b>REJECT the null hypothesis</b> (H₀).<br><br>
    Several features — notably smoking, anxiety, allergy, wheezing, coughing, shortness of breath,
    swallowing difficulty, and chest pain — show p-values well below 0.05 in logistic regression analysis.
    This provides strong statistical evidence that independent variables DO significantly affect lung cancer outcomes.
    The balanced class distribution (52%/48%) further validates our cleaned dataset.""", "success")

# ─────────────────────────────────────────────
#  PAGE 6 · MODEL
# ─────────────────────────────────────────────
elif st.session_state.page == "model":
    page_header("06 / 07", "MODEL — LOGISTIC REGRESSION",
                "Trained on 9 features · StandardScaler pipeline · 70/30 split")

    acc = accuracy_score(y_test,y_pred)
    prec = precision_score(y_test,y_pred)
    rec = recall_score(y_test,y_pred)
    f1s = f1_score(y_test,y_pred)
    auc = roc_auc_score(y_test,y_prob)
    cm  = confusion_matrix(y_test,y_pred)

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: metric_card(f"{acc*100:.2f}%", "Accuracy", GREEN)
    with c2: metric_card(f"{prec*100:.2f}%", "Precision", CYAN)
    with c3: metric_card(f"{rec*100:.2f}%", "Recall", ORANGE)
    with c4: metric_card(f"{f1s*100:.2f}%", "F1-Score", PURPLE)
    with c5: metric_card(f"{auc:.4f}", "ROC-AUC", RED)

    section_divider()

    c_left, c_right = st.columns(2)

    with c_left:
        # Confusion matrix
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Negative','Predicted Positive'],
            y=['Actual Negative','Actual Positive'],
            text=[[str(cm[i][j]) for j in range(2)] for i in range(2)],
            texttemplate='<b>%{text}</b>',
            textfont=dict(family='Orbitron',size=22,color='#ffffff'),
            colorscale=[[0,'rgba(4,8,16,.9)'],[0.5,'rgba(0,100,160,.4)'],[1,'rgba(0,245,255,.6)']],
            showscale=False,
            hovertemplate='%{y}<br>%{x}<br>Count: %{z}<extra></extra>'
        ))
        fig.add_annotation(x='Predicted Negative',y='Actual Negative',
                           text="TN",showarrow=False,
                           font=dict(family='IBM Plex Mono',size=10,color=GREEN),yshift=-22)
        fig.add_annotation(x='Predicted Positive',y='Actual Negative',
                           text="FP",showarrow=False,
                           font=dict(family='IBM Plex Mono',size=10,color=RED),yshift=-22)
        fig.add_annotation(x='Predicted Negative',y='Actual Positive',
                           text="FN",showarrow=False,
                           font=dict(family='IBM Plex Mono',size=10,color=RED),yshift=-22)
        fig.add_annotation(x='Predicted Positive',y='Actual Positive',
                           text="TP",showarrow=False,
                           font=dict(family='IBM Plex Mono',size=10,color=GREEN),yshift=-22)
        fig = apply_theme(fig, "Confusion Matrix", 400)
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0,1],y=[0,1],mode='lines',
            line=dict(color='rgba(180,180,180,.3)',dash='dash',width=1.5),
            name='Random Classifier',
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            line=dict(color=CYAN, width=3),
            name=f'Logistic Regression (AUC = {auc:.4f})',
            fill='tozeroy',
            fillcolor='rgba(0,245,255,.06)',
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
        ))
        fig = apply_theme(fig, f"ROC Curve | AUC = {auc:.4f}", 400)
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=.55,y=.1)
        )
        st.plotly_chart(fig, use_container_width=True)

    section_divider()

    # Metrics comparison bar
    metrics_df = {
        'Metric': ['Accuracy','Precision','Recall','F1-Score','ROC-AUC'],
        'Score':  [acc, prec, rec, f1s, auc],
        'Color':  [GREEN, CYAN, ORANGE, PURPLE, RED]
    }
    fig = go.Figure()
    for i, row in enumerate(zip(metrics_df['Metric'],metrics_df['Score'],metrics_df['Color'])):
        name, score, color = row
        fig.add_trace(go.Bar(
            name=name, x=[name], y=[score*100],
            marker=dict(color=color,opacity=.8,
                        line=dict(color=color,width=1)),
            text=f"{score*100:.2f}%", textposition='outside',
            textfont=dict(family='Orbitron',size=11,color='#e8f4ff'),
            hovertemplate=f'{name}: %{{y:.2f}}%<extra></extra>'
        ))
    fig.add_hline(y=90,line_dash='dot',line_color='rgba(255,255,255,.2)',
                  annotation_text="90% threshold",
                  annotation_font=dict(family='IBM Plex Mono',color='rgba(255,255,255,.4)',size=10))
    fig = apply_theme(fig, "Complete Model Performance Overview", 360)
    fig.update_layout(showlegend=False, yaxis=dict(range=[0,110]))
    st.plotly_chart(fig, use_container_width=True)

    section_divider()

    # Confusion matrix breakdown
    tn,fp,fn,tp = cm.ravel()
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card(str(tp), "True Positives", GREEN)
    with c2: metric_card(str(tn), "True Negatives", CYAN)
    with c3: metric_card(str(fp), "False Positives", ORANGE)
    with c4: metric_card(str(fn), "False Negatives", RED)

    section_divider()

    # ── Live Prediction Tool
    st.markdown(f"""
    <div style="font-family:'Orbitron',monospace;font-size:1rem;font-weight:700;
                color:#e8f4ff;letter-spacing:.08em;margin-bottom:6px;">
        LIVE PREDICTION TOOL
    </div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:.8rem;
                color:rgba(0,245,255,.6);margin-bottom:24px;">
        Enter patient symptoms below to get an instant prediction
    </div>
    """, unsafe_allow_html=True)

    labels_map = {
        'smoking':              'Smoking',
        'anxiety':              'Anxiety',
        'allergy':              'Allergy',
        'wheezing':             'Wheezing',
        'alcohol consuming':    'Alcohol Consuming',
        'coughing':             'Coughing',
        'shortness of breath':  'Shortness of Breath',
        'swallowing difficulty':'Swallowing Difficulty',
        'chest pain':           'Chest Pain',
    }

    user_input = {}
    cols = st.columns(3)
    for i, feat in enumerate(MODEL_FEATURES):
        with cols[i % 3]:
            val = st.selectbox(labels_map[feat], [0,1],
                               format_func=lambda x: "Yes (1)" if x==1 else "No (0)",
                               key=f"pred_{feat}")
            user_input[feat] = val

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("RUN PREDICTION", key="predict_btn"):
        input_df = pd.DataFrame([user_input])
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0]

        pos_prob = probability[1]*100
        neg_prob = probability[0]*100

        if prediction == 1:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(255,58,92,.15),rgba(4,8,16,.95));
                        border:1px solid rgba(255,58,92,.5);border-radius:14px;
                        padding:28px;text-align:center;margin-top:16px;">
              <div style="font-size:2.5rem;margin-bottom:8px;">⚠️</div>
              <div style="font-family:'Orbitron',monospace;font-size:1.4rem;font-weight:700;
                          color:{RED};text-shadow:0 0 24px {RED}88;margin-bottom:10px;">
                LUNG CANCER: POSITIVE
              </div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:.9rem;
                          color:rgba(255,200,200,.85);">
                Model Confidence: <b style="color:{RED};">{pos_prob:.1f}%</b>
                &nbsp;|&nbsp; Negative Probability: {neg_prob:.1f}%
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(0,255,153,.1),rgba(4,8,16,.95));
                        border:1px solid rgba(0,255,153,.45);border-radius:14px;
                        padding:28px;text-align:center;margin-top:16px;">
              <div style="font-size:2.5rem;margin-bottom:8px;">✅</div>
              <div style="font-family:'Orbitron',monospace;font-size:1.4rem;font-weight:700;
                          color:{GREEN};text-shadow:0 0 24px {GREEN}88;margin-bottom:10px;">
                LUNG CANCER: NEGATIVE
              </div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:.9rem;
                          color:rgba(180,255,210,.85);">
                Model Confidence: <b style="color:{GREEN};">{neg_prob:.1f}%</b>
                &nbsp;|&nbsp; Positive Probability: {pos_prob:.1f}%
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pos_prob,
            title={'text':"Cancer Risk Probability (%)",
                   'font':{'family':'Orbitron','size':13,'color':'#e8f4ff'}},
            number={'font':{'family':'Orbitron','size':28,
                            'color':RED if prediction==1 else GREEN},'suffix':'%'},
            gauge={
                'axis':{'range':[0,100],'tickcolor':'#9abcd6',
                        'tickfont':{'family':'IBM Plex Mono','size':9}},
                'bar':{'color':RED if prediction==1 else GREEN,'thickness':.25},
                'bgcolor':'rgba(5,12,22,.8)',
                'bordercolor':'rgba(0,245,255,.2)',
                'steps':[
                    {'range':[0,30],'color':'rgba(0,255,153,.08)'},
                    {'range':[30,60],'color':'rgba(255,179,71,.08)'},
                    {'range':[60,100],'color':'rgba(255,58,92,.08)'},
                ],
                'threshold':{'line':{'color':ORANGE,'width':2},'thickness':.75,'value':50}
            }
        ))
        fig = apply_theme(fig, "", 280)
        st.plotly_chart(fig, use_container_width=True)

    section_divider()
    info_box("""<b>Model Summary:</b><br>
    Algorithm: Logistic Regression (with StandardScaler pipeline)<br>
    Features Used: 9 (after backward elimination from 15)<br>
    Train/Test Split: 70% / 30% (random_state=42)<br>
    Accuracy: 92.34% &nbsp;|&nbsp; ROC-AUC: 0.9852 (Outstanding)<br>
    The model demonstrates excellent discriminative ability with minimal false negatives,
    which is critical in a medical screening context.""", "success")