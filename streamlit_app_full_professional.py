# streamlit_app_full_professional.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import glob, io, zipfile, os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# --- PROFESSIONAL LIBRARIES (expected installed) ---
try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    import statsmodels.api as sm
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except Exception:
    HAS_PMDARIMA = False

from scipy.stats import entropy as shannon_entropy
# reportlab imports (for PDF)
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table as RLTable
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import letter
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

st.set_page_config(layout="wide", page_title="Research Dashboard — Professional")

# ---------------------------
# Utilities
# ---------------------------
def load_combined(path="/mnt/data/combined_fixed.xlsx"):
    if Path(path).exists():
        df = pd.read_excel(path, engine="openpyxl")
        return df
    # fallback: try to combine CSVs
    csvs = sorted(glob.glob("/mnt/data/*.csv"))
    if csvs:
        frames = []
        for f in csvs:
            df = pd.read_csv(f)
            df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
            date_col = next((c for c in df.columns if 'date' in c or 'time' in c), None)
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df['source_file'] = Path(f).stem
            frames.append(df)
        comb = pd.concat(frames, ignore_index=True, sort=False)
        date_col = next((c for c in comb.columns if 'date' in c or 'time' in c), None)
        if date_col:
            comb = comb.dropna(subset=[date_col])
            comb = comb.sort_values(by=date_col).reset_index(drop=True)
        return comb
    raise FileNotFoundError("No combined_fixed.xlsx or CSVs found in /mnt/data")

def ensure_price_col(df, col):
    """Return a numeric Series for the price column, or stop with an error."""
    if col not in df.columns:
        st.error(f"Column {col} not found.")
        st.stop()
    ser = pd.to_numeric(df[col], errors='coerce')
    if ser.dropna().empty:
        st.error(f"Column {col} has no numeric values.")
        st.stop()
    return ser

def compute_shannon_entropy(returns, bins=50):
    r = returns.dropna()
    if r.empty:
        return np.nan
    hist, edges = np.histogram(r, bins=bins, density=True)
    hist = hist[hist>0]
    if hist.size == 0:
        return 0.0
    return shannon_entropy(hist)

def compute_returns(series):
    return series.pct_change().dropna()

# ---------------------------
# SAFE ADF function (fixes NameError + constant-series error)
# ---------------------------
def adf_test(series):
    """
    Safe wrapper for augmented dickey-fuller test.
    Returns a dict with results, or a human-readable string if not applicable.
    """
    if not HAS_STATSMODELS:
        return "statsmodels not installed"

    ser = pd.to_numeric(series, errors='coerce').dropna()
    if ser.size < 10:
        return "Not enough data for ADF (min 10 non-null samples)"
    if ser.nunique() <= 1:
        return "ADF not applicable: series is constant"

    try:
        res = adfuller(ser, autolag='AIC')
        return {
            'adf_stat': res[0],
            'p_value': res[1],
            'used_lag': res[2],
            'nobs': res[3],
            'critical_values': res[4]
        }
    except ValueError as e:
        return f"ADF ValueError: {e}"
    except Exception as e:
        return f"ADF error: {e}"

# ---------------------------
# Analytics modules
# ---------------------------
def run_garch(returns, p=1, q=1):
    if not HAS_ARCH:
        return None, "arch not installed"
    r = (returns - returns.mean()) * 100  # scale to percent for arch
    am = arch_model(r, vol='Garch', p=p, q=q, mean='Zero', dist='normal')
    res = am.fit(disp='off')
    cond_vol = res.conditional_volatility / 100.0
    return cond_vol, res

def run_pca_on_returns(df, cols, n_components=3):
    if not HAS_SKLEARN:
        return None, "sklearn not installed"
    X = df[cols].pct_change().dropna().fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(Xs)
    return {'pca': pca, 'components': pcs, 'explained_variance': pca.explained_variance_ratio_, 'index': X.index}

def pca_volatility_clusters(df, price_col, n_clusters=3):
    if not HAS_SKLEARN:
        return None, "sklearn not installed"
    ret = df[price_col].pct_change().fillna(0)
    feat = pd.DataFrame({
        'v_short': ret.rolling(10).std().fillna(0),
        'v_long': ret.rolling(60).std().fillna(0)
    }).dropna()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(feat)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(Xs)
    labels = pd.Series(kmeans.labels_, index=feat.index)
    return labels, feat

def regime_transition_matrix(regimes_series):
    r = regimes_series.dropna().astype(int)
    if r.empty:
        return None
    prev = r.shift(1).dropna()
    curr = r.loc[prev.index]
    mat = pd.crosstab(prev, curr, normalize='index')
    return mat

def var_fit_and_forecast(df, cols, steps=5):
    if not HAS_STATSMODELS:
        return None, "statsmodels not installed"
    data = df[cols].pct_change().dropna()
    if data.shape[0] < 10:
        return None, "Not enough data for VAR"
    model = VAR(data)
    sel = model.select_order(15)
    lag = sel.selected_orders.get('aic', 1) if hasattr(sel, "selected_orders") else 1
    res = model.fit(lag)
    fc = res.forecast(data.values[-res.k_ar:], steps=steps)
    # create forecast index by extending the datetime index by frequency if present
    try:
        last = data.index[-1]
        freq = pd.infer_freq(data.index)
        if freq is not None:
            idx = pd.date_range(start=last, periods=steps+1, freq=freq)[1:]
        else:
            idx = pd.RangeIndex(start=len(data), stop=len(data)+steps)
    except Exception:
        idx = pd.RangeIndex(start=len(data), stop=len(data)+steps)
    fc_df = pd.DataFrame(fc, index=idx, columns=data.columns)
    return fc_df, res

def auto_arima_forecast(series, steps=10):
    if not HAS_PMDARIMA:
        return None, "pmdarima not installed"
    s = series.dropna()
    if len(s) < 20:
        return None, "Not enough data for auto_arima"
    model = pm.auto_arima(s, seasonal=False, stepwise=True, suppress_warnings=True)
    fc = model.predict(n_periods=steps)
    idx = pd.RangeIndex(start=len(s), stop=len(s)+steps)
    return pd.Series(fc, index=idx), model

# ---------------------------
# PDF Report (optional)
# ---------------------------
def generate_pdf_report(summary_text, tables, out_path="/mnt/data/research_report_professional.pdf"):
    if not HAS_REPORTLAB:
        raise RuntimeError("reportlab not installed")
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(out_path, pagesize=letter)
    story = []
    story.append(Paragraph("Research Dashboard Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 12))
    for name, df in tables.items():
        story.append(Paragraph(name, styles['Heading3']))
        data = [df.columns.tolist()] + df.head(20).values.tolist()
        story.append(RLTable(data))
        story.append(Spacer(1, 12))
    doc.build(story)
    return out_path

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Research Dashboard — Professional (fixed ADF)")

# Load file (auto load from /mnt/data or show uploader)
df = None
default_path = "/mnt/data/combined_fixed.xlsx"
if Path(default_path).exists():
    try:
        df = pd.read_excel(default_path, engine="openpyxl")
        st.sidebar.success("Loaded combined_fixed.xlsx from /mnt/data")
    except Exception as e:
        st.sidebar.warning(f"Failed to load default combined file: {e}")

if df is None:
    uploaded = st.sidebar.file_uploader("Upload combined_fixed.xlsx or CSV", type=["xlsx","xls","csv"])
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded, engine="openpyxl")
            st.sidebar.success(f"Loaded {uploaded.name}")
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
            st.stop()
    else:
        st.info("No combined file found — please upload or place combined_fixed.xlsx in /mnt/data")
        st.stop()

# normalize columns & index
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
date_col = next((c for c in df.columns if 'date' in c or 'time' in c), None)
if date_col is None:
    st.error("No date/time column found in dataset.")
    st.stop()

df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
df.set_index(date_col, inplace=True)

# sidebar filters
st.sidebar.header("Filters")
instruments = ["All"] + (sorted(df['source_file'].unique()) if 'source_file' in df.columns else [])
instr = st.sidebar.selectbox("Instrument", instruments)
if instr != "All":
    df_plot = df[df['source_file'] == instr].copy()
else:
    df_plot = df.copy()

price_choices = [c for c in df_plot.columns if c in ['open','high','low','close']]
if not price_choices:
    st.error("No OHLC columns found.")
    st.stop()

price_col = st.sidebar.selectbox("Price column for analysis", price_choices, index=price_choices.index('close') if 'close' in price_choices else 0)
# date range
start_date = st.sidebar.date_input("Start date", df_plot.index.min().date())
end_date = st.sidebar.date_input("End date", df_plot.index.max().date())
df_plot = df_plot.loc[start_date:end_date]

# ensure numeric price column
price_ser = ensure_price_col(df_plot, price_col)
df_plot[price_col] = price_ser

# price plot
st.subheader("Price Series")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[price_col], mode='lines', name=price_col))
fig.update_layout(height=350)
st.plotly_chart(fig, use_container_width=True)

# returns and stats
returns = df_plot[price_col].pct_change().dropna()
st.subheader("Basic Statistics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Observations", f"{len(returns)}")
col2.metric("Mean Return", f"{returns.mean():.6f}")
col3.metric("Std Return", f"{returns.std():.6f}")
col4.metric("Shannon Entropy", f"{compute_shannon_entropy(returns):.6f}")

# ADF
st.subheader("ADF Test")
adf_res = adf_test(df_plot[price_col])
st.write(adf_res)

# ACF/PACF
st.subheader("ACF / PACF")
if HAS_STATSMODELS:
    try:
        acf_vals = acf(pd.to_numeric(df_plot[price_col], errors='coerce').dropna(), nlags=40)
        pacf_vals = pacf(pd.to_numeric(df_plot[price_col], errors='coerce').dropna(), nlags=40)
        fig_acf = go.Figure(); fig_acf.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals)); fig_acf.update_layout(title='ACF')
        fig_pacf = go.Figure(); fig_pacf.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals)); fig_pacf.update_layout(title='PACF')
        st.plotly_chart(fig_acf, use_container_width=True)
        st.plotly_chart(fig_pacf, use_container_width=True)
    except Exception as e:
        st.warning(f"ACF/PACF failed: {e}")
else:
    st.warning("statsmodels not installed — ACF/PACF disabled")

# Hurst
st.metric("Hurst exponent (approx)", hurst_exponent(df_plot[price_col]) if 'hurst_exponent' in globals() else "N/A")

# PCA on returns
st.subheader("PCA on Returns")
if HAS_SKLEARN:
    try:
        pca_res = run_pca_on_returns(df_plot, price_choices, n_components=min(5, len(price_choices)))
        if isinstance(pca_res, dict):
            ev = pca_res['explained_variance']
            st.write("Explained variance ratio:", np.round(ev,4))
            figp = go.Figure(); figp.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(ev))], y=ev))
            st.plotly_chart(figp, use_container_width=True)
    except Exception as e:
        st.warning(f"PCA failed: {e}")
else:
    st.warning("sklearn not installed — PCA disabled")

# PCA-vol clusters
st.subheader("PCA-Vol Clusters")
if HAS_SKLEARN:
    try:
        labels, feat = pca_volatility_clusters(df_plot, price_col, n_clusters=3)
        if labels is not None:
            st.dataframe(feat.tail(10))
            st.write("Cluster counts:", labels.value_counts())
            df_plot['vol_cluster'] = labels
    except Exception as e:
        st.warning(f"PCA-vol clustering failed: {e}")
else:
    st.warning("sklearn not installed — Vol clustering disabled")

# Regime clustering & transition matrix
st.subheader("Regime Clustering")
if HAS_SKLEARN:
    try:
        ret = df_plot[price_col].pct_change().fillna(0)
        feat = pd.DataFrame({'mean': ret.rolling(20).mean().fillna(0), 'vol': ret.rolling(20).std().fillna(0)})
        feat = feat.dropna()
        if len(feat) > 10:
            scaler = StandardScaler(); Xs = scaler.fit_transform(feat)
            try:
                gmm = GaussianMixture(n_components=2, random_state=0).fit(Xs)
                labels = gmm.predict(Xs)
            except Exception:
                km = KMeans(n_clusters=2, random_state=0).fit(Xs)
                labels = km.labels_
            regimes = pd.Series(labels, index=feat.index)
            regimes_full = pd.Series(np.nan, index=df_plot.index)
            regimes_full.loc[regimes.index] = regimes
            df_plot['regime'] = regimes_full
            st.write("Regime counts:", regimes_full.value_counts(dropna=True))
            trans = regime_transition_matrix(regimes_full)
            if trans is not None:
                st.subheader("Transition Matrix")
                st.dataframe(trans)
    except Exception as e:
        st.warning(f"Regime clustering failed: {e}")
else:
    st.warning("sklearn not installed — regime clustering disabled")

# GARCH
st.subheader("GARCH (1,1)")
if HAS_ARCH:
    if len(returns) < 50:
        st.info("Not enough data for GARCH (<50).")
    else:
        try:
            cond_vol, garch_res = run_garch(returns)
            if cond_vol is not None:
                # align cond_vol with df_plot index
                cond_vol_full = cond_vol.reindex(df_plot.index).fillna(method='ffill')
                st.line_chart(cond_vol_full)
                st.text(garch_res.summary().as_text())
        except Exception as e:
            st.warning(f"GARCH failed: {e}")
else:
    st.warning("arch not installed — GARCH disabled")

# VAR forecasting
st.subheader("VAR Forecast")
var_cols = st.multiselect("Select columns for VAR (use OHLC)", price_choices, default=[price_col])
steps = st.sidebar.number_input("Forecast steps", min_value=1, max_value=60, value=10)
if HAS_STATSMODELS and var_cols:
    try:
        fc_df, var_res = var_fit_and_forecast(df_plot, var_cols, steps=int(steps))
        if fc_df is not None:
            st.dataframe(fc_df)
            st.line_chart(fc_df)
    except Exception as e:
        st.warning(f"VAR failed: {e}")
else:
    st.warning("statsmodels not installed or no columns selected — VAR disabled")

# Auto-ARIMA forecast
st.subheader("Auto ARIMA Forecast (single series)")
if HAS_PMDARIMA:
    try:
        s = df_plot[price_col].pct_change().dropna()
        if len(s) >= 30:
            arima_fc, arima_model = auto_arima_forecast(s, steps=int(steps))
            if arima_fc is not None:
                st.line_chart(arima_fc)
                st.write("Auto-ARIMA done.")
        else:
            st.info("Need >=30 points for auto-ARIMA.")
    except Exception as e:
        st.warning(f"Auto-ARIMA failed: {e}")
else:
    st.warning("pmdarima not installed — auto-ARIMA disabled")

# Rolling entropy
st.subheader("Rolling Entropy (window=60)")
try:
    roll_entropy = df_plot[price_col].pct_change().rolling(60).apply(lambda x: compute_shannon_entropy(pd.Series(x.dropna())), raw=False)
    st.line_chart(roll_entropy)
except Exception as e:
    st.warning(f"Entropy computation failed: {e}")

# PDF export
st.subheader("Export PDF report")
if st.button("Generate PDF"):
    if not HAS_REPORTLAB:
        st.error("reportlab not installed - PDF export disabled")
    else:
        summary_text = f"Dataset rows: {len(df_plot)}; Price column: {price_col}; Date range: {df_plot.index.min()} to {df_plot.index.max()}."
        tables = {"Basic stats": pd.DataFrame({'mean_return':[returns.mean()], 'std_return':[returns.std()]})}
        try:
            pdf_path = generate_pdf_report(summary_text, tables)
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f, file_name="research_report_professional.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")

st.success("App loaded (professional build).")
