# streamlit_app_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import glob, io, zipfile
from datetime import datetime
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide", page_title="OHLC Research Dashboard (robust loader)", initial_sidebar_state="expanded")

# ---------- Data loader ----------
@st.cache_data
def try_load_default():
    """Try to find an Excel/CSV in /mnt/data and return (df, path)."""
    # Preferred exact file first
    preferred = "/mnt/data/combined_multisheet_fixed_6dp_v2.xlsx"
    try:
        df = pd.read_excel(preferred, sheet_name="Combined", engine="openpyxl")
        return df, preferred
    except Exception:
        pass

    # Try any xlsx/xls in /mnt/data (Combined sheet first, then sheet 0)
    candidates = sorted(glob.glob("/mnt/data/*.xlsx")) + sorted(glob.glob("/mnt/data/*.xls"))
    for c in candidates:
        try:
            df = pd.read_excel(c, sheet_name="Combined", engine="openpyxl")
            return df, c
        except Exception:
            try:
                df = pd.read_excel(c, sheet_name=0, engine="openpyxl")
                return df, c
            except Exception:
                continue

    # Try CSVs as fallback
    csvs = sorted(glob.glob("/mnt/data/*.csv"))
    for c in csvs:
        try:
            df = pd.read_csv(c)
            return df, c
        except Exception:
            continue

    # Nothing found
    raise FileNotFoundError("No suitable Excel/CSV file found in /mnt/data. Use the uploader in the sidebar.")

def load_from_uploaded(uploaded):
    """Load a user uploaded file (BytesIO)."""
    # try Excel first
    try:
        return pd.read_excel(uploaded, sheet_name="Combined", engine="openpyxl")
    except Exception:
        pass
    try:
        return pd.read_excel(uploaded, sheet_name=0, engine="openpyxl")
    except Exception:
        pass
    # try CSV
    uploaded.seek(0)
    return pd.read_csv(uploaded)

# ---------- Small helpers / indicators ----------
def sma(series, window): return series.rolling(window=window, min_periods=1).mean()
def ema(series, span): return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up/ma_down
    return 100 - (100/(1+rs))

def macd(series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast); slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def true_range(df, high_col, low_col, close_col):
    prev_close = df[close_col].shift(1)
    tr1 = df[high_col] - df[low_col]
    tr2 = (df[high_col] - prev_close).abs()
    tr3 = (df[low_col] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def atr(df, high_col, low_col, close_col, period=14):
    tr = true_range(df, high_col, low_col, close_col)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def hurst_exponent(ts, min_window=10, max_window=None):
    import numpy as _np, scipy.stats as stats
    x = _np.array(ts.dropna().astype(float))
    n = x.size
    if n < min_window:
        return np.nan
    if max_window is None:
        max_window = n // 2
    sizes = _np.floor(_np.logspace(np.log10(min_window), np.log10(max_window), num=20)).astype(int)
    sizes = _np.unique(sizes[sizes>1])
    rs = []
    for s in sizes:
        if s >= n: break
        num_segments = n // s
        seg_rs = []
        for i in range(num_segments):
            seg = x[i*s:(i+1)*s]
            if seg.size < 2: continue
            Z = seg - seg.mean()
            Y = _np.cumsum(Z)
            R = Y.max() - Y.min()
            S = seg.std(ddof=1)
            if S > 0:
                seg_rs.append(R/S)
        if len(seg_rs)>0:
            rs.append(_np.mean(seg_rs))
    if len(rs) < 2:
        return np.nan
    slope, _, _, _, _ = stats.linregress(_np.log(sizes[:len(rs)]), _np.log(rs))
    return slope

def adf_test(series):
    ser = series.dropna().astype(float)
    if ser.size < 10:
        return None
    res = adfuller(ser, autolag='AIC')
    return {'adf_stat': res[0], 'p_value': res[1], 'used_lag': res[2], 'nobs': res[3], 'crit_vals': res[4]}

def plot_acf_pacf(series, nlags=40):
    ser = series.dropna().astype(float)
    acfs = acf(ser, nlags=nlags, fft=True)
    pacfs = pacf(ser, nlags=nlags, method='ld')
    fig_acf = go.Figure(); fig_acf.add_trace(go.Bar(x=list(range(len(acfs))), y=acfs)); fig_acf.update_layout(title='ACF', xaxis_title='lag', yaxis_title='acf')
    fig_pacf = go.Figure(); fig_pacf.add_trace(go.Bar(x=list(range(len(pacfs))), y=pacfs)); fig_pacf.update_layout(title='PACF', xaxis_title='lag', yaxis_title='pacf')
    return fig_acf, fig_pacf

def regime_clustering(df, price_col='close', window=20, n_clusters=2):
    ret = df[price_col].pct_change().fillna(0)
    roll_mean = ret.rolling(window=window).mean().fillna(0)
    roll_vol = ret.rolling(window=window).std().fillna(0)
    features = pd.DataFrame({'ret_mean': roll_mean, 'ret_vol': roll_vol}, index=df.index).dropna()
    scaler = StandardScaler(); X = scaler.fit_transform(features)
    gmm = GaussianMixture(n_components=n_clusters, random_state=0)
    labels = gmm.fit_predict(X)
    regimes = pd.Series(index=features.index, data=labels)
    return regimes, features

# ---------- CoV & report ----------
def compute_cov_and_corr(df):
    d = df.copy()
    d.columns = [c.strip().lower().replace(' ', '_') for c in d.columns]
    date_col = None
    for c in d.columns:
        if any(k in c for k in ('date','time','timestamp')):
            date_col = c; break
    for c in ['open','high','low','close']:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    if 'close' not in d.columns:
        raise ValueError("No 'close' column present.")
    if date_col:
        d[date_col] = pd.to_datetime(d[date_col], errors='coerce')
        d = d.sort_values(by=date_col).reset_index(drop=True)
    d = d.dropna(subset=['close']).copy()
    d['return'] = d['close'].pct_change()
    ret = d['return'].dropna()
    mean_ret = ret.mean(); std_ret = ret.std(ddof=1)
    cov_ret = std_ret / (abs(mean_ret) if mean_ret != 0 else np.nan)
    cov_per_instrument = None
    if 'source_file' in d.columns:
        rows = []
        for name,g in d.groupby('source_file'):
            r = g['close'].pct_change().dropna()
            if r.size>0:
                rows.append({'source_file': name, 'mean_return': r.mean(), 'std_return': r.std(ddof=1),
                             'cov': r.std(ddof=1)/(abs(r.mean()) if r.mean()!=0 else np.nan), 'n': int(r.size)})
        cov_per_instrument = pd.DataFrame(rows)
    ohlc_cols = [c for c in ['open','high','low','close'] if c in d.columns]
    corr_matrix = d[ohlc_cols].corr() if len(ohlc_cols) else None
    summary = {'mean_return': mean_ret, 'std_return': std_ret, 'coefficient_of_variation_return': cov_ret, 'n_returns': int(ret.size)}
    return summary, corr_matrix, cov_per_instrument

def make_report_zip(summary, corr_df, cov_per_instrument):
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('cofr_summary.csv', pd.DataFrame(list(summary.items()), columns=['metric','value']).to_csv(index=False))
        if corr_df is not None:
            zf.writestr('ohlc_correlation.csv', corr_df.to_csv(index=True))
        if cov_per_instrument is not None:
            zf.writestr('cov_per_instrument.csv', cov_per_instrument.to_csv(index=False))
        zf.writestr('README.txt', "CoV report: CoV = std(return)/|mean(return)|. Returns computed as close.pct_change().")
    mem.seek(0)
    return mem

# ---------- App UI and logic ----------
st.title("OHLC Research Dashboard - Robust")

# attempt to load default dataset; if not found show uploader
data_df = None; data_path = None
try:
    data_df, data_path = try_load_default()
    st.sidebar.success(f"Loaded data from: {data_path}")
except FileNotFoundError as e:
    st.sidebar.warning(str(e))
    uploaded = st.sidebar.file_uploader("Upload cleaned combined Excel (.xlsx/.xls) or CSV", type=['xlsx','xls','csv'])
    if uploaded is None:
        st.info("No data available. Please upload the cleaned Excel/CSV in the sidebar or place it in /mnt/data.")
        st.stop()
    try:
        data_df = load_from_uploaded(uploaded)
        st.sidebar.success(f"Loaded uploaded file: {uploaded.name}")
    except Exception as ex:
        st.sidebar.error(f"Failed to parse uploaded file: {ex}")
        st.stop()

# normalize df
df = data_df.copy()
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

# find date column
date_col = None
for c in df.columns:
    if any(k in c for k in ('date','time','timestamp')):
        date_col = c; break
if date_col is None:
    st.error("No date-like column found (headers containing 'date'/'time'/'timestamp').")
    st.stop()

df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)

# controls
instruments = sorted(df['source_file'].unique()) if 'source_file' in df.columns else ["All"]
selected_instrument = st.sidebar.selectbox("Instrument", ["All"] + instruments)
value_column = st.sidebar.selectbox("Price column", [c for c in df.columns if c in ["open","high","low","close"]])
start_date = st.sidebar.date_input("Start date", df[date_col].min().date())
end_date = st.sidebar.date_input("End date", df[date_col].max().date())

# filter
data = df.copy()
if selected_instrument != "All":
    data = data[data['source_file'] == selected_instrument]
data = data.set_index(date_col)
data = data.loc[start_date:end_date].reset_index()

price = data[value_column].astype(float)
data['ret'] = price.pct_change(); data['logret'] = np.log(price).diff()

# advanced metrics
adf_price = adf_test(price); adf_returns = adf_test(data['ret'])
fig_acf, fig_pacf = plot_acf_pacf(price, nlags=40)
hurst = hurst_exponent(price, min_window=10)
regimes, features = regime_clustering(data, price_col=value_column, window=20, n_clusters=2)
data = data.join(regimes.rename('regime'), how='left')

# KPIs & charts
c1, c2, c3 = st.columns(3)
c1.metric("Start", str(data[date_col].min()))
c2.metric("End", str(data[date_col].max()))
c3.metric(f"Mean {value_column}", f"{price.mean():.6f}")

fig = go.Figure()
for r in sorted(data['regime'].dropna().unique()):
    seg = data[data['regime'] == r]
    fig.add_trace(go.Scatter(x=seg[date_col], y=seg[value_column], mode='lines', name=f"Regime {int(r)}"))
fig.update_layout(title=f"{value_column.upper()} segmented by regimes", height=450)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Stationarity & Autocorrelation")
st.write("ADF (price):", adf_price)
st.write("ADF (returns):", adf_returns)
col_acf, col_pacf = st.columns(2)
with col_acf: st.plotly_chart(fig_acf, use_container_width=True)
with col_pacf: st.plotly_chart(fig_pacf, use_container_width=True)

st.subheader("Hurst exponent")
st.write(f"Hurst ≈ {hurst:.4f}")

st.subheader("Regime features (rolling mean & vol)")
st.dataframe(features.join(regimes.rename('regime')).tail(200))
st.download_button("Download regimes CSV", features.join(regimes.rename('regime')).reset_index().to_csv(index=False), "regimes.csv", "text/csv")

# CoV report
st.markdown("---")
st.header("Generate CoV report (zip)")

if st.button("Generate CoV Report ZIP"):
    try:
        summary, corr_df, cov_per_instrument = compute_cov_and_corr(data)
        zipbuf = make_report_zip(summary, corr_df, cov_per_instrument)
        st.success("Report generated.")
        st.download_button("Download CoV report (ZIP)", zipbuf, "cofr_report.zip", "application/zip")
        st.subheader("Summary"); st.table(pd.DataFrame(list(summary.items()), columns=['metric','value']))
        if corr_df is not None:
            st.subheader("OHLC correlation matrix"); st.dataframe(corr_df)
        if cov_per_instrument is not None:
            st.subheader("CoV per instrument"); st.dataframe(cov_per_instrument)
    except Exception as e:
        st.error(f"Failed to create CoV report: {e}")
