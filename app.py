"""
Tamil Nadu & Puducherry Climate Dashboard (2022)
Run locally:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ------------------------------------------------------------------ page setup
st.set_page_config(page_title="TN & Puducherry Climate 2022", page_icon="🌦️", layout="wide")

st.markdown(
    """
<style>
.stApp {background: radial-gradient(1200px 600px at 10% -10%, #1d3b53 0%, #0b1622 55%, #070d14 100%);}
[data-testid="stSidebar"] {background: #0d1b2a; border-right: 1px solid #1f3a52;}
.hero {padding: 1.4rem 1.6rem; border-radius: 18px; margin-bottom: 1rem;
       background: linear-gradient(120deg, #ff9966 0%, #ff5e62 40%, #7f5af0 100%);
       box-shadow: 0 10px 30px rgba(0,0,0,.35);}
.hero h1 {margin:0; color:white; font-size: 2.1rem;}
.hero p  {margin:.3rem 0 0; color:#fff; opacity:.92;}
.kpi {border-radius: 16px; padding: .9rem .8rem; background: rgba(255,255,255,.06);
      border: 1px solid rgba(255,255,255,.12); backdrop-filter: blur(6px);}
.kpi .l {font-size:.62rem; letter-spacing:0; white-space:nowrap; text-transform:uppercase; color:#9fb6c9;}
.kpi .v {font-size:1.3rem; white-space:nowrap; font-weight:700; color:#fff; line-height:1.2;}
.kpi .s {font-size:.72rem; color:#7fd1b9;}
.insight {padding:.7rem 1rem; margin:.4rem 0; border-left: 4px solid #ff9966;
          background: rgba(255,255,255,.05); border-radius: 8px; color:#e8f0f7;}
</style>
""",
    unsafe_allow_html=True,
)

import inspect
PW = {"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters else {"use_container_width": True}
DW = {"width": "stretch"} if "width" in inspect.signature(st.dataframe).parameters else {"use_container_width": True}

PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=50, b=10))
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASON = {1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer", 5: "Summer",
          6: "SW Monsoon", 7: "SW Monsoon", 8: "SW Monsoon", 9: "SW Monsoon",
          10: "NE Monsoon", 11: "NE Monsoon", 12: "NE Monsoon"}
SEASON_ORDER = ["Winter", "Summer", "SW Monsoon", "NE Monsoon"]
# Labels inferred from the average rainfall of each code in the dataset
COND = {"c3": "Clear / Dry", "c1": "Light rain", "c2": "Moderate rain", "c0": "Heavy rain"}
COND_COLORS = {"Clear / Dry": "#ffd166", "Light rain": "#4cc9f0", "Moderate rain": "#4361ee", "Heavy rain": "#7209b7"}
VARS = ["temp", "sfc_temp", "soilt", "hum", "rain", "gust", "cloud", "pres", "vis", "u", "v", "wind"]


# ------------------------------------------------------------------ data
@st.cache_data
def load():
    d = pd.read_csv("data/district_daily.csv", parse_dates=["date"])
    m = pd.read_csv("data/city_month.csv")
    d["month"] = d.date.dt.month
    d["season"] = d.month.map(SEASON)
    return d, m


daily, citym = load()


def wavg(f, key):
    """Average of VARS grouped by `key`, weighted by number of cities in each district."""
    w = f["n_city"]
    g = f.assign(**{v: f[v] * w for v in VARS}).groupby(key)[VARS + ["n_city"]].sum()
    g[VARS] = g[VARS].div(g["n_city"], axis=0)
    return g


# ------------------------------------------------------------------ sidebar filters
st.sidebar.title("🎛️ Filters")
states = st.sidebar.multiselect("State / UT", sorted(daily.State.unique()), default=sorted(daily.State.unique()))
dist_options = sorted(daily[daily.State.isin(states)].District.unique())
districts = st.sidebar.multiselect("District (empty = all)", dist_options)
dmin, dmax = daily.date.min().date(), daily.date.max().date()
rng = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if isinstance(rng, (tuple, list)) and len(rng) == 2:
    start, end = rng
else:
    start, end = dmin, dmax
st.sidebar.caption("Data: 2022 daily weather, 512 towns, 40 districts.")

f = daily[daily.State.isin(states) & (daily.date.dt.date >= start) & (daily.date.dt.date <= end)]
if districts:
    f = f[f.District.isin(districts)]
if f.empty:
    st.warning("No data for these filters. Please widen the selection.")
    st.stop()

# ------------------------------------------------------------------ hero + KPIs
st.markdown(
    f"""<div class="hero"><h1>🌦️ Tamil Nadu & Puducherry Climate Explorer</h1>
    <p>{start:%d %b %Y} → {end:%d %b %Y} &nbsp;|&nbsp; {f.District.nunique()} district(s) &nbsp;|&nbsp; daily weather, 2022</p></div>""",
    unsafe_allow_html=True,
)

by_day = wavg(f, "date")
by_day["temp_max"] = f.groupby("date").temp_max.max()
by_day["temp_min"] = f.groupby("date").temp_min.min()
cond_tot = f[list(COND)].sum()
rainy_pct = 100 * (cond_tot.c0 + cond_tot.c1 + cond_tot.c2) / cond_tot.sum()
hot_day = by_day.temp_max.idxmax()

kpis = [
    ("Avg temperature", f"{by_day.temp.mean():.1f} °C", f"min {by_day.temp_min.min():.1f} · max {by_day.temp_max.max():.1f}"),
    ("Hottest reading", f"{by_day.temp_max.max():.1f} °C", f"on {hot_day:%d %b}"),
    ("Total rainfall", f"{by_day.rain.sum():,.0f} mm", "area average"),
    ("Avg humidity", f"{by_day.hum.mean():.0f} %", f"cloud cover {by_day.cloud.mean():.0f} %"),
    ("Avg wind", f"{by_day.wind.mean() * 3.6:.1f} km/h", f"peak gust {by_day.gust.max() * 3.6:.0f} km/h"),
    ("Rainy town-days", f"{rainy_pct:.0f} %", "light + moderate + heavy"),
]
cols = st.columns(6)
for c, (l, v, s) in zip(cols, kpis):
    c.markdown(f'<div class="kpi"><div class="l">{l}</div><div class="v">{v}</div><div class="s">{s}</div></div>',
               unsafe_allow_html=True)
st.write("")

t1, t2, t3, t4, t5, t6 = st.tabs(["📈 Trends", "🗺️ Map", "🔥 Compare districts", "⛈️ Conditions & seasons",
                                  "🌬️ Wind & relationships", "💡 Insights & data"])

# ------------------------------------------------------------------ TAB 1 : trends
with t1:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=by_day.index, y=by_day.rain, name="Rainfall (mm)", marker_color="rgba(76,201,240,.55)", secondary_y=True)
    fig.add_scatter(x=by_day.index, y=by_day.temp, name="Daily temp", line=dict(color="rgba(255,153,102,.45)", width=1), secondary_y=False)
    fig.add_scatter(x=by_day.index, y=by_day.temp.rolling(7, min_periods=1).mean(), name="7-day avg temp",
                    line=dict(color="#ff5e62", width=3), secondary_y=False)
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
    fig.update_yaxes(title_text="Rainfall (mm/day)", secondary_y=True, showgrid=False)
    fig.update_layout(title="Temperature vs rainfall through the year", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.15), **PLOT)
    st.plotly_chart(fig, **PW)

    mon_rain = by_day.groupby(by_day.index.month).rain.sum()
    c1, c2 = st.columns(2)
    with c1:
        figm = px.bar(x=[MONTHS[i - 1] for i in mon_rain.index], y=mon_rain.values, color=mon_rain.values,
                      color_continuous_scale="Blues", labels=dict(x="", y="Rainfall (mm)", color="mm"),
                      title="Monthly rainfall")
        figm.update_layout(coloraxis_showscale=False, **PLOT)
        st.plotly_chart(figm, **PW)
    with c2:
        mt = by_day.groupby(by_day.index.month).agg(avg=("temp", "mean"), mx=("temp_max", "max"), mn=("temp_min", "min"))
        x = [MONTHS[i - 1] for i in mt.index]
        figt = go.Figure()
        figt.add_scatter(x=x, y=mt.mx, name="Max", line=dict(color="#ff5e62"))
        figt.add_scatter(x=x, y=mt.mn, name="Min", line=dict(color="#4cc9f0"), fill="tonexty", fillcolor="rgba(255,153,102,.15)")
        figt.add_scatter(x=x, y=mt.avg, name="Avg", line=dict(color="white", dash="dot"))
        figt.update_layout(title="Monthly temperature range (°C)", **PLOT)
        st.plotly_chart(figt, **PW)

# ------------------------------------------------------------------ TAB 2 : map
with t2:
    metric = st.radio("Colour the map by", ["Avg temperature (°C)", "Total rainfall (mm)", "Avg humidity (%)", "Avg gust (m/s)"],
                      horizontal=True)
    months_in = sorted(f.month.unique())
    cm = citym[citym.State.isin(states) & citym.month.isin(months_in)]
    if districts:
        cm = cm[cm.District.isin(districts)]
    cg = cm.groupby(["City", "District", "State"]).agg(
        lat=("latitude", "mean"), lon=("longitude", "mean"), t=("temp_sum", "sum"), h=("hum_sum", "sum"),
        g=("gust_sum", "sum"), r=("rain_sum", "sum"), n=("days", "sum")).reset_index()
    # rain_sum counts every pincode in the town, so divide to get a per-location figure
    cg["Avg temperature (°C)"] = cg.t / cg.n
    cg["Avg humidity (%)"] = cg.h / cg.n
    cg["Avg gust (m/s)"] = cg.g / cg.n
    cg["Total rainfall (mm)"] = cg.r / cg.n * (len(months_in) * 30.4)
    scale = {"Avg temperature (°C)": "Turbo", "Total rainfall (mm)": "Blues", "Avg humidity (%)": "Teal", "Avg gust (m/s)": "Magma"}[metric]
    figmap = px.scatter_map(cg, lat="lat", lon="lon", color=metric, size=cg[metric].clip(lower=0.1), size_max=14,
                            hover_name="City", hover_data={"District": True, "lat": False, "lon": False},
                            color_continuous_scale=scale, zoom=6.2, height=620, map_style="carto-darkmatter")
    figmap.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figmap, **PW)
    st.caption("The map uses the months covered by your date range. Rainfall is the approximate total for those months.")

# ------------------------------------------------------------------ TAB 3 : compare districts
with t3:
    opt = {"Avg temperature (°C)": "temp", "Rainfall (mm/day)": "rain", "Humidity (%)": "hum", "Wind speed (m/s)": "wind"}
    lab = st.selectbox("Compare districts by", list(opt))
    var = opt[lab]
    dm = wavg(f, ["District", "month"])[var].unstack("month")
    dm.columns = [MONTHS[i - 1] for i in dm.columns]
    dm = dm.loc[dm.mean(axis=1).sort_values(ascending=False).index]
    fh = px.imshow(dm, aspect="auto", color_continuous_scale="Turbo" if var == "temp" else "Blues",
                   labels=dict(color=lab), title=f"{lab}: district × month heatmap", height=max(450, 18 * len(dm)))
    fh.update_layout(**PLOT)
    st.plotly_chart(fh, **PW)

    rank = wavg(f, "District")[var].sort_values()
    n = min(10, len(rank))
    r1, r2 = st.columns(2)
    top = rank.tail(n)
    low = rank.head(n)
    r1.plotly_chart(px.bar(x=top.values, y=top.index, orientation="h", title=f"Top {n} highest",
                           color=top.values, color_continuous_scale="OrRd", labels=dict(x=lab, y="")).update_layout(coloraxis_showscale=False, **PLOT),
                    **PW)
    r2.plotly_chart(px.bar(x=low.values, y=low.index, orientation="h", title=f"Top {n} lowest",
                           color=low.values, color_continuous_scale="GnBu", labels=dict(x=lab, y="")).update_layout(coloraxis_showscale=False, **PLOT),
                    **PW)

# ------------------------------------------------------------------ TAB 4 : conditions & seasons
with t4:
    c1, c2 = st.columns(2)
    cdf = pd.DataFrame({"Condition": [COND[k] for k in COND], "Days": [cond_tot[k] for k in COND]})
    fd = px.pie(cdf, names="Condition", values="Days", hole=0.55, color="Condition", color_discrete_map=COND_COLORS,
                title="Weather condition mix (town-days)")
    fd.update_layout(**PLOT)
    c1.plotly_chart(fd, **PW)

    mc = f.groupby("month")[list(COND)].sum()
    mc = mc.div(mc.sum(axis=1), axis=0) * 100
    fs = go.Figure()
    for k, name in COND.items():
        fs.add_bar(x=[MONTHS[i - 1] for i in mc.index], y=mc[k], name=name, marker_color=COND_COLORS[name])
    fs.update_layout(barmode="stack", title="Share of conditions by month (%)", **PLOT)
    c2.plotly_chart(fs, **PW)

    by_day_s = by_day.assign(season=by_day.index.month.map(SEASON))
    ss = by_day_s.groupby("season").agg(rain=("rain", "sum"), temp=("temp", "mean"), hum=("hum", "mean")).reindex(SEASON_ORDER).dropna()
    fz = make_subplots(specs=[[{"secondary_y": True}]])
    fz.add_bar(x=ss.index, y=ss.rain, name="Rainfall (mm)", marker_color="#4361ee", secondary_y=False)
    fz.add_scatter(x=ss.index, y=ss.temp, name="Avg temp (°C)", mode="lines+markers", line=dict(color="#ff5e62", width=3), secondary_y=True)
    fz.update_layout(title="Season scoreboard: rainfall and temperature", **PLOT)
    st.plotly_chart(fz, **PW)
    st.caption("Seasons: Winter (Jan-Feb), Summer (Mar-May), SW Monsoon (Jun-Sep), NE Monsoon (Oct-Dec). "
               "Condition labels were inferred from the average rainfall of each code in the dataset.")

# ------------------------------------------------------------------ TAB 5 : wind & relationships
with t5:
    w1, w2 = st.columns(2)
    fw = f.copy()
    fw["dir"] = (270 - np.degrees(np.arctan2(fw.v, fw.u))) % 360        # direction wind blows FROM
    fw["sector"] = ((fw.dir + 11.25) // 22.5 % 16).astype(int)
    fw["cls"] = pd.cut(fw.wind, [0, 2, 4, 6, 8, 100], labels=["0-2", "2-4", "4-6", "6-8", "8+"], include_lowest=True)
    rose = fw.groupby(["sector", "cls"], observed=False).size().unstack(fill_value=0)
    rose = rose.reindex(range(16), fill_value=0)
    rose = rose / rose.values.sum() * 100
    fr = go.Figure()
    for cls, colr in zip(rose.columns, ["#caf0f8", "#90e0ef", "#00b4d8", "#0077b6", "#03045e"]):
        fr.add_barpolar(r=rose[cls], theta=rose.index * 22.5, name=f"{cls} m/s", marker_color=colr, width=20)
    fr.update_layout(title="Wind rose (where wind comes from)", **PLOT,
                     polar=dict(bgcolor="rgba(0,0,0,0)", angularaxis=dict(direction="clockwise", rotation=90,
                                tickvals=[0, 90, 180, 270], ticktext=["N", "E", "S", "W"]),
                                radialaxis=dict(ticksuffix="%")))
    w1.plotly_chart(fr, **PW)

    corr_cols = {"temp": "Temp", "hum": "Humidity", "rain": "Rain", "cloud": "Cloud", "pres": "Pressure",
                 "wind": "Wind", "gust": "Gust", "vis": "Visibility"}
    cc = by_day[list(corr_cols)].rename(columns=corr_cols).corr()
    fc = px.imshow(cc, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="What moves together? (correlation)")
    fc.update_layout(**PLOT)
    w2.plotly_chart(fc, **PW)

    fsc = px.scatter(by_day.reset_index(), x="hum", y="rain", color="temp", color_continuous_scale="Turbo",
                     labels=dict(hum="Humidity (%)", rain="Rainfall (mm/day)", temp="Temp °C"), title="Humid days are rainier days")
    fsc.update_layout(**PLOT)
    st.plotly_chart(fsc, **PW)

# ------------------------------------------------------------------ TAB 6 : insights & data
with t6:
    st.subheader("Auto-generated insights (they change with your filters)")
    mon_t = by_day.groupby(by_day.index.month).temp.mean()
    mon_r = by_day.groupby(by_day.index.month).rain.sum()
    dist_r = wavg(f, "District").rain
    dist_t = wavg(f, "District").temp
    seas_r = by_day_s.groupby("season").rain.sum()
    ne_share = 100 * seas_r.get("NE Monsoon", 0) / max(seas_r.sum(), 1e-9)
    r_hum = by_day.hum.corr(by_day.rain)
    insights = [
        f"🔥 The hottest month is **{MONTHS[mon_t.idxmax() - 1]}** ({mon_t.max():.1f} °C on average); the coolest is **{MONTHS[mon_t.idxmin() - 1]}** ({mon_t.min():.1f} °C).",
        f"🌧️ **{MONTHS[mon_r.idxmax() - 1]}** is the wettest month with about {mon_r.max():.0f} mm of area-average rain.",
        f"📍 **{dist_r.idxmax()}** gets the most daily rain ({dist_r.max():.2f} mm/day), while **{dist_r.idxmin()}** is the driest ({dist_r.min():.2f} mm/day).",
        f"🌡️ **{dist_t.idxmax()}** is the warmest district ({dist_t.max():.1f} °C) and **{dist_t.idxmin()}** the coolest ({dist_t.min():.1f} °C).",
        f"🌀 The North-East monsoon (Oct-Dec) brings {ne_share:.0f}% of the rain in the selected period.",
        f"💧 Humidity and rainfall have a correlation of {r_hum:.2f}: " + ("humid days tend to be rainy days." if r_hum > 0.3 else "the link is weak in this selection."),
    ]
    for t in insights:
        st.markdown(f'<div class="insight">{t}</div>', unsafe_allow_html=True)

    st.subheader("Data")
    show = f.drop(columns=["month", "season"]).round(2)
    st.dataframe(show, height=300, **DW)
    st.download_button("⬇️ Download filtered data (CSV)", show.to_csv(index=False).encode(), "filtered_weather.csv", "text/csv")
