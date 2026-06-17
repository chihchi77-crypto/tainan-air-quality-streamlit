from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="臺南 6/16 車流、空氣品質與氣象曲線圖",
    page_icon="🌿",
    layout="wide",
)

DATA_PATH = Path("data/DATA.csv")
FALLBACK_DATA_PATH = Path("data/official_air_traffic_data.csv")

AIR_POLLUTANTS = {
    "AQI": {"column": "aqi", "unit": "", "left_title": "AQI"},
    "PM2.5": {"column": "pm25", "unit": "µg/m³", "left_title": "PM2.5"},
    "PM10": {"column": "pm10", "unit": "µg/m³", "left_title": "PM10"},
    "二氧化硫 SO₂": {"column": "so2", "unit": "ppb", "left_title": "SO₂"},
    "二氧化氮 NO₂": {"column": "no2", "unit": "ppb", "left_title": "NO₂"},
    "一氧化碳 CO": {"column": "co", "unit": "ppm", "left_title": "CO"},
    "臭氧 O₃": {"column": "o3", "unit": "ppb", "left_title": "O₃"},
}

WEATHER_INDICATORS = {
    "溫度": {"column": "temperature_c", "unit": "°C", "left_title": "溫度"},
    "體感溫度": {"column": "apparent_temperature_c", "unit": "°C", "left_title": "體感溫度"},
    "相對濕度 RH": {"column": "rh", "unit": "%", "left_title": "相對濕度"},
    "降雨機率": {"column": "rain_probability", "unit": "%", "left_title": "降雨機率"},
    "風速": {"column": "wind_speed", "unit": "m/s", "left_title": "風速"},
    "風級": {"column": "wind_level", "unit": "級", "left_title": "風級"},
}

NUMERIC_COLUMNS = [
    "aqi", "pm25", "pm10", "so2", "no2", "co", "o3", "co_8h", "o3_8h",
    "nmhc", "wind_speed", "wind_dir", "rh", "total", "car", "motorcycle",
    "bus", "truck", "co2_kg_h", "temperature_c", "apparent_temperature_c",
    "rain_probability"
]
TEXT_COLUMNS = ["weather", "wind_dir_text", "comfort"]


def beaufort_level(speed: float) -> int:
    if pd.isna(speed):
        return 0
    thresholds = [0.2, 1.5, 3.3, 5.4, 7.9, 10.7, 13.8, 17.1, 20.7, 24.4, 28.4, 32.6]
    for level, upper in enumerate(thresholds):
        if speed <= upper:
            return level
    return 12


def wind_degree_to_text(degree: float) -> str:
    if pd.isna(degree):
        return "暫缺"
    directions = ["北風", "東北風", "東風", "東南風", "南風", "西南風", "西風", "西北風"]
    index = int(((degree + 22.5) % 360) // 45)
    return directions[index]


def get_aqi_level(aqi: float):
    if pd.isna(aqi):
        return "無資料", "#f5f5f5", "#999999"
    if aqi <= 50:
        return "良好", "#dff3ee", "#009865"
    if aqi <= 100:
        return "普通", "#fffde3", "#ffde33"
    if aqi <= 150:
        return "對敏感族群不健康", "#fff0df", "#ff9933"
    if aqi <= 200:
        return "對所有族群不健康", "#f8dfe6", "#cc0033"
    if aqi <= 300:
        return "非常不健康", "#eadff2", "#660099"
    return "危害", "#eadde2", "#7e0023"


def show_aqi_legend() -> None:
    st.markdown(
        """
        <div class="aqi-title">空氣品質指標（AQI）</div>
        <div class="aqi-legend">
            <div class="aqi-box good"><strong>良好</strong><br>0～50</div>
            <div class="aqi-box moderate"><strong>普通</strong><br>51～100</div>
            <div class="aqi-box sensitive"><strong>對敏感族群不健康</strong><br>101～150</div>
            <div class="aqi-box unhealthy"><strong>對所有族群不健康</strong><br>151～200</div>
            <div class="aqi-box very-unhealthy"><strong>非常不健康</strong><br>201～300</div>
            <div class="aqi-box hazardous"><strong>危害</strong><br>301～500</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_data_path() -> Path:
    if DATA_PATH.exists():
        return DATA_PATH
    if FALLBACK_DATA_PATH.exists():
        return FALLBACK_DATA_PATH
    return DATA_PATH


def read_data(uploaded_file=None) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        path = get_data_path()
        if not path.exists():
            st.error("找不到正式資料檔。請將 6/16 正式 DATA.csv 放入 data 資料夾，路徑應為 data/DATA.csv。")
            st.stop()
        df = pd.read_csv(path)

    df.columns = df.columns.str.strip()
    if "datetime" not in df.columns:
        st.error("CSV 缺少 datetime 欄位。請確認第一列欄位名稱包含 datetime。")
        st.stop()

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            df[column] = np.nan

    for column in TEXT_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    required = ["aqi", "pm25", "pm10", "so2", "no2", "co", "o3", "rh",
                "total", "car", "motorcycle", "bus", "truck", "co2_kg_h"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"CSV 缺少必要欄位：{', '.join(missing)}")
        st.stop()

    df["wind_level"] = df["wind_speed"].apply(beaufort_level)
    df["wind_dir_text"] = df.apply(
        lambda row: row["wind_dir_text"] if str(row["wind_dir_text"]).strip() else wind_degree_to_text(row["wind_dir"]),
        axis=1,
    )
    return df


def format_date_label(date_value) -> str:
    date_obj = pd.Timestamp(date_value)
    return f"{date_obj.year}年{date_obj.month}月{date_obj.day}日"


def build_date_options(df: pd.DataFrame) -> dict:
    dates = sorted(df["datetime"].dt.date.unique())
    return {format_date_label(date): str(date) for date in dates}


def build_time_options(df: pd.DataFrame, selected_date: str) -> list[str]:
    target = pd.Timestamp(selected_date).date()
    times = df[df["datetime"].dt.date == target]["datetime"].dt.strftime("%H:%M").tolist()
    return sorted(times)


def filter_selected_date(df: pd.DataFrame, selected_date: str) -> pd.DataFrame:
    target = pd.Timestamp(selected_date).date()
    return df[df["datetime"].dt.date == target].copy()


def filter_selected_time(df: pd.DataFrame, selected_date: str, selected_time: str) -> pd.DataFrame:
    target_datetime = pd.to_datetime(f"{selected_date} {selected_time}")
    return df[df["datetime"] == target_datetime].copy()


def add_selected_time_marker(fig: go.Figure, selected_datetime) -> None:
    fig.add_shape(
        type="line", x0=selected_datetime, x1=selected_datetime, y0=0, y1=1,
        xref="x", yref="paper", line=dict(color="#cc0033", width=2, dash="dash")
    )
    fig.add_annotation(
        x=selected_datetime, y=1, xref="x", yref="paper", text="選擇時段",
        showarrow=False, xanchor="left", yanchor="bottom", font=dict(color="#cc0033", size=13)
    )


def add_weather_icons(fig: go.Figure, df: pd.DataFrame, y_col: str) -> None:
    if "weather" not in df.columns:
        return
    icon_map = {"下雨": "🌧️", "雷雨": "⛈️", "多雲": "☁️", "晴": "☀️", "陰": "☁️"}
    y_min, y_max = df[y_col].min(), df[y_col].max()
    y_pos = y_min - (y_max - y_min) * 0.12 if y_max != y_min else y_min
    for _, row in df.iterrows():
        weather_text = str(row.get("weather", "")).strip()
        if weather_text:
            fig.add_annotation(
                x=row["datetime"], y=y_pos, xref="x", yref="y",
                text=icon_map.get(weather_text, "☁️"), showarrow=False, font=dict(size=18)
            )


def build_air_wind_chart(df: pd.DataFrame, pollutant_name: str, selected_datetime=None) -> go.Figure:
    cfg = AIR_POLLUTANTS[pollutant_name]
    y_col = cfg["column"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    unit = cfg["unit"]
    display_name = pollutant_name if unit == "" else f"{pollutant_name}（{unit}）"

    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df[y_col], mode="lines+markers", name=display_name,
        hovertemplate="%{x|%m/%d %H:%M}<br>" + display_name + ": %{y}<extra></extra>"
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["wind_level"], mode="lines+markers", name="風級",
        hovertemplate="%{x|%m/%d %H:%M}<br>風級: %{y}<extra></extra>"
    ), secondary_y=True)

    for _, row in df.iterrows():
        if not pd.isna(row.get("wind_dir")):
            fig.add_annotation(
                x=row["datetime"], y=row["wind_level"], xref="x", yref="y2",
                text="➤", showarrow=False, textangle=float(row["wind_dir"]) - 90,
                font=dict(size=22, color="#2faa4a"), hovertext=f"風向 {row['wind_dir']}°"
            )

    add_selected_time_marker(fig, selected_datetime)
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=30, b=20),
                      hovermode="x unified", legend=dict(orientation="h", y=-0.18, x=0.40))
    fig.update_xaxes(title_text="時間")
    fig.update_yaxes(title_text=cfg["left_title"], secondary_y=False)
    fig.update_yaxes(title_text="風級", secondary_y=True,
                     range=[0, max(3.5, df["wind_level"].max() + 0.5)])
    return fig


def build_weather_chart(df: pd.DataFrame, weather_name: str, selected_datetime=None) -> go.Figure:
    cfg = WEATHER_INDICATORS[weather_name]
    y_col = cfg["column"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    display_name = f"{weather_name}（{cfg['unit']}）"

    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df[y_col], mode="lines+markers+text", name=display_name,
        text=df[y_col].round(1), textposition="top center",
        hovertemplate="%{x|%m/%d %H:%M}<br>" + display_name + ": %{y}<extra></extra>"
    ), secondary_y=False)

    if y_col != "wind_level":
        secondary_col, secondary_name, secondary_title = "wind_level", "風級", "風級"
    else:
        secondary_col, secondary_name, secondary_title = "rain_probability", "降雨機率（%）", "降雨機率"

    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df[secondary_col], mode="lines+markers", name=secondary_name,
        hovertemplate="%{x|%m/%d %H:%M}<br>" + secondary_name + ": %{y}<extra></extra>"
    ), secondary_y=True)

    add_weather_icons(fig, df, y_col)
    add_selected_time_marker(fig, selected_datetime)
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=30, b=20),
                      hovermode="x unified", legend=dict(orientation="h", y=-0.18, x=0.40))
    fig.update_xaxes(title_text="時間")
    fig.update_yaxes(title_text=cfg["left_title"], secondary_y=False)
    fig.update_yaxes(title_text=secondary_title, secondary_y=True)
    return fig


def build_traffic_chart(df: pd.DataFrame, selected_datetime=None) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["datetime"], y=df["total"], name="總車流／偵測次數"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["co2_kg_h"], mode="lines+markers", name="CO₂ kg/h"), secondary_y=True)
    add_selected_time_marker(fig, selected_datetime)
    fig.update_layout(height=420, hovermode="x unified", margin=dict(l=20, r=20, t=30, b=20),
                      legend=dict(orientation="h", y=-0.18, x=0.40))
    fig.update_xaxes(title_text="時間")
    fig.update_yaxes(title_text="車輛偵測次數", secondary_y=False)
    fig.update_yaxes(title_text="kg CO₂/h", secondary_y=True)
    return fig


def show_metric_cards(df: pd.DataFrame) -> None:
    latest = df.iloc[-1]
    aqi_status, aqi_bg, aqi_border = get_aqi_level(latest["aqi"])
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-box" style="background:{aqi_bg}; border-left-color:{aqi_border};">
            <div class="metric-label">空氣品質指標 AQI</div>
            <div class="metric-value">{latest['aqi']:.0f}</div>
            <div class="metric-sub">{aqi_status}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-box" style="background:#f7f8f1;">
            <div class="metric-label">總車流／偵測次數</div>
            <div class="metric-value">{int(latest['total']):,}</div>
            <div class="metric-sub">YOLOv8 車輛偵測結果</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-box" style="background:#eef6f8;">
            <div class="metric-label">CO₂ 推估量</div>
            <div class="metric-value">{latest['co2_kg_h']:.2f}</div>
            <div class="metric-sub">kg CO₂/h</div>
        </div>""", unsafe_allow_html=True)


def show_weather_cards(df: pd.DataFrame) -> None:
    latest = df.iloc[-1]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("溫度", f"{latest.get('temperature_c', np.nan):.1f} °C")
    c2.metric("體感溫度", f"{latest.get('apparent_temperature_c', np.nan):.1f} °C")
    c3.metric("相對濕度", f"{latest.get('rh', np.nan):.0f} %")
    c4.metric("降雨機率", f"{latest.get('rain_probability', np.nan):.0f} %")
    c5.metric("天氣狀況", str(latest.get("weather", "暫缺")))
    c6.metric("舒適度", str(latest.get("comfort", "暫缺")))
    st.caption(
        f"風向：{latest.get('wind_dir_text', '暫缺')}｜"
        f"風向角度：{latest.get('wind_dir', np.nan):.0f}°｜"
        f"風速：{latest.get('wind_speed', np.nan):.1f} m/s｜"
        f"風級：{latest.get('wind_level', np.nan):.0f}"
    )


def explain_latest(df: pd.DataFrame) -> str:
    latest = df.iloc[-1]
    car_ratio = latest["car"] / latest["total"] * 100 if latest["total"] else 0
    heavy_ratio = (latest["bus"] + latest["truck"]) / latest["total"] * 100 if latest["total"] else 0
    aqi_status, _, _ = get_aqi_level(latest["aqi"])
    return f"""
    本時段為 **{latest['datetime'].strftime('%Y/%m/%d %H:%M')}**，
    車輛偵測次數為 **{int(latest['total']):,}**，
    其中汽車占比約 **{car_ratio:.1f}%**，
    公車與卡車等大型車占比約 **{heavy_ratio:.1f}%**。

    本時段 CO₂ 推估量約 **{latest['co2_kg_h']:.2f} kg CO₂/h**。

    同時段空氣品質指標 AQI 為 **{latest['aqi']:.0f}**，
    分級為 **{aqi_status}**；
    PM2.5 為 **{latest['pm25']:.1f} µg/m³**，
    二氧化氮 NO₂ 為 **{latest['no2']:.1f} ppb**，
    一氧化碳 CO 為 **{latest['co']:.2f} ppm**，
    臭氧 O₃ 為 **{latest['o3']:.1f} ppb**。

    氣象條件方面，本時段溫度約 **{latest.get('temperature_c', np.nan):.1f} °C**，
    體感溫度約 **{latest.get('apparent_temperature_c', np.nan):.1f} °C**，
    相對濕度約 **{latest.get('rh', np.nan):.0f}%**，
    天氣狀況為 **{latest.get('weather', '暫缺')}**，
    降雨機率約 **{latest.get('rain_probability', np.nan):.0f}%**，
    風向為 **{latest.get('wind_dir_text', '暫缺')}**，
    舒適度為 **{latest.get('comfort', '暫缺')}**。

    目前僅能進行「車流、空氣品質與氣象資料的時序對照」，
    不應直接宣稱單一路口車流造成光化測站空氣品質變化。
    """


st.markdown("""
<style>
.main-title {font-size: 34px; font-weight: 800; margin-bottom: 0.2rem;}
.date-box {background-color: #f4f7ef; border-left: 6px solid #7bbf43; padding: 14px 18px;
border-radius: 10px; font-size: 20px; font-weight: 700; margin: 12px 0 20px 0;}
.metric-box {padding: 18px 20px; border-radius: 12px; border-left: 7px solid #7bbf43;
box-shadow: 0 4px 14px rgba(0,0,0,0.06); min-height: 112px; margin-bottom: 16px;}
.metric-label {font-size: 15px; color: #555; font-weight: 700;}
.metric-value {font-size: 32px; font-weight: 800; margin-top: 6px;}
.metric-sub {color: #666; font-size: 14px; margin-top: 2px;}
.aqi-title {text-align: center; font-size: 18px; font-weight: 800; padding: 10px;
background: #eeeeee; margin-top: 16px; border-radius: 8px 8px 0 0;}
.aqi-legend {display: grid; grid-template-columns: repeat(6, 1fr); margin-bottom: 22px; border: 1px solid #ffffff;}
.aqi-box {padding: 18px 10px; min-height: 76px; border-right: 2px solid #ffffff; font-size: 16px; line-height: 1.6;}
.aqi-box.good {background: #dff3ee;}
.aqi-box.moderate {background: #fffde3;}
.aqi-box.sensitive {background: #fff0df;}
.aqi-box.unhealthy {background: #f8dfe6;}
.aqi-box.very-unhealthy {background: #eadff2;}
.aqi-box.hazardous {background: #eadde2;}
.stRadio > div {gap: 0.45rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">▌臺南 6/16 空氣品質、氣象與車流曲線圖</div>', unsafe_allow_html=True)
st.caption("正式 DATA.csv｜臺南光化測站空品資料 × 氣象資料 × YOLOv8 車流統計 × 風向風速對照")

with st.sidebar:
    st.header("資料設定")
    st.caption("目前網站預設讀取 data/DATA.csv")
    uploaded_file = st.file_uploader("臨時上傳其他 CSV 資料", type=["csv"])
    st.info("正式成果請將 6/16 正式資料命名為 DATA.csv，並放在專案的 data 資料夾。")

raw_df = read_data(uploaded_file)
date_options = build_date_options(raw_df)

select_col1, select_col2 = st.columns(2)

with select_col1:
    selected_date_label = st.selectbox("選擇資料項目之日期", list(date_options.keys()), index=0)

selected_date_value = date_options[selected_date_label]
time_options = build_time_options(raw_df, selected_date_value)

with select_col2:
    default_index = time_options.index("00:00") if "00:00" in time_options else 0
    selected_time = st.selectbox("選擇資料項目之時間", time_options, index=default_index)

selected_datetime = pd.to_datetime(f"{selected_date_value} {selected_time}")

st.markdown(f'<div class="date-box">時間段：{selected_date_label}　{selected_time}</div>', unsafe_allow_html=True)
show_aqi_legend()

day_df = filter_selected_date(raw_df, selected_date_value)
selected_df = filter_selected_time(raw_df, selected_date_value, selected_time)
st.caption(


if day_df.empty:
    st.error(f"目前沒有 {selected_date_label} 的資料。請確認 CSV 是否包含該日期。")
    st.stop()

if selected_df.empty:
    st.warning(f"目前沒有 {selected_date_label} {selected_time} 的資料。請確認 CSV 是否包含該時段。")
    st.stop()

show_metric_cards(selected_df)
show_weather_cards(selected_df)

indicator_type = st.radio("選擇指標類型", ["空品指標", "天氣指標"], horizontal=True)

if indicator_type == "空品指標":
    selected_air_indicator = st.radio("空品指標", list(AIR_POLLUTANTS.keys()), horizontal=True)
    st.plotly_chart(build_air_wind_chart(day_df, selected_air_indicator, selected_datetime), use_container_width=True)
else:
    selected_weather_indicator = st.radio("天氣指標", list(WEATHER_INDICATORS.keys()), horizontal=True)
    st.plotly_chart(build_weather_chart(day_df, selected_weather_indicator, selected_datetime), use_container_width=True)

tab1, tab2, tab3, tab4 = st.tabs(["交通排放對照表", "氣象資料表", "原始資料表", "研究判讀"])

with tab1:
    st.subheader("車流量與 CO₂ 推估量")
    st.plotly_chart(build_traffic_chart(day_df, selected_datetime), use_container_width=True)

with tab2:
    st.subheader("氣象資料")
    weather_cols = ["datetime", "temperature_c", "apparent_temperature_c", "rh",
                    "weather", "rain_probability", "wind_speed", "wind_level",
                    "wind_dir", "wind_dir_text", "comfort"]
    existing_weather_cols = [col for col in weather_cols if col in day_df.columns]
    st.dataframe(
    day_df[existing_weather_cols],
    use_container_width=True,
    height=900,
)

with tab3:
    st.subheader("原始資料表")
    st.dataframe(
    day_df,
    use_container_width=True,
    height=900,
)

with tab4:
    st.subheader("此時段判讀")
    st.markdown(explain_latest(selected_df))
    st.warning("研究限制：光化測站屬於區域性測站，不等於單一路口即時污染濃度。本研究應定位為趨勢對照與可能性分析，不直接做因果宣稱。")
