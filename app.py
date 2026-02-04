from src.ai_engine import run_forecast
from src.data_engine import get_all_stock_data, get_live_news
from src.risk_engine import calculate_var
from src.sentiment_engine import get_headline_sentiment
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
# We only need the indicator function now


# Initialize session state for theme if it doesn't exist
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True  # Default to Dark Mode

# Sidebar toggle for Theme
with st.sidebar:
    st.title("Settings")
    theme_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    st.session_state.dark_mode = theme_toggle

# Define your color palettes for each mode
if st.session_state.dark_mode:
    # DARK PALETTE
    bg_color = "#0f172a"
    text_color = "#f8fafc"
    card_bg = "#1e293b"
    border_color = "#334155"
    accent_color = "#38bdf8"  # Sky Blue
else:
    # WHITE PALETTE
    bg_color = "#ffffff"
    text_color = "#1e293b"
    card_bg = "#f1f5f9"
    border_color = "#e2e8f0"
    accent_color = "#0284c7"  # Darker Blue


# Set Page Config
st.set_page_config(page_title="VisionAI Stock Monitor",
                   layout="wide", page_icon="📈")


st.markdown(f"""
    <style>
    /* 1. Main App Background and Global Text */
    .stApp {{
        background-color: {bg_color};
        color: {text_color} !important;
    }}

    /* 2. Universal Text Visibility (H1 to P) */
    h1, h2, h3, h4, p, span, label {{
        color: {text_color} !important;
    }}

    /* 3. Metric Card Customization */
    [data-testid="stMetric"] {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        border-radius: 12px;
        padding: 15px;
    }}

    /* Fix: Ensure the Metric Label (top) and Value (bottom) are visible */
    [data-testid="stMetricLabel"] p, [data-testid="stMetricValue"] div {{
        color: {text_color} !important;
    }}

    /* 4. News Cards */
    .news-card {{
        background-color: {card_bg};
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid {accent_color};
        margin-bottom: 10px;
        color: {text_color} !important;
    }}

    /* 5. Same-Size Tabs with Theme-Aware Colors */
    .stTabs [data-baseweb="tab-list"] {{
        display: flex;
        gap: 0px;
    }}

    .stTabs [data-baseweb="tab"] {{
        flex-grow: 1;
        text-align: center;
        background-color: {card_bg} !important;
        border: 1px solid {border_color};
    }}

    /* Fix: Tab text color (Active and Inactive) */
    .stTabs [data-baseweb="tab"] div {{
        color: {text_color} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

st.title(" AI STOCK INTELLIGENCE SYSTEM", anchor=None,
         help=None, text_alignment="center")

# --- INPUT SECTION ---
ticker = st.text_input(
    "Enter Asset Ticker (e.g., RELIANCE.NS)", value="RELIANCE.NS").upper()
analyze_btn = st.button("RUN SYSTEM ANALYSIS", use_container_width=True)

if analyze_btn:
    df_all = get_all_stock_data(ticker)

    if df_all is None or df_all.empty:
        st.error(f"❌ Error: No data found for {ticker}.")
    else:
        # 1. FETCH DATA
        forecast_1d, _, forecast_conf = run_forecast(df_all)
        news_list = get_live_news(ticker)
        curr_p = float(df_all['Close'].iloc[-1])
        target_p = float(forecast_1d) if forecast_1d is not None else curr_p

        # 2. TOP METRICS
        st.subheader(f"Results for {ticker}")
        m1, m2 = st.columns(2)
        m1.metric("Live Price", f"₹{curr_p:,.2f}")

        delta_p = target_p - curr_p
        m2.metric("AI Target (Tomorrow)",
                  f"₹{target_p:,.2f}", delta=f"{delta_p:,.2f}")

        st.divider()

        # 3. TABS
        tab1, tab2, tab3 = st.tabs(
            ["📊 Price Chart", "📰 Top Headlines", "🤖 Prediction Reliability"])

        with tab1:
            # 1. Create the figure first
            df_plot = df_all.tail(60).copy()
            fig_cand = go.Figure(data=[go.Candlestick(
                x=df_plot.index, open=df_plot['Open'], high=df_plot['High'],
                low=df_plot['Low'], close=df_plot['Close'], name="Price"
            )])

            # --- PASTE THE CHART THEME SYNC CODE HERE ---
            # --- CHART THEME SYNC ---
            # Choose a Plotly template based on your toggle
            chart_template = "plotly_dark" if st.session_state.dark_mode else "plotly_white"

            fig_cand.update_layout(
                template=chart_template,
                paper_bgcolor=bg_color,  # Use your dynamic bg_color variable
                plot_bgcolor=bg_color,
                # Force axis numbers to follow theme
                font=dict(color=text_color),
                xaxis=dict(gridcolor=border_color),
                yaxis=dict(gridcolor=border_color),
                height=500,
                xaxis_rangeslider_visible=False
            )
            chart_template = "plotly_dark" if st.session_state.dark_mode else "plotly_white"

            fig_cand.update_layout(
                template=chart_template,
                paper_bgcolor=bg_color,  # Match app background
                plot_bgcolor=bg_color,
                font=dict(color=text_color),  # Axis labels visibility
                xaxis=dict(gridcolor=border_color),
                yaxis=dict(gridcolor=border_color),
                height=500,
                xaxis_rangeslider_visible=False
            )

            # 3. Finally, display the finished chart
            st.plotly_chart(fig_cand, use_container_width=True, theme=None)

        with tab2:
            st.subheader("📰 Latest Headlines with Sentiment")
            from src.sentiment_engine import get_headline_sentiment

            if news_list:
                for item in news_list[:5]:
                    # Get the dot and the raw score
                    dot, score = get_headline_sentiment(item)

                    # Display the headline with its sentiment indicator
                    st.markdown(f"""
                    <div class="news-card">
                        <span style="font-size: 1.2rem; margin-right: 10px;">{dot}</span>
                        {item}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No headlines found.")

        with tab3:
            st.subheader("🤖 Prediction Reliability")

            # 1. Color Logic for the Score
            if forecast_conf > 90:
                conf_color = "#22c55e"  # Vibrant Green
                status_text = "High Confidence"
            elif forecast_conf > 75:
                conf_color = "#f59e0b"  # Amber/Yellow
                status_text = "Moderate Confidence"
            else:
                conf_color = "#ef4444"  # Signal Red
                status_text = "Low Confidence (Caution)"

            # 2. Display the Score in a Visual Card
            st.markdown(f"""
                <div style="background-color: {card_bg}; padding: 25px; border-radius: 15px; border: 2px solid {conf_color}; text-align: center; margin-bottom: 25px;">
                    <h3 style="color: {text_color}; margin-bottom: 10px; opacity: 0.8;">{status_text}</h3>
                    <h1 style="color: {conf_color}; font-size: 4rem; margin: 0;">{forecast_conf:.1f}%</h1>
                    <p style="color: {text_color}; margin-top: 10px; font-weight: bold;">Reliability Rating for {ticker}</p>
                </div>
            """, unsafe_allow_html=True)

            st.divider()

            # 3. Dynamic News Headlines
            st.subheader("📰 Latest Headlines")
            if news_list:
                from src.sentiment_engine import get_headline_sentiment
                for item in news_list[:5]:
                    dot, _ = get_headline_sentiment(item)
                    st.markdown(
                        f'<div class="news-card"><span style="margin-right:10px;">{dot}</span>{item}</div>', unsafe_allow_html=True)
            else:
                st.info("No headlines found for this asset.")
else:
    st.info("Enter a ticker to start.")
