import streamlit as st
import pandas as pd
import json
from pathlib import Path

LOG_PATH = Path("logs/interaction_log.jsonl")

st.set_page_config(page_title="Phinite Agent Dashboard", layout="wide")
st.title("📊 Phinite AI Agent Dashboard")

# Load log data
if not LOG_PATH.exists():
    st.warning("No log file found.")
    st.stop()

# Read all log lines
with open(LOG_PATH) as f:
    lines = f.readlines()

# Parse each JSON line into a dict
data = [json.loads(line) for line in lines]
df = pd.DataFrame(data)

# Convert timestamp column
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sidebar filters
st.sidebar.header("Filters")
agent_filter = st.sidebar.multiselect("Agent", df["agent"].unique(), default=df["agent"].unique())
channel_filter = st.sidebar.multiselect("Channel", df["channel"].unique(), default=df["channel"].unique())

# Apply filters
filtered_df = df[df["agent"].isin(agent_filter) & df["channel"].isin(channel_filter)]

# Main stats
st.subheader("🧠 Summary Stats")
col1, col2, col3 = st.columns(3)
col1.metric("Total Queries", len(filtered_df))
col2.metric("Total Cost ($)", f"{filtered_df['cost'].sum():.5f}")
col3.metric("Average Latency (s)", f"{filtered_df['latency'].mean():.2f}")

# Charts
st.subheader("📈 Tokens Used by Agent")
st.bar_chart(filtered_df.groupby("agent")["tokens"].sum())

st.subheader("📉 Latency Distribution")
st.line_chart(filtered_df.set_index("timestamp")["latency"])

# Show raw data
with st.expander("🔍 View Raw Logs"):
    st.dataframe(filtered_df.sort_values("timestamp", ascending=False), use_container_width=True)
