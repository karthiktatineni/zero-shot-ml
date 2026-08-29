import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.express as px

# Configuration
st.set_page_config(page_title="Zero-Shot PFN Dashboard", layout="wide")
st.title("Zero-Shot PFN Training Dashboard")

# Setup autorefresh (every 5 seconds)
# Streamlit >= 1.35 supports st.experimental_rerun, but easiest is st_autorefresh if installed,
# otherwise we can use st.empty() and time.sleep in a loop.
# Let's use a native Streamlit loop if possible, or just expect the user to use an autorefresh component.
# Actually, Streamlit 1.37+ has `st.fragment(run_every="5s")`. Let's try that if available, otherwise fallback.
# For simplicity and compatibility, we'll just read the file and use a manual button, 
# or use st.rerun() with a sleep if a checkbox is checked.

run_dir = st.text_input("Run Directory", "runs/main_run")
metrics_file = Path(run_dir) / "metrics.jsonl"

auto_refresh = st.checkbox("Auto-Refresh (5s)", value=True)

@st.cache_data(ttl=5)
def load_data(filepath: Path):
    if not filepath.exists():
        return pd.DataFrame()
    
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return pd.DataFrame(records)

df = load_data(metrics_file)

if df.empty:
    st.warning(f"No metrics found in {metrics_file}. Start training to generate data.")
else:
    # Latest metrics
    latest = df.iloc[-1]
    
    current_step = int(latest['step'])
    total_steps = st.number_input("Total Steps (for progress bar)", min_value=1, value=62500)
    progress_val = min(current_step / total_steps, 1.0)
    st.progress(progress_val, text=f"Training Progress: {current_step} / {total_steps} steps ({progress_val:.1%})")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Step", f"{current_step}")
    col2.metric("Loss", f"{latest['loss']:.4f}")
    col3.metric("Learning Rate", f"{latest['lr']:.2e}")
    col4.metric("Speed", f"{latest['speed_eps']:.1f} ep/s")
    
    st.subheader("Training Loss")
    # Smoothed loss
    df['smoothed_loss'] = df['loss'].rolling(window=max(1, len(df)//20)).mean()
    
    fig_loss = px.line(df, x='step', y=['loss', 'smoothed_loss'], 
                       labels={'value': 'Loss', 'step': 'Step'},
                       title='Training Loss over Time')
    st.plotly_chart(fig_loss, use_container_width=True)
    
    st.subheader("Learning Rate")
    fig_lr = px.line(df, x='step', y='lr', log_y=True,
                     labels={'lr': 'Learning Rate', 'step': 'Step'},
                     title='Learning Rate Schedule')
    st.plotly_chart(fig_lr, use_container_width=True)

if auto_refresh:
    import time
    time.sleep(5)
    st.rerun()
