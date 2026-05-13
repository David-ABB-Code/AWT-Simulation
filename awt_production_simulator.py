
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict

# ════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════
st.set_page_config(page_title="AWT Production Simulator", layout="wide", initial_sidebar_state="expanded")

# ABB Colours
C_RED = '#FF000F'
C_LIL = '#6764f6'
C_RED1 = '#ff957e'
C_LIL1 = '#93a1ff'
C_LIL2 = '#e4e7ff'

# ════════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ════════════════════════════════════════════════════════════════
def run_simulation(cycle_times, resources, daily_minutes, sim_days, product_mix, inter_arrival):
    """
    Runs the discrete event simulation.
    """
    np.random.seed(42)
    
    SIM_TIME = sim_days * daily_minutes
    N = int(SIM_TIME / inter_arrival) + 120
    
    # Generate product types and launch times
    ptypes = ['AWT210' if np.random.random() < product_mix else 'AWT420/424' for _ in range(N)]
    launch_times = [i * inter_arrival for i in range(N)]
    
    def process_stage(stage, units_ready):
        cap = resources[stage]
        ct = cycle_times[stage]
        avail = [0.0] * cap
        finish = {}
        waits = []
        busy = 0.0
        
        for ready_time, idx in sorted(units_ready, key=lambda x: x[0]):
            best = min(range(cap), key=lambda x: avail[x])
            start = max(ready_time, avail[best])
            waits.append(start - ready_time)
            f = start + ct
            avail[best] = f
            finish[idx] = f
            busy += ct
        
        return finish, waits, busy
    
    # Stage 1.1: Door Assembly
    f_11, w_11, b_11 = process_stage('1.1', [(launch_times[i], i) for i in range(N)])
    
    # Kitting 4.1
    f_kit, w_kit, b_kit = process_stage('4.1', [(f_11[i], i) for i in range(N)])
    
    # Path A: AWT210
    a210 = [i for i in range(N) if ptypes[i] == 'AWT210']
    f_21a, w_21a, b_21a = process_stage('2.1A', [(f_11[i], i) for i in a210])
    f_31a, w_31a, b_31a = process_stage('3.1A', [(f_21a[i], i) for i in a210])
    
    # Path B: AWT420/424
    a424 = [i for i in range(N) if ptypes[i] == 'AWT420/424']
    f_22b, w_22b, b_22b = process_stage('2.2B', [(f_11[i], i) for i in a424])
    f_32b, w_32b, b_32b = process_stage('3.2B', [(f_22b[i], i) for i in a424])
    
    # Merge paths
    path_done = {**{i: f_31a[i] for i in a210}, **{i: f_32b[i] for i in a424}}
    
    # Finals 5.1
    ready_finals = [(max(path_done[i], f_kit[i]), i) for i in range(N)]
    f_51, w_51, b_51 = process_stage('5.1', ready_finals)
    
    # Filter completions within SIM_TIME
    completed = [(f_51[i], i) for i in range(N) if f_51[i] <= SIM_TIME]
    completions = [t for t, _ in completed]
    lead_times = [f_51[i] - launch_times[i] for _, i in completed]
    p_done = [ptypes[i] for _, i in completed]
    
    n_complete = len(completions)
    throughput_day = n_complete / sim_days
    achieved_takt = daily_minutes / throughput_day if throughput_day > 0 else 0
    avg_lead_time = np.mean(lead_times) if lead_times else 0
    
    awt210_count = p_done.count('AWT210')
    awt424_count = p_done.count('AWT420/424')
    
    # Daily throughput
    daily_comp = defaultdict(int)
    for t in completions:
        daily_comp[int(t // daily_minutes)] += 1
    daily_tp = [daily_comp.get(d, 0) for d in range(sim_days)]
    
    # Utilisation
    tp_210 = awt210_count / sim_days
    tp_424 = awt424_count / sim_days
    util = {
        '1.1': cycle_times['1.1'] * throughput_day / (resources['1.1'] * daily_minutes) * 100,
        '2.1A': cycle_times['2.1A'] * tp_210 / (resources['2.1A'] * daily_minutes) * 100,
        '2.2B': cycle_times['2.2B'] * tp_424 / (resources['2.2B'] * daily_minutes) * 100,
        '3.1A': cycle_times['3.1A'] * tp_210 / (resources['3.1A'] * daily_minutes) * 100,
        '3.2B': cycle_times['3.2B'] * tp_424 / (resources['3.2B'] * daily_minutes) * 100,
        '4.1': cycle_times['4.1'] * throughput_day / (resources['4.1'] * daily_minutes) * 100,
        '5.1': cycle_times['5.1'] * throughput_day / (resources['5.1'] * daily_minutes) * 100,
    }
    util = {k: min(v, 100.0) for k, v in util.items()}
    
    # Average wait times
    trim = int(N * 0.05)
    wait_raw = {
        '1.1': w_11, '2.1A': w_21a, '2.2B': w_22b,
        '3.1A': w_31a, '3.2B': w_32b, '4.1': w_kit, '5.1': w_51,
    }
    avg_wait = {s: np.mean(w[trim:-trim]) if len(w) > 2*trim else np.mean(w) if w else 0
                for s, w in wait_raw.items()}
    
    theo_cap = {s: (daily_minutes / cycle_times[s]) * resources[s] for s in cycle_times.keys()}
    
    return {
        'n_complete': n_complete,
        'throughput_day': throughput_day,
        'achieved_takt': achieved_takt,
        'avg_lead_time': avg_lead_time,
        'awt210_count': awt210_count,
        'awt424_count': awt424_count,
        'utilisation': util,
        'avg_wait': avg_wait,
        'theo_cap': theo_cap,
        'daily_tp': daily_tp,
        'lead_times': lead_times,
    }

# ════════════════════════════════════════════════════════════════
# SIDEBAR: INPUT CONTROLS
# ════════════════════════════════════════════════════════════════
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown("---")

daily_minutes = st.sidebar.slider("Daily Working Time (min)", 300, 480, 395, 5)
sim_days = st.sidebar.slider("Simulation Period (days)", 5, 90, 30, 5)
product_mix = st.sidebar.slider("AWT210 Mix (%)", 0, 100, 50, 5) / 100
inter_arrival = st.sidebar.slider("Inter-arrival Time (min)", 2, 30, 10, 1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📦 Cycle Times (min)")

cycle_times = {
    '1.1': st.sidebar.number_input("1.1 - Door Assembly", min_value=1, value=5),
    '2.1A': st.sidebar.number_input("2.1A - AWT210 Case", min_value=1, value=20),
    '2.2B': st.sidebar.number_input("2.2B - AWT420/424 Case", min_value=1, value=20),
    '3.1A': st.sidebar.number_input("3.1A - AWT210 Sub-Assy", min_value=1, value=15),
    '3.2B': st.sidebar.number_input("3.2B - AWT420/424 Sub-Assy", min_value=1, value=20),
    '4.1': st.sidebar.number_input("4.1 - Kitting", min_value=1, value=9),
    '5.1': st.sidebar.number_input("5.1 - Finals", min_value=1, value=20),
}

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Resources (Machines/Operators)")

resources = {
    '1.1': st.sidebar.number_input("1.1 - Door Assembly", min_value=1, value=1, key='r11'),
    '2.1A': st.sidebar.number_input("2.1A - AWT210 Case", min_value=1, value=1, key='r21a'),
    '2.2B': st.sidebar.number_input("2.2B - AWT420/424 Case", min_value=1, value=1, key='r22b'),
    '3.1A': st.sidebar.number_input("3.1A - AWT210 Sub-Assy", min_value=1, value=1, key='r31a'),
    '3.2B': st.sidebar.number_input("3.2B - AWT420/424 Sub-Assy", min_value=1, value=1, key='r32b'),
    '4.1': st.sidebar.number_input("4.1 - Kitting", min_value=1, value=1, key='r41'),
    '5.1': st.sidebar.number_input("5.1 - Finals", min_value=1, value=2, key='r51'),
}

run_button = st.sidebar.button("🚀 Run Simulation", use_container_width=True)

# ════════════════════════════════════════════════════════════════
# RUN SIMULATION & DISPLAY
# ════════════════════════════════════════════════════════════════
if run_button or 'results' not in st.session_state:
    with st.spinner("⏳ Running simulation..."):
        results = run_simulation(cycle_times, resources, daily_minutes, sim_days, product_mix, inter_arrival)
        st.session_state.results = results
else:
    results = st.session_state.results

# ════════════════════════════════════════════════════════════════
# TITLE & HEADER METRICS
# ════════════════════════════════════════════════════════════════
st.markdown("# 🏭 AWT Production Line Simulator")
st.markdown(f"**Simulation:** {sim_days} days × {daily_minutes} min/day | **Mix:** {int(product_mix*100)}% AWT210 / {100-int(product_mix*100)}% AWT420/424")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Units Completed", f"{results['n_complete']}", delta=None)
with col2:
    st.metric("Throughput", f"{results['throughput_day']:.2f} u/day")
with col3:
    st.metric("Achieved Takt", f"{results['achieved_takt']:.2f} min/u")
with col4:
    st.metric("Lead Time", f"{results['avg_lead_time']:.0f} min")
with col5:
    st.metric("Product Mix", f"{results['awt210_count']} | {results['awt424_count']}")

st.markdown("---")

# ════════════════════════════════════════════════════════════════
# UTILISATION CHART
# ════════════════════════════════════════════════════════════════
stage_names = {
    '1.1': '1.1 Door Assy',
    '2.1A': '2.1A AWT210 Case',
    '2.2B': '2.2B AWT420/424 Case',
    '3.1A': '3.1A AWT210 Sub',
    '3.2B': '3.2B AWT420/424 Sub',
    '4.1': '4.1 Kitting',
    '5.1': '5.1 Finals',
}
stages = list(cycle_times.keys())
util_vals = [results['utilisation'][s] for s in stages]
labels = [stage_names[s] for s in stages]

fig_util = go.Figure()
fig_util.add_trace(go.Bar(
    y=labels,
    x=util_vals,
    orientation='h',
    marker=dict(
        color=util_vals,
        colorscale=[[0, C_LIL1], [0.7, C_LIL], [0.99, C_RED], [1, C_RED]],
        colorbar=dict(title="Util %", thickness=15),
        line=dict(width=0.5, color='white'),
    ),
    text=[f'{v:.1f}%' for v in util_vals],
    textposition='outside',
))
fig_util.add_vline(x=100, line_dash="dash", line_color="#888", annotation_text="100% Cap", annotation_position="top")
fig_util.update_layout(
    title="Stage Utilisation (%)",
    xaxis_title="Utilisation (%)",
    height=350,
    margin=dict(l=0, r=50, t=40, b=0),
    showlegend=False,
    template='plotly_white',
)
fig_util.update_xaxes(range=[0, 120])

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_util, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# WAIT TIME & CAPACITY CHARTS
# ════════════════════════════════════════════════════════════════
wait_vals = [results['avg_wait'][s] for s in stages]
cap_vals = [results['theo_cap'][s] for s in stages]

fig_wait = go.Figure()
fig_wait.add_trace(go.Bar(
    y=labels,
    x=wait_vals,
    orientation='h',
    marker=dict(
        color=[C_RED if w >= 10 else C_LIL if w >= 2 else C_LIL1 for w in wait_vals],
        line=dict(width=0.5, color='white'),
    ),
    text=[f'{v:.1f}m' for v in wait_vals],
    textposition='outside',
))
fig_wait.update_layout(
    title="Average Queue Wait Time (min)",
    xaxis_title="Wait Time (min)",
    height=350,
    margin=dict(l=0, r=50, t=40, b=0),
    showlegend=False,
    template='plotly_white',
)

with col2:
    st.plotly_chart(fig_wait, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# CAPACITY VS THROUGHPUT
# ════════════════════════════════════════════════════════════════
fig_cap = go.Figure()
fig_cap.add_trace(go.Bar(
    y=labels,
    x=cap_vals,
    orientation='h',
    marker=dict(color=[C_LIL if util_vals[i] < 99 else C_RED for i in range(len(stages))]),
    name='Theoretical Capacity',
    text=[f'{v:.1f}' for v in cap_vals],
    textposition='outside',
))
fig_cap.add_vline(x=results['throughput_day'], line_dash="dash", line_color=C_RED, line_width=2.5,
                   annotation_text=f"System Throughput: {results['throughput_day']:.1f} u/day",
                   annotation_position="top right")
fig_cap.update_layout(
    title="Theoretical Capacity vs System Throughput",
    xaxis_title="Units/Day",
    height=350,
    margin=dict(l=0, r=50, t=40, b=0),
    showlegend=False,
    template='plotly_white',
)

st.plotly_chart(fig_cap, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# DAILY THROUGHPUT & LEAD TIME
# ════════════════════════════════════════════════════════════════
col1, col2 = st.columns(2)

with col1:
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Bar(
        x=list(range(1, sim_days + 1)),
        y=results['daily_tp'],
        marker=dict(color=C_LIL, line=dict(width=0.5, color='white')),
        name='Daily Output',
    ))
    fig_daily.add_hline(y=results['throughput_day'], line_dash="dash", line_color=C_RED, line_width=2,
                         annotation_text=f"Avg: {results['throughput_day']:.1f} u/day",
                         annotation_position="top right")
    fig_daily.update_layout(
        title="Daily Throughput",
        xaxis_title="Simulation Day",
        yaxis_title="Units Completed",
        height=350,
        margin=dict(l=0, r=50, t=40, b=0),
        showlegend=False,
        template='plotly_white',
    )
    st.plotly_chart(fig_daily, use_container_width=True)

with col2:
    fig_lt = go.Figure()
    fig_lt.add_trace(go.Histogram(
        x=results['lead_times'],
        nbinsx=30,
        marker=dict(color=C_LIL, line=dict(width=0.5, color='white')),
        name='Lead Time',
    ))
    fig_lt.add_vline(x=results['avg_lead_time'], line_dash="dash", line_color=C_RED, line_width=2,
                      annotation_text=f"Mean: {results['avg_lead_time']:.0f} min",
                      annotation_position="top right")
    fig_lt.update_layout(
        title="Lead Time Distribution (min)",
        xaxis_title="Lead Time (min)",
        yaxis_title="Frequency",
        height=350,
        margin=dict(l=0, r=50, t=40, b=0),
        showlegend=False,
        template='plotly_white',
    )
    st.plotly_chart(fig_lt, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# DETAILED STAGE TABLE
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📊 Detailed Stage Analysis")

df_stages = pd.DataFrame({
    'Stage': [stage_names[s] for s in stages],
    'Cycle Time (min)': [cycle_times[s] for s in stages],
    'Resources': [resources[s] for s in stages],
    'Capacity (u/day)': [f"{results['theo_cap'][s]:.1f}" for s in stages],
    'Utilisation (%)': [f"{results['utilisation'][s]:.1f}" for s in stages],
    'Avg Wait (min)': [f"{results['avg_wait'][s]:.1f}" for s in stages],
})

st.dataframe(df_stages, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("*Adjust settings in the left sidebar and click **Run Simulation** to update all charts.*")
