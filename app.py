
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.database import (
    initialize_database, seed_demo_data, get_readings,
    get_reading_count, get_household_count, get_database_path,
    get_households, insert_reading, create_alarm, get_alarms,
    get_alarm_count, acknowledge_alarm,
)

# Install with: python -m pip install streamlit-autorefresh
try:
    from streamlit_autorefresh import st_autorefresh
    AUTO_REFRESH_AVAILABLE = True
except ImportError:
    AUTO_REFRESH_AVAILABLE = False

st.set_page_config(
    page_title="GridRakshak | Distribution Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #07131f;
        background-image:
            linear-gradient(
                90deg,
                rgba(7, 19, 31, 0.42),
                rgba(7, 19, 31, 0.12) 55%,
                rgba(7, 19, 31, 0.32)
            ),
            url("/app/static/gridrakshak-substation.svg");
        background-position: center, center center;
        background-size: cover, cover;
        background-repeat: no-repeat, no-repeat;
        background-attachment: fixed;
    }

    .stApp,
    .stApp p,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp label,
    .stApp [data-testid="stMetricLabel"],
    .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stCaptionContainer"] {
        color: #e2edf5;
    }

    [data-testid="stHeader"] {
        background: rgba(7, 19, 31, 0.78);
    }

    [data-testid="stSidebar"] {
        background: rgba(7, 19, 31, 0.88);
        border-right: 1px solid rgba(56, 189, 248, 0.12);
    }

    [data-testid="stMainBlockContainer"] {
        background: transparent;
    }

    @media (max-width: 700px) {
        .stApp {
            background-attachment: scroll;
            background-position: center, 57% center;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# DEMO DATA
# ==========================================================

def generate_demo_data():
    rng = np.random.default_rng(42)
    timestamps = pd.date_range(
        end=pd.Timestamp.now().floor("5min"),
        periods=288,
        freq="5min",
    )
    rows = []

    for number in range(1, 10):
        household = f"H{number:02d}"
        feeder = f"F{((number - 1) // 3) + 1:02d}"

        voltages = rng.normal(230, 3, len(timestamps))
        currents = rng.uniform(0.3, 5.0, len(timestamps))

        for i, ts in enumerate(timestamps):
            voltage = float(voltages[i])
            current = float(currents[i])
            anomaly = False

            if household in {"H02", "H05", "H08"} and i in {65, 66, 67}:
                anomaly = True
                if i == 65:
                    current = 14.0
                elif i == 66:
                    voltage, current = 190.0, 8.0
                else:
                    current = 0.1

            power = max(0, voltage * current * 0.98)
            energy = power * (5 / 60) / 1000

            rows.append({
                "Timestamp": ts,
                "Household": household,
                "Feeder": feeder,
                "Voltage_V": round(voltage, 2),
                "Current_A": round(current, 3),
                "Power_W": round(power, 2),
                "Energy_kWh": round(energy, 6),
                "Demo_Anomaly": anomaly,
            })

    return pd.DataFrame(rows)


# ==========================================================
# ANOMALY DETECTION
# ==========================================================

def detect_anomalies(data):
    result = data.copy()
    features = ["Voltage_V", "Current_A", "Power_W"]

    for col in features:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    result["Demo_Anomaly"] = (
        result["Demo_Anomaly"].fillna(False).astype(bool)
    )
    result = result.dropna(subset=features).copy()
    result["AI_Anomaly"] = False

    if len(result) >= 20:
        try:
            from sklearn.ensemble import IsolationForest

            model = IsolationForest(
                n_estimators=150,
                contamination=0.03,
                random_state=42,
            )
            result["AI_Anomaly"] = model.fit_predict(
                result[features]
            ) == -1
        except ImportError:
            for col in features:
                median = result[col].median()
                mad = (result[col] - median).abs().median()
                if mad > 0:
                    result["AI_Anomaly"] |= (
                        (result[col] - median).abs() > 6 * mad
                    )

    voltage_bad = (
        (result["Voltage_V"] < 207)
        | (result["Voltage_V"] > 253)
    )
    current_high = result["Current_A"] > 10
    power_high = result["Power_W"] > 3000

    result["Electrical_Threshold_Alert"] = (
        voltage_bad | current_high | power_high
    )
    result["Anomaly"] = (
        result["AI_Anomaly"]
        | result["Electrical_Threshold_Alert"]
        | result["Demo_Anomaly"]
    )

    result["Anomaly_Type"] = "Normal"
    result.loc[result["AI_Anomaly"], "Anomaly_Type"] = (
        "AI-detected unusual pattern"
    )
    result.loc[voltage_bad, "Anomaly_Type"] = (
        "Voltage outside demo limits"
    )
    result.loc[current_high, "Anomaly_Type"] = "High current"
    result.loc[power_high, "Anomaly_Type"] = "High power"
    result.loc[result["Demo_Anomaly"], "Anomaly_Type"] = (
        "Simulated abnormal event"
    )

    def severity(row):
        if not row["Anomaly"]:
            return "Normal"
        if (
            row["Voltage_V"] < 190 or row["Voltage_V"] > 260
            or row["Current_A"] > 20 or row["Power_W"] > 5000
        ):
            return "Critical"
        if (
            row["Voltage_V"] < 207 or row["Voltage_V"] > 253
            or row["Current_A"] > 10 or row["Power_W"] > 3000
        ):
            return "High"
        if row["AI_Anomaly"] or row["Demo_Anomaly"]:
            return "Medium"
        return "Low"

    result["Severity"] = result.apply(severity, axis=1)
    return result


# ==========================================================
# ALARM SYNCHRONIZATION
# ==========================================================

def synchronize_alarms(data):
    if data.empty:
        return

    latest_indices = set(
        data.sort_values("Timestamp")
        .groupby("Household")
        .tail(1)
        .index
    )

    for index, row in data.iterrows():
        demo_event = bool(row["Demo_Anomaly"])
        threshold_event = bool(row["Electrical_Threshold_Alert"])
        latest_ai_event = (
            index in latest_indices and bool(row["AI_Anomaly"])
        )

        if not (demo_event or threshold_event or latest_ai_event):
            continue

        ts = row["Timestamp"]
        if pd.isna(ts):
            continue

        household = str(row["Household"])
        feeder = str(row["Feeder"])
        event_type = str(row["Anomaly_Type"])

        key_text = (
            f"{household}|{feeder}|{ts.isoformat()}|{event_type}"
        )
        alarm_key = hashlib.sha256(
            key_text.encode("utf-8")
        ).hexdigest()

        message = (
            f"{event_type} detected for {household} on feeder {feeder}. "
            f"Voltage={row['Voltage_V']:.1f} V, "
            f"Current={row['Current_A']:.2f} A, "
            f"Power={row['Power_W']:.1f} W."
        )

        create_alarm(
            alarm_key=alarm_key,
            timestamp=ts,
            household_id=household,
            feeder_id=feeder,
            severity=row["Severity"],
            anomaly_type=event_type,
            message=message,
            voltage_v=float(row["Voltage_V"]),
            current_a=float(row["Current_A"]),
            power_w=float(row["Power_W"]),
        )


def latest_by_household(data):
    return (
        data.sort_values("Timestamp")
        .groupby("Household", as_index=False)
        .tail(1)
        .sort_values("Household")
    )


def show_table(data, columns=None):
    if data.empty:
        st.info("No records match this selection.")
        return

    if columns:
        data = data[[c for c in columns if c in data.columns]]

    if "Timestamp" in data.columns:
        data = data.sort_values("Timestamp", ascending=False)

    st.dataframe(data, width="stretch", hide_index=True)


def render_scada_asset_tree(data, feeder_ids):
    latest = latest_by_household(data)
    node_ids = {"Substation"}
    node_ids.update(feeder_ids)
    node_ids.update(latest["Household"].astype(str))

    selected_node = st.session_state.get("scada_selected_node", "Substation")
    if selected_node not in node_ids:
        selected_node = "Substation"
        st.session_state["scada_selected_node"] = selected_node

    st.title("Substation SCADA | Logical Asset Tree")
    st.caption(
        "Operator view of configured feeder-to-meter relationships. "
        "This research prototype displays simulated or manually entered "
        "data; it is not connected to live SCADA or substation equipment."
    )
    st.info(
        "The database contains feeder and household-meter data only. "
        "Busbars, breakers, transformers, and protection relays are not "
        "configured and are intentionally not shown as live assets."
    )

    with st.container(horizontal=True):
        st.metric("Feeders", len(feeder_ids), border=True)
        st.metric("Monitored meters", latest["Household"].nunique(), border=True)
        st.metric("Latest site load", f"{latest['Power_W'].sum():,.0f} W", border=True)
        st.metric("Active alarms", get_alarm_count("Active"), border=True)

    tree_column, detail_column = st.columns([1, 1.6])

    with tree_column:
        with st.container(border=True):
            st.subheader("Asset hierarchy")
            st.caption("Select a node to inspect its latest status and history.")

            with st.expander("🏭 Distribution substation", expanded=True):
                if st.button(
                    "View substation",
                    key="scada_select_substation",
                    type="primary" if selected_node == "Substation" else "secondary",
                    width="stretch",
                ):
                    st.session_state["scada_selected_node"] = "Substation"
                    selected_node = "Substation"

                for feeder_id in feeder_ids:
                    feeder_latest = latest[latest["Feeder"] == feeder_id]
                    flagged = feeder_latest[feeder_latest["Anomaly"]]
                    if flagged.empty:
                        feeder_status = "Normal"
                        icon = "🟢"
                    else:
                        severity_order = ["Critical", "High", "Medium", "Low"]
                        feeder_status = next(
                            (
                                severity
                                for severity in severity_order
                                if (flagged["Severity"] == severity).any()
                            ),
                            "Anomaly",
                        )
                        icon = {
                            "Critical": "🔴",
                            "High": "🟠",
                            "Medium": "🟡",
                            "Low": "🔵",
                        }.get(feeder_status, "🟡")

                    with st.expander(
                        f"{icon} Feeder {feeder_id} · {feeder_status}",
                        expanded=(
                            selected_node == feeder_id
                            or selected_node
                            in feeder_latest["Household"].astype(str).tolist()
                        ),
                    ):
                        if st.button(
                            f"View feeder {feeder_id}",
                            key=f"scada_select_feeder_{feeder_id}",
                            type=(
                                "primary"
                                if selected_node == feeder_id
                                else "secondary"
                            ),
                            width="stretch",
                        ):
                            st.session_state["scada_selected_node"] = feeder_id
                            selected_node = feeder_id

                        for _, meter in feeder_latest.iterrows():
                            household = str(meter["Household"])
                            meter_severity = str(meter["Severity"])
                            meter_icon = {
                                "Normal": "🟢",
                                "Critical": "🔴",
                                "High": "🟠",
                                "Medium": "🟡",
                                "Low": "🔵",
                            }.get(meter_severity, "⚪")
                            if st.button(
                                f"  └─ {meter_icon} Meter {household} · "
                                f"{meter_severity}",
                                key=f"scada_select_meter_{household}",
                                type=(
                                    "primary"
                                    if selected_node == household
                                    else "secondary"
                                ),
                                width="stretch",
                            ):
                                st.session_state["scada_selected_node"] = household
                                selected_node = household

    with detail_column:
        with st.container(border=True):
            if selected_node == "Substation":
                st.subheader("Substation summary")
                latest_flagged = latest[latest["Anomaly"]]
                status = "Attention required" if not latest_flagged.empty else "Normal"
                st.metric("Overall status", status)
                st.metric("Latest flagged meters", len(latest_flagged))
                st.subheader("Latest exceptions")
                show_table(
                    latest_flagged,
                    [
                        "Household", "Feeder", "Voltage_V", "Current_A",
                        "Power_W", "Anomaly_Type", "Severity",
                    ],
                )
            elif selected_node in feeder_ids:
                feeder_latest = latest[latest["Feeder"] == selected_node]
                feeder_history = data[data["Feeder"] == selected_node]
                feeder_alarms = get_alarms(
                    status="Active", feeder_id=selected_node, limit=1000
                )
                st.subheader(f"Feeder {selected_node}")
                metric_columns = st.columns(3)
                metric_columns[0].metric(
                    "Meters", feeder_latest["Household"].nunique()
                )
                metric_columns[1].metric(
                    "Latest load", f"{feeder_latest['Power_W'].sum():,.0f} W"
                )
                metric_columns[2].metric(
                    "Latest exceptions",
                    int(feeder_latest["Anomaly"].sum()),
                )
                st.metric("Active alarms", len(feeder_alarms))
                trend = (
                    feeder_history.groupby("Timestamp", as_index=False)["Power_W"]
                    .sum()
                )
                st.plotly_chart(
                    px.line(
                        trend,
                        x="Timestamp",
                        y="Power_W",
                        title="Feeder aggregate load history",
                        labels={"Power_W": "Aggregate load (W)"},
                    ),
                    width="stretch",
                )
                st.subheader("Latest meter status")
                show_table(
                    feeder_latest,
                    [
                        "Household", "Voltage_V", "Current_A", "Power_W",
                        "Anomaly_Type", "Severity",
                    ],
                )
                st.subheader("Active feeder alarms")
                show_table(feeder_alarms)
            else:
                household_data = data[data["Household"].astype(str) == selected_node]
                meter_latest = latest[
                    latest["Household"].astype(str) == selected_node
                ].iloc[0]
                meter_alarms = get_alarms(
                    status="All", feeder_id=str(meter_latest["Feeder"]), limit=1000
                )
                meter_alarms = meter_alarms[
                    meter_alarms["Household"].astype(str) == selected_node
                ]
                st.subheader(f"Household meter {selected_node}")
                metric_columns = st.columns(4)
                metric_columns[0].metric("Feeder", meter_latest["Feeder"])
                metric_columns[1].metric(
                    "Voltage", f"{meter_latest['Voltage_V']:.1f} V"
                )
                metric_columns[2].metric(
                    "Current", f"{meter_latest['Current_A']:.2f} A"
                )
                metric_columns[3].metric(
                    "Power", f"{meter_latest['Power_W']:.1f} W"
                )
                st.caption(
                    f"Latest status: {meter_latest['Severity']} · "
                    f"{meter_latest['Timestamp']}"
                )
                st.plotly_chart(
                    px.line(
                        household_data,
                        x="Timestamp",
                        y="Voltage_V",
                        title="Meter voltage history",
                    ),
                    width="stretch",
                )
                st.plotly_chart(
                    px.line(
                        household_data,
                        x="Timestamp",
                        y="Power_W",
                        title="Meter power history",
                    ),
                    width="stretch",
                )
                st.subheader("Meter alarm history")
                show_table(meter_alarms)


# ==========================================================
# INITIALIZATION
# ==========================================================

initialize_database()

if get_reading_count() == 0:
    seed_demo_data(generate_demo_data())

# Refresh entire application every 15 seconds when package is installed.
if AUTO_REFRESH_AVAILABLE:
    st_autorefresh(
        interval=15_000,
        limit=None,
        key="gridrakshak_auto_refresh",
    )

df = get_readings()

if df.empty:
    st.error("No readings found in the database.")
    st.stop()

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"], format="mixed", errors="coerce"
)
df["Demo_Anomaly"] = df["Demo_Anomaly"].fillna(False).astype(bool)
df = df.dropna(
    subset=[
        "Timestamp", "Household", "Feeder",
        "Voltage_V", "Current_A", "Power_W"
    ]
)

if df.empty:
    st.error("No valid readings are available.")
    st.stop()

df = detect_anomalies(df)
synchronize_alarms(df)

household_ids = sorted(df["Household"].astype(str).unique())
feeder_ids = sorted(df["Feeder"].astype(str).unique())

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("⚡ GridRakshak")
st.sidebar.caption("Distribution Intelligence Platform")

if AUTO_REFRESH_AVAILABLE:
    st.sidebar.success("Auto-refresh: every 15 seconds")
else:
    st.sidebar.warning("Auto-refresh package not installed")
    if st.sidebar.button("Refresh dashboard now"):
        st.rerun()

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Substation SCADA Tree",
        "Household Monitoring",
        "Feeder Localisation",
        "Substation Alarms",
        "Database Management",
    ],
)

st.sidebar.divider()
st.sidebar.metric("Registered households", get_household_count())
st.sidebar.metric("Stored readings", f"{get_reading_count():,}")
st.sidebar.metric("Total alarms", get_alarm_count())
st.sidebar.caption("Prototype · Simulated/manual data")

# ==========================================================
# OVERVIEW
# ==========================================================

if page == "Overview":
    st.title("GridRakshak — Distribution Overview")
    st.caption("Customer-to-feeder monitoring and persistent alarm management.")

    latest = latest_by_household(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Households", get_household_count())
    c2.metric("Readings stored", f"{get_reading_count():,}")
    c3.metric("Anomalous readings", int(df["Anomaly"].sum()))
    c4.metric("Active alarms", get_alarm_count("Active"))

    st.subheader("Latest household readings")
    show_table(latest, [
        "Timestamp", "Household", "Feeder", "Voltage_V",
        "Current_A", "Power_W", "Anomaly", "Severity"
    ])

    st.subheader("Feeder-wise latest power")
    feeder_load = latest.groupby("Feeder", as_index=False)["Power_W"].sum()
    fig = px.bar(
        feeder_load, x="Feeder", y="Power_W",
        labels={"Power_W": "Latest household power (W)"},
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Recent anomalies")
    show_table(
        df[df["Anomaly"]].sort_values("Timestamp", ascending=False).head(20),
        [
            "Timestamp", "Household", "Feeder", "Voltage_V",
            "Current_A", "Power_W", "Anomaly_Type", "Severity"
        ],
    )

# ==========================================================
# HOUSEHOLD MONITORING
# ==========================================================

elif page == "Substation SCADA Tree":
    render_scada_asset_tree(df, feeder_ids)

elif page == "Household Monitoring":
    st.title("Household Monitoring")
    selected = st.selectbox("Select household", household_ids)
    household_data = df[df["Household"] == selected].sort_values("Timestamp")

    if household_data.empty:
        st.warning("No readings available.")
    else:
        latest = household_data.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Feeder", latest["Feeder"])
        c2.metric("Voltage", f"{latest['Voltage_V']:.1f} V")
        c3.metric("Current", f"{latest['Current_A']:.2f} A")
        c4.metric("Power", f"{latest['Power_W']:.1f} W")

        st.plotly_chart(
            px.line(
                household_data, x="Timestamp", y="Voltage_V",
                title=f"{selected} — Voltage history",
            ),
            width="stretch",
        )
        st.plotly_chart(
            px.line(
                household_data, x="Timestamp", y="Power_W",
                title=f"{selected} — Power history",
            ),
            width="stretch",
        )

        st.subheader("Anomaly records")
        show_table(
            household_data[household_data["Anomaly"]],
            [
                "Timestamp", "Voltage_V", "Current_A", "Power_W",
                "Energy_kWh", "Anomaly_Type", "Severity"
            ],
        )

# ==========================================================
# FEEDER LOCALISATION AND SUMMARY
# ==========================================================

elif page == "Feeder Localisation":
    st.title("Feeder Localisation and Anomaly Summary")
    selected_feeder = st.selectbox("Select feeder", feeder_ids)
    feeder_data = df[df["Feeder"] == selected_feeder]
    latest = latest_by_household(feeder_data)

    feeder_alarms = get_alarms(
        status="All", feeder_id=selected_feeder, limit=1000
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Households", latest["Household"].nunique())
    c2.metric("Latest anomalous households", int(latest["Anomaly"].sum()))
    c3.metric("Active alarms", get_alarm_count("Active") if feeder_alarms.empty
              else int((feeder_alarms["Status"] == "Active").sum()))
    c4.metric("Historical alarms", len(feeder_alarms))

    st.subheader("Feeder readings")
    show_table(latest, [
        "Timestamp", "Household", "Voltage_V", "Current_A",
        "Power_W", "Anomaly", "Anomaly_Type", "Severity"
    ])

    st.subheader("Feeder anomaly summary")
    summary = (
        feeder_data.groupby("Household")
        .agg(
            Reading_Count=("Timestamp", "count"),
            Anomaly_Count=("Anomaly", "sum"),
            Maximum_Current_A=("Current_A", "max"),
            Maximum_Power_W=("Power_W", "max"),
        )
        .reset_index()
    )
    show_table(summary)

    st.subheader("Affected customers")
    affected = latest[latest["Anomaly"]]
    if affected.empty:
        st.success("No latest readings are flagged on this feeder.")
    else:
        st.warning(f"{len(affected)} household(s) have flagged latest readings.")
        show_table(affected, [
            "Household", "Feeder", "Voltage_V", "Current_A",
            "Power_W", "Anomaly_Type", "Severity"
        ])

    st.subheader("Historical alarms on this feeder")
    show_table(feeder_alarms)

# ==========================================================
# SUBSTATION ALARMS
# ==========================================================

elif page == "Substation Alarms":
    st.title("Substation Alarm Management")
    st.caption(
        "Acknowledgement confirms operator review; it does not repair "
        "the underlying fault."
    )

    a, b, c, d = st.columns(4)
    with a:
        status_filter = st.selectbox(
            "Status", ["Active", "Acknowledged", "All"]
        )
    with b:
        severity_filter = st.selectbox(
            "Severity", ["All", "Critical", "High", "Medium", "Low"]
        )
    with c:
        feeder_filter = st.selectbox("Feeder", ["All"] + feeder_ids)
    with d:
        household_filter = st.selectbox(
            "Household", ["All"] + household_ids
        )

    search_text = st.text_input(
        "Search alarm message or anomaly type",
        placeholder="For example: voltage or current",
    )

    alarms = get_alarms(
        status=status_filter,
        feeder_id=feeder_filter,
        limit=1000,
    )

    if not alarms.empty:
        if severity_filter != "All":
            alarms = alarms[alarms["Severity"] == severity_filter]
        if household_filter != "All":
            alarms = alarms[alarms["Household"] == household_filter]
        if search_text.strip():
            needle = search_text.strip().lower()
            mask = (
                alarms["Message"].fillna("").str.lower().str.contains(
                    needle, regex=False
                )
                | alarms["Anomaly_Type"].fillna("").str.lower().str.contains(
                    needle, regex=False
                )
            )
            alarms = alarms[mask]

    c1, c2, c3 = st.columns(3)
    c1.metric("Active alarms — all", get_alarm_count("Active"))
    c2.metric("Acknowledged — all", get_alarm_count("Acknowledged"))
    c3.metric("Filtered results", len(alarms))

    st.subheader("Alarm history")
    show_table(alarms, [
        "Alarm_ID", "Timestamp", "Household", "Feeder",
        "Severity", "Anomaly_Type", "Message", "Status",
        "Acknowledged_At"
    ])

    st.subheader("Acknowledge an active alarm")
    active = (
        alarms[alarms["Status"] == "Active"]
        if not alarms.empty else pd.DataFrame()
    )

    if active.empty:
        st.info("No active alarms match the selected filters.")
    else:
        choices = {
            f"#{int(row['Alarm_ID'])} | {row['Severity']} | "
            f"{row['Household']} | {row['Feeder']} | {row['Anomaly_Type']}":
            int(row["Alarm_ID"])
            for _, row in active.iterrows()
        }
        chosen_label = st.selectbox("Choose alarm to acknowledge", list(choices))
        if st.button("Acknowledge selected alarm", type="primary"):
            success = acknowledge_alarm(choices[chosen_label])
            if success:
                st.success(f"Alarm {choices[chosen_label]} acknowledged.")
            else:
                st.warning(
                    "This alarm may already have been acknowledged. "
                    "Refresh the page and check its status."
                )
            st.rerun()

# ==========================================================
# DATABASE MANAGEMENT
# ==========================================================

elif page == "Database Management":
    st.title("Database Management")
    c1, c2, c3 = st.columns(3)
    c1.metric("Households", get_household_count())
    c2.metric("Readings", f"{get_reading_count():,}")
    c3.metric("Alarms", get_alarm_count())

    st.write("**SQLite database path**")
    st.code(get_database_path())

    st.subheader("Add a manual test reading")
    st.caption(
        "These are manually entered test values, not live meter readings."
    )

    with st.form("manual_reading_form"):
        a, b = st.columns(2)
        with a:
            household = st.selectbox("Household ID", household_ids)
            voltage = st.number_input(
                "Voltage (V)", 0.0, 500.0, 230.0, step=1.0
            )
            current = st.number_input(
                "Current (A)", 0.0, 200.0, 2.0, step=0.1
            )
        with b:
            feeder = st.selectbox("Feeder ID", feeder_ids)
            power = st.number_input(
                "Power (W)", 0.0, 50000.0, 460.0, step=10.0
            )
            energy = st.number_input(
                "Energy increment (kWh)", 0.0, 1000.0, 0.0383,
                step=0.001, format="%.6f"
            )

        submitted = st.form_submit_button(
            "Save reading", type="primary", width="stretch"
        )

        if submitted:
            try:
                insert_reading(
                    household_id=household,
                    feeder_id=feeder,
                    voltage_v=voltage,
                    current_a=current,
                    power_w=power,
                    energy_kwh=energy,
                    timestamp=datetime.now(),
                    demo_anomaly=False,
                )
                st.success("Reading saved to SQLite.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save reading: {exc}")

    st.subheader("Registered households")
    show_table(get_households())

    st.subheader("Latest database readings")
    show_table(get_readings(limit=50))

# ==========================================================
# FOOTER
# ==========================================================

st.divider()
st.caption(
    "GridRakshak research prototype. Data is simulated or manually entered. "
    "It is not connected to live meters, SCADA, or a real substation. "
    "Demonstration thresholds are not operational protection settings."
)
