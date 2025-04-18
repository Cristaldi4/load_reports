# streamlit_load_report.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Office Building Load Report", layout="wide")
st.title("🏢 Office Building Load Profile Dashboard")
st.markdown("Prepared by Grid Discovery  ")
st.markdown("---")

# Load pre-existing CSV data with corrected filename
file_path = "198126_load_dashboard/198126_Essex County_Large Office.csv"
data = pd.read_csv(file_path)
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.sort_values('timestamp')

# Add season column globally so all sections can use it
season_mapping = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
}
data['season'] = data['timestamp'].dt.month.map(season_mapping)

# Ensure column names are correct
expected_columns = ['timestamp', 'load_kW', 'energy_kWh', 'cooling_kWh', 'exterior_lighting_kWh', 'fans_kWh', 'interior_equipment_kWh', 'interior_lighting_kWh']
missing_columns = [col for col in expected_columns if col not in data.columns]
if missing_columns:
    st.error(f"Missing required column(s): {', '.join(missing_columns)}")
    st.stop()

# Calculate metrics needed for summary
annual_energy_consumption = data['energy_kWh'].sum()
peak_demand = data['load_kW'].max()
average_load = data['load_kW'].mean()
load_factor = annual_energy_consumption / (peak_demand * 8760)
monthly_avg_demand = data.resample('ME', on='timestamp')['load_kW'].mean()
annual_demand_charge_rate = 4.7075
summer_demand_charge_rate = 16.1987
all_in_summer_demand_charge = annual_demand_charge_rate + summer_demand_charge_rate
monthly_avg_demand = monthly_avg_demand.reset_index()
monthly_avg_demand['month_num'] = monthly_avg_demand['timestamp'].dt.month
monthly_avg_demand['avg_demand_charge'] = monthly_avg_demand.apply(
    lambda row: row['load_kW'] * all_in_summer_demand_charge if row['month_num'] in [6, 7, 8, 9]
    else row['load_kW'] * annual_demand_charge_rate,
    axis=1
)
avg_demand_charge_total = monthly_avg_demand['avg_demand_charge'].sum()

# Sidebar navigation
st.sidebar.title("Report Sections")
section = st.sidebar.radio("Jump to:", [
    "Executive Summary",
    "Descriptive Statistics",
    "Equipment Usage",
    "Monthly Peak and Average Demand",
    "Monthly Demand Charges",
    "Top 10 Peak Days",
    "Hourly Load Heatmap",
    "Daily Energy Consumption",
    "Load Variability",
    "Annual Load Duration Curve",
    "Peak Day Load Duration Curve",
    "Load Duration Curve Comparison: Annual vs Peak Day",
    "Seasonal Weekly Load Profiles",
    "Average Daily Load for Each Season",
    "Load Trends for Top 3 Peak Days"
])

if section == "Executive Summary":
    st.subheader("📋 Executive Summary")

    st.markdown("**Description:** Large office complex in Newark, NJ")
    st.markdown("**Compared to:** 8-story 750k square foot office building in Essex County, NJ")
    st.markdown("**Equipment:**")
    st.markdown("• HVAC system: Packaged variable air volume with gas boiler reheat")
    st.markdown("• Lighting: T12 incandescent")
    st.markdown(f"**Estimated annual demand savings:** ${avg_demand_charge_total:,.2f}")
    st.markdown(f"**Building load factor:** {load_factor:.2f}")

    st.markdown("---")
    st.markdown("**About Grid Discovery:**")
    st.markdown("""
    Grid Discovery is a NJ-based software startup on a mission to power America with cheap, renewable energy.

    Our technology combines millions of consumption, generation, and geospatial data points to target ideal buildings for on-site storage and generation projects.
    """)

    st.markdown("---")
    st.markdown("**Disclaimer:**")
    st.markdown("""
    All data in this report has been pulled and processed from the ComStock and ResStock databases.
    National Renewable Energy Laboratory maintains the database for the purpose of identifying which building stock improvements save the most energy and money.

    This report should be used for analytical purposes only. All decisions should be consulted and reviewed with a licensed engineer.
    """)

elif section == "Equipment Usage":
    st.subheader("🛠️ Equipment Usage Over Time") 

    end_use_columns = [
        'cooling_kWh',
        'exterior_lighting_kWh',
        'fans_kWh',
        'interior_equipment_kWh',
        'interior_lighting_kWh'
    ]

    # Convert timestamp if needed
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Optional: resample to daily for smoother curves if the data is noisy
    daily_end_use = data.set_index('timestamp')[end_use_columns].resample('D').sum().reset_index()

    # Build Plotly figure
    fig = go.Figure()
    for col in end_use_columns:
        fig.add_trace(go.Scatter(
            x=daily_end_use['timestamp'],
            y=daily_end_use[col],
            mode='lines',
            name=col.replace('_kWh', '').replace('_', ' ').title()
        ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Energy (kWh)",
        xaxis=dict(
            tickformat="%b %d",
            tickangle=45
        ),
        legend_title="Component",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

    st.markdown("---")

    st.subheader("📊 Equipment Usage Contributions")

    end_use_columns = [
        'cooling_kWh',
        'exterior_lighting_kWh',
        'fans_kWh',
        'interior_equipment_kWh',
        'interior_lighting_kWh'
    ]

    contribution_mode = st.radio("View by:", ["Total Contribution", "Daily Average Contribution"], horizontal=True)
    chart_type = st.radio("Chart Type:", ["Bar Chart", "Pie Chart"], horizontal=True)

    # Calculate values
    if contribution_mode == "Total Contribution":
        total_energy = data['energy_kWh'].sum()
        contributions = data[end_use_columns].sum() / total_energy
        title_suffix = "(Total)"
    else:
        daily_energy = data.resample('D', on='timestamp')['energy_kWh'].sum()
        daily_components = data.resample('D', on='timestamp')[end_use_columns].sum()
        contributions = (daily_components.div(daily_energy, axis=0)).mean()
        title_suffix = "(Daily Avg)"

    contributions = contributions.sort_values(ascending=False)
    labels = [col.replace('_kWh', '').replace('_', ' ').title() for col in contributions.index]
    values = contributions.values

    # Bar Chart
    if chart_type == "Bar Chart":
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=values,
                text=[f"{v:.1%}" for v in values],
                textposition='outside',
                marker_color='steelblue'
            )
        ])
        fig.update_layout(
            title=f"Equipment Usage Contributions {title_suffix}",
            xaxis_title="End-Use Component",
            yaxis_title="Share of Energy Consumption",
            yaxis_tickformat=".0%",
            height=500
        )

    # Pie Chart
    else:
        fig = go.Figure(data=[
            go.Pie(
                labels=labels,
                values=values,
                textinfo='label+percent',
                insidetextorientation='radial',
                marker=dict(colors=px.colors.sequential.Viridis)
            )
        ])
        fig.update_layout(
            title=f"Equipment Usage Contributions {title_suffix}",
            height=500
        )

    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

    st.markdown("---")

    st.subheader("⛰️ Stacked Area Plot of Equipment Usage Over Time")

    end_use_columns = [
        'cooling_kWh',
        'exterior_lighting_kWh',
        'fans_kWh',
        'interior_equipment_kWh',
        'interior_lighting_kWh'
    ]

    # Ensure data is sorted by time
    data_sorted = data.sort_values('timestamp')

    fig = go.Figure()

    for col in end_use_columns:
        fig.add_trace(go.Scatter(
            x=data_sorted['timestamp'],
            y=data_sorted[col],
            mode='lines',
            stackgroup='one',
            name=col.replace('_kWh', '').replace('_', ' ').title()
        ))

    fig.update_layout(
    margin=dict(b=100), 
    xaxis_title="Date",
    yaxis_title="Energy (kWh)",
    height=500,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.3,         # Negative y moves legend below chart
        xanchor="center",
        x=0.5
    )
)

    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

    st.markdown("---")

    st.subheader("📊 Equipment Usage for Peak Day")

    daily_peaks = data.resample('D', on='timestamp')['load_kW'].max()
    max_peak_day = daily_peaks.idxmax().date()
    day_data = data[data['timestamp'].dt.date == max_peak_day].copy()

    if not day_data.empty:
        fig = go.Figure()

        for col in end_use_columns:
            fig.add_trace(go.Scatter(
                x=day_data['timestamp'],
                y=day_data[col],
                mode='lines',
                name=col.replace('_kWh', '').replace('_', ' ').title()
            ))

        fig.update_layout(
            title=f"Peak Day: {max_peak_day.strftime('%b %d')}",
            xaxis_title="Time of Day",
            yaxis_title="Energy (kWh)",
            xaxis=dict(
                tickformat="%H:%M",
                dtick=3600000,  # 1 hour in ms
            ),
            height=500,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.3,
                xanchor="center",
                x=0.5
            ),
            margin=dict(b=100)
        )

        st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})
    else:
        st.warning("No data available for the peak demand day.")


elif section == "Descriptive Statistics":
    st.subheader("📊 Descriptive Statistics")

    # Recalculate in case context needs them here
    average_load_daily = data.resample('D', on='timestamp')['load_kW'].mean()
    average_load_weekly = data.resample('W', on='timestamp')['load_kW'].mean()
    average_load_monthly = data.resample('ME', on='timestamp')['load_kW'].mean()
    daily_energy_consumption = data.resample('D', on='timestamp')['energy_kWh'].sum()

    capacity_factor = average_load / peak_demand
    peak_shaving_potential = peak_demand - average_load
    grid_discovery_opportunity = (1 - load_factor) * peak_demand

    # Display metrics
    st.markdown(f"**Total Annual Energy Consumption:** {annual_energy_consumption:,.2f} kWh")
    st.markdown(f"**Peak Demand:** {peak_demand:.2f} kW")
    st.markdown(f"**Average Demand:** {average_load:.2f} kW")
    st.markdown(f"**Peak Shaving Potential:** {peak_shaving_potential:.2f} kW")
    st.markdown(f"**Load Factor:** {load_factor:.2f}")
    st.markdown(f"**Daily Average Load:** {average_load_daily.mean():.2f} kW")
    st.markdown(f"**Weekly Average Load:** {average_load_weekly.mean():.2f} kW")
    st.markdown(f"**Monthly Average Load:** {average_load_monthly.mean():.2f} kW")
    st.markdown(f"**Grid Discovery Opportunity:** {grid_discovery_opportunity:.2f}")


elif section == "Monthly Demand Charges":
    st.subheader("📈 Monthly Demand Charges")

    monthly_peaks = data.loc[data.groupby(data['timestamp'].dt.to_period('M'))['load_kW'].idxmax()].copy()
    monthly_peaks['month'] = monthly_peaks['timestamp'].dt.strftime('%b %Y')
    monthly_peaks['month_num'] = monthly_peaks['timestamp'].dt.month

    monthly_peaks['demand_charge'] = monthly_peaks.apply(
        lambda row: row['load_kW'] * all_in_summer_demand_charge if row['month_num'] in [6, 7, 8, 9]
        else row['load_kW'] * annual_demand_charge_rate,
        axis=1
    )
    monthly_peaks['month_short'] = pd.to_datetime(monthly_peaks['timestamp']).dt.strftime('%b')
    monthly_avg_demand['month_short'] = pd.to_datetime(monthly_avg_demand['timestamp']).dt.strftime('%b')

    # Interactive Plotly-style stacked bar chart (non-interactive config)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15, subplot_titles=(
        "Monthly Demand Charges: Peak-based vs Avg-based",
        "Monthly Demand Charge Reduction Opportunity"
    ))

    fig.add_trace(go.Bar(x=monthly_peaks["month_short"], y=monthly_peaks["demand_charge"], name="Peak-based Charge", marker_color="tomato"), row=1, col=1)
    fig.add_trace(go.Bar(x=monthly_avg_demand["month_short"], y=monthly_avg_demand["avg_demand_charge"], name="Avg-based Charge", marker_color="steelblue", opacity=0.7), row=1, col=1)

    savings = monthly_peaks["demand_charge"].values - monthly_avg_demand["avg_demand_charge"].values
    fig.add_trace(go.Bar(x=monthly_peaks["month_short"], y=savings, name="Potential Savings", marker_color="green"), row=2, col=1)

    fig.update_layout(
        height=600,
        showlegend=True,
        barmode='group',
    )
    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

    st.markdown(f"**Total Annual Demand Charges:** ${monthly_peaks['demand_charge'].sum():,.2f}")
    st.markdown(f"**Total Annual Demand Charges (Avg-Based):** ${avg_demand_charge_total:,.2f}")
    st.markdown(f"**Potential Annual Demand Savings:** ${monthly_peaks['demand_charge'].sum() - avg_demand_charge_total:,.2f}")



elif section == "Top 10 Peak Days":
    st.subheader("🔟 Top 10 Peak Days")

    # Identify top 10 daily peaks
    top_10_peak_days = data.resample('D', on='timestamp')['load_kW'].max().nlargest(10)

    # Extract timestamp of each day's peak
    top_10_peaks_with_time = []
    for day in top_10_peak_days.index:
        day_data = data[data['timestamp'].dt.date == day.date()]
        peak_row = day_data.loc[day_data['load_kW'].idxmax()]
        top_10_peaks_with_time.append({
            'Date': day.strftime('%Y-%m-%d'),
            'Time of Peak': peak_row['timestamp'].strftime('%H:%M'),
            'Peak Load (kW)': round(peak_row['load_kW'], 2)
        })

    # Display in table format
    top_10_df = pd.DataFrame(top_10_peaks_with_time)
    st.dataframe(top_10_df)


elif section == "Hourly Load Heatmap":
    st.subheader("🕒 Hourly Load Heatmap (24hr × 365 Days)")

    heatmap_data = data.copy()
    heatmap_data['date'] = heatmap_data['timestamp'].dt.date
    heatmap_data['hour'] = heatmap_data['timestamp'].dt.hour

    pivot_table = heatmap_data.pivot_table(
        index='hour',
        columns='date',
        values='load_kW',
        aggfunc='mean'
    )

    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=[d.strftime('%b %d') for d in pivot_table.columns],
        y=[f"{h}:00" for h in pivot_table.index],
        colorscale='Viridis',
        colorbar=dict(title='Load (kW)')
    ))

    fig.update_layout(
        xaxis_nticks=30,
        xaxis_title='Date',
        yaxis_title='Hour of Day',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

elif section == "Monthly Peak and Average Demand":
    st.subheader("📈 Monthly Peak and Average Demand")

    monthly_peak_profile = data.resample('ME', on='timestamp')['load_kW'].max()
    monthly_avg_profile = data.resample('ME', on='timestamp')['load_kW'].mean()
    months = monthly_peak_profile.index.strftime('%b %Y')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=months,
        y=monthly_peak_profile.values,
        name='Peak Demand (kW)',
        marker_color='darkblue'
    ))
    fig.add_trace(go.Scatter(
        x=months,
        y=monthly_avg_profile.values,
        name='Avg Demand (kW)',
        mode='lines+markers',
        line=dict(color='orange', dash='dash')
    ))

    for x, peak_y, avg_y in zip(months, monthly_peak_profile.values, monthly_avg_profile.values):
        label_y = avg_y + (peak_y - avg_y) * 0.15
        fig.add_annotation(x=x, y=label_y, text=f"{avg_y:.1f}", showarrow=False,
                            font=dict(size=10, color="orange"), align="center")

    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Demand (kW)',
        barmode='group',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

elif section == "Daily Energy Consumption":
    st.subheader("📅 Daily Energy Consumption")

    daily_energy_consumption = data.resample('D', on='timestamp')['energy_kWh'].sum().reset_index()
    daily_energy_consumption['label'] = daily_energy_consumption['timestamp'].dt.strftime('%b %d')

    fig = px.line(
        daily_energy_consumption,
        x='label',
        y='energy_kWh',
        labels={'label': 'Date', 'energy_kWh': 'Energy Consumption (kWh)'},
    )
    fig.update_layout(xaxis_title='Date', yaxis_title='Energy Consumption (kWh)')
    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})


elif section == "Load Variability":
    st.subheader("📈 Load Variability")

    average_load_daily = data.resample('D', on='timestamp')['load_kW'].mean()
    average_load_weekly = data.resample('W', on='timestamp')['load_kW'].mean()
    average_load_monthly = data.resample('ME', on='timestamp')['load_kW'].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=average_load_daily.index, y=average_load_daily.values,
        mode='lines', name='Daily Avg Load', line=dict(color='steelblue')
    ))
    fig.add_trace(go.Scatter(
        x=average_load_weekly.index, y=average_load_weekly.values,
        mode='lines', name='Weekly Avg Load', line=dict(color='orange', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=average_load_monthly.index, y=average_load_monthly.values,
        mode='lines+markers', name='Monthly Avg Load', line=dict(color='green', dash='dot')
    ))

    fig.update_layout(
        xaxis=dict(
            title='Date',
            tickformat='%b %d',
            tickangle=45
        ),
        yaxis=dict(title='Average Load (kW)'),
        height=550,
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.3,
            xanchor='center',
            x=0.5
        )
    )

    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})


elif section == "Annual Load Duration Curve":
    st.subheader("📉 Annual Load Duration Curve")

    # Sort load data in ascending order
    sorted_load = data['load_kW'].sort_values(ascending=False).reset_index(drop=True)

    # Convert 15-minute intervals to cumulative hours
    hour_indices = np.arange(1, len(sorted_load) + 1) / 4  # 4 intervals per hour

    # Define thresholds
    peak_demand = sorted_load.max()
    base_load_threshold = sorted_load.quantile(0.05)
    critical_load_threshold = 0.75 * peak_demand

    # Find load periods
    high_load_periods = data[data['load_kW'] > data['load_kW'].quantile(0.9)]
    critical_load_periods = data[data['load_kW'] >= critical_load_threshold]

    # Plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hour_indices,
        y=sorted_load,
        mode='lines',
        name='Load Duration Curve',
        line=dict(color='steelblue')
    ))

    # Add threshold lines
    fig.add_hline(y=base_load_threshold, line_dash="dash", line_color="green",
                  annotation_text=f"Base Load: {base_load_threshold:.1f} kW", annotation_position="top right")
    fig.add_hline(y=critical_load_threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Critical Load: {critical_load_threshold:.1f} kW", annotation_position="top right")

    fig.update_layout(
        xaxis_title='Cumulative Hours',
        yaxis_title='Load (kW)',
        height=500,
        legend=dict(orientation="h", y=-0.2)
    )

    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

    # Summary metrics
    st.write(f"**Base Load Threshold (kW):** {round(base_load_threshold, 2)}")
    st.write(f"**Critical Load Threshold (kW):** {round(critical_load_threshold, 2)}")
    st.write(f"**High Load Periods (>90th percentile):** {len(high_load_periods)}")
    st.write(f"**Critical Load Periods (≥75% of Peak Demand):** {len(critical_load_periods)}")

    st.markdown("---")

    st.subheader("📈 Slope Analysis with Percentile Thresholds")

    # Sort the load data
    ldc = data['load_kW'].sort_values(ascending=False).reset_index(drop=True)

    # Use cumulative hour indices (15-minute interval → 4 per hour)
    hour_indices = np.arange(1, len(ldc) + 1) / 4

    # Percentile thresholds
    p90 = ldc.quantile(0.90)
    p50 = ldc.quantile(0.50)
    p10 = ldc.quantile(0.10)

    # Slope calculations
    slope_high_mid = (p50 - p90) / (0.50 - 0.90)
    slope_mid_low = (p10 - p50) / (0.10 - 0.50)

    # Plot with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hour_indices, y=ldc, mode='lines', name='Load Duration Curve (kW)', line=dict(color='blue')))
    fig.add_hline(y=p90, line_dash='dash', line_color='green',
                  annotation_text=f"90th percentile: {p90:.1f} kW", annotation_position="top right")
    fig.add_hline(y=p50, line_dash='dash', line_color='orange',
                  annotation_text=f"50th percentile: {p50:.1f} kW", annotation_position="top right")
    fig.add_hline(y=p10, line_dash='dash', line_color='red',
                  annotation_text=f"10th percentile: {p10:.1f} kW", annotation_position="top right")

    fig.update_layout(
        xaxis_title='Cumulative Hours',
        yaxis_title='Load (kW)',
        height=500,
        legend=dict(orientation="h", y=-0.2)
    )

    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

    # Display metrics
    st.markdown("### Slope Summary")
    st.write(f"90th percentile load: {p90:.2f} kW")
    st.write(f"50th percentile load: {p50:.2f} kW")
    st.write(f"10th percentile load: {p10:.2f} kW")
    st.markdown("---")
    st.write(f"**Slope (90% to 50%)**: {slope_high_mid:.2f} kW/percentile")
    st.write(f"**Slope (50% to 10%)**: {slope_mid_low:.2f} kW/percentile")


    st.markdown("---")

    st.subheader("📊 Slope Analysis with Rate of Change")

    # Sort the load data
    ldc = data['load_kW'].sort_values(ascending=False).reset_index(drop=True)

    # Cumulative hours based on 15-min interval
    hour_indices = np.arange(1, len(ldc) + 1) / 4

    # First-order difference
    ldc_derivative = ldc.diff().fillna(0)

    # Create dual-axis plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # LDC trace
    fig.add_trace(
        go.Scatter(x=hour_indices, y=ldc, name="Load Duration Curve (kW)", line=dict(color='blue')),
        secondary_y=False
    )

    # Rate of Change trace
    fig.add_trace(
        go.Scatter(x=hour_indices, y=ldc_derivative, name="Rate of Change (ΔkW)", line=dict(color='red', dash='dash')),
        secondary_y=True
    )

    # Layout
    fig.update_layout(
        xaxis_title="Cumulative Hours",
        yaxis_title="Load (kW)",
        yaxis2_title="Rate of Change (ΔkW)",
        height=500,
        legend=dict(orientation="h", y=-0.2)
    )

    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})



elif section == "Peak Day Load Duration Curve":
    st.subheader("📅 Peak Day Load Duration Curve")

    # Identify the peak day
    daily_peaks = data.resample('D', on='timestamp')['load_kW'].max()
    max_peak_day = daily_peaks.idxmax().date()

    # Pull and sort the peak day load data
    day_data = data[data['timestamp'].dt.date == max_peak_day].copy()
    ldc_day = day_data['load_kW'].sort_values(ascending=False).reset_index(drop=True)

    # Convert intervals to cumulative hours
    hour_indices = np.arange(1, len(ldc_day) + 1) / 4

    # Plot
    fig = px.line(
        x=hour_indices,
        y=ldc_day,
        labels={'x': 'Cumulative Hours', 'y': 'Load (kW)'}
    )
    fig.update_layout(
        title=f'Peak Day Load Duration Curve: {max_peak_day.strftime("%B %d")}',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

    st.markdown("---")

    st.subheader("📈 Slope Analysis with Percentile Thresholds")

    # Find peak day
    daily_peaks = data.resample('D', on='timestamp')['load_kW'].max()
    max_peak_day = daily_peaks.idxmax().date()
    day_data = data[data['timestamp'].dt.date == max_peak_day].copy()
    ldc_day = day_data['load_kW'].sort_values(ascending=False).reset_index(drop=True)

    # Convert index to hours (15-minute intervals = 0.25 hours each)
    hour_indices = np.arange(1, len(ldc_day) + 1) / 4

    # Get percentiles and slopes
    p90 = ldc_day.quantile(0.90)
    p50 = ldc_day.quantile(0.50)
    p10 = ldc_day.quantile(0.10)

    slope_day_high_mid = (p50 - p90) / (0.50 - 0.90)
    slope_day_mid_low = (p10 - p50) / (0.10 - 0.50)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hour_indices, y=ldc_day, mode='lines',
                             name='Load Duration Curve (kW)', line=dict(color='blue')))
    fig.add_hline(y=p90, line_dash='dash', line_color='green',
                  annotation_text=f"90th percentile: {p90:.1f} kW", annotation_position="top right")
    fig.add_hline(y=p50, line_dash='dash', line_color='orange',
                  annotation_text=f"50th percentile: {p50:.1f} kW", annotation_position="top right")
    fig.add_hline(y=p10, line_dash='dash', line_color='red',
                  annotation_text=f"10th percentile: {p10:.1f} kW", annotation_position="top right")

    fig.update_layout(
        xaxis_title='Cumulative Hours',
        yaxis_title='Load (kW)',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

    st.markdown("### Slope Analysis")
    st.write(f"90th percentile load: {p90:.2f} kW")
    st.write(f"50th percentile load: {p50:.2f} kW")
    st.write(f"10th percentile load: {p10:.2f} kW")
    st.write(f"**Slope (90% to 50%)**: {slope_day_high_mid:.2f} kW/percentile")
    st.write(f"**Slope (50% to 10%)**: {slope_day_mid_low:.2f} kW/percentile")

    st.markdown("---")

    st.subheader("📊 Slope Analysis with Rate of Change")

    # Identify peak day
    daily_peaks = data.resample('D', on='timestamp')['load_kW'].max()
    max_peak_day = daily_peaks.idxmax().date()
    day_data = data[data['timestamp'].dt.date == max_peak_day].copy()

    # Load Duration Curve & Derivative
    ldc_day = day_data['load_kW'].sort_values(ascending=False).reset_index(drop=True)
    ldc_derivative = ldc_day.diff().fillna(0)

    # Convert 15-min intervals to cumulative hours
    hour_indices = np.arange(1, len(ldc_day) + 1) / 4

    # Plot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=hour_indices, y=ldc_day, mode='lines',
                   name='Load Duration Curve (kW)', line=dict(color='blue')),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=hour_indices, y=ldc_derivative, mode='lines',
                   name='Rate of Change (ΔkW)', line=dict(color='red', dash='dash')),
        secondary_y=True
    )

    fig.update_layout(
        xaxis_title='Cumulative Hours',
        yaxis_title='Load (kW)',
        yaxis2_title='Rate of Change (ΔkW)',
        height=500,
        legend=dict(orientation="h", y=-0.2)
    )
    

    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})


elif section == "Load Duration Curve Comparison: Annual vs Peak Day":
    st.subheader("📊 Load Duration Curve Comparison: Annual vs Peak Day")

    # Annual Load Duration Curve
    ldc_annual = data['load_kW'].sort_values(ascending=False).reset_index(drop=True)
    p90_a = ldc_annual.quantile(0.90)
    p50_a = ldc_annual.quantile(0.50)
    p10_a = ldc_annual.quantile(0.10)

    slope_90_50_a = (p50_a - p90_a) / (0.50 - 0.90)
    slope_50_10_a = (p10_a - p50_a) / (0.10 - 0.50)

    # Peak Day Load Duration Curve
    daily_peaks = data.resample('D', on='timestamp')['load_kW'].max()
    max_peak_day = daily_peaks.idxmax().date()
    ldc_day = data[data['timestamp'].dt.date == max_peak_day]['load_kW'].sort_values(ascending=False).reset_index(drop=True)
    p90_d = ldc_day.quantile(0.90)
    p50_d = ldc_day.quantile(0.50)
    p10_d = ldc_day.quantile(0.10)

    slope_90_50_d = (p50_d - p90_d) / (0.50 - 0.90)
    slope_50_10_d = (p10_d - p50_d) / (0.10 - 0.50)

    # Percent change in slope
    slope_90_50_diff_pct = ((slope_90_50_d - slope_90_50_a) / abs(slope_90_50_a)) * 100
    slope_50_10_diff_pct = ((slope_50_10_d - slope_50_10_a) / abs(slope_50_10_a)) * 100

    # Assemble summary table
    comparison_df = pd.DataFrame({
        'Metric': ['90th Percentile Load', '50th Percentile Load', '10th Percentile Load',
                   'Slope (90–50%)', 'Slope (50–10%)'],
        'Annual (kW)': [f"{p90_a:.2f}", f"{p50_a:.2f}", f"{p10_a:.2f}", f"{slope_90_50_a:.2f}", f"{slope_50_10_a:.2f}"],
        f'Peak Day: {max_peak_day.strftime("%b %d")}': [f"{p90_d:.2f}", f"{p50_d:.2f}", f"{p10_d:.2f}",
                                                            f"{slope_90_50_d:.2f}", f"{slope_50_10_d:.2f}"],
        'Peak Day vs. Annual (%)': ['', '', '', f"{slope_90_50_diff_pct:.1f}%", f"{slope_50_10_diff_pct:.1f}%"]
    })

    st.dataframe(comparison_df)

    # Interpretation
    st.markdown("### 🧠 Why the Comparison Matters")
    st.markdown("""
    While the peak day will always be more extreme:

    - The degree to which it deviates from the annual curve tells you how spiky and rare those peaks are.
    - If your peak day is only **10–15% higher** than the annual curve → your system is **fairly balanced**.
    - If your peak day is **2× steeper** or more → you have **major short-duration peaks** that could be shaved for savings.

    **This analysis helps answer:**
    > “Is my peak behavior just a bad day — or a recurring risk?”
    """)

elif section == "Seasonal Weekly Load Profiles":
    st.subheader("📆 Seasonal Weekly Load Profiles")

    season_mapping = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    data['season'] = data['timestamp'].dt.month.map(season_mapping)
    data['day_of_week'] = data['timestamp'].dt.day_name()

    fig = go.Figure()
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        season_data = data[data['season'] == season]
        weekly_avg = season_data.groupby('day_of_week')['load_kW'].mean()
        weekly_avg = weekly_avg.reindex(weekdays)
        fig.add_trace(go.Scatter(x=weekly_avg.index, y=weekly_avg.values, mode='lines', name=f'{season} Profile'))

    fig.update_layout(
        title='Seasonal Load Profiles',
        xaxis_title='Day of the Week',
        yaxis_title='Weekly Average Load (kW)',
        legend_title='Season',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

elif section == "Average Daily Load for Each Season":
    st.subheader("🌤️ Average Daily Load for Each Season")

    fig = go.Figure()
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        season_data = data[data['season'] == season].copy()
        season_data['hour'] = season_data['timestamp'].dt.hour
        hourly_avg = season_data.groupby('hour')['load_kW'].mean()
        fig.add_trace(go.Scatter(x=hourly_avg.index, y=hourly_avg.values, mode='lines', name=f'{season} Profile'))

    fig.update_layout(
        title='Average Daily Load for Each Season',
        xaxis_title='Hour of the Day',
        yaxis_title='Average Load (kW)',
        xaxis=dict(tickmode='array', tickvals=list(range(24)), ticktext=[f'{h:02}:00' for h in range(24)]),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

elif section == "Load Trends for Top 3 Peak Days":
    st.subheader("🚀 Load Trends for Top 3 Peak Days")

    def ordinal(n):
        return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"

    top_3_peak_days = data.resample('D', on='timestamp')['load_kW'].max().nlargest(3)
    fig = go.Figure()

    max_energy_global = 0
    for day in top_3_peak_days.index:
        daily_data = data[data['timestamp'].dt.date == day.date()].copy()
        daily_data['hour'] = daily_data['timestamp'].dt.hour
        daily_data['minute'] = daily_data['timestamp'].dt.minute
        daily_data['hour_decimal'] = daily_data['hour'] + daily_data['minute'] / 60.0
        label = f"{day.strftime('%B')} {ordinal(day.day)}"
        fig.add_trace(go.Scatter(x=daily_data['hour_decimal'], y=daily_data['load_kW'], mode='lines', name=label))
        max_energy_global = max(max_energy_global, daily_data['load_kW'].max())

    fig.update_layout(
        title='Load Trends for Top 3 Peak Days',
        xaxis=dict(title='Hour of the Day', tickmode='array', tickvals=list(range(24)), ticktext=[f"{h:02}:00" for h in range(24)]),
        yaxis=dict(title='Load (kW)', range=[0, max_energy_global]),
        yaxis2=dict(overlaying='y', side='right', title='% of Load', range=[0, 100]),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})
