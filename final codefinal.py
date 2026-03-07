import streamlit as st
import pandas as pd
import plotly.express as px
import glob
import os

st.set_page_config(
    page_title="Classroom Behavior Dashboard",
    page_icon="🎓",
    layout="wide"
)

# Custom Styling (Even with dark mode, we can add some flair)
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #f9a01b;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-container {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

st.title("🎓 Classroom Behavior Dashboard")

@st.cache_data(ttl=10) # Cache but allow updating every 10 seconds if data changes
def load_data():
    # Find newest excel sheet or csv
    excel_files = glob.glob("*.xlsx")
    csv_files = glob.glob("*.csv")
    
    all_files = excel_files + csv_files
    if not all_files:
        return None, None
        
    # Get the latest modified file
    latest_file = max(all_files, key=os.path.getmtime)
    
    try:
        if latest_file.endswith('.xlsx'):
            df = pd.read_excel(latest_file)
        else:
            df = pd.read_csv(latest_file)
        return df, latest_file
    except Exception as e:
        st.error(f"Error loading {latest_file}: {e}")
        return None, None

df, filename = load_data()

if df is None:
    st.warning("No Excel or CSV data files found in the directory. Please ensure the program has run and generated data.")
else:
    st.markdown(f"**Loaded Data From:** `{filename}`")
    
    # Try to identify main columns regardless of exact casing
    cols = list(df.columns)
    col_lower = {c.lower(): c for c in cols}
    
    behavior_col = col_lower.get('behavior') or col_lower.get('label')
    score_col = col_lower.get('score') or col_lower.get('confidence %') or col_lower.get('confidence')
    
    # Optional metrics at the top
    metric1, metric2, metric3 = st.columns(3)
    with metric1:
        st.metric("Total Records", len(df))
    with metric2:
        if behavior_col:
            st.metric("Unique Behaviors", df[behavior_col].nunique())
    with metric3:
        if score_col:
            # Handle if the column is categorical or string by converting to numeric if possible
            try:
                mean_score = pd.to_numeric(df[score_col], errors='coerce').mean()
                st.metric(f"Average {score_col.title()}", f"{mean_score:.2f}")
            except:
                pass
            
    st.markdown("---")
    
    # Visualizations row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Trend Line Graph")
        # Line graph for score, confidence or pitch
        if score_col:
            # Convert to numeric just in case there are strings
            df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
            
            fig_line = px.line(df, y=score_col, title=f"Trend over Time ({score_col.title()})",
                               markers=True, line_shape="spline",
                               color_discrete_sequence=['#ff4b4b'])
            fig_line.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            # Fallback to a numeric column
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                 y_col = numeric_cols[0]
                 fig_line = px.line(df, y=y_col, title=f"Trend over Time ({y_col})",
                                    markers=True, line_shape="spline",
                                    color_discrete_sequence=['#ff4b4b'])
                 fig_line.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                 st.plotly_chart(fig_line, use_container_width=True)
            else:
                 st.info("No numeric columns found for line graph.")

    with col2:
        st.subheader("🥧 Behavior Distribution")
        if behavior_col:
            behavior_counts = df[behavior_col].value_counts().reset_index()
            behavior_counts.columns = [behavior_col, 'Count']
            
            fig_pie = px.pie(behavior_counts, names=behavior_col, values='Count', 
                             title="Behavior Breakdown",
                             hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Plotly)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', 
                                  hoverinfo="label+percent+name")
            fig_pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No 'Behavior' column found for Pie Chart.")

    st.markdown("---")
    
    # Raw Data Section
    st.subheader("📋 Raw Data (Excel Sheet Data)")
    st.markdown("This raw data is directly taken from the latest generated excel sheet.")
    
    # st.dataframe with custom height
    st.dataframe(
        df,
        use_container_width=True,
        height=400
    )
