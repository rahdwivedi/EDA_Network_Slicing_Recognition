
# 1. Imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from scipy import stats
from io import StringIO
from datetime import datetime
import time
import warnings
import os

#------------------------------------------------------------------------------------------------------------------------------------------------------
  
# 2. Page Configuration
st.set_page_config(page_title="Network Slice EDA Dashboard", layout="wide")

#------------------------------------------------------------------------------------------------------------------------------------------------------

# 3. Hardcoded Credentials (same as you did)
USER_CREDENTIALS = {
    "admin": "admin123",
    "analyst": "net2025"
}

#------------------------------------------------------------------------------------------------------------------------------------------------------

# 4. Login Page
def login_page():
    st.title("🔐 Login to Network Slice Dashboard")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
        else:
            st.error("❌ Invalid username or password")

#------------------------------------------------------------------------------------------------------------------------------------------------------
  
# 5. Load Dataset
import pandas as pd
import warnings

# Optional: Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load CSV files instead of using SQL Server
df_test = pd.read_csv("test_dataset.csv")
df_train = pd.read_csv("train_dataset.csv")


#------------------------------------------------------------------------------------------------------------------------------------------------------

# 6. Sidebar & Filters
def sidebar_filters():
    st.sidebar.header("Filter Options")
    selected_category = st.sidebar.selectbox(
        "Select Analysis Category:",
        ["All", "Descriptive Analysis", "Diagnostic Analysis", "Predictive Analysis","EDA"]
    )
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
    return selected_category

#------------------------------------------------------------------------------------------------------------------------------------------------------
  
# 7. Main Routing Logic

# =============================== #
# 📊 Descriptive Analysis
# =============================== #
def network_slcice_eda():
    st.title("Network Slicing Recognition - Exploratory Data Analysis")
    
    # --- Descriptive Analysis ---
    if selected_filter == "All" or selected_filter == "Descriptive Analysis":
        # --- KPIs Section ---
        st.header("Key Performance Indicators - Descriptive Analysis")
    
        # KPIs based on Descriptive Analysis
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Records", df_train.shape[0])
        col2.metric("Average Packet Delay (ms)", round(df_train["Packet_delay"].mean(), 2))
        col3.metric("Average Packet Loss Rate", f"{df_train['Packet_Loss_Rate'].mean():.6f}")
        col4.metric("Most Common Slice", df_train["slice_type_label"].mode()[0])
        col5.metric("Avg IoT Devices per Record", round(df_train["IoT_Devices"].mean(), 2))

    
    # --- Descriptive Analysis ---
    if selected_filter == "All" or selected_filter == "Descriptive Analysis":
        st.header("Descriptive Analysis")

        # Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Usage Count by Application Category")
            
            usage_data = {
                'App': [
                    'AR/VR/Gaming',
                    'Healthcare',
                    'Industry 4.0',
                    'Public Safety',
                    'Smart City & Home',
                    'Smart Transportation',
                    'Smartphone'
                ],
                'Usage_Count': [
                    df_train['AR_VR_Gaming'].sum(),
                    df_train['Healthcare'].sum(),
                    df_train['Industry_4_0'].sum(),
                    df_train['Public_Safety'].sum(),
                    df_train['Smart_City_Home'].sum(),
                    df_train['Smart_Transportation'].sum(),
                    df_train['Smartphone'].sum()
                ]
            }
        
            usage_df = pd.DataFrame(usage_data).sort_values(by='Usage_Count', ascending=False)
        
            # Matplotlib chart rendering with Seaborn
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.barplot(data=usage_df, x='Usage_Count', y='App', hue='App', dodge=False, palette='viridis', ax=ax1, legend=False)
            ax1.set_title('Usage Count by Application Category')
            ax1.set_xlabel('Usage Count')
            ax1.set_ylabel('Application')
            st.pyplot(fig1)
        
        with col2:
            st.subheader("Packet Delay Distribution with Min/Max Lines")
        
            min_delay = df_train['Packet_delay'].min()
            max_delay = df_train['Packet_delay'].max()
        
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.histplot(df_train['Packet_delay'], bins=30, kde=True, color='skyblue', ax=ax2)
            ax2.axvline(min_delay, color='green', linestyle='--', linewidth=2, label=f"Min: {min_delay}")
            ax2.axvline(max_delay, color='red', linestyle='--', linewidth=2, label=f"Max: {max_delay}")
            ax2.set_title('Distribution of Packet Delay with Min and Max Highlighted')
            ax2.set_xlabel('Packet Delay')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            st.pyplot(fig2)

        # Row 2
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Average Packet Loss Rate by Time Slot")
        
            packet_loss_by_time = (
                df_train.groupby('Time')['Packet_Loss_Rate']
                .mean()
                .reset_index(name='Avg_Packet_Loss')
                .sort_values(by='Avg_Packet_Loss', ascending=False)
            )
        
            top_time_slot = packet_loss_by_time.iloc[0]
        
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            sns.barplot(
                data=packet_loss_by_time,
                x='Time',
                y='Avg_Packet_Loss',
                hue='Time',
                dodge=False,
                palette='flare',
                legend=False,
                ax=ax3
            )
            ax3.set_title('Average Packet Loss Rate by Time Slot')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Average Packet Loss Rate')
            ax3.axhline(
                y=top_time_slot['Avg_Packet_Loss'],
                color='red',
                linestyle='--',
                label=f"Highest: {top_time_slot['Time']}"
            )
            ax3.legend()
            plt.setp(ax3.get_xticklabels(), rotation=45)
            st.pyplot(fig3)
        
        with col4:
            st.subheader("Total Records: LTE vs 5G")
        
            lte_5g_usage = (
                df_train.groupby('LTE_5G')
                .size()
                .reset_index(name='Total_Records')
                .sort_values(by='Total_Records', ascending=False)
            )
        
            fig4, ax4 = plt.subplots(figsize=(8, 5))
            sns.barplot(
                data=lte_5g_usage,
                x='LTE_5G',
                y='Total_Records',
                hue='LTE_5G',
                dodge=False,
                palette='coolwarm',
                legend=False,
                ax=ax4
            )
            ax4.set_title('Total Records: LTE vs 5G')
            ax4.set_xlabel('Network Type')
            ax4.set_ylabel('Total Records')
            st.pyplot(fig4)


        # Row 3
        col5, col6 = st.columns(2)
        
        with col5:
            st.subheader("Top 5 High-Latency Records")
        
            top_5_latency = df_train.sort_values(by='Packet_delay', ascending=False).head(5)
            top_5_latency_reset = top_5_latency.reset_index()
        
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=top_5_latency_reset,
                x='index',
                y='Packet_delay',
                hue='index',
                dodge=False,
                palette='Reds',
                legend=False,
                ax=ax5
            )
            ax5.set_title('Top 5 High-Latency Records')
            ax5.set_xlabel('Record Index')
            ax5.set_ylabel('Packet Delay')
            st.pyplot(fig5)
        
        with col6:
            st.subheader("Total IoT Devices by Time Slot")
        
            iot_by_time = (
                df_train.groupby('Time')['IoT_Devices']
                .sum()
                .reset_index(name='Total_IoT')
                .sort_values(by='Total_IoT', ascending=False)
            )
        
            fig6, ax6 = plt.subplots(figsize=(12, 6))
            sns.barplot(
                data=iot_by_time,
                x='Time',
                y='Total_IoT',
                hue='Time',
                dodge=False,
                palette='mako',
                legend=False,
                ax=ax6
            )
            ax6.set_title('Total IoT Devices by Time Slot')
            ax6.set_xlabel('Time')
            ax6.set_ylabel('Total IoT Devices')
            plt.setp(ax6.get_xticklabels(), rotation=45)
            st.pyplot(fig6)


        # Row 4
        col7, col8 = st.columns(2)
        
        with col7:
            st.subheader("Application Usage Spread by LTE vs 5G")
        
            # Group and rename columns
            app_spread = (
                df_train.groupby('LTE_5G')[['AR_VR_Gaming', 'Healthcare', 'Smart_Transportation']]
                .sum()
                .reset_index()
            )
            app_spread.rename(columns={
                'AR_VR_Gaming': 'Gaming',
                'Smart_Transportation': 'Transport'
            }, inplace=True)
        
            # Melt for grouped bar plot
            app_spread_melted = app_spread.melt(
                id_vars='LTE_5G',
                var_name='Application',
                value_name='Usage_Count'
            )
        
            fig7, ax7 = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=app_spread_melted,
                x='LTE_5G',
                y='Usage_Count',
                hue='Application',
                palette='Set2',
                ax=ax7
            )
            ax7.set_title('Application Usage Spread by Network Type (LTE vs 5G)')
            ax7.set_xlabel('Network Type')
            ax7.set_ylabel('Total Usage Count')
            ax7.legend(title='Application')
            st.pyplot(fig7)
        
        with col8:
            st.subheader("Top 3 Time Slots for Healthcare Usage")
        
            healthcare_by_time = (
                df_train.groupby('Time')['Healthcare']
                .sum()
                .reset_index(name='Total_Healthcare')
                .sort_values(by='Total_Healthcare', ascending=False)
            )
            top_3_healthcare = healthcare_by_time.head(3)
        
            fig8, ax8 = plt.subplots(figsize=(8, 5))
            sns.barplot(
                data=top_3_healthcare,
                x='Time',
                y='Total_Healthcare',
                hue='Time',
                dodge=False,
                palette='Blues',
                legend=False,
                ax=ax8
            )
            ax8.set_title('Top 3 Time Slots for Healthcare Usage')
            ax8.set_xlabel('Time')
            ax8.set_ylabel('Total Healthcare Usage')
            st.pyplot(fig8)


        # Row 5
        col9, col10 = st.columns(2)
        
        with col9:
            st.subheader("Non-GBR Usage per Slice")
        
            # Group by slice type and label, summing Non-GBR
            non_gbr_usage = (
                df_train.groupby(['slice_Type', 'slice_type_label'])['Non_GBR']
                .sum()
                .reset_index(name='Non_GBR_Total')
            )
        
            # Create the plot
            fig9, ax9 = plt.subplots(figsize=(12, 6))
            sns.barplot(
                data=non_gbr_usage,
                x='slice_Type',
                y='Non_GBR_Total',
                hue='slice_type_label',
                dodge=True,
                palette='coolwarm',
                ax=ax9
            )
            ax9.set_title('Non-GBR Usage per Slice Type')
            ax9.set_xlabel('Slice Type')
            ax9.set_ylabel('Total Non-GBR Usage')
            ax9.legend(title='Slice Type Label')
            st.pyplot(fig9)
        
        with col10:
            st.subheader("Avg Packet Loss per Application")
        
            # Compute averages only for active usages
            avg_loss_arvr = df_train.loc[df_train['AR_VR_Gaming'] > 0, 'Packet_Loss_Rate'].mean()
            avg_loss_healthcare = df_train.loc[df_train['Healthcare'] > 0, 'Packet_Loss_Rate'].mean()
            avg_loss_transport = df_train.loc[df_train['Smart_Transportation'] > 0, 'Packet_Loss_Rate'].mean()
        
            avg_loss_df = pd.DataFrame({
                'App': ['AR/VR/Gaming', 'Healthcare', 'Smart Transportation'],
                'Avg_Loss': [avg_loss_arvr, avg_loss_healthcare, avg_loss_transport]
            })
        
            # Create the plot
            fig10, ax10 = plt.subplots(figsize=(8, 5))
            sns.barplot(
                data=avg_loss_df,
                x='App',
                y='Avg_Loss',
                hue='App',
                dodge=False,
                palette='muted',
                legend=False,
                ax=ax10
            )
            ax10.set_title('Average Packet Loss Rate per Application')
            ax10.set_xlabel('Application')
            ax10.set_ylabel('Average Packet Loss Rate')
            ax10.tick_params(axis='x', rotation=15)
            st.pyplot(fig10)


        # Row 6
        col11, col12 = st.columns(2)
        
        with col11:
            st.subheader("LTE/5G Category Frequency")
        
            # Count category frequencies
            category_counts = df_train['LTE_5g_Category'].value_counts().reset_index()
            category_counts.columns = ['LTE_5g_Category', 'Record_Count']
        
            # Create the plot
            fig11, ax11 = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=category_counts,
                x='LTE_5g_Category',
                y='Record_Count',
                hue='LTE_5g_Category',
                dodge=False,
                palette='pastel',
                legend=False,
                ax=ax11
            )
            ax11.set_title('Frequency of LTE/5G Categories')
            ax11.set_xlabel('LTE/5G Category')
            ax11.set_ylabel('Record Count')
            ax11.tick_params(axis='x', rotation=45)
            st.pyplot(fig11)
        
        with col12:
            st.subheader("Delay Fluctuation by Time")
        
            # Average packet delay by time
            delay_by_time = (
                df_train.groupby('Time')['Packet_delay']
                .mean()
                .reset_index(name='Avg_Delay')
            )
        
            # Sort for visualization
            delay_sorted_desc = delay_by_time.sort_values(by='Avg_Delay', ascending=False)
        
            # Create the plot
            fig12, ax12 = plt.subplots(figsize=(12, 6))
            sns.barplot(
                data=delay_sorted_desc,
                x='Time',
                y='Avg_Delay',
                hue='Time',
                dodge=False,
                palette='coolwarm',
                legend=False,
                ax=ax12
            )
            ax12.set_title('Average Packet Delay by Time')
            ax12.set_xlabel('Time')
            ax12.set_ylabel('Average Packet Delay')
            ax12.tick_params(axis='x', rotation=45)
            st.pyplot(fig12)


        # Row 7
        col13, _ = st.columns([1, 0.1])  # Single column for a wide bar chart
        
        with col13:
            st.subheader("Slice Type Label Breakdown")
        
            # Map slice_Type to human-readable labels
            slice_label_map = {
                1: 'eMBB',
                2: 'URLLC',
                3: 'mMTC'
            }
        
            # Apply mapping
            df_train['Slice_Label'] = df_train['slice_Type'].map(slice_label_map)
        
            # Count records per slice label
            slice_counts = (
                df_train.groupby('Slice_Label')
                .size()
                .reset_index(name='Record_Count')
            )
        
            # --- Visualization ---
            fig13, ax13 = plt.subplots(figsize=(8, 5))
            sns.barplot(
                data=slice_counts,
                x='Slice_Label',
                y='Record_Count',
                hue='Slice_Label',  # Assign hue to avoid warning
                dodge=False,
                palette='Set2',
                legend=False,
                ax=ax13
            )
            ax13.set_title('Record Count by Slice Type Label')
            ax13.set_xlabel('Slice Type Label')
            ax13.set_ylabel('Record Count')
            st.pyplot(fig13)

#------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # --- Diagnostic Analysis ---
    if selected_filter == "All" or selected_filter == "Diagnostic Analysis":
        # --- KPIs Section ---
        st.header("Key Performance Indicators - Diagnostic Analysis")
    
        # KPIs based on Diagnostic Analysis
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("High Delay Records (>100ms)", df_train[df_train["Packet_delay"] > 100].shape[0])
        col2.metric("Zero Packet Loss Records", df_train[df_train["Packet_Loss_Rate"] == 0].shape[0])
        col3.metric("GBR Usage (%)", f"{(df_train['GBR'].sum() / df_train.shape[0] * 100):.2f}%")
        col4.metric("LTE vs 5G Ratio", f"{df_train['LTE_5G'].value_counts().to_dict().get(0, 0)}:{df_train['LTE_5G'].value_counts().to_dict().get(1, 0)}")
        col5.metric("Smartphone Usage Rate (%)", f"{(df_train['Smartphone'].sum() / df_train.shape[0] * 100):.2f}%")
    
    
    # --- Diagnostic Analysis ---
    if selected_filter == "All" or selected_filter == "Diagnostic Analysis":
        st.header("Diagnostic Analysis")

        # Row 1
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Use Case Combinations Leading to Highest Delay")
            
            # Calculate average delay by use case combinations
            delay_by_use_case = (
                df_train.groupby(['AR_VR_Gaming', 'Healthcare', 'Industry_4_0'])['Packet_delay']
                .mean().reset_index(name='Avg_Delay').sort_values(by='Avg_Delay', ascending=False)
            ).head(10)
            
            # Create readable combo label
            delay_by_use_case['Combo'] = delay_by_use_case.apply(
                lambda row: f"AR:{row['AR_VR_Gaming']}, HC:{row['Healthcare']}, I4.0:{row['Industry_4_0']}", axis=1)
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=delay_by_use_case, x='Avg_Delay', y='Combo', hue='Combo', palette='magma', ax=ax, legend=False)
            ax.set_title('Top 10 Use Case Combinations → Avg Delay')
            st.pyplot(fig)
            plt.close(fig)
       
        with col2:
            st.subheader("IoT vs Non‑IoT Traffic by Slice Type")
            iot_slice_counts = (
                df_train.groupby(['IoT_Devices', 'slice_type_label'])
                .size().reset_index(name='Count')
            )
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=iot_slice_counts, x='IoT_Devices', y='Count', hue='slice_type_label', dodge=True, palette='Set2', ax=ax)
            ax.set_title('IoT vs Non‑IoT Traffic by Slice')
            st.pyplot(fig)

        # Row 2
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Avg Packet Delay by LTE/5G Category")
            
            # Calculate average packet delay by LTE/5G category
            avg_latency = (
                df_train.groupby('LTE_5g_Category')['Packet_delay']
                .mean().reset_index(name='Avg_Latency')
            )
        
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=avg_latency,
                x='LTE_5g_Category',
                y='Avg_Latency',
                hue='LTE_5g_Category',  # Add hue to use palette without warning
                palette='muted',
                dodge=False,
                ax=ax,
                legend=False  # Hide redundant legend
            )
            ax.set_title('Avg Packet Delay · LTE/5G Category')
            st.pyplot(fig)
            plt.close(fig)
      
        with col4:
            st.subheader("Traffic by Slice During Peak Hours")
            traffic_by_time_slice = (
                df_train.groupby(['Time', 'slice_type_label'])
                .size().reset_index(name='Count')
            )
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=traffic_by_time_slice, x='Time', y='Count', hue='slice_type_label', palette='tab10', ax=ax)
            ax.set_title('Traffic by Slice Type Over Time')
            plt.setp(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
        
        # Row 3
        col5, col6 = st.columns(2)
        with col5:
            st.subheader("Delay Std Dev by Slice Type")
            
            # Compute standard deviation of Packet Delay per slice type
            delay_std = df_train.groupby('slice_type_label')['Packet_delay'].std().reset_index(name='StdDev')
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=delay_std,
                x='slice_type_label',
                y='StdDev',
                hue='slice_type_label',    # Add hue to avoid palette warning
                palette='magma',
                dodge=False,
                ax=ax,
                legend=False               # Disable redundant legend
            )
            
            ax.set_title('Packet Delay Std Dev · Slice Type')
            st.pyplot(fig)
            plt.close(fig)

 
        with col6:
            st.subheader("Slice Usage for Public Safety Apps")
        
            # Filter for Public Safety apps
            ps = df_train[df_train['Public_Safety'] == 1]
            pst = ps.groupby('slice_type_label').size().reset_index(name='Count')
        
            # Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(
                data=pst,
                x='slice_type_label',
                y='Count',
                hue='slice_type_label',   # Add hue to avoid FutureWarning
                palette='coolwarm',
                dodge=False,
                ax=ax,
                legend=False              # Remove redundant legend
            )
            
            ax.set_title('Public Safety App → Slice Allocation')
            st.pyplot(fig)
            plt.close(fig)

        
        # Row 4
        # Row 4
        col7, col8 = st.columns(2)
        
        with col7:
            st.subheader("Std Dev of Packet Delay by Time Slot")
        
            # Calculate standard deviation
            dt = df_train.groupby('Time')['Packet_delay'].std().reset_index(name='StdDev')
        
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=dt,
                x='Time',
                y='StdDev',
                hue='Time',            # Required to use palette safely
                palette='mako',
                dodge=False,
                ax=ax,
                legend=False           # Avoid legend repetition
            )
            plt.setp(ax.get_xticklabels(), rotation=45)
            ax.set_title('Delay Variability by Time Slot')
            
            st.pyplot(fig)
            plt.close(fig)            # ✅ Prevent too many open figures

        
        with col8:
            st.subheader("Avg Delay · GBR vs Non‑GBR")
        
            # Compute averages
            avg_gbr = df_train[df_train['GBR'] == 1]['Packet_delay'].mean()
            avg_non = df_train[df_train['Non_GBR'] == 1]['Packet_delay'].mean()
        
            # Create dataframe
            tmp = pd.DataFrame({'Traffic': ['GBR', 'Non‑GBR'], 'AvgDelay': [avg_gbr, avg_non]})
        
            # Plot
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(
                data=tmp,
                x='Traffic',
                y='AvgDelay',
                hue='Traffic',         # ✅ Required for using palette safely
                palette='Set2',
                dodge=False,
                ax=ax,
                legend=False           # ✅ No legend needed for this visual
            )
            ax.set_title('Average Delay: GBR vs Non‑GBR')
            st.pyplot(fig)
            plt.close(fig)             # ✅ Prevent figure overload

        
        # Row 5
        col9, col10 = st.columns(2)
        with col9:
            st.subheader("Packet Loss by AR/VR & Smartphone Usage")
            loss_pair = (
                df_train.groupby(['AR_VR_Gaming', 'Smartphone'])['Packet_Loss_Rate']
                .mean().reset_index(name='Avg_Loss')
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=loss_pair, x='Avg_Loss', y='Smartphone', hue='AR_VR_Gaming', palette='coolwarm', ax=ax)
            ax.set_title('Packet Loss → AR/VR & Smartphone')
            st.pyplot(fig)

        with col10:
            st.subheader("Overused Slices During High Traffic")
            high = df_train[(df_train['Packet_Loss_Rate'] > 0.03) | (df_train['Packet_delay'] > 100)]
            over = high.groupby('slice_type_label').size().reset_index(name='Count')
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Assign 'slice_type_label' to hue to avoid FutureWarning
            sns.barplot(
                data=over, 
                x='Count', 
                y='slice_type_label', 
                palette='autumn', 
                hue='slice_type_label', 
                legend=False, 
                dodge=False, 
                ax=ax
            )
        
            ax.set_title('Overused Slice Types (High Delay/Loss)')
            st.pyplot(fig)

        
        # Row 6
        col11, col12 = st.columns(2)
        with col11:
            st.subheader("Delay vs Packet Loss · Slice Type")
            dl = (
                df_train.groupby('slice_type_label')
                .agg(Avg_Delay=('Packet_delay','mean'), Avg_Loss=('Packet_Loss_Rate','mean'))
                .reset_index()
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=dl, x='Avg_Delay', y='Avg_Loss', hue='slice_type_label', palette='Set2', s=100, ax=ax)
            for i, r in dl.iterrows():
                ax.text(r['Avg_Delay']+0.5, r['Avg_Loss'], r['slice_type_label'])
            ax.set_title('Delay vs Packet Loss: Slice Type')
            st.pyplot(fig)


        with col12:
            st.subheader("Device Type & Slice Congestion")
            
            # Group and aggregate
            cong = (
                df_train.groupby(['IoT_Devices', 'Smartphone', 'slice_type_label'])
                .size()
                .reset_index(name='Count')
                .sort_values('Count', ascending=False)
                .head(10)
            )
            
            # Create label for y-axis
            cong['Combo'] = cong.apply(lambda r: f"IoT:{r['IoT_Devices']}|SP:{r['Smartphone']}|Slice:{r['slice_type_label']}", axis=1)
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Fix FutureWarning by using hue='Combo' and legend=False
            sns.barplot(data=cong, x='Count', y='Combo', palette='YlOrRd', hue='Combo', legend=False, dodge=False, ax=ax)
            
            ax.set_title('Device‑Slice Congestion Pairs')
            st.pyplot(fig)
            plt.close(fig)  # Prevent memory warning
        
             
        # Row 7
        col13, col14 = st.columns(2)
        with col13:
            st.subheader("Avg Delay for Healthcare Apps")
            hdelay = df_train.loc[df_train['Healthcare']==1,'Packet_delay'].mean()
            fig, ax = plt.subplots(figsize=(6,4))
            sns.barplot(x=['Healthcare'], y=[hdelay], color='steelblue', ax=ax)
            ax.set_ylim(0, hdelay*1.5)
            ax.set_title('Avg Packet Delay – Healthcare Apps')
            st.pyplot(fig)
        
        with col14:
            st.subheader("Avg Loss by LTE Category for Smart City Apps")
            sc = df_train[df_train['Smart_City_Home']==1]
            al = sc.groupby('LTE_5g_Category')['Packet_Loss_Rate'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(8,5))
            sns.barplot(data=al, x='LTE_5g_Category', y='Packet_Loss_Rate', color='mediumseagreen', ax=ax)
            ax.set_title('Avg Packet Loss – Smart City by Network')
            st.pyplot(fig)
        
        # Row 8
        col15, _ = st.columns([1,0.1])
        with col15:
            st.subheader("Top 10 Mixed App Use Cases → Packet Loss")
            loss_mix = (
                df_train.groupby(['Healthcare','AR_VR_Gaming','Smart_Transportation'])['Packet_Loss_Rate']
                .mean().reset_index(name='Avg_Loss').sort_values('Avg_Loss', ascending=False).head(10)
            )
            loss_mix['Combo'] = loss_mix.apply(lambda r: f"HC:{r['Healthcare']} AR:{r['AR_VR_Gaming']} ST:{r['Smart_Transportation']}", axis=1)
            fig, ax = plt.subplots(figsize=(12,6))
            sns.barplot(data=loss_mix, x='Avg_Loss', y='Combo', color='tomato', dodge=False, ax=ax)
            ax.set_title('Top 10 Mixed App Use Cases – Packet Loss')
            st.pyplot(fig)

#------------------------------------------------------------------------------------------------------------------------------------------------------

    # --- Predictive Analysis  ---
    # --- Predictive Analysis ---
    if selected_filter == "All" or selected_filter == "Predictive Analysis":
        st.header("Key Performance Indicators - Predictive Analysis")

        # --- Example: Define KPI variables using your dataset (train_df/test_df) ---
        
        # Most frequent LTE_5g_Category and its most common slice label
        lte_slice_counts = df_train.groupby(['LTE_5g_Category', 'slice_type_label']).size().reset_index(name='count')
        top_lte_slice = lte_slice_counts.sort_values('count', ascending=False).iloc[0]
        
        lte_category = top_lte_slice['LTE_5g_Category']
        slice_label = top_lte_slice['slice_type_label']
        allocation_count = top_lte_slice['count']
        
        # GBR packet loss difference between GBR and Non-GBR
        gbr_loss = df_train[df_train['GBR'] == 1]['Packet_Loss_Rate'].mean()
        nongbr_loss = df_train[df_train['Non_GBR'] == 1]['Packet_Loss_Rate'].mean()
        diff = abs(gbr_loss - nongbr_loss)
        trend = "Higher" if gbr_loss > nongbr_loss else "Lower"
        
        # High smartphone delay check
        high_delay_rows = df_train[(df_train['Smartphone'] == 1) & (df_train['Packet_delay'] > 80)]
        
        # Total IoT devices
        total_iot_devices = df_train['IoT_Devices'].sum()
        
        # Top application type by total counts
        app_cols = ['AR_VR_Gaming', 'Healthcare', 'Industry_4_0', 'Public_Safety', 
                    'Smart_City_Home', 'Smart_Transportation', 'Smartphone']
        
        app_usage_totals = df_train[app_cols].sum().sort_values(ascending=False)
        top_app = app_usage_totals.idxmax()
        top_app_value = app_usage_totals.max()

        
        # --- KPI Layout with Smaller Font Sizes to Prevent Text Truncation ---
        col1, col2, col3, col4, col5 = st.columns([1.3, 1.3, 1.3, 1.1, 1.1])
        
        with col1:
            st.markdown("**Top 5G Category Impact**", unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:20px; font-weight:bold'>{lte_category} ➝ {slice_label}</div>"
                f"<div style='color:green; font-size:14px'>↑ Allocations: {allocation_count}</div>",
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown("**GBR Traffic Loss Trend**", unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:20px; font-weight:bold'>{trend} Loss in GBR</div>"
                f"<div style='color:green; font-size:14px'>↑ Diff: {diff:.3f}</div>",
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown("**Smartphone Delay Alert**", unsafe_allow_html=True)
            if not high_delay_rows.empty:
                delay_value = high_delay_rows.iloc[0]['Packet_delay']
                st.markdown(
                    f"<div style='font-size:20px; font-weight:bold'>⚠ High Latency</div>"
                    f"<div style='color:green; font-size:14px'>↑ {delay_value:.2f} ms</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='font-size:20px; font-weight:bold'>✅ Normal</div>"
                    f"<div style='color:green; font-size:14px'>All below 80 ms</div>",
                    unsafe_allow_html=True
                )
        
        with col4:
            st.markdown("**IoT Devices**", unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:20px; font-weight:bold'>{int(total_iot_devices)}</div>",
                unsafe_allow_html=True
            )
        
        with col5:
            st.markdown("**Top App Usage**", unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:20px; font-weight:bold'>{top_app}</div>"
                f"<div style='color:green; font-size:14px'>↑ Total: {int(top_app_value)}</div>",
                unsafe_allow_html=True
            )

    
   # --- Predictive Analysis ---
    if selected_filter == "All" or selected_filter == "Predictive Analysis":
        st.header("Predictive Analysis")

    # Define target and features
        target_column = 'slice_type_label'
        selected_features = [
            'Packet_Loss_Rate', 'Packet_delay', 'IoT', 'LTE_5G', 'GBR', 'Non_GBR',
            'AR_VR_Gaming', 'Healthcare', 'Industry_4_0', 'IoT_Devices',
            'Public_Safety', 'Smart_City_Home', 'Smart_Transportation', 'Smartphone'
        ]


        # Row 1: Top Influential Features by Correlation with Target (Categorical Target)
        col1, _ = st.columns([1, 0.1])
        with col1:
            st.subheader("Feature Correlation with Target (Cramér's V)")
            from scipy.stats import chi2_contingency
        
            def cramers_v(x, y):
                confusion_matrix = pd.crosstab(x, y)
                chi2 = chi2_contingency(confusion_matrix)[0]
                n = confusion_matrix.sum().sum()
                phi2 = chi2 / n
                r, k = confusion_matrix.shape
                return np.sqrt(phi2 / min(k - 1, r - 1))
        
            cat_features = [col for col in selected_features if df_train[col].nunique() <= 10]
            cramers_results = []
            for col in cat_features:
                try:
                    v = cramers_v(df_train[col], df_train[target_column])
                    cramers_results.append((col, v))
                except:
                    continue
        
            cramers_df = pd.DataFrame(cramers_results, columns=['Feature', 'Cramers_V']).sort_values(by='Cramers_V', ascending=False)
        
            fig, ax = plt.subplots(figsize=(10, 6))
            # Explicitly set hue=None to avoid FutureWarning
            sns.barplot(data=cramers_df, x='Cramers_V', y='Feature', palette='mako', hue='Feature', legend=False, ax=ax)
            ax.set_title("Top Features Correlated with Target")
            st.pyplot(fig)


        # Row 2: Numeric Feature Importance (Correlation Heatmap)
        col2, _ = st.columns([1, 0.1])
        with col2:
            st.subheader("Numerical Feature Correlation with Target")
        
            num_features = [col for col in selected_features if pd.api.types.is_numeric_dtype(df_train[col])]
            df_corr = df_train[num_features + [target_column]].copy()
        
            # If target is categorical, encode it numerically
            if df_corr[target_column].dtype == 'object':
                df_corr[target_column] = df_corr[target_column].astype('category').cat.codes
        
            corr = df_corr.corr()[[target_column]].drop(target_column).sort_values(by=target_column, ascending=False)
        
            fig, ax = plt.subplots(figsize=(8, 6))
            # Add hue=None to suppress FutureWarning
            sns.barplot(x=corr[target_column], y=corr.index, palette='viridis', hue=corr.index, legend=False, ax=ax)
            ax.set_title("Feature Correlation with Target")
            st.pyplot(fig)
   
        # Row 3: Distribution of Target Column
        col3, _ = st.columns([1, 0.1])
        with col3:
            st.subheader("Distribution of Target Labels")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(data=df_train, x=target_column, palette='Set3', hue=df_train[target_column], legend=False, ax=ax)
            ax.set_title("Target Label Distribution")
            st.pyplot(fig)

    
        # Row 4: Mean Feature Values per Target Class
        col4, _ = st.columns([1, 0.1])
        with col4:
            st.subheader("Mean of Features by Target Class")
            grouped_means = df_train.groupby(target_column)[num_features].mean()
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(grouped_means.T, cmap="YlGnBu", annot=True, fmt=".2f", ax=ax)
            ax.set_title("Mean Feature Values per Class")
            st.pyplot(fig)
    
        # Row 5: Top App Feature Usage per Slice Type (if exists)
        app_features = ['AR_VR_Gaming', 'Healthcare', 'Industry_4_0', 'Public_Safety',
                        'Smart_City_Home', 'Smart_Transportation', 'Smartphone']
        available_app_features = [col for col in app_features if col in df_train.columns]
    
        if available_app_features:
            col5, _ = st.columns([1, 0.1])
            with col5:
                st.subheader("Top Application Usage by Slice Type")
                app_usage = df_train.groupby(target_column)[available_app_features].sum().T
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.heatmap(app_usage, annot=True, fmt=".0f", cmap="coolwarm", ax=ax)
                ax.set_title("Application Usage per Slice Type")
                st.pyplot(fig)

    # Row 6: Forecasting Peak Usage Times
    col6, _ = st.columns([1, 0.1])
    with col6:
        st.subheader("Forecasting Peak Usage Times")
        peak_usage = df_train.groupby('Time').size().reset_index(name='Usage').sort_values(by='Usage', ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=peak_usage, x='Time', y='Usage', palette='Blues_d', hue='Time', legend=False, ax=ax)
        ax.set_title("Peak Network Usage by Time Interval")
        plt.setp(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)


    # Row 7: Correlation of Packet Delay & Loss
    col7, _ = st.columns([1, 0.1])
    with col7:
        st.subheader("Correlation of Packet Delay & Loss")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df_train, x='Packet_delay', y='Packet_Loss_Rate', alpha=0.6, ax=ax)
        ax.set_title("Scatter Plot of Delay vs Packet Loss")
        st.pyplot(fig)

    # Row 8: Predict Slice Based on IoT Density
    col8, _ = st.columns([1, 0.1])
    with col8:
        st.subheader("IoT Device Density vs Slice Type")
        iot_slice = df_train.groupby(['IoT_Devices', 'slice_type_label']).size().reset_index(name='Count')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=iot_slice, x='IoT_Devices', y='Count', hue='slice_type_label', ax=ax)
        ax.set_title("Slice Allocation per IoT Device Count")
        st.pyplot(fig)

    # Row 9: Anomaly Detection in Delay Spikes
    col9, _ = st.columns([1, 0.1])
    with col9:
        st.subheader("Anomaly Detection in Delay Spikes")
        anomaly_df = df_train[df_train['Packet_delay'] > 250]
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(anomaly_df['Packet_delay'], bins=15, kde=True, color='red', ax=ax)
        ax.set_title("Packet Delay Spikes (Delay > 250 ms)")
        st.pyplot(fig)

    # Row 10: Impact of LTE/5G on Delay
    col10, _ = st.columns([1, 0.1])
    with col10:
        st.subheader("LTE/5G Impact on Delay")
        lte_delay = df_train.groupby('LTE_5G')['Packet_delay'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=lte_delay, x='LTE_5G', y='Packet_delay', palette='Set2', hue='LTE_5G', legend=False, ax=ax)
        ax.set_title("Average Delay by LTE/5G")
        st.pyplot(fig)

#------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # --- EDA Analysis ---
    if selected_filter == "All" or selected_filter == "EDA":
        st.header("Exploratory Data Analysis (EDA)")

        # Row 1 – Correlation Matrix
        col_corr, _ = st.columns([1, 0.1])
        with col_corr:
            st.subheader("Correlation Matrix – Pearson")
        
            # Filter numeric columns only
            numeric_cols = df_train.select_dtypes(include=['int64', 'float64']).columns
        
            # Compute correlation matrix
            corr_matrix = df_train[numeric_cols].corr(method='pearson')
        
            # Plot heatmap
            fig, ax = plt.subplots(figsize=(14, 10))
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                ax=ax
            )
            ax.set_title('Correlation Matrix (Pearson)', fontsize=16)
            plt.tight_layout()
            st.pyplot(fig)


        # Row 2 – Correlation Matrix (Spearman)
        col_spearman, _ = st.columns([1, 0.1])
        with col_spearman:
            st.subheader("Correlation Matrix – Spearman")
        
            # Filter numeric columns
            numeric_cols = df_train.select_dtypes(include=['int64', 'float64']).columns
        
            # Compute Spearman correlation matrix
            spearman_corr = df_train[numeric_cols].corr(method='spearman')
        
            # Plot heatmap
            fig, ax = plt.subplots(figsize=(14, 10))
            sns.heatmap(
                spearman_corr,
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                ax=ax
            )
            ax.set_title('Correlation Matrix (Spearman)', fontsize=16)
            plt.tight_layout()
            st.pyplot(fig)


        # Row 3 – Packet Loss Rate & Delay vs Slice Type
        col_loss_delay, _ = st.columns([1, 0.1])
        with col_loss_delay:
            st.subheader("Packet Loss & Delay vs Slice Type")
        
            # Binning Packet_Loss_Rate
            loss_bins = [0, 0.01, 0.02, 0.05, 0.1, 1.0]
            loss_labels = ['0-0.01', '0.01-0.02', '0.02-0.05', '0.05-0.1', '0.1+']
            df_train['Loss_Rate_Bin'] = pd.cut(df_train['Packet_Loss_Rate'], bins=loss_bins, labels=loss_labels)
        
            # Binning Packet_delay
            delay_bins = [0, 10, 20, 50, 100, df_train['Packet_delay'].max()]
            delay_labels = ['0-10', '10-20', '20-50', '50-100', '100+']
            df_train['Delay_Bin'] = pd.cut(df_train['Packet_delay'], bins=delay_bins, labels=delay_labels)
        
            # Set style
            sns.set(style="whitegrid")
        
            # Plotting
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
            # Grouped Bar Chart for Packet Loss Rate
            sns.countplot(
                x='Loss_Rate_Bin',
                hue='slice_type_label',
                data=df_train,
                palette='Set3',
                ax=axes[0]
            )
            axes[0].set_title('Packet Loss Rate vs Slice Type')
            axes[0].set_xlabel('Packet Loss Rate Bins')
            axes[0].set_ylabel('Count')
            axes[0].legend(title='Slice Type', loc='upper right')
        
            # Grouped Bar Chart for Packet Delay
            sns.countplot(
                x='Delay_Bin',
                hue='slice_type_label',
                data=df_train,
                palette='Set2',
                ax=axes[1]
            )
            axes[1].set_title('Packet Delay vs Slice Type')
            axes[1].set_xlabel('Packet Delay Bins')
            axes[1].set_ylabel('Count')
            axes[1].legend(title='Slice Type', loc='upper right')
        
            plt.tight_layout()
            st.pyplot(fig)


        # Row 4 – Count of LTE 5G Category
        col_lte_count, _ = st.columns([1, 0.1])
        with col_lte_count:
            st.subheader("Count of Network Slice Categories (LTE_5g_Category)")
        
            # Set style
            sns.set(style="whitegrid")
        
            # Plotting
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(
                x='LTE_5g_Category',
                data=df_train,
                hue='LTE_5g_Category',   # Avoid duplicate legend
                palette='pastel',
                legend=False,
                ax=ax
            )
        
            # Add count labels
            for p in ax.patches:
                count = p.get_height()
                ax.annotate(f'{int(count)}', (p.get_x() + p.get_width() / 2., count),
                            ha='center', va='bottom', fontsize=10)
        
            ax.set_title('Count of Network Slice Categories (LTE_5g_Category)', fontsize=14)
            ax.set_xlabel('LTE 5G Slice Category', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
            plt.tight_layout()
            st.pyplot(fig)


        # Row 5 – Pie Chart of LTE_5g_Category
        col_lte_pie, _ = st.columns([1, 0.1])
        with col_lte_pie:
            st.subheader("LTE_5g_Category Distribution (Pie Chart)")
        
            # Count values and prepare labels
            category_counts = df_train['LTE_5g_Category'].value_counts().sort_index()
            labels = [str(i) for i in category_counts.index]
            colors = sns.color_palette('pastel')[0:len(labels)]
        
            # Plotting
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(
                category_counts,
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                textprops={'fontsize': 14}
            )
        
            ax.set_title('LTE_5g_Category Distribution (Pie Chart)', fontsize=16)
            ax.axis('equal')  # Keep pie chart as circle
        
            plt.tight_layout()
            st.pyplot(fig)


        # Row 6 – Mutual Information Scores Bar Chart
        col_mi, _ = st.columns([1, 0.1])
        with col_mi:
            st.subheader("Mutual Information Scores – Feature Importance")
        
            from sklearn.feature_selection import mutual_info_classif
        
            # Step 1: Separate features and target
            X = df_train.drop(columns=['slice_type_label'])
            y = df_train['slice_type_label']
        
            # Step 2: One-hot encode categorical features
            X_encoded = pd.get_dummies(X)
        
            # Step 3: Compute mutual information scores
            mi_scores = mutual_info_classif(X_encoded, y, discrete_features='auto', random_state=42)
        
            # Step 4: Build DataFrame and sort
            mi_df = pd.DataFrame({'Feature': X_encoded.columns, 'MI Score': mi_scores})
            mi_df = mi_df.sort_values(by='MI Score', ascending=False)
        
            # Step 5: Plot the MI scores
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x='MI Score', y='Feature', data=mi_df, palette='viridis', hue='Feature', legend=False, ax=ax)
            ax.set_title("Mutual Information Scores for Predicting Slice Type")
            ax.set_xlabel("Mutual Information Score")
            ax.set_ylabel("Feature")
        
            plt.tight_layout()
            st.pyplot(fig)

#-------------------------------------------------------------------------------------------------------------------------------------------------

# 8. App Entry Point
if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login_page()
    else:
        selected_filter = sidebar_filters()
        network_slcice_eda()
