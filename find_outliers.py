import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Outlier Detector", layout="centered")
st.title("📊 Outlier Detection Dashboard")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 🔍 Data Preview", df.head())

    numeric_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()

    if numeric_cols:
        column = st.selectbox("📈 Select a numeric column for outlier detection", numeric_cols)

        min_val = float(df[column].min())
        max_val = float(df[column].max())

        st.write(f"Range of **{column}**: {min_val:.2f} to {max_val:.2f}")

        lower_threshold, upper_threshold = st.slider(
            "🔧 Adjust the thresholds for detecting outliers",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val)
        )

        # Detect outliers
        is_outlier = (df[column] < lower_threshold) | (df[column] > upper_threshold)
        outliers = df[is_outlier]
        inliers = df[~is_outlier]
        mean_value = df[column].mean()

        # Plot
        st.write("### 📉 Outlier Visualization")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(x=df[column], ax=ax, color="lightblue", width=0.3)

        # Scatter inliers and outliers
        ax.scatter(inliers[column], [0.05] * len(inliers), color='green', label='Inliers', alpha=0.6, s=50)
        ax.scatter(outliers[column], [-0.05] * len(outliers), color='red', label='Outliers', alpha=0.8, s=50)

        # Threshold lines
        ax.axvline(lower_threshold, color='red', linestyle='--', label='Lower Threshold')
        ax.axvline(upper_threshold, color='red', linestyle='--', label='Upper Threshold')

        # Mean line
        ax.axvline(mean_value, color='blue', linestyle=':', label=f'Mean ({mean_value:.2f})')

        ax.set_yticks([])
        ax.set_title(f"Boxplot of {column} with Inliers, Outliers, and Mean")
        ax.legend()
        st.pyplot(fig)

        # Summary
        st.write("### 📊 Summary Statistics")
        stats = df[column].describe().to_frame().T
        stats["Mean"] = mean_value
        stats["Outlier Count"] = len(outliers)
        stats["Inlier Count"] = len(inliers)
        st.dataframe(stats)

        st.write("### 🚨 Detected Outliers")
        st.dataframe(outliers.reset_index(drop=True))
    else:
        st.warning("⚠️ No numeric columns found in your file.")
else:
    st.info("👆 Please upload a CSV file to get started.")

