import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression

# App Title
st.set_page_config(page_title="Interactive Data Visualization Dashboard", layout="wide")
st.title("📊 Interactive Data Visualization Dashboard")
st.markdown("### Explore, Visualize, and Analyze Your Data Easily!")

# File Upload
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    # Tabs for better navigation
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Charts", "📄 Data", "💡 Insights", "⚙️ Tools"])
    
    with tab1:
        st.subheader("📊 Create Your Chart")

        chart_type = st.selectbox("Select Chart Type", ["Line", "Bar", "Scatter", "Pie"])
        x_col = st.selectbox("Select X-axis", df.columns)
        y_col = st.selectbox("Select Y-axis", df.columns)

        if chart_type == "Line":
            fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
        elif chart_type == "Bar":
            fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
        elif chart_type == "Pie":
            cat_cols = df.select_dtypes(include='object').columns.tolist()
            num_cols = df.select_dtypes(include='number').columns.tolist()
            if len(cat_cols) > 0 and len(num_cols) > 0:
                fig = px.pie(df, names=cat_cols[0], values=num_cols[0], title="Pie Chart")
            else:
                st.warning("⚠️ Pie chart requires at least 1 categorical and 1 numeric column.")
                fig = None

        if 'fig' in locals() and fig:
            st.plotly_chart(fig, use_container_width=True)

            # Download chart button
            try:
                img_bytes = fig.to_image(format="png")
                st.download_button(
                    label="📥 Download Chart as PNG",
                    data=img_bytes,
                    file_name="chart.png",
                    mime="image/png"
                )
            except Exception:
                st.info("Download not available for this chart type.")

    with tab2:
        st.subheader("📋 Data Overview")
        st.dataframe(df, use_container_width=True)

        if st.button("Show Summary"):
            st.dataframe(df.describe())

        st.subheader("🔎 Search Data")
        search_term = st.text_input("Enter keyword to filter data:")
        if search_term:
            filtered_data = df[df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)]
            st.dataframe(filtered_data)

    with tab3:
        st.subheader("💡 Data Insights")

        if 'Sales' in df.columns:
            st.write("📈 Highest Sales:", df['Sales'].max())
            st.write("📉 Lowest Sales:", df['Sales'].min())
            st.write("💰 Average Sales:", round(df['Sales'].mean(), 2))

        if 'Profit' in df.columns and 'Sales' in df.columns:
            st.subheader("📉 Correlation Heatmap")
            numeric_df = df.select_dtypes(include=np.number)
            if not numeric_df.empty:
                fig_corr = px.imshow(numeric_df.corr(), text_auto=True, title="Correlation Heatmap")
                st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("📊 Simple Profit Prediction (Optional Feature)")
        if 'Sales' in df.columns and 'Profit' in df.columns:
            X = df[['Sales']]
            y = df['Profit']
            model = LinearRegression().fit(X, y)
            new_sales = st.number_input("Enter Sales to predict Profit:", min_value=0)
            if st.button("Predict Profit"):
                pred = model.predict([[new_sales]])[0]
                st.success(f"Predicted Profit: {pred:.2f}")

    with tab4:
        st.subheader("⚙️ Tools Used")
        st.markdown("""
        - 🐍 **Python** – Programming language for logic  
        - 📊 **Streamlit** – For web dashboard  
        - 📈 **Plotly** – For interactive charts  
        - 📑 **Pandas** – For data handling  
        - 📘 **CSV File** – Dataset input  
        """)
else:
    st.info("Please upload a CSV file to begin.")
