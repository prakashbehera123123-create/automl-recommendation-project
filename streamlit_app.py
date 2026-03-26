import streamlit as st
import pandas as pd
import time
from source.pipeline import final_recommendation_engine
from source.pipeline import run_recommendation

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AutoML Recommender",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------
# CUSTOM CSS (UI Styling)
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #0b0f19;
}
.stApp {
    background: linear-gradient(135deg, #0b0f19, #111827);
    color: white;
}
.title {
    font-size: 48px;
    font-weight: 800;
    color: #5bffc8;
}
.subtitle {
    font-size: 18px;
    color: #9ca3af;
}
.upload-box {
    border: 2px dashed #5bffc8;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
}
.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #1f2937;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown('<div class="title">AutoML Algorithm Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload → Analyze → Get Best ML Model</div>', unsafe_allow_html=True)

st.write("")

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload your dataset",
    type=["csv", "xlsx", "json"]
)

# Show uploaded file name
if uploaded_file:
    st.success(f"✅ Uploaded: {uploaded_file.name}")

# -------------------------------
# PROCESS BUTTON
# -------------------------------
if uploaded_file:

    if st.button("⚡ Run AutoML Recommendation"):

        with st.spinner("Processing your dataset..."):
            time.sleep(1)  # smooth UX

            # -------------------------------
            # FILE HANDLING
            # -------------------------------
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)

            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)

            elif uploaded_file.name.endswith(".json"):
                df = pd.read_json(uploaded_file)

            else:
                st.error("Unsupported file format")
                st.stop()

            # -------------------------------
            # RUN YOUR PIPELINE
            # -------------------------------
            try:
                result = run_recommendation(df)

            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        # -------------------------------
        # RESULTS UI
        # -------------------------------
        st.markdown("## 📊 Results")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("🏆 Final Algorithm", result["final_algorithm"])
        col2.metric("📈 Confidence", f"{result['confidence']*100:.2f}%")
        col3.metric("🧠 Meta Model", result["meta_prediction"])
        col4.metric("Similarity Engine", result['similarity_prediction'])

        st.write("")

        # Detailed results
        st.markdown("### 🔍 Detailed Predictions")
        st.json(result)

        # -------------------------------
        # OPTIONAL: FEATURE DISPLAY
        # -------------------------------
        

# -------------------------------
# FOOTER
# -------------------------------
st.write("---")
st.markdown("Built by Pk AI Platform")