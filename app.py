import streamlit as st
import pandas as pd
from groq import Groq

# ===== API KEY =====
GROQ_API_KEY = "YOUR_GROQ_API_KEY"

client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="CET Predictor", layout="wide")

st.title("🎓 CET College & Branch Predictor")

# ===== FILE UPLOAD =====
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully ✅")
    st.dataframe(df.head())

    # ===== INPUT =====
    rank = st.number_input("Enter CET Rank", min_value=1)

    if st.button("Predict 🚀"):

        # ===== FILTER LOGIC =====
        filtered = df[df['cutoff_rank'].astype(int) >= rank]

        # Sort by cutoff (closest match)
        filtered = filtered.sort_values(by="cutoff_rank")

        if filtered.empty:
            st.warning("No results found 😢")
        else:
            st.subheader("🎯 Suggested Colleges, Branches & Codes")

            result = filtered[
                ['college_name', 'branch', 'branch_code', 'cutoff_rank']
            ].head(10)

            st.dataframe(result)

            # ===== AI SUGGESTION =====
            prompt = f"""
            A student got CET rank {rank}.

            Based on this data:
            {result.to_string()}

            Suggest best branch and college with reason.
            """

            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}]
            )

            st.subheader("🤖 AI Recommendation")
            st.write(response.choices[0].message.content)
