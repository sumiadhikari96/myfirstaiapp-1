from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate

import streamlit as st
import os

# Set your Google API key from Streamlit secrets
os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']

# Prompt Template requesting metrics from basic to advanced
HR_Metrics_template = (
    "Provide {number} HR metrics on the topic: '{topic}'. "
    "Cover a range from basic to advanced level. {formula_instruction} "
    "Present the output in a clear and simple format."
)

HR_Metrics_prompt = PromptTemplate(
    template=HR_Metrics_template,
    input_variables=['number', 'topic', 'formula_instruction']
)

# Load Gemini model
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# Chain
HR_Metrics_chain = HR_Metrics_prompt | gemini_model

# --- Streamlit UI ---

st.set_page_config(page_title="HR Metrics Generator", layout="centered")
st.title("📊 HR Metrics Generator - Sumedha")
st.subheader("From Beginner to Advanced – Generate Powerful HR Metrics")

st.markdown("Use this tool to generate HR metrics for **any** topic – customized for your specific need!")

# Topic Input (Open-ended)
topic = st.text_input("🧠 Enter your HR Topic", placeholder="e.g. Remote Work Productivity, Training ROI, Candidate Experience", help="You can enter any topic related to HR you'd like metrics on")

# Number selector
number = st.slider("🔢 Number of Metrics", min_value=1, max_value=10, value=3, step=1, help="Select how many metrics to generate")

# Formula toggle
include_formula = st.checkbox("➕ Include Simple Formulas", value=True)

# Generate button
if st.button("🚀 Generate HR Metrics"):
    if not topic.strip():
        st.warning("Please enter a topic to generate HR metrics.")
    else:
        formula_instruction = "Include a simple formula for each metric." if include_formula else "Do not include formulas."
        
        with st.spinner("Generating HR Metrics from easy to advanced..."):
            response = HR_Metrics_chain.invoke({
                "number": number,
                "topic": topic,
                "formula_instruction": formula_instruction
            })

            # Output
            st.markdown("### 📌 Generated Metrics (Easy ➡️ Advanced)")
            st.markdown(response.content)

            st.success("✅ Metrics generated successfully!")

            st.markdown("---")
            st.markdown("### 📈 Summary & Use")
            st.markdown(f"These metrics on **{topic}** range from foundational to complex. Use them for deeper insights, performance tracking, or reporting.")

st.caption("Built with 💙 using LangChain & Gemini")
