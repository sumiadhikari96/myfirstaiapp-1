from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate

import streamlit as st
import os

# Set your Google API key from Streamlit secrets
os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']

# Prompt Template: precise HR metrics with calculations
HR_Metrics_template = (
    "Provide {number} precise and widely-used HR metrics for the topic: '{topic}'. "
    "{formula_instruction} Each metric should include a clear definition and a practical calculation formula. "
    "Keep the language professional and the format easy to read."
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
st.subheader("HR Metrics with Calculations")

st.markdown("This tool gives you **precise and ready-to-use HR metrics** for any HR-related topic.")

# Open-ended topic input
topic = st.text_input(
    "🧠 Enter Your HR Topic",
    placeholder="e.g. Recruitment Efficiency, Employee Turnover, Learning Impact",
    help="Enter any HR-related topic you'd like detailed metrics for."
)

# Number of metrics
number = st.slider(
    "🔢 Number of Metrics",
    min_value=1, max_value=10, value=3, step=1,
    help="Choose how many metrics to generate"
)

# Checkbox to include formulas
include_formula = st.checkbox("🧮 Include Calculation Formulas", value=True)

# Generate button
if st.button("🚀 Generate HR Metrics"):
    if not topic.strip():
        st.warning("Please enter a topic before generating.")
    else:
        formula_instruction = "Include a clear calculation formula for each metric." if include_formula else "Do not include formulas."
        
        with st.spinner("Generating precise HR metrics..."):
            response = HR_Metrics_chain.invoke({
                "number": number,
                "topic": topic,
                "formula_instruction": formula_instruction
            })

            # Output
            st.markdown("### 📌 Generated Metrics")
            st.markdown(response.content)

            st.success("✅ Metrics generated successfully!")

            st.markdown("---")
            st.markdown("### 📈 How to Use")
            st.markdown(f"Use these **precise metrics** on **{topic}** to support HR analysis, decision-making, and reporting. Ideal for dashboards, performance reviews, or executive summaries.")

st.caption("Built with 💙 using LangChain & Google Gemini")
