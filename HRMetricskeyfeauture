from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate

import streamlit as st
import os

# Set your Google API key from Streamlit secrets
os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']

# Prompt Template with option to include formulas
HR_Metrics_template = (
    "Give me {number} HR Metrics on the topic '{topic}'. "
    "{formula_instruction} Present the output in a clear, simple format."
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
st.subheader("Generate insightful HR Metrics with simple formulas")

st.markdown("Use this tool to generate HR metrics for any HR domain. Ideal for HR Professionals")

# Topic options
topics = [
    "Employee Engagement", "Attrition", "Performance Management", "Learning & Development",
    "Recruitment", "Compensation & Benefits", "HR Analytics", "Workforce Planning"
]

# UI Components
topic = st.selectbox("🔍 Select HR Topic", options=topics, help="Pick a domain to generate relevant metrics")

number = st.slider("🔢 Number of Metrics", min_value=1, max_value=10, value=3, step=1, help="Select how many metrics to generate")

include_formula = st.checkbox("➕ Include Simple Formulas", value=True)

# Generate button
if st.button("🚀 Generate HR Metrics"):
    formula_instruction = "Include a simple formula for each metric." if include_formula else "Do not include formulas."
    
    # Invoke the chain
    with st.spinner("Generating HR Metrics..."):
        response = HR_Metrics_chain.invoke({
            "number": number,
            "topic": topic,
            "formula_instruction": formula_instruction
        })

        # Display results
        st.markdown("### 📌 Generated Metrics")
        st.markdown(response.content)

        st.success("✅ Metrics generated successfully!")

        st.markdown("---")
        st.markdown("### 📈 Summary & Use")
        st.markdown(f"These metrics can help in understanding and improving **{topic}** processes. "
                    "Use them in HR dashboards, performance reviews, or data analysis.")

st.caption("Built with 💙 using LangChain & Google Gemini")
