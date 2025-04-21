from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import LLMChain
from langchain import PromptTemplate
import streamlit as st
import os

# Set API key
os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']

# Create prompt template for interview question generation
Interview_prompt_template = """
Generate {number} interview questions for the job role of {job_role}.
The candidate has {experience} years of experience.
The difficulty level should be {difficulty}.
{skills_clause}
Ensure the questions are relevant and diverse.
"""

# Define the template with variables
Interview_prompt = PromptTemplate(
    template=Interview_prompt_template,
    input_variables=["number", "job_role", "experience", "difficulty", "skills_clause"]
)

# Initialize the Gemini model
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# Chain prompt with the model
Interview_chain = Interview_prompt | gemini_model

# Streamlit UI
st.header("InterviewGuru" - by Sumedha")

st.subheader("Generate Interview Questions for Any Job Role")

# User Inputs
job_role = st.text_input("Job Role (e.g., Data Analyst, Software Engineer)")

experience = st.number_input("Years of Experience", min_value=0, max_value=30, value=2, step=1)

difficulty = st.selectbox("Difficulty Level", ["Easy", "Medium", "Hard"])

skills = st.text_input("Specific Skills (optional, comma-separated)")

number = st.number_input("Number of Questions", min_value=1, max_value=20, value=5, step=1)

# Generate Button
if st.button("Generate"):
    skills_clause = f"Focus on these skills: {skills}." if skills else "No specific skills provided."
    
    interview_questions = Interview_chain.invoke({
        "number": number,
        "job_role": job_role,
        "experience": experience,
        "difficulty": difficulty,
        "skills_clause": skills_clause
    })

    st.markdown("### Generated Interview Questions")
    st.write(interview_questions.content)
