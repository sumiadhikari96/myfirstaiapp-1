from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import LLMChain
from langchain import PromptTemplate

import streamlit as st
import os

# Set your API key from Streamlit secrets
os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']

# Email generation prompt template
email_template = "Write {number} email(s) on the subject: '{subject}' with a {tone} tone."

# PromptTemplate to format user input into prompt
email_prompt = PromptTemplate(template=email_template, input_variables=['number', 'subject', 'tone'])

# Initialize Gemini model
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# Create LLM chain
email_chain = email_prompt | gemini_model

# Streamlit UI
st.header("📧 AI Email Generator - Sumedha")

st.subheader("Create custom emails using Generative AI")

subject = st.text_input("✉️ Subject", placeholder="e.g. Product launch announcement")

tone = st.text_input("🎭 Tone", placeholder="e.g. formal, friendly, persuasive")

number = st.number_input("📬 Number of Emails", min_value=1, max_value=5, value=1, step=1)

if st.button("Generate Emails"):
    if subject and tone:
        emails = email_chain.invoke({"number": number, "subject": subject, "tone": tone})
        st.write(emails.content)
    else:
        st.warning("Please fill in both Subject and Tone fields.")
