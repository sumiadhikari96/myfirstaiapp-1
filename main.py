from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import LLMChain
from langchain import PromptTemplate

import streamlit as st
import os

os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']

# Create prompt template for generating tweets

HR_Metrics_template = "Give me {number} HR Metrics on {topic}"

Hr_Metrics_prompt = PromptTemplate(template = HR_Metrics_template, input_variables = ['number', 'topic'])

# Initialize Google's Gemini model
gemini_model = ChatGoogleGenerativeAI(model = "gemini-1.5-flash-latest")


# Create LLM chain using the prompt template and model
HR_Metrics_chain = HR_Metrics_prompt | gemini_model 


import streamlit as st

st.header("HR Metrics Generator - Sumedha")

st.subheader("Generate HR Metrics using Generative AI")

topic = st.text_input("Topic")

number = st.number_input("Number of tweets", min_value = 1, max_value = 10, value = 1, step = 1)

if st.button("Generate"):
    HR_Metrics = HR_Metrics_chain.invoke({"number" : number, "topic" : topic})
    st.write(HR_Metrics.content)
    
