import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import io
import requests

# 🔑 Set your OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]  # or replace with your key directly

# 📄 GitHub raw CSV URL
CSV_URL = "https://raw.githubusercontent.com/SwapnilGautama/CloudInsights/main/SoftwareCompany_2025_Data.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_URL)
    df['Month'] = pd.to_datetime(df['Month'])
    return df

# 🧠 GPT-powered query interpreter
from openai import OpenAI

client = OpenAI(api_key=openai.api_key)

def ask_gpt(user_query, df_sample):
    prompt = f"""
You are a data analyst. Given a dataset with these columns:
{', '.join(df_sample.columns)}

The user asked: "{user_query}"

Generate a Python pandas code snippet that filters and analyzes the dataset to provide:
1. Revenue and Cost for the client
2. Breakup of revenue by Type (Fixed_Position vs Project)
3. Breakup of cost between Onshore and Offshore

Just return the pandas code, no explanation.
Assume the dataframe is named df.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content

# 📊 Plot helpers
def plot_bar(data, title, ylabel):
    fig, ax = plt.subplots()
    data.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    st.pyplot(fig)

# 🚀 Main App
st.set_page_config(page_title="Cloud Insights Chatbot", page_icon="💬")
st.title("💬 Cloud Insights Chatbot")

df = load_data()
user_query = st.text_input("Ask a question like:", "Show revenue and cost breakdown for BMW")

if user_query:
    try:
        st.markdown("Generating insights...")
        code = ask_gpt(user_query, df.head(3))
        st.code(code, language='python')

                # 👇 Execute GPT-generated code safely
        local_vars = {'df': df.copy()}

        # ✨ Strip markdown code block markers before exec
        clean_code = code.strip().strip("`").replace("python", "").strip()

        exec(clean_code, {}, local_vars)

        # 🎯 Expecting: result, summary1, summary2
        if 'result' in local_vars:
            st.dataframe(local_vars['result'], use_container_width=True)

        if 'summary1' in local_vars:
            st.subheader("🔹 Revenue by Type")
            st.dataframe(local_vars['summary1'])
            plot_bar(local_vars['summary1'], "Revenue by Type", "Revenue")

        if 'summary2' in local_vars:
            st.subheader("🔹 Cost Split (Onshore vs Offshore)")
            st.dataframe(local_vars['summary2'])
            plot_bar(local_vars['summary2'], "Cost by Location", "Cost")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
