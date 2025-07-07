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
def ask_gpt(user_query, df_sample):
    prompt = f"""
You are a data analyst. Given a dataset with these columns:
{', '.join(df_sample.columns)}

The user asked: "{user_query}"

Generate a Python pandas code snippet that filters and analyzes the dataset to provide:
1. Revenue and Cost for the client (if specified — case-insensitive)
2. Breakup of revenue by Type (Fixed_Position vs Project)
3. Breakup of cost between Onshore and Offshore
4. If no client is specified, return total revenue and cost for all data.

- Make all string filtering (e.g., Client == 'BMW') case-insensitive using .str.lower().
- Return the result as:
  - result → filtered dataframe
  - summary1 → revenue grouped by Type
  - summary2 → cost split by Onshore and Offshore

Just return the Python pandas code, no explanation.
Assume the dataframe is named df.
    """

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content

# 📊 Plot helpers
def plot_bar(data, title, ylabel):
    fig, ax = plt.subplots(figsize=(6, 4))
    data.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    st.pyplot(fig)

# 🚀 Main App
st.set_page_config(page_title="Cloud Insights Chatbot", page_icon="💬", layout="wide")
st.title("💬 Cloud Insights Chatbot")

df = load_data()

# ✅ Add sidebar listing unique clients
with st.sidebar:
    st.markdown("### 🧾 Clients in Dataset")
    for client in sorted(df["Client"].unique()):
        st.markdown(f"- {client}")

user_query = st.text_input("Ask a question like:", "Show revenue and cost breakdown for BMW")

if user_query:
    try:
        st.markdown("Generating insights...")
        code = ask_gpt(user_query, df.head(3))

        # 👇 Execute GPT-generated code safely
        local_vars = {'df': df.copy()}
        clean_code = code.strip().strip("`").replace("python", "").strip()
        exec(clean_code, {}, local_vars)

        if 'result' in local_vars:
            agg = local_vars['result'].groupby("Type").agg({
                "Revenue": "sum",
                "Cost": "sum",
                "Resources_Total": "sum"
            }).reset_index()

            agg["Revenue ($M)"] = (agg["Revenue"] / 1_000_000).round(2)
            agg["Cost ($M)"] = (agg["Cost"] / 1_000_000).round(2)
            agg.rename(columns={"Resources_Total": "Total Resources"}, inplace=True)

            # 🗒️ Summary Text
            st.subheader("📌 Key Insights Summary")
            for _, row in agg.iterrows():
                st.markdown(f"- **The total revenue is ${row['Revenue ($M)']}M and total cost is ${row['Cost ($M)']}M for `{row['Type']}` engagements.**")

            # 📊 Aggregated Summary by Type
            st.subheader("📊 Summary by Type (Aggregated)")
            col1, col2 = st.columns([1.1, 1])

            with col1:
                st.dataframe(agg[["Type", "Revenue ($M)", "Cost ($M)", "Total Resources"]], use_container_width=True, height=350)

            with col2:
                fig, ax1 = plt.subplots(figsize=(6, 4))
                ax2 = ax1.twinx()

                ax1.bar(agg["Type"], agg["Revenue ($M)"], label="Revenue ($M)", color="skyblue")
                ax2.plot(agg["Type"], agg["Cost ($M)"], label="Cost ($M)", color="red", marker="o")

                ax1.set_ylabel("Revenue ($M)")
                ax2.set_ylabel("Cost ($M)")
                ax1.set_title("Revenue and Cost by Type")
                ax1.legend(loc="upper left")
                ax2.legend(loc="upper right")

                st.pyplot(fig)

            # 📈 MoM Revenue vs Cost Chart
            st.subheader("📈 Monthly Revenue vs Cost Trend")
            monthly = local_vars['result'].groupby("Month").agg({
                "Revenue": "sum",
                "Cost": "sum"
            }).sort_index()

            fig, ax1 = plt.subplots(figsize=(8, 4))
            ax2 = ax1.twinx()

            ax1.bar(monthly.index.strftime("%b %Y"), monthly["Revenue"] / 1_000_000, label="Revenue ($M)", color="lightgreen")
            ax2.plot(monthly.index.strftime("%b %Y"), monthly["Cost"] / 1_000_000, label="Cost ($M)", color="orange", marker="o")

            ax1.set_ylabel("Revenue ($M)")
            ax2.set_ylabel("Cost ($M)")
            ax1.set_title("Monthly Revenue vs Cost")
            ax1.set_xticklabels(monthly.index.strftime("%b %Y"), rotation=45)
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")

            st.pyplot(fig)

            # 📋 Full project + fixed data at bottom
            st.subheader("📋 Project-wise and Fixed Position Data")
            st.dataframe(local_vars['result'], use_container_width=True, height=400)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
