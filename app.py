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
    lowered_query = user_query.lower()

    if any(word in lowered_query for word in ["total", "overall", "aggregate"]):
        prompt = f"""
You are a data analyst. Given a dataset with these columns:
{', '.join(df_sample.columns)}

The user asked: "{user_query}"

Generate a Python pandas code snippet that analyzes the full dataset (no client filtering) and returns:
- result → full dataframe with only needed columns
- summary1 → revenue grouped by Type
- summary2 → cost split by Onshore and Offshore

Assume the dataframe is named df. Do not filter by client.
        """
    else:
        prompt = f"""
You are a data analyst. Given a dataset with these columns:
{', '.join(df_sample.columns)}

The user asked: "{user_query}"

Generate a Python pandas code snippet that filters and analyzes the dataset to provide:
1. Revenue and Cost for the client (case-insensitive match)
2. Breakup of revenue by Type (Fixed_Position vs Project)
3. Breakup of cost between Onshore and Offshore

Return the result as:
- result → filtered dataframe
- summary1 → revenue grouped by Type
- summary2 → cost split by Onshore and Offshore

Just return the Python pandas code, no explanation.
Assume the dataframe is named df.
Use case-insensitive filtering.
        """

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content

# 📊 Plot helpers
def plot_dual_axis_bar(agg):
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

# 🚀 Main App
st.set_page_config(page_title="Cloud Insights Chatbot", page_icon="💬", layout="wide")
st.title("💬 Cloud Insights Chatbot")

df = load_data()

# ✅ Sidebar with client list
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
            result_df = local_vars['result']

            # Ensure needed columns are present
            required_cols = ['Type', 'Revenue', 'Cost']
            for col in required_cols:
                if col not in result_df.columns:
                    result_df[col] = None

            if 'Resources_Total' not in result_df.columns:
                result_df['Resources_Total'] = 0

            agg = result_df.groupby("Type", dropna=False).agg({
                "Revenue": "sum",
                "Cost": "sum",
                "Resources_Total": "sum"
            }).reset_index()

            agg["Revenue ($M)"] = (agg["Revenue"] / 1_000_000).round(2)
            agg["Cost ($M)"] = (agg["Cost"] / 1_000_000).round(2)
            agg.rename(columns={"Resources_Total": "Total Resources"}, inplace=True)

            # 🧮 Total Summary
            total_revenue = result_df["Revenue"].sum() / 1_000_000
            total_cost = result_df["Cost"].sum() / 1_000_000
            total_resources = result_df["Resources_Total"].sum()

            st.subheader("📌 Total Summary")
            st.markdown(f"- **Total Revenue:** ${total_revenue:.2f}M")
            st.markdown(f"- **Total Cost:** ${total_cost:.2f}M")
            st.markdown(f"- **Total Resources:** {int(total_resources)}")

            # 📊 Aggregated Summary by Type
            st.subheader("📊 Summary by Type (Aggregated)")
            col1, col2 = st.columns([1.1, 1])

            with col1:
                st.dataframe(agg[["Type", "Revenue ($M)", "Cost ($M)", "Total Resources"]], use_container_width=True, height=350)

            with col2:
                plot_dual_axis_bar(agg)

            # 📋 Full detail
            st.subheader("📋 Project-wise and Fixed Position Data")
            st.dataframe(result_df, use_container_width=True, height=400)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
