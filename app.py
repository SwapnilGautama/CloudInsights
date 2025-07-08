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

    The user asked: "{user_query.lower()}"

    Generate a Python pandas code snippet that filters and analyzes the dataset to provide:
    1. If the user asks for 'total', 'overall', 'aggregate', or 'company-wide', show revenue and cost across the **entire dataset**.
    2. If a client is mentioned, filter by that client (case-insensitive).
    3. Provide:
        - Total revenue and cost
        - Revenue by 'Type' (Fixed_Position vs Project)
        - Cost split by Onshore vs Offshore (Location_Onshore and Location_Offshore)

    Assume the dataframe is called df.
    - Use `.str.lower()` for string comparisons
    - Return the following variables:
        - result → filtered df
        - summary1 → revenue by Type
        - summary2 → cost by Onshore/Offshore

    Just return executable Python code, no explanation.
    """

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content

# 🕊️ Plot helpers
def plot_bar(data, title, ylabel):
    fig, ax = plt.subplots(figsize=(6, 4))
    data.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    st.pyplot(fig)

# ✨ Detect if user is asking for full client report
def is_full_client_report(query):
    return "full client report" in query.lower() or "client-wise summary" in query.lower()

# ⚖️ Generate full client report visuals
def show_full_client_report(df):

    agg_df = df.groupby("Client").agg({
        "Revenue": "sum",
        "Cost": "sum",
        "Resources_Total": "sum"
    }).reset_index()

    agg_df["Revenue ($M)"] = (agg_df["Revenue"] / 1_000_000).round(2)
    agg_df["Cost ($M)"] = (agg_df["Cost"] / 1_000_000).round(2)
    agg_df.rename(columns={"Resources_Total": "Total Resources"}, inplace=True)

    # Add total row
    total_row = pd.DataFrame({
        "Client": ["Total"],
        "Revenue ($M)": [agg_df["Revenue ($M)"].sum().round(2)],
        "Cost ($M)": [agg_df["Cost ($M)"].sum().round(2)],
        "Total Resources": [agg_df["Total Resources"].sum()]
    })

    display_df = pd.concat([agg_df[["Client", "Revenue ($M)", "Cost ($M)", "Total Resources"]], total_row], ignore_index=True)

    st.subheader("📊 Client-wise Summary Table")
    st.dataframe(display_df, use_container_width=True)

    # Pie Charts
    st.subheader("🞈 Client-wise Breakup (Pie Charts)")
    col1, col2, col3 = st.columns(3)

    with col1:
        fig1, ax1 = plt.subplots()
        ax1.pie(agg_df["Cost ($M)"], labels=agg_df["Client"], autopct='%1.1f%%')
        ax1.set_title("Cost by Client")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.pie(agg_df["Revenue ($M)"], labels=agg_df["Client"], autopct='%1.1f%%')
        ax2.set_title("Revenue by Client")
        st.pyplot(fig2)

    with col3:
        fig3, ax3 = plt.subplots()
        ax3.pie(agg_df["Total Resources"], labels=agg_df["Client"], autopct='%1.1f%%')
        ax3.set_title("Resources by Client")
        st.pyplot(fig3)

    # MoM Revenue Trend
    st.subheader("📈 Monthly Revenue by Client")
    monthly_client = df.groupby(["Month", "Client"])["Revenue"].sum().reset_index()
    pivot_df = monthly_client.pivot(index="Month", columns="Client", values="Revenue").fillna(0)

    fig, ax = plt.subplots(figsize=(10, 4))
    pivot_df.plot(ax=ax)
    ax.set_ylabel("Revenue")
    ax.set_title("MoM Revenue Trend by Client")
    ax.legend(title="Client")
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

        if is_full_client_report(user_query):
            show_full_client_report(df)
        else:
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

                # 📒 Summary Text
                st.subheader("🖌️ Key Insights Summary")
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
