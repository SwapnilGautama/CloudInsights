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

Respond with executable Python pandas code using the dataframe `df`. Do the following:

1. If the user asks for 'total', 'overall', 'aggregate', or 'company-wide', show revenue and cost across the entire dataset.
2. If a client is mentioned, filter by that client (case-insensitive).
3. If user asks to compare all clients or generate a report for all clients, then:
   - Generate a summary table grouped by client with Revenue, Cost, Resources_Total
   - Also return monthly revenue trend by client for plotting

Always return:
- result: filtered df
- summary1: revenue by Type
- summary2: cost by Location
- client_summary: groupby client table (if relevant)
- monthly_trend: monthly trend by client (if relevant)

Avoid any return statements. Do not wrap in functions. No explanation.
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
    st.markdown("### 🗒️ Clients in Dataset")
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

        result = local_vars.get("result")
        summary1 = local_vars.get("summary1")
        summary2 = local_vars.get("summary2")
        client_summary = local_vars.get("client_summary")
        monthly_trend = local_vars.get("monthly_trend")

        if result is not None:
            agg = result.groupby("Type").agg({
                "Revenue": "sum",
                "Cost": "sum",
                "Resources_Total": "sum"
            }).reset_index()

            agg["Revenue ($M)"] = (agg["Revenue"] / 1_000_000).round(2)
            agg["Cost ($M)"] = (agg["Cost"] / 1_000_000).round(2)
            agg.rename(columns={"Resources_Total": "Total Resources"}, inplace=True)

            st.subheader("📜 Key Insights Summary")
            for _, row in agg.iterrows():
                st.markdown(f"- **The total revenue is ${row['Revenue ($M)']}M and total cost is ${row['Cost ($M)']}M for `{row['Type']}` engagements.**")

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

            st.subheader("📈 Monthly Revenue vs Cost Trend")
            monthly = result.groupby("Month").agg({"Revenue": "sum", "Cost": "sum"}).sort_index()
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

            st.subheader("📋 Project-wise and Fixed Position Data")
            st.dataframe(result, use_container_width=True, height=400)

        # New block: client comparison & reporting
        if client_summary is not None:
            st.subheader("📊 Client-wise Summary Table")
            client_summary["Revenue ($M)"] = (client_summary["Revenue"] / 1_000_000).round(2)
            client_summary["Cost ($M)"] = (client_summary["Cost"] / 1_000_000).round(2)
            client_summary = client_summary.rename(columns={"Resources_Total": "Resources"})

            total_row = pd.DataFrame({
                "Client": ["Total"],
                "Revenue ($M)": [client_summary["Revenue ($M)"].sum()],
                "Cost ($M)": [client_summary["Cost ($M)"].sum()],
                "Resources": [client_summary["Resources"].sum()]
            })
            full_table = pd.concat([client_summary[["Client", "Revenue ($M)", "Cost ($M)", "Resources"]], total_row])
            st.dataframe(full_table, use_container_width=True)

            st.subheader("🎈 Distribution by Client")
            col1, col2, col3 = st.columns(3)
            with col1:
                fig1, ax1 = plt.subplots()
                ax1.pie(client_summary["Cost ($M)"], labels=client_summary["Client"], autopct='%1.1f%%')
                ax1.set_title("Cost Distribution")
                st.pyplot(fig1)
            with col2:
                fig2, ax2 = plt.subplots()
                ax2.pie(client_summary["Revenue ($M)"], labels=client_summary["Client"], autopct='%1.1f%%')
                ax2.set_title("Revenue Distribution")
                st.pyplot(fig2)
            with col3:
                fig3, ax3 = plt.subplots()
                ax3.pie(client_summary["Resources"], labels=client_summary["Client"], autopct='%1.1f%%')
                ax3.set_title("Resources Distribution")
                st.pyplot(fig3)

        if monthly_trend is not None:
            st.subheader("📅 Monthly Revenue Trend by Client")
            pivot = monthly_trend.pivot(index="Month", columns="Client", values="Revenue").fillna(0)
            fig, ax = plt.subplots(figsize=(10, 5))
            pivot = pivot / 1_000_000
            pivot.plot(ax=ax, marker="o")
            ax.set_title("Monthly Revenue by Client")
            ax.set_ylabel("Revenue ($M)")
            ax.set_xlabel("Month")
            ax.legend(title="Client")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
