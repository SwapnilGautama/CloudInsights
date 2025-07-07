import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt

# 🔑 Set OpenAI API Key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 📄 Load CSV from GitHub
CSV_URL = "https://raw.githubusercontent.com/SwapnilGautama/CloudInsights/main/SoftwareCompany_2025_Data.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_URL)
    df["Month"] = pd.to_datetime(df["Month"])
    return df

# 📊 Plot dual-axis chart
def plot_dual_axis_bar(agg):
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()
    ax1.bar(agg["Type"], agg["Revenue ($M)"], color="skyblue", label="Revenue ($M)")
    ax2.plot(agg["Type"], agg["Cost ($M)"], color="red", marker="o", label="Cost ($M)")
    ax1.set_ylabel("Revenue ($M)")
    ax2.set_ylabel("Cost ($M)")
    ax1.set_title("Revenue and Cost by Type")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    st.pyplot(fig)

# 🧠 GPT Query Generator
def ask_gpt(user_query, df_sample):
    prompt = f"""
You are a data analyst. The user wants revenue and cost insights from this dataset:
{', '.join(df_sample.columns)}

User said: "{user_query}"

Return Python pandas code that:
1. Filters the data for the relevant client (case-insensitive).
2. Returns three variables:
   - result: filtered rows
   - summary1: revenue by Type
   - summary2: cost split by Location (Onshore/Offshore)

Return only the code. DataFrame name is df.
"""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

# 🚀 Streamlit UI
st.set_page_config(page_title="Cloud Insights Chatbot", page_icon="💬", layout="wide")
st.title("💬 Cloud Insights Chatbot")

df = load_data()

with st.sidebar:
    st.markdown("### 🧾 Clients in Dataset")
    for client in sorted(df["Client"].unique()):
        st.markdown(f"- {client}")

user_query = st.text_input("Ask a question like:", "Show revenue and cost breakdown for BMW")

if user_query:
    try:
        lowered = user_query.lower()
        st.markdown("Generating insights...")

        if any(word in lowered for word in ["total", "overall", "aggregate"]):
            result = df.copy()
            summary1 = result.groupby("Type", dropna=False)["Revenue"].sum().reset_index()

            if "Location" in result.columns:
                summary2 = result.groupby("Location", dropna=False)["Cost"].sum().reset_index()
            else:
                summary2 = pd.DataFrame(columns=["Location", "Cost"])

        else:
            code = ask_gpt(user_query, df.head(3))
            local_vars = {"df": df.copy()}
            exec(code.strip().strip("`").replace("python", ""), {}, local_vars)
            result = local_vars["result"]
            summary1 = local_vars["summary1"]
            summary2 = local_vars["summary2"]

        # Ensure columns exist
        if "Resources_Total" not in result.columns:
            result["Resources_Total"] = 0
        if "Type" not in result.columns:
            result["Type"] = "Unknown"

        agg = result.groupby("Type", dropna=False).agg({
            "Revenue": "sum",
            "Cost": "sum",
            "Resources_Total": "sum"
        }).reset_index()

        agg["Revenue ($M)"] = (agg["Revenue"] / 1_000_000).round(2)
        agg["Cost ($M)"] = (agg["Cost"] / 1_000_000).round(2)
        agg.rename(columns={"Resources_Total": "Total Resources"}, inplace=True)

        # 🔢 Summary Block
        st.subheader("📌 Total Summary")
        st.markdown(f"- **Total Revenue:** ${result['Revenue'].sum() / 1_000_000:.2f}M")
        st.markdown(f"- **Total Cost:** ${result['Cost'].sum() / 1_000_000:.2f}M")
        st.markdown(f"- **Total Resources:** {int(result['Resources_Total'].sum())}")

        # 📊 Type Summary
        st.subheader("📊 Summary by Type (Aggregated)")
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.dataframe(agg[["Type", "Revenue ($M)", "Cost ($M)", "Total Resources"]],
                         use_container_width=True, height=350)
        with col2:
            plot_dual_axis_bar(agg)

        # 📋 Raw Table
        st.subheader("📋 Project-wise and Fixed Position Data")
        st.dataframe(result, use_container_width=True, height=400)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
