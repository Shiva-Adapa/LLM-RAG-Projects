import os
import html
from io import BytesIO

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Inventory Intelligence",
    layout="wide"
)

st.title("AI Inventory Intelligence Dashboard")
st.caption("Product-level sales, inventory decisions, AI agent insights, and PDF reports")


# =========================
# API SETUP
# =========================

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FIX 1: Corrected model name from "gpt-5.4-mini" to "gpt-4o-mini"
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)


# =========================
# LOAD DATA
# =========================

st.sidebar.header("Data Source")

uploaded_file = st.sidebar.file_uploader(
    "Upload Excel file",
    type=["xlsx"]
)

# FIX 2: Removed hardcoded local file path — upload is now required
if uploaded_file is None:
    st.warning("Please upload your Excel file using the sidebar to get started.")
    st.stop()

file_source = uploaded_file

sales_df = pd.read_excel(file_source, sheet_name="Sales")
stock_df = pd.read_excel(file_source, sheet_name="Stock")
products_df = pd.read_excel(file_source, sheet_name="Products")


# =========================
# DATA PREPARATION
# =========================

sales_products_df = sales_df.merge(
    products_df[["ProductID", "ProductName", "Category"]],
    on="ProductID",
    how="left"
)

sales_summary = (
    sales_products_df
    .groupby(["ProductID", "ProductName", "Category"])
    .agg(
        TotalSalesAED=("SalesAmountAED", "sum"),
        TotalUnitsSold=("UnitsSold", "sum"),
        Transactions=("SalesID", "count")
    )
    .reset_index()
)

stock_summary = (
    stock_df
    .groupby("ProductID")
    .agg(
        TotalOnHandQty=("OnHandQty", "sum"),
        TotalAvailableQty=("AvailableQty", "sum"),
        TotalStockValueAED=("StockValueAED", "sum")
    )
    .reset_index()
)

sales_stock_df = sales_summary.merge(
    stock_summary,
    on="ProductID",
    how="left"
)


# =========================
# DECISION LOGIC
# =========================

def product_decision(row):
    if row["TotalSalesAED"] >= 30000 and row["TotalAvailableQty"] <= 20:
        return "RESTOCK"
    elif row["TotalSalesAED"] < 15000 and row["TotalAvailableQty"] > 100:
        return "REVIEW / OVERSTOCK"
    elif row["TotalSalesAED"] >= 30000:
        return "KEEP"
    else:
        return "MONITOR"


sales_stock_df["Decision"] = sales_stock_df.apply(product_decision, axis=1)


# =========================
# FILTERS
# =========================

st.sidebar.header("Filters")

category_options = sorted(sales_stock_df["Category"].dropna().unique())
decision_options = sorted(sales_stock_df["Decision"].dropna().unique())

selected_categories = st.sidebar.multiselect(
    "Category",
    category_options,
    default=category_options
)

selected_decisions = st.sidebar.multiselect(
    "Decision",
    decision_options,
    default=decision_options
)

filtered_df = sales_stock_df[
    (sales_stock_df["Category"].isin(selected_categories)) &
    (sales_stock_df["Decision"].isin(selected_decisions))
]


# =========================
# SUMMARIES
# =========================

sales_by_category = (
    filtered_df
    .groupby("Category")["TotalSalesAED"]
    .sum()
    .reset_index()
    .sort_values(by="TotalSalesAED", ascending=False)
)

decision_count = (
    filtered_df["Decision"]
    .value_counts()
    .reset_index()
)

decision_count.columns = ["Decision", "Count"]

product_ai_summary = (
    filtered_df
    .sort_values(by="TotalSalesAED", ascending=False)
)


# =========================
# AGENT TOOLS
# =========================

def get_sales_by_category():
    if sales_by_category.empty:
        return "No sales by category available for the current filters."
    return sales_by_category.to_string(index=False)


def get_overstock_products():
    result_df = filtered_df[
        filtered_df["Decision"] == "REVIEW / OVERSTOCK"
    ][
        [
            "ProductName",
            "Category",
            "TotalSalesAED",
            "TotalAvailableQty",
            "TotalStockValueAED"
        ]
    ]

    if result_df.empty:
        return "No overstock products found in the current filtered data."

    return result_df.to_string(index=False)


def get_restock_products():
    result_df = filtered_df[
        filtered_df["Decision"] == "RESTOCK"
    ][
        [
            "ProductName",
            "Category",
            "TotalSalesAED",
            "TotalAvailableQty",
            "TotalStockValueAED"
        ]
    ]

    if result_df.empty:
        return "No restock products found in the current filtered data."

    return result_df.to_string(index=False)


def get_product_summary():
    if product_ai_summary.empty:
        return "No product summary available for the current filters."

    return product_ai_summary[
        [
            "ProductName",
            "Category",
            "TotalSalesAED",
            "TotalUnitsSold",
            "Transactions",
            "TotalAvailableQty",
            "TotalStockValueAED",
            "Decision"
        ]
    ].to_string(index=False)


# =========================
# KPIs
# =========================

total_sales = filtered_df["TotalSalesAED"].sum()
total_stock_value = filtered_df["TotalStockValueAED"].sum()
total_products = len(filtered_df)
review_count = filtered_df["Decision"].str.contains("REVIEW").sum()
restock_count = filtered_df["Decision"].str.contains("RESTOCK").sum()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Sales AED", f"{total_sales:,.0f}")
col2.metric("Stock Value AED", f"{total_stock_value:,.0f}")
col3.metric("Products", f"{total_products:,}")
col4.metric("Review / Overstock", review_count)
col5.metric("Restock Alerts", restock_count)


# =========================
# EXECUTIVE RECOMMENDATIONS
# =========================

st.subheader("AI Executive Recommendations")

if st.button("Generate Executive Recommendations"):
    with st.spinner("Generating executive recommendations..."):

        recommendation_prompt = f"""
            You are a senior business analyst.

            Based on this product-level sales and inventory summary:

            {get_product_summary()}

            Generate:
            1. Top Risk
            2. Top Opportunity
            3. Recommended Action

            Keep each response short and executive-level.
        """

        rec_response = llm.invoke(recommendation_prompt)
        recommendation_text = rec_response.content

        sections = recommendation_text.split("\n\n")

        risk_text = sections[0] if len(sections) > 0 else "No risk identified."
        opportunity_text = sections[1] if len(sections) > 1 else "No opportunity identified."
        action_text = sections[2] if len(sections) > 2 else "No action identified."

        rec_col1, rec_col2, rec_col3 = st.columns(3)

        rec_col1.error(risk_text)
        rec_col2.success(opportunity_text)
        rec_col3.warning(action_text)


# =========================
# TABS
# =========================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dashboard",
    "Product Decisions",
    "AI Agent",
    "Raw Data",
    "About Project"
])


# =========================
# TAB 1: DASHBOARD
# =========================

with tab1:
    st.subheader("Sales by Category")

    if sales_by_category.empty:
        st.warning("No data available for the selected filters.")
    else:
        st.bar_chart(sales_by_category.set_index("Category"))

    st.subheader("Decision Count")

    if decision_count.empty:
        st.warning("No decisions available for the selected filters.")
    else:
        st.bar_chart(decision_count.set_index("Decision"))

    st.subheader("Product Summary")
    st.dataframe(
        product_ai_summary[
            [
                "ProductID",
                "ProductName",
                "Category",
                "TotalSalesAED",
                "TotalUnitsSold",
                "Transactions",
                "TotalAvailableQty",
                "TotalStockValueAED",
                "Decision"
            ]
        ],
        use_container_width=True
    )


# =========================
# TAB 2: PRODUCT DECISIONS
# =========================

with tab2:
    st.subheader("Product-Level Decisions")

    decision_table = filtered_df[
        [
            "ProductID",
            "ProductName",
            "Category",
            "TotalSalesAED",
            "TotalUnitsSold",
            "Transactions",
            "TotalAvailableQty",
            "TotalStockValueAED",
            "Decision"
        ]
    ]

    st.dataframe(decision_table, use_container_width=True)


# =========================
# TAB 3: AI AGENT WITH LANGCHAIN
# =========================

with tab3:
    st.subheader("AI Business Agent")

    st.info(
        "The agent selects an analytical tool, then LangChain generates the final business response."
    )

    question = st.text_input(
        "Ask a question about sales, inventory, categories, overstock, or restocking"
    )

    if st.button("Ask AI Agent"):

        if not question:
            st.warning("Please enter a question first.")

        else:
            with st.spinner("Selecting the right analytical tool..."):

                tool_selection_prompt = f"""
                    You are an AI business analyst agent.

                    Available tools:

                    1. sales_by_category
                    Use this when the user asks about:
                    - category performance
                    - top categories
                    - sales by category
                    - revenue by category

                    2. overstock_products
                    Use this when the user asks about:
                    - overstock
                    - weak products
                    - excess inventory
                    - products needing review
                    - cash tied in stock

                    3. restock_products
                    Use this when the user asks about:
                    - low stock
                    - replenishment
                    - restocking
                    - shortage risk

                    4. product_summary
                    Use this when the user asks a broad or general business question.

                    User Question:
                    {question}

                    Return ONLY one tool name from this list:
                    sales_by_category
                    overstock_products
                    restock_products
                    product_summary
                """

                # FIX 3: Corrected OpenAI API call pattern
                # Old (incorrect): client.responses.create(...) with output_text
                # New (correct):   client.chat.completions.create(...) with choices[0].message.content
                tool_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "user", "content": tool_selection_prompt}
                    ]
                )

                selected_tool = tool_response.choices[0].message.content.strip()

                if "sales_by_category" in selected_tool:
                    tool_data = get_sales_by_category()
                elif "overstock_products" in selected_tool:
                    tool_data = get_overstock_products()
                elif "restock_products" in selected_tool:
                    tool_data = get_restock_products()
                else:
                    selected_tool = "product_summary"
                    tool_data = get_product_summary()

            with st.spinner("LangChain is generating the business insight..."):

                agent_prompt = PromptTemplate(
                    input_variables=["selected_tool", "tool_data", "question"],
                    template="""
                        You are a senior business analyst AI agent.

                        The following business data was retrieved using the most relevant analytical tool.

                        Tool Used:
                        {selected_tool}

                        Tool Output:
                        {tool_data}

                        User Question:
                        {question}

                        Give:
                        - business insight
                        - risks if any
                        - recommendation

                        Use clear, practical business language.
                    """
                )

                final_prompt = agent_prompt.format(
                    selected_tool=selected_tool,
                    tool_data=tool_data,
                    question=question
                )

                response = llm.invoke(final_prompt)

                st.subheader("Selected Tool")
                st.code(selected_tool)

                st.subheader("Tool Output")
                st.text(tool_data)

                st.subheader("AI Insight")
                st.write(response.content)

                # PDF REPORT
                buffer = BytesIO()
                doc = SimpleDocTemplate(buffer)
                styles = getSampleStyleSheet()
                elements = []

                safe_question = html.escape(question).replace("\n", "<br/>")
                safe_tool = html.escape(selected_tool).replace("\n", "<br/>")
                safe_insight = html.escape(response.content).replace("\n", "<br/>")

                elements.append(Paragraph("AI Inventory Intelligence Report", styles["Title"]))
                elements.append(Spacer(1, 12))

                elements.append(Paragraph(f"<b>Question:</b><br/>{safe_question}", styles["BodyText"]))
                elements.append(Spacer(1, 12))

                elements.append(Paragraph(f"<b>Selected Tool:</b><br/>{safe_tool}", styles["BodyText"]))
                elements.append(Spacer(1, 12))

                elements.append(Paragraph(f"<b>AI Insight:</b><br/>{safe_insight}", styles["BodyText"]))

                doc.build(elements)

                pdf_data = buffer.getvalue()
                buffer.close()

                st.download_button(
                    label="Download PDF Report",
                    data=pdf_data,
                    file_name="AI_Business_Report.pdf",
                    mime="application/pdf"
                )


# =========================
# TAB 4: RAW DATA
# =========================

with tab4:
    st.subheader("Sales Data")
    st.dataframe(sales_df, use_container_width=True)

    st.subheader("Stock Data")
    st.dataframe(stock_df, use_container_width=True)

    st.subheader("Products Data")
    st.dataframe(products_df, use_container_width=True)


# =========================
# TAB 5: ABOUT PROJECT
# =========================

with tab5:
    st.title("About This Project")

    st.markdown("""
**Lana Al-Maradni** · AI-Augmented Analytics Specialist · Dubai, UAE

---

This dashboard was designed and built from scratch to demonstrate how AI
can move inventory analytics beyond static reporting — toward intelligent,
automated decision-making. The dataset was purpose-built to reflect real
supply chain dynamics common in UAE and GCC markets.

---

### 🔍 Business Problem

Traditional dashboards show data. They don't recommend decisions.

Companies across the UAE and GCC struggle with:
- Excess inventory tying up cash with no visibility
- Low stock on high-performing products causing lost revenue
- Slow, manual reporting cycles that delay action
- No intelligent layer to turn data into executive recommendations

---

### ✅ Solution

This application bridges that gap by combining:

1. **Sales analytics** — product-level revenue and transaction summaries
2. **Inventory analysis** — on-hand quantity, available stock, stock value
3. **Decision logic** — rule-based engine flagging RESTOCK, KEEP, MONITOR, REVIEW / OVERSTOCK
4. **AI agent** — selects the most relevant analytical tool per business question
5. **Executive recommendations** — AI-generated Top Risk, Opportunity, and Action
6. **PDF reporting** — downloadable executive report per session

---

### 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data | Python · pandas · Excel |
| AI | OpenAI API · LangChain |
| Dashboard | Streamlit |
| Reporting | ReportLab PDF |

---

### 🏗️ Architecture

```
Data Layer        →  Excel + pandas
Logic Layer       →  Product-level inventory decision rules
AI Layer          →  Tool selection + LangChain response generation
Presentation      →  Streamlit interactive dashboard
Reporting Layer   →  Downloadable PDF executive reports
```

---

### 👤 About the Author

**Lana Al-Maradni** is an AI-Augmented Analytics Specialist based in Dubai, UAE,
with 19+ years of experience in data and business analytics — including humanitarian
analytics with UN/OCHA across Syria, Jordan, and South Sudan, covering needs
assessments, response analysis, reach vs. target reporting, and humanitarian access
monitoring across complex operational environments.

Now focused on building AI-native analytics solutions for UAE and GCC markets,
combining deep domain expertise with modern AI tooling including LangChain,
OpenAI API, Power BI, and workflow automation.

📍 Dubai, UAE · 💼 Open to AI Analytics & Consulting roles

🔗 [GitHub](https://github.com/lana-almaradni)
""")