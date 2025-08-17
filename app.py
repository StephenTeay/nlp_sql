import streamlit as st
import pandas as pd
import os
from langchain.chains import create_sql_query_chain
from langchain_gemini import Gemini, GeminiEmbeddings
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from operator import itemgetter
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_core.example_selector import SemanticSimilarityExampleSelector
from langchain_core.pydantic import BaseModel, Field
from langchain.chains.gemini_tools import create_extraction_chain_pydantic
from typing import List, Optional
from langchain.memory import ChatMessageHistory
from langchain_community.utilities.sql_database import SQLDatabase

# --- Configuration and Initialization ---

# Check for required secrets
if "GEMINI_API_KEY" not in st.secrets or "LANGCHAIN_API_KEY" not in st.secrets or "db" not in st.secrets:
    st.error("Missing API keys or database credentials. Please check your .streamlit/secrets.toml file.")
    st.stop()

# Set environment variables from Streamlit secrets
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Use session state to persist components across reruns
if 'history' not in st.session_state:
    st.session_state.history = ChatMessageHistory()
if 'db' not in st.session_state:
    st.session_state.db_user = st.secrets["db"]["user"]
    st.session_state.db_password = st.secrets["db"]["password"]
    st.session_state.db_host = st.secrets["db"]["host"]
    st.session_state.db_name = st.secrets["db"]["name"]
    try:
        st.session_state.db = SQLDatabase.from_uri(f"mysql+pymysql://{st.session_state.db_user}:{st.session_state.db_password}@{st.session_state.db_host}/{st.session_state.db_name}")
        st.success("Successfully connected to the database!")
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        st.session_state.db = None
        st.stop()
if 'llm' not in st.session_state:
    st.session_state.llm = Gemini(
        model="gemini-1.5-flash",
        temperature=0.0,
        max_output_tokens=1024
    )
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = GeminiEmbeddings()

# --- Helper Functions ---

# Function to get table details from a CSV file
def get_table_details():
    try:
        table_description = pd.read_csv("table_description.csv")
        table_details = ""
        # Corrected iterrows() method
        for index, row in table_description.iterrows():
            table_details += f"Table Name: {row['Table']}\n"
            table_details += f"Table Description: {row['Description']}\n"
            table_details += f"Columns: {row['Columns']}\n\n"
        return table_details
    except FileNotFoundError:
        st.warning("`table_description.csv` not found. Table descriptions will not be used for query generation.")
        return ""

# Pydantic model for table extraction
class Table(BaseModel):
    name: str = Field(..., description="Name of the table in SQL database")

# Function to extract table names from the Pydantic model output
def get_tables(tables: List[Table]) -> List[str]:
    return [table.name for table in tables]

# --- LangChain Component Setup ---

# Few-shot examples for dynamic selection
examples = [
    {
        "question": "What is the total sales amount for each product in the last month?",
        "query": "SELECT product_id, SUM(sales_amount) AS total_sales FROM sales WHERE sale_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH) GROUP BY product_id;"
    },
    {
        "question": "How many customers made purchases last week?",
        "query": "SELECT COUNT(DISTINCT customer_id) AS total_customers FROM sales WHERE sale_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY);"
    }
]

# Set up the FewShotPromptTemplate for dynamic few-shot learning
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{question}\nSQLQuery:"),
        ("ai", "{query}"),
    ]
)

vectorstore = Chroma.from_texts(
    [example["question"] for example in examples],
    st.session_state.embedding_model
)

example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,
    input_keys=["question"],
)

# Corrected FewShotPromptTemplate initialization
few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    input_variables=["question", "table_info"],
)

# Prompt for the final answer
rephrase_answer_prompt = PromptTemplate.from_template(
    """
    Given the SQL query results, provide a concise answer to the question.
    Question: {question}   
    SQL Query: {query}
    SQL Query Results: {results}
    Answer:
    """
)

# Final prompt for the SQL query chain
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a MySQL expert. Given an input question, create a syntactically correct SQL query to answer the question.\n\nHere is the relevant table info: {table_info}\n\nBelow is an example of a question and its corresponding SQL query."),
        few_shot_prompt,
        MessagesPlaceholder(variable_name="history"), # Use this for chat history
        ("human", "{question}"),
    ]
)

# Extraction chain for table selection
table_details = get_table_details()
if table_details:
    table_details_prompt = f"""
    Return the names of ALL the SQL tables that might be relevant to the provided question. The tables are:
    {table_details}
    Remember to include ALL POTENTIALLY RELEVANT tables, even if you are not sure they are relevant.
    """
    select_table_chain = create_extraction_chain_pydantic(Table, st.session_state.llm, system_message=table_details_prompt)
    select_table_chain = select_table_chain | get_tables
else:
    select_table_chain = lambda x: {"table_names_to_use": ""}

# Main chain components
generate_query_chain = create_sql_query_chain(st.session_state.llm, st.session_state.db, final_prompt)
execute_query_tool = QuerySQLDatabaseTool(db=st.session_state.db)
rephrase_answer_chain = rephrase_answer_prompt | st.session_state.llm | RunnablePassthrough()

# Full end-to-end chain with follow-up questions and table selection
full_chain = (
    RunnablePassthrough.assign(
        table_names_to_use=itemgetter("question") | select_table_chain
    )
    .assign(
        query=generate_query_chain
    )
    .assign(
        results=itemgetter("query") | execute_query_tool
    )
    | rephrase_answer_chain
)

# --- Streamlit UI ---

st.title("Natural Language to SQL with LangChain 🤖")
st.write("Ask a question about your `classicmodels` database and I'll use AI to generate and execute a SQL query to find the answer.")
st.write("---")

# Display chat history
for msg in st.session_state.history.messages:
    if msg.type == "human":
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Handle user input
user_question = st.chat_input("Ask a question about the database...")

if user_question:
    # Add user question to history
    st.session_state.history.add_user_message(user_question)
    with st.chat_message("user"):
        st.markdown(user_question)

    # Run the full chain and display progress
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = full_chain.invoke({
                    "question": user_question,
                    "history": st.session_state.history.messages,
                    "table_info": table_details,
                })
                st.markdown(response)

                # Add generated AI response to history
                st.session_state.history.add_ai_message(response)
                
                # Use expanders to show the "behind-the-scenes" process
                with st.expander("Show details"):
                    # Get relevant info for display from the chain's steps
                    tables_used = itemgetter("table_names_to_use").invoke(full_chain.invoke({
                        "question": user_question,
                        "history": st.session_state.history.messages,
                        "table_info": table_details,
                    }, {"steps_to_run": ["table_names_to_use"]}))
                    
                    generated_query = itemgetter("query").invoke(full_chain.invoke({
                        "question": user_question,
                        "history": st.session_state.history.messages,
                        "table_info": table_details,
                    }, {"steps_to_run": ["query"]}))

                    query_results = itemgetter("results").invoke(full_chain.invoke({
                        "question": user_question,
                        "history": st.session_state.history.messages,
                        "table_info": table_details,
                    }, {"steps_to_run": ["results"]}))
                    
                    st.subheader("Relevant Tables")
                    st.write(tables_used if tables_used else "No specific tables selected (or `table_description.csv` is missing).")
                    
                    st.subheader("Generated SQL Query")
                    st.code(generated_query, language="sql")
                    
                    st.subheader("Query Results")
                    st.write(query_results)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.history.add_ai_message(f"I encountered an error while processing your request: {e}")

# Add a reset button to clear the conversation
if st.button("Reset Conversation"):
    st.session_state.history = ChatMessageHistory()
    st.rerun()

