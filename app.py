import streamlit as st
import pandas as pd
import os
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from operator import itemgetter
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains import create_extraction_chain_pydantic
from typing import List, Optional
from langchain.memory import ChatMessageHistory
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser

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
    st.session_state.llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.0,
        max_output_tokens=1024
    )
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# --- Helper Functions ---

# Function to get table details from a CSV file
def get_table_details():
    try:
        table_description = pd.read_csv("table_description.csv")
        table_details = ""
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

# Initialize vectorstore and example selector once
@st.cache_resource
def initialize_vectorstore():
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
    
    vectorstore = Chroma.from_texts(
        [example["question"] for example in examples],
        st.session_state.embedding_model
    )
    
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
        input_keys=["question"],
    )
    
    return examples, example_selector

examples, example_selector = initialize_vectorstore()

# Set up the example prompt template
example_prompt = PromptTemplate.from_template(
    "User input: {question}\nSQL query: {query}"
)

# Few-shot prompt template
few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="You are a MySQL expert. Here are some examples of questions and their corresponding SQL queries:",
    suffix="Now, generate a SQL query for the following question:\nUser input: {question}\nSQL query:",
    input_variables=["question"],
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
        ("system", "You are a MySQL expert. Given an input question and table information, create a syntactically correct SQL query.\n\nHere is the relevant table info: {table_info}"),
        MessagesPlaceholder(variable_name="few_shot_examples"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# Get table details
table_details = get_table_details()

# Table selection function - simplified approach
def select_relevant_tables(question: str) -> str:
    if table_details:
        # For now, return all table details - you can make this smarter
        return table_details
    return ""

# Main chain components
generate_query_chain = create_sql_query_chain(st.session_state.llm, st.session_state.db)
execute_query_tool = QuerySQLDatabaseTool(db=st.session_state.db)
rephrase_answer_chain = rephrase_answer_prompt | st.session_state.llm | StrOutputParser()

# Simplified chain that works step by step
def process_question(question: str, history_messages):
    try:
        # Step 1: Get relevant table info
        relevant_tables = select_relevant_tables(question)
        
        # Step 2: Generate SQL query
        query_input = {
            "question": question,
            "table_info": relevant_tables
        }
        generated_query = generate_query_chain.invoke(query_input)
        
        # Step 3: Execute query
        query_results = execute_query_tool.invoke({"query": generated_query})
        
        # Step 4: Generate final answer
        answer_input = {
            "question": question,
            "query": generated_query,
            "results": query_results
        }
        final_answer = rephrase_answer_chain.invoke(answer_input)
        
        return final_answer, generated_query, query_results, relevant_tables
        
    except Exception as e:
        raise Exception(f"Error in processing: {str(e)}")

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

    # Run the chain and display progress
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Process the question
                response, generated_query, query_results, tables_used = process_question(
                    user_question, 
                    st.session_state.history.messages
                )
                
                # Display the response
                st.markdown(response)

                # Add generated AI response to history
                st.session_state.history.add_ai_message(response)
                
                # Use expanders to show the "behind-the-scenes" process
                with st.expander("Show details"):
                    st.subheader("Relevant Tables")
                    if tables_used:
                        st.text(tables_used)
                    else:
                        st.write("No specific tables selected (or `table_description.csv` is missing).")
                    
                    st.subheader("Generated SQL Query")
                    st.code(generated_query, language="sql")
                    
                    st.subheader("Query Results")
                    st.text(str(query_results))

            except Exception as e:
                error_msg = f"I encountered an error while processing your request: {e}"
                st.error(error_msg)
                st.session_state.history.add_ai_message(error_msg)

# Add a reset button to clear the conversation
if st.button("Reset Conversation"):
    st.session_state.history = ChatMessageHistory()
    st.rerun()
