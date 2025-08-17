import streamlit as st
import pandas as pd
import os
import sqlite3
import tempfile
import asyncio
import nest_asyncio
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
import re
import io
import time


# Fix the event loop issue
nest_asyncio.apply()

# --- Configuration and Initialization ---

# Check for required secrets (only need API keys now)
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Missing GEMINI_API_KEY. Please check your .streamlit/secrets.toml file.")
    st.stop()

# Set environment variables from Streamlit secrets
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
if "LANGCHAIN_API_KEY" in st.secrets:
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Initialize LLM and embedding model
if 'llm' not in st.session_state:
    st.session_state.llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.0,
        max_output_tokens=1024
    )
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize chat history
if 'history' not in st.session_state:
    st.session_state.history = ChatMessageHistory()

# --- Helper Functions ---

def create_database_from_csv_files(csv_files):
    """Create SQLite database from uploaded CSV files"""
    # Create temporary database
    db_path = tempfile.mktemp(suffix='.db')
    conn = sqlite3.connect(db_path)
    
    table_info = ""
    
    for csv_file in csv_files:
        # Read CSV
        df = pd.read_csv(csv_file)
        
        # Clean table name (remove extension and special characters)
        table_name = re.sub(r'[^a-zA-Z0-9_]', '_', csv_file.name.split('.')[0].lower())
        
        # Save to SQLite
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Generate table info
        table_info += f"Table Name: {table_name}\n"
        table_info += f"Columns: {', '.join(df.columns.tolist())}\n"
        table_info += f"Sample Data: {df.head(2).to_dict('records')}\n\n"
    
    conn.close()
    return db_path, table_info

def create_database_from_excel(excel_file):
    """Create SQLite database from uploaded Excel file"""
    # Create temporary database
    db_path = tempfile.mktemp(suffix='.db')
    conn = sqlite3.connect(db_path)
    
    # Read all sheets
    excel_data = pd.read_excel(excel_file, sheet_name=None)
    
    table_info = ""
    
    for sheet_name, df in excel_data.items():
        # Clean table name
        table_name = re.sub(r'[^a-zA-Z0-9_]', '_', sheet_name.lower())
        
        # Save to SQLite
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Generate table info
        table_info += f"Table Name: {table_name} (from sheet: {sheet_name})\n"
        table_info += f"Columns: {', '.join(df.columns.tolist())}\n"
        table_info += f"Sample Data: {df.head(2).to_dict('records')}\n\n"
    
    conn.close()
    return db_path, table_info

def execute_sql_file(sql_file, db_path):
    """Execute SQL file against the database"""
    conn = sqlite3.connect(db_path)
    
    # Read SQL file content
    sql_content = sql_file.read().decode('utf-8')
    
    # Split SQL statements (simple approach)
    statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
    
    results = []
    for statement in statements:
        try:
            cursor = conn.execute(statement)
            if statement.upper().startswith('SELECT'):
                result = cursor.fetchall()
                results.append(f"Query: {statement}\nResults: {result}\n")
            else:
                conn.commit()
                results.append(f"Executed: {statement}\n")
        except Exception as e:
            results.append(f"Error in statement '{statement}': {e}\n")
    
    conn.close()
    return '\n'.join(results)

# Initialize vectorstore and example selector once
@st.cache_resource
def initialize_examples():
    # Simple examples without vectorstore to avoid async issues
    examples = [
        {
            "question": "What is the total sales amount for each product?",
            "query": "SELECT product_name, SUM(sales_amount) AS total_sales FROM sales GROUP BY product_name;"
        },
        {
            "question": "How many customers are there?",
            "query": "SELECT COUNT(DISTINCT customer_id) AS total_customers FROM customers;"
        },
        {
            "question": "Show me the top 5 products by sales",
            "query": "SELECT product_name, SUM(sales_amount) as total FROM sales GROUP BY product_name ORDER BY total DESC LIMIT 5;"
        }
    ]
    
    return examples

# Set up prompt templates
def setup_prompts():
    example_prompt = PromptTemplate.from_template(
        "User input: {question}\nSQL query: {query}"
    )

    rephrase_answer_prompt = PromptTemplate.from_template(
        """
        Given the SQL query results, provide a concise and helpful answer to the question.
        Question: {question}   
        SQL Query: {query}
        SQL Query Results: {results}
        
        Answer:
        """
    )
    
    return example_prompt, rephrase_answer_prompt

# Process question function
def process_question(question: str, db, table_info):
    try:
        # Step 1: Generate SQL query
        query_input = {
            "question": question,
            "table_info": table_info
        }
        generate_query_chain = create_sql_query_chain(st.session_state.llm, db)
        generated_query = generate_query_chain.invoke(query_input)
        
        # Step 2: Execute query
        execute_query_tool = QuerySQLDatabaseTool(db=db)
        query_results = execute_query_tool.invoke({"query": generated_query})
        
        # Step 3: Generate final answer
        example_prompt, rephrase_answer_prompt = setup_prompts()
        rephrase_answer_chain = rephrase_answer_prompt | st.session_state.llm | StrOutputParser()
        
        answer_input = {
            "question": question,
            "query": generated_query,
            "results": query_results
        }
        final_answer = rephrase_answer_chain.invoke(answer_input)
        
        return final_answer, generated_query, query_results
        
    except Exception as e:
        raise Exception(f"Error in processing: {str(e)}")

# --- Streamlit UI ---

st.title("📊 Natural Language to SQL with File Upload")
st.write("Upload your data files (CSV, Excel, or SQL) and ask questions about your data using natural language!")

# Sidebar for file uploads
with st.sidebar:
    st.header("📁 Upload Your Data")
    
    upload_option = st.radio(
        "Choose your data source:",
        ["CSV Files", "Excel File", "Excel + SQL File"]
    )
    
    db_path = None
    table_info = ""
    
    if upload_option == "CSV Files":
        csv_files = st.file_uploader(
            "Upload CSV files",
            type=['csv'],
            accept_multiple_files=True,
            help="Upload one or more CSV files. Each file will become a table."
        )
        
        if csv_files:
            with st.spinner("Creating database from CSV files..."):
                try:
                    db_path, table_info = create_database_from_csv_files(csv_files)
                    st.success(f"✅ Created database with {len(csv_files)} tables!")
                    
                    with st.expander("View Table Information"):
                        st.text(table_info)
                        
                except Exception as e:
                    st.error(f"Error creating database: {e}")
    
    elif upload_option == "Excel File":
        excel_file = st.file_uploader(
            "Upload Excel file",
            type=['xlsx', 'xls'],
            help="Upload an Excel file. Each sheet will become a table."
        )
        
        if excel_file:
            with st.spinner("Creating database from Excel file..."):
                try:
                    db_path, table_info = create_database_from_excel(excel_file)
                    st.success("✅ Created database from Excel file!")
                    
                    with st.expander("View Table Information"):
                        st.text(table_info)
                        
                except Exception as e:
                    st.error(f"Error creating database: {e}")
    
    elif upload_option == "Excel + SQL File":
        excel_file = st.file_uploader(
            "Upload Excel file",
            type=['xlsx', 'xls'],
            help="Upload an Excel file for data"
        )
        
        sql_file = st.file_uploader(
            "Upload SQL file",
            type=['sql'],
            help="Upload a SQL file to modify the database"
        )
        
        if excel_file:
            with st.spinner("Creating database from Excel file..."):
                try:
                    db_path, table_info = create_database_from_excel(excel_file)
                    st.success("✅ Created database from Excel file!")
                    
                    if sql_file:
                        with st.spinner("Executing SQL file..."):
                            try:
                                sql_results = execute_sql_file(sql_file, db_path)
                                st.success("✅ SQL file executed!")
                                
                                with st.expander("SQL Execution Results"):
                                    st.text(sql_results)
                            except Exception as e:
                                st.error(f"Error executing SQL file: {e}")
                    
                    with st.expander("View Table Information"):
                        st.text(table_info)
                        
                except Exception as e:
                    st.error(f"Error creating database: {e}")

# Main chat interface
st.write("---")

# Initialize database connection in session state
if db_path and db_path not in st.session_state.get('processed_files', set()):
    try:
        st.session_state.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        st.session_state.table_info = table_info
        
        # Keep track of processed files to avoid re-processing
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = set()
        st.session_state.processed_files.add(db_path)
        
        st.success("🎉 Database ready! You can now ask questions.")
        
        # Show available tables
        try:
            available_tables = st.session_state.db.get_usable_table_names()
            st.info(f"📋 Available tables: {', '.join(available_tables)}")
        except:
            pass
            
    except Exception as e:
        st.error(f"❌ Error connecting to database: {e}")
        if 'db' in st.session_state:
            del st.session_state.db

# Display chat history
for msg in st.session_state.history.messages:
    if msg.type == "human":
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Handle user input
if 'db' in st.session_state:
    user_question = st.chat_input("Ask a question about your data...")
    
    if user_question:
        # Add user question to history
        st.session_state.history.add_user_message(user_question)
        with st.chat_message("user"):
            st.markdown(user_question)

        # Process the question
        with st.chat_message("assistant"):
            # Create a placeholder for step-by-step updates
            status_placeholder = st.empty()
            
            try:
                status_placeholder.write("🤖 Analyzing your data...")
                
                response, generated_query, query_results = process_question(
                    user_question, 
                    st.session_state.db,
                    st.session_state.table_info
                )
                
                # Clear status and show final response
                status_placeholder.empty()
                
                # Display the response
                st.markdown(response)

                # Add response to history
                st.session_state.history.add_ai_message(response)
                
                # Show details in expander
                with st.expander("🔍 Show Details"):
                    st.subheader("Generated SQL Query")
                    st.code(generated_query, language="sql")
                    
                    st.subheader("Query Results")
                    st.text(str(query_results))

            except Exception as e:
                status_placeholder.empty()
                error_msg = f"❌ I encountered an error: {e}"
                st.error(error_msg)
                st.session_state.history.add_ai_message(error_msg)
                
                # Show debug info
                with st.expander("🐛 Debug Information"):
                    st.write("Database tables:", st.session_state.db.get_usable_table_names())
                    st.write("Table info available:", bool(st.session_state.table_info))
                    st.write("Error details:", str(e))
else:
    st.info("👆 Please upload your data files using the sidebar to get started!")

# Reset conversation button
if st.button("🔄 Reset Conversation"):
    st.session_state.history = ChatMessageHistory()
    if 'db' in st.session_state:
        del st.session_state.db
    if 'table_info' in st.session_state:
        del st.session_state.table_info
    st.rerun()
