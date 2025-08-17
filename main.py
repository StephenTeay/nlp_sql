import streamlit as st
import pandas as pd
import os
import sqlite3
import tempfile
import re
import traceback
import numpy as np

from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# --- Configuration and Initialization ---

# Check for required secrets
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Missing GOOGLE_API_KEY. Please check your .streamlit/secrets.toml file.")
    st.stop()

# Set environment variables from Streamlit secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
if "LANGCHAIN_API_KEY" in st.secrets:
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Initialize chat history
if 'history' not in st.session_state:
    st.session_state.history = ChatMessageHistory()

# --- Helper Functions ---

def get_column_summary(df, col):
    """Get detailed summary for a column"""
    non_null = df[col].dropna()
    n_non_null = len(non_null)
    
    # Determine sample size safely
    sample_size = min(3, n_non_null)
    sample_values = non_null.sample(sample_size).tolist() if n_non_null > 0 else []
    
    # Detect column type
    col_type = str(df[col].dtype)
    if col_type.startswith('datetime'):
        col_type = "datetime"
    elif np.issubdtype(df[col].dtype, np.number):
        col_type = "numeric"
    elif col_type == 'object':
        if df[col].apply(lambda x: isinstance(x, str)).all():
            col_type = "text"
        else:
            col_type = "mixed"
    
    # Get value range for numeric columns
    value_range = ""
    if col_type == "numeric" and n_non_null > 0:
        value_range = f" | Range: [{non_null.min()}, {non_null.max()}]"
    
    # Detect date format
    date_format = ""
    if col_type == "datetime" and n_non_null > 0:
        try:
            date_format = pd.to_datetime(non_null).dt.strftime('%Y-%m-%d').iloc[0]
            date_format = f" | Format: {date_format[:10]}"
        except:
            pass
    
    return {
        "name": col,
        "type": col_type,
        "sample": sample_values,
        "non_null_count": n_non_null,
        "total_count": len(df),
        "summary": f"- {col}: {col_type}{value_range}{date_format} | Sample: {sample_values}"
    }

def create_database_from_csv_files(csv_files):
    """Create SQLite database from uploaded CSV files"""
    db_path = tempfile.mktemp(suffix='.db')
    conn = sqlite3.connect(db_path)
    
    table_info = ""
    schema_description = ""
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        table_name = re.sub(r'[^a-zA-Z0-9_]', '_', csv_file.name.split('.')[0].lower())
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Build detailed schema description
        schema_description += f"### Table: {table_name}\n"
        schema_description += f"- Rows: {len(df)}, Columns: {len(df.columns)}\n"
        
        column_summaries = []
        for col in df.columns:
            col_summary = get_column_summary(df, col)
            column_summaries.append(col_summary)
            schema_description += col_summary["summary"] + "\n"
        
        schema_description += "\n"
        
        # Table info for UI
        table_info += f"Table Name: {table_name}\n"
        table_info += f"Columns: {', '.join(df.columns.tolist())}\n"
        table_info += f"Sample Data: {df.head(2).to_dict('records')}\n\n"
    
    conn.close()
    return db_path, table_info, schema_description

def create_database_from_excel(excel_file):
    """Create SQLite database from uploaded Excel file"""
    db_path = tempfile.mktemp(suffix='.db')
    conn = sqlite3.connect(db_path)
    
    excel_data = pd.read_excel(excel_file, sheet_name=None)
    
    table_info = ""
    schema_description = ""
    
    for sheet_name, df in excel_data.items():
        table_name = re.sub(r'[^a-zA-Z0-9_]', '_', sheet_name.lower())
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Build detailed schema description
        schema_description += f"### Table: {table_name} (from sheet: '{sheet_name}')\n"
        schema_description += f"- Rows: {len(df)}, Columns: {len(df.columns)}\n"
        
        column_summaries = []
        for col in df.columns:
            col_summary = get_column_summary(df, col)
            column_summaries.append(col_summary)
            schema_description += col_summary["summary"] + "\n"
        
        schema_description += "\n"
        
        # Table info for UI
        table_info += f"Table Name: {table_name} (from sheet: {sheet_name})\n"
        table_info += f"Columns: {', '.join(df.columns.tolist())}\n"
        table_info += f"Sample Data: {df.head(2).to_dict('records')}\n\n"
    
    conn.close()
    return db_path, table_info, schema_description

def execute_sql_file(sql_file, db_path):
    """Execute SQL file against the database"""
    conn = sqlite3.connect(db_path)
    
    sql_content = sql_file.read().decode('utf-8')
    
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

def generate_fallback_query(question, available_tables):
    question_lower = question.lower()
    
    if not available_tables:
        return "SELECT 'No tables available' as message;"
    
    first_table = available_tables[0]
    
    if any(word in question_lower for word in ['count', 'how many', 'number']):
        return f"SELECT COUNT(*) as count FROM {first_table};"
    elif any(word in question_lower for word in ['show', 'display', 'see', 'what']):
        return f"SELECT * FROM {first_table} LIMIT 10;"
    elif 'columns' in question_lower or 'fields' in question_lower:
        return f"PRAGMA table_info({first_table});"
    elif 'tables' in question_lower:
        return "SELECT name FROM sqlite_master WHERE type='table';"
    elif any(word in question_lower for word in ['sum', 'total']):
        return f"SELECT * FROM {first_table} LIMIT 5;"
    else:
        return f"SELECT * FROM {first_table} LIMIT 5;"

def process_question(question: str, db, schema_description):
    try:
        st.write("🔍 Step 1: Analyzing database schema...")
        
        available_tables = db.get_usable_table_names()
        st.write(f"📋 Found tables: {', '.join(available_tables)}")
        
        st.write("🤖 Step 2: Generating SQL with enhanced understanding...")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.0,
            max_output_tokens=1000,
            google_api_key=st.secrets["GOOGLE_API_KEY"],
            client_options={"api_endpoint": "generativelanguage.googleapis.com"}
        )
        
        # Enhanced prompt with schema analysis instructions
        prompt = f"""
You are an expert data analyst and SQL developer. Your task is to analyze the database schema and generate an accurate SQLite query based on the user's question.

### Database Schema Description:
{schema_description}

### User Question:
{question}

### Instructions:
1. Carefully analyze the schema to identify the most relevant table(s) and column(s)
2. For date-related questions:
   - Look for columns with names like 'date', 'time', 'day', 'created_at', 'timestamp'
   - Check the sample values to confirm date format
3. For amount/numeric questions:
   - Look for columns with names like 'amount', 'value', 'price', 'total', 'cost', 'revenue'
   - Check the sample values to confirm numeric format
4. When finding maximum/minimum values:
   - First identify the correct numeric column
   - Then find the corresponding row with that value
   - Include all relevant columns in the result
5. If multiple tables are needed, determine proper JOIN conditions using foreign keys
6. Only use columns that actually exist in the schema
7. If you're unsure about a column's meaning:
   - Examine the sample values for clues
   - Consider the column name in context
   - Look at the table name for context
8. For date comparisons, use SQLite date functions like DATE() if needed
9. Always return the full row when answering "which" or "what" questions
10. Prefer explicit column names over SELECT *

### Important:
- Only output the SQL query
- Do not include any explanations or markdown formatting
- Use table aliases for clarity when joining tables
- Use LIMIT only when explicitly requested
- Handle null values appropriately with COALESCE if needed
- Format dates using SQLite's date formatting functions

### SQL Query:
"""
        
        try:
            st.write("📡 Making LLM request with enhanced schema understanding...")
            llm_response = llm.invoke(prompt)
            generated_query = llm_response.content.strip()
            
            # Clean up query
            if "```" in generated_query:
                generated_query = generated_query.split("```")[1].strip()
                if generated_query.startswith("sql"):
                    generated_query = generated_query[3:].strip()
            
            # Remove any non-SQL text after the query
            if ';' in generated_query:
                generated_query = generated_query.split(';')[0] + ';'
            else:
                generated_query += ';'
                
            # Remove any trailing text after the semicolon
            generated_query = generated_query.split(';')[0] + ';'
            
            st.write("✅ LLM generated query successfully")
            st.write(f"📝 Query to execute: {generated_query}")
            
        except Exception as e:
            st.warning(f"⚠️ LLM failed with error: {str(e)[:100]}")
            st.write("🛠️ Using rule-based fallback query...")
            generated_query = generate_fallback_query(question, available_tables)
            st.write(f"🔧 Generated fallback query: {generated_query}")
        
        st.write("🔧 Step 3: Executing query...")
        
        try:
            execute_query_tool = QuerySQLDatabaseTool(db=db)
            query_results = execute_query_tool.invoke({"query": generated_query})
            st.write("✅ Query executed successfully")
        
        except Exception as e:
            st.warning(f"⚠️ LangChain execution failed, trying direct SQLite...")
            
            try:
                db_path = db.database_uri.replace('sqlite:///', '')
                conn = sqlite3.connect(db_path, timeout=10.0)
                cursor = conn.execute(generated_query)
                query_results = cursor.fetchall()
                conn.close()
                st.write("✅ Direct SQLite execution successful")
            
            except Exception as e2:
                st.error(f"❌ Both execution methods failed: {str(e2)[:200]}")
                query_results = f"Execution Error: {e2}"
        
        st.write("💭 Step 4: Formatting results...")
        
        if isinstance(query_results, str) and "Error" in query_results:
            final_answer = f"❌ Query execution failed: {query_results}"
        elif not query_results:
            final_answer = "No results found for your query."
        else:
            if isinstance(query_results, list) and len(query_results) > 0:
                # Convert to DataFrame for better display if possible
                try:
                    if isinstance(query_results[0], tuple) or isinstance(query_results[0], list):
                        # Try to get column names
                        try:
                            col_names = [desc[0] for desc in cursor.description] 
                        except:
                            try:
                                col_names = [f"Column_{i+1}" for i in range(len(query_results[0]))]
                            except:
                                col_names = ["Result"]
                        
                        df = pd.DataFrame(query_results, columns=col_names)
                        final_answer = f"📊 **Results for '{question}':**\n\n"
                        final_answer += df.to_markdown(index=False)
                    else:
                        final_answer = f"📊 **Results for '{question}':**\n\n"
                        for i, row in enumerate(query_results[:10], 1):
                            final_answer += f"{i}. {row}\n"
                        if len(query_results) > 10:
                            final_answer += f"\n... and {len(query_results)-10} more rows"
                except:
                    final_answer = f"📊 **Results for '{question}':**\n\n"
                    for i, row in enumerate(query_results[:10], 1):
                        final_answer += f"{i}. {row}\n"
                    if len(query_results) > 10:
                        final_answer += f"\n... and {len(query_results)-10} more rows"
            else:
                final_answer = f"Results: {query_results}"
        
        st.write("✅ Process complete!")
        
        return final_answer, generated_query, query_results
        
    except Exception as e:
        error_msg = f"❌ Error in process_question: {str(e)}"
        st.error(error_msg)
        st.write("🐛 Debug Information:", str(e))
        traceback.print_exc()
        return error_msg, "", ""

# --- Streamlit UI ---

st.title("📊 Advanced Natural Language to SQL Analyzer")
st.write("Upload your data files and ask questions - I'll intelligently analyze your schema!")

# Sidebar for file uploads
with st.sidebar:
    st.header("📁 Upload Your Data")
    
    upload_option = st.radio(
        "Choose your data source:",
        ["CSV Files", "Excel File", "Excel + SQL File"]
    )
    
    db_path = None
    table_info = ""
    schema_description = ""
    
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
                    db_path, table_info, schema_description = create_database_from_csv_files(csv_files)
                    st.success(f"✅ Created database with {len(csv_files)} tables!")
                    
                    with st.expander("View Table Information"):
                        st.text(table_info)
                        
                    with st.expander("View Schema Analysis"):
                        st.text(schema_description)
                        
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
                    db_path, table_info, schema_description = create_database_from_excel(excel_file)
                    st.success("✅ Created database from Excel file!")
                    
                    with st.expander("View Table Information"):
                        st.text(table_info)
                        
                    with st.expander("View Schema Analysis"):
                        st.text(schema_description)
                        
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
                    db_path, table_info, schema_description = create_database_from_excel(excel_file)
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
                        
                    with st.expander("View Schema Analysis"):
                        st.text(schema_description)
                        
                except Exception as e:
                    st.error(f"Error creating database: {e}")

# Main chat interface
st.write("---")

# Initialize database connection in session state
if db_path and db_path not in st.session_state.get('processed_files', set()):
    try:
        st.session_state.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        st.session_state.table_info = table_info
        st.session_state.schema_description = schema_description
        
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = set()
        st.session_state.processed_files.add(db_path)
        
        st.success("🎉 Database ready! You can now ask questions.")
        
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
        st.session_state.history.add_user_message(user_question)
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            try:
                response, generated_query, query_results = process_question(
                    user_question, 
                    st.session_state.db,
                    st.session_state.schema_description
                )
                
                st.markdown(response)
                st.session_state.history.add_ai_message(response)
                
                with st.expander("🔍 Show Details"):
                    st.subheader("Generated SQL Query")
                    st.code(generated_query, language="sql")
                    
                    st.subheader("Raw Query Results")
                    st.text(str(query_results)[:2000] + ("..." if len(str(query_results)) > 2000 else ""))

            except Exception as e:
                error_msg = f"❌ Processing failed: {str(e)}"
                st.error(error_msg)
                st.session_state.history.add_ai_message(error_msg)
                traceback.print_exc()
                
else:
    st.info("👆 Please upload your data files using the sidebar to get started!")

# Reset conversation button
if st.button("🔄 Reset Conversation"):
    st.session_state.history = ChatMessageHistory()
    if 'db' in st.session_state:
        del st.session_state.db
    if 'table_info' in st.session_state:
        del st.session_state.table_info
    if 'schema_description' in st.session_state:
        del st.session_state.schema_description
    if 'processed_files' in st.session_state:
        del st.session_state.processed_files
    st.rerun()
