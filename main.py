import streamlit as st
import pandas as pd
import os
import sqlite3
import tempfile
import re
import traceback
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import hashlib
import json
from typing import Dict, List, Tuple, Optional
import time
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

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

# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state.history = ChatMessageHistory()
if 'query_cache' not in st.session_state:
    st.session_state.query_cache = {}
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'bookmarked_queries' not in st.session_state:
    st.session_state.bookmarked_queries = []
if 'conversation_context' not in st.session_state:
    st.session_state.conversation_context = []
if 'dashboard_items' not in st.session_state:
    st.session_state.dashboard_items = []
if 'data_quality_report' not in st.session_state:
    st.session_state.data_quality_report = {}

# --- Helper Functions ---

def get_column_summary(df, col):
    """Enhanced column summary with data quality metrics"""
    non_null = df[col].dropna()
    n_non_null = len(non_null)
    
    sample_size = min(3, n_non_null)
    sample_values = non_null.sample(sample_size).tolist() if n_non_null > 0 else []
    
    # Enhanced type detection - Fixed the ambiguous Series issue
    col_type = str(df[col].dtype)
    if col_type.startswith('datetime'):
        col_type = "datetime"
    elif np.issubdtype(df[col].dtype, np.number):
        col_type = "numeric"
    elif col_type == 'object':
        # Fix: Check if non_null is not empty before applying the check
        if n_non_null > 0:
            string_check = df[col].dropna().apply(lambda x: isinstance(x, str))
            if len(string_check) > 0 and string_check.all():
                col_type = "text"
            else:
                col_type = "mixed"
        else:
            col_type = "mixed"
    
    # Data quality metrics
    null_percentage = (len(df) - n_non_null) / len(df) * 100 if len(df) > 0 else 0
    duplicate_count = df[col].duplicated().sum()
    unique_count = df[col].nunique()
    
    # Value range for numeric columns
    value_range = ""
    if col_type == "numeric" and n_non_null > 0:
        value_range = f" | Range: [{non_null.min():.2f}, {non_null.max():.2f}]"
        # Detect potential outliers
        q1, q3 = non_null.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr > 0:  # Avoid division by zero
            outliers = len(non_null[(non_null < q1 - 1.5*iqr) | (non_null > q3 + 1.5*iqr)])
            if outliers > 0:
                value_range += f" | Outliers: {outliers}"
    
    # Date format detection
    date_format = ""
    if col_type == "datetime" and n_non_null > 0:
        try:
            date_sample = pd.to_datetime(non_null)
            if len(date_sample) > 0:
                date_format = f" | Format: {date_sample.dt.strftime('%Y-%m-%d').iloc[0][:10]}"
        except:
            pass
    
    # Categorical analysis
    categorical_info = ""
    if col_type == "text" and unique_count < 50:
        categorical_info = f" | Categories: {unique_count}"
        if unique_count <= 10 and n_non_null > 0:
            top_values = df[col].value_counts().head(3).to_dict()
            categorical_info += f" | Top: {top_values}"
    
    return {
        "name": col,
        "type": col_type,
        "sample": sample_values,
        "non_null_count": n_non_null,
        "total_count": len(df),
        "null_percentage": null_percentage,
        "unique_count": unique_count,
        "duplicate_count": duplicate_count,
        "summary": f"- {col}: {col_type}{value_range}{date_format}{categorical_info} | Nulls: {null_percentage:.1f}% | Sample: {sample_values}"
    }

def detect_relationships(dfs: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    """Detect potential relationships between tables"""
    relationships = {}
    table_names = list(dfs.keys())
    
    for i, table1 in enumerate(table_names):
        relationships[table1] = []
        for j, table2 in enumerate(table_names):
            if i != j:
                # Look for common column names (potential foreign keys)
                common_cols = set(dfs[table1].columns) & set(dfs[table2].columns)
                for col in common_cols:
                    if col.lower().endswith('id') or 'id' in col.lower():
                        relationships[table1].append(f"{table2}.{col}")
                
                # Look for similar column names
                for col1 in dfs[table1].columns:
                    for col2 in dfs[table2].columns:
                        if col1 != col2 and (col1.lower() in col2.lower() or col2.lower() in col1.lower()):
                            if 'id' in col1.lower() or 'id' in col2.lower():
                                relationships[table1].append(f"{table2}.{col2}")
    
    return relationships

def validate_sql_query(query: str, available_tables: List[str]) -> Tuple[bool, str]:
    """Validate SQL query before execution"""
    # Clean and normalize the query first
    query_clean = query.strip()
    if not query_clean:
        return False, "Empty query provided"
    
    # Remove common prefixes that might interfere
    if query_clean.startswith("```sql"):
        query_clean = query_clean[6:].strip()
    elif query_clean.startswith("```"):
        query_clean = query_clean[3:].strip()
    
    # Remove trailing ```
    if query_clean.endswith("```"):
        query_clean = query_clean[:-3].strip()
    
    query_upper = query_clean.upper()
    
    # Basic syntax checks - be more flexible with whitespace and formatting
    query_start = query_upper.lstrip()  # Remove leading whitespace
    if not query_start.startswith(('SELECT', 'WITH')):
        return False, f"Only SELECT queries are allowed for security reasons. Query starts with: '{query_clean[:20]}...'"
    
    # Check for dangerous keywords
    dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            return False, f"Query contains potentially dangerous keyword: {keyword}"
    
    # Check if query is too complex (basic heuristic)
    if query_clean.count('SELECT') > 5:
        return False, "Query appears to be too complex. Please simplify."
    
    # Check for table existence - make this more lenient
    if available_tables:
        table_referenced = False
        for table in available_tables:
            if table.lower() in query_clean.lower():
                table_referenced = True
                break
        
        if not table_referenced:
            return False, f"Query should reference at least one of these tables: {', '.join(available_tables)}"
    
    return True, "Query validation passed"

def optimize_query_suggestions(query: str, table_info: Dict) -> List[str]:
    """Suggest query optimizations"""
    suggestions = []
    query_upper = query.upper()
    
    # Suggest LIMIT if not present and no aggregation
    if 'LIMIT' not in query_upper and not any(agg in query_upper for agg in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'GROUP BY']):
        suggestions.append("💡 Consider adding LIMIT to avoid large result sets")
    
    # Suggest WHERE clause for large tables
    if 'WHERE' not in query_upper:
        suggestions.append("💡 Consider adding WHERE clause to filter results")
    
    # Check for SELECT *
    if 'SELECT *' in query_upper:
        suggestions.append("💡 Consider selecting specific columns instead of *")
    
    return suggestions

def decompose_complex_query(question: str) -> List[str]:
    """Break down complex questions into simpler parts"""
    question_lower = question.lower()
    steps = []
    
    # Check for multiple conditions
    if ' and ' in question_lower or ' also ' in question_lower:
        parts = re.split(r'\s+and\s+|\s+also\s+', question_lower)
        if len(parts) > 1:
            for i, part in enumerate(parts, 1):
                steps.append(f"Step {i}: {part.strip()}")
    
    # Check for comparisons
    elif any(word in question_lower for word in ['compare', 'versus', 'vs', 'difference']):
        steps.append("Step 1: Get data for first item")
        steps.append("Step 2: Get data for second item") 
        steps.append("Step 3: Calculate comparison")
    
    # Check for ranking questions
    elif any(word in question_lower for word in ['top', 'bottom', 'highest', 'lowest', 'best', 'worst']):
        steps.append("Step 1: Get all relevant data")
        steps.append("Step 2: Sort by the relevant metric")
        steps.append("Step 3: Return top/bottom results")
    
    return steps if len(steps) > 1 else []

def generate_visualization(df: pd.DataFrame, query: str, question: str) -> Optional[Dict]:
    """Generate appropriate visualizations based on data"""
    if df.empty or len(df.columns) == 0:
        return None
    
    viz_config = {"type": None, "config": {}}
    
    # Detect numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Try to parse date columns
    for col in categorical_cols[:]:
        try:
            pd.to_datetime(df[col])
            datetime_cols.append(col)
            categorical_cols.remove(col)
        except:
            continue
    
    question_lower = question.lower()
    
    # Time series plot
    if datetime_cols and numeric_cols:
        viz_config["type"] = "time_series"
        viz_config["config"] = {
            "x": datetime_cols[0],
            "y": numeric_cols[0],
            "title": f"{numeric_cols[0]} over {datetime_cols[0]}"
        }
    
    # Bar chart for categorical vs numeric
    elif categorical_cols and numeric_cols and len(df) <= 50:
        viz_config["type"] = "bar"
        viz_config["config"] = {
            "x": categorical_cols[0],
            "y": numeric_cols[0],
            "title": f"{numeric_cols[0]} by {categorical_cols[0]}"
        }
    
    # Histogram for single numeric column
    elif len(numeric_cols) == 1 and len(df) > 10:
        viz_config["type"] = "histogram"
        viz_config["config"] = {
            "x": numeric_cols[0],
            "title": f"Distribution of {numeric_cols[0]}"
        }
    
    # Scatter plot for two numeric columns
    elif len(numeric_cols) >= 2:
        viz_config["type"] = "scatter"
        viz_config["config"] = {
            "x": numeric_cols[0],
            "y": numeric_cols[1],
            "title": f"{numeric_cols[1]} vs {numeric_cols[0]}"
        }
    
    # Pie chart for small categorical data
    elif len(categorical_cols) == 1 and len(df) <= 20:
        value_counts = df[categorical_cols[0]].value_counts()
        viz_config["type"] = "pie"
        viz_config["config"] = {
            "values": value_counts.values,
            "names": value_counts.index,
            "title": f"Distribution of {categorical_cols[0]}"
        }
    
    return viz_config if viz_config["type"] else None

def create_visualization(df: pd.DataFrame, viz_config: Dict):
    """Create and display visualization"""
    try:
        if viz_config["type"] == "time_series":
            df_plot = df.copy()
            df_plot[viz_config["config"]["x"]] = pd.to_datetime(df_plot[viz_config["config"]["x"]])
            fig = px.line(df_plot, 
                         x=viz_config["config"]["x"], 
                         y=viz_config["config"]["y"],
                         title=viz_config["config"]["title"])
        
        elif viz_config["type"] == "bar":
            fig = px.bar(df, 
                        x=viz_config["config"]["x"], 
                        y=viz_config["config"]["y"],
                        title=viz_config["config"]["title"])
        
        elif viz_config["type"] == "histogram":
            fig = px.histogram(df, 
                              x=viz_config["config"]["x"],
                              title=viz_config["config"]["title"])
        
        elif viz_config["type"] == "scatter":
            fig = px.scatter(df, 
                            x=viz_config["config"]["x"], 
                            y=viz_config["config"]["y"],
                            title=viz_config["config"]["title"])
        
        elif viz_config["type"] == "pie":
            fig = go.Figure(data=[go.Pie(
                labels=viz_config["config"]["names"],
                values=viz_config["config"]["values"]
            )])
            fig.update_layout(title=viz_config["config"]["title"])
        
        else:
            return None
        
        fig.update_layout(height=400)
        return fig
    
    except Exception as e:
        st.warning(f"Could not create visualization: {str(e)}")
        return None

def perform_statistical_analysis(df: pd.DataFrame) -> Dict:
    """Perform basic statistical analysis"""
    analysis = {}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 1:
        # Descriptive statistics
        analysis["descriptive"] = df[numeric_cols].describe().to_dict()
        
        # Correlation analysis for multiple numeric columns
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            analysis["correlations"] = corr_matrix.to_dict()
            
            # Find strong correlations
            strong_corr = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_corr.append({
                            "col1": numeric_cols[i],
                            "col2": numeric_cols[j], 
                            "correlation": corr_val
                        })
            analysis["strong_correlations"] = strong_corr
        
        # Outlier detection
        outliers = {}
        for col in numeric_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            outlier_count = len(df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)])
            if outlier_count > 0:
                outliers[col] = outlier_count
        analysis["outliers"] = outliers
    
    return analysis

def detect_anomalies(df: pd.DataFrame) -> List[str]:
    """Simple anomaly detection"""
    anomalies = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 10:
            # Z-score based anomaly detection
            z_scores = np.abs(stats.zscore(col_data))
            anomaly_count = len(z_scores[z_scores > 3])
            if anomaly_count > 0:
                anomalies.append(f"Found {anomaly_count} potential anomalies in {col} (z-score > 3)")
    
    return anomalies

def simple_clustering(df: pd.DataFrame, n_clusters: int = 3) -> Optional[pd.DataFrame]:
    """Perform simple clustering analysis"""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2 or len(df) < 10:
            return None
        
        # Prepare data
        X = df[numeric_cols].fillna(df[numeric_cols].mean())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to dataframe
        result_df = df.copy()
        result_df['Cluster'] = clusters
        
        return result_df
        
    except Exception as e:
        return None

def create_database_from_csv_files(csv_files):
    """Enhanced database creation with relationship detection"""
    db_path = tempfile.mktemp(suffix='.db')
    conn = sqlite3.connect(db_path)
    
    table_info = ""
    schema_description = ""
    dfs = {}
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        table_name = re.sub(r'[^a-zA-Z0-9_]', '_', csv_file.name.split('.')[0].lower())
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        dfs[table_name] = df
        
        # Enhanced schema description with data quality
        schema_description += f"### Table: {table_name}\n"
        schema_description += f"- Rows: {len(df)}, Columns: {len(df.columns)}\n"
        
        for col in df.columns:
            col_summary = get_column_summary(df, col)
            schema_description += col_summary["summary"] + "\n"
        
        schema_description += "\n"
        table_info += f"Table Name: {table_name}\n"
        table_info += f"Columns: {', '.join(df.columns.tolist())}\n"
        table_info += f"Sample Data: {df.head(2).to_dict('records')}\n\n"
    
    # Detect relationships
    relationships = detect_relationships(dfs)
    if any(relationships.values()):
        schema_description += "### Detected Relationships:\n"
        for table, relations in relationships.items():
            if relations:
                schema_description += f"- {table}: {', '.join(relations)}\n"
        schema_description += "\n"
    
    # Store data quality report
    st.session_state.data_quality_report = {
        table: {col: get_column_summary(df, col) for col in df.columns}
        for table, df in dfs.items()
    }
    
    conn.close()
    return db_path, table_info, schema_description

def create_database_from_excel(excel_file):
    """Enhanced Excel processing"""
    db_path = tempfile.mktemp(suffix='.db')
    conn = sqlite3.connect(db_path)
    
    excel_data = pd.read_excel(excel_file, sheet_name=None)
    
    table_info = ""
    schema_description = ""
    dfs = {}
    
    for sheet_name, df in excel_data.items():
        table_name = re.sub(r'[^a-zA-Z0-9_]', '_', sheet_name.lower())
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        dfs[table_name] = df
        
        schema_description += f"### Table: {table_name} (from sheet: '{sheet_name}')\n"
        schema_description += f"- Rows: {len(df)}, Columns: {len(df.columns)}\n"
        
        for col in df.columns:
            col_summary = get_column_summary(df, col)
            schema_description += col_summary["summary"] + "\n"
        
        schema_description += "\n"
        table_info += f"Table Name: {table_name} (from sheet: {sheet_name})\n"
        table_info += f"Columns: {', '.join(df.columns.tolist())}\n"
        table_info += f"Sample Data: {df.head(2).to_dict('records')}\n\n"
    
    # Detect relationships
    relationships = detect_relationships(dfs)
    if any(relationships.values()):
        schema_description += "### Detected Relationships:\n"
        for table, relations in relationships.items():
            if relations:
                schema_description += f"- {table}: {', '.join(relations)}\n"
        schema_description += "\n"
    
    # Store data quality report
    st.session_state.data_quality_report = {
        table: {col: get_column_summary(df, col) for col in df.columns}
        for table, df in dfs.items()
    }
    
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

def get_query_hash(query: str) -> str:
    """Generate hash for query caching"""
    return hashlib.md5(query.encode()).hexdigest()

def cache_query_result(query: str, result: any):
    """Cache query results"""
    query_hash = get_query_hash(query)
    st.session_state.query_cache[query_hash] = {
        'result': result,
        'timestamp': datetime.now(),
        'query': query
    }

def get_cached_result(query: str) -> Optional[any]:
    """Retrieve cached query result"""
    query_hash = get_query_hash(query)
    if query_hash in st.session_state.query_cache:
        cached = st.session_state.query_cache[query_hash]
        # Cache expires after 1 hour
        if datetime.now() - cached['timestamp'] < timedelta(hours=1):
            return cached['result']
        else:
            del st.session_state.query_cache[query_hash]
    return None

def add_to_query_history(question: str, query: str, result: any):
    """Add query to history"""
    history_item = {
        'timestamp': datetime.now(),
        'question': question,
        'query': query,
        'result_preview': str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
    }
    st.session_state.query_history.insert(0, history_item)
    
    # Keep only last 50 queries
    if len(st.session_state.query_history) > 50:
        st.session_state.query_history = st.session_state.query_history[:50]

def generate_enhanced_query(question: str, db, schema_description, context: List[str] = None):
    """Enhanced query generation with context and multi-step processing - Fixed query cleaning"""
    try:
        available_tables = db.get_usable_table_names()
        
        # Check for cached result
        cached_result = get_cached_result(question)
        if cached_result:
            st.info("📋 Using cached result")
            return cached_result
        
        # Decompose complex queries
        query_steps = decompose_complex_query(question)
        if query_steps:
            st.info("🔄 Breaking down complex question into steps:")
            for step in query_steps:
                st.write(f"- {step}")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.0,
            max_output_tokens=1000,
            google_api_key=st.secrets["GOOGLE_API_KEY"],
            client_options={"api_endpoint": "generativelanguage.googleapis.com"}
        )
        
        # Build context from conversation history
        context_str = ""
        if context and st.session_state.conversation_context:
            recent_context = st.session_state.conversation_context[-3:]  # Last 3 interactions
            context_str = "\n### Previous Conversation Context:\n"
            for ctx in recent_context:
                context_str += f"Q: {ctx['question']}\nA: {ctx['result'][:200]}...\n\n"
        
        # Enhanced prompt with context and relationship awareness
        prompt = f"""
You are an expert data analyst and SQL developer. Generate accurate SQLite queries based on the database schema and user questions.

### Database Schema Description:
{schema_description}

{context_str}

### User Question:
{question}

### Enhanced Instructions:
1. **Schema Analysis**: Carefully analyze table relationships and foreign keys
2. **Context Awareness**: Consider previous questions and results when relevant
3. **Data Quality**: Handle potential null values and data inconsistencies
4. **Query Optimization**: Write efficient queries with appropriate WHERE clauses
5. **Multi-step Logic**: Break complex questions into logical steps if needed
6. **Relationship Joins**: Use detected relationships for accurate joins
7. **Semantic Understanding**: Match user intent with appropriate aggregations

### Query Requirements:
- Only output valid SQLite syntax
- Use proper table aliases for joins
- Include helpful column aliases
- Handle edge cases (nulls, empty results)
- Optimize for performance when possible
- Return meaningful results that directly answer the question
- Start with SELECT (no markdown formatting)

### Important: Return ONLY the SQL query, no explanations or markdown formatting.

SQL Query:
"""
        
        llm_response = llm.invoke(prompt)
        generated_query = llm_response.content.strip()
        
        # Enhanced query cleaning
        if "```" in generated_query:
            # Extract query from markdown code blocks
            parts = generated_query.split("```")
            for part in parts:
                if part.strip().startswith(("SELECT", "select", "WITH", "with")) or "SELECT" in part.upper():
                    generated_query = part.strip()
                    break
        
        # Remove common prefixes
        if generated_query.startswith("sql"):
            generated_query = generated_query[3:].strip()
        elif generated_query.startswith("SQL"):
            generated_query = generated_query[3:].strip()
        
        # Ensure query ends with semicolon
        if not generated_query.endswith(';'):
            generated_query += ';'
        
        # Clean up multiple semicolons
        if ';;' in generated_query:
            generated_query = generated_query.replace(';;', ';')
        
        # If query still has multiple statements, take only the first one
        if generated_query.count(';') > 1:
            generated_query = generated_query.split(';')[0] + ';'
        
        return generated_query
        
    except Exception as e:
        st.warning(f"⚠️ Enhanced query generation failed: {str(e)}")
        # Fallback to simple generation
        return generate_fallback_query(question, available_tables)

def generate_fallback_query(question, available_tables):
    """Enhanced fallback query generation"""
    question_lower = question.lower()
    
    if not available_tables:
        return "SELECT 'No tables available' as message;"
    
    first_table = available_tables[0]
    
    # More sophisticated pattern matching
    if any(word in question_lower for word in ['count', 'how many', 'number of']):
        return f"SELECT COUNT(*) as total_count FROM {first_table};"
    elif any(word in question_lower for word in ['average', 'avg', 'mean']):
        return f"SELECT * FROM {first_table} LIMIT 5;"  # Will be enhanced based on numeric columns
    elif any(word in question_lower for word in ['maximum', 'max', 'highest', 'largest']):
        return f"SELECT * FROM {first_table} ORDER BY rowid DESC LIMIT 1;"
    elif any(word in question_lower for word in ['minimum', 'min', 'lowest', 'smallest']):
        return f"SELECT * FROM {first_table} ORDER BY rowid ASC LIMIT 1;"
    elif any(word in question_lower for word in ['sum', 'total']):
        return f"SELECT * FROM {first_table} LIMIT 10;"
    elif any(word in question_lower for word in ['group', 'category', 'type']):
        return f"SELECT * FROM {first_table} GROUP BY rowid LIMIT 10;"
    elif 'columns' in question_lower or 'fields' in question_lower:
        return f"PRAGMA table_info({first_table});"
    elif 'tables' in question_lower:
        return "SELECT name FROM sqlite_master WHERE type='table';"
    else:
        return f"SELECT * FROM {first_table} LIMIT 10;"

def process_question_enhanced(question: str, db, schema_description):
    """Enhanced question processing with all new features - Fixed return values"""
    try:
        st.write("🔍 Step 1: Analyzing database schema...")
        
        available_tables = db.get_usable_table_names()
        st.write(f"📋 Found tables: {', '.join(available_tables)}")
        
        st.write("🤖 Step 2: Generating enhanced SQL query...")
        
        # Generate query with context
        context = st.session_state.conversation_context[-3:] if st.session_state.conversation_context else []
        generated_query = generate_enhanced_query(question, db, schema_description, context)
        
        st.write(f"📝 Generated query: {generated_query}")
        
        # Validate query
        is_valid, validation_msg = validate_sql_query(generated_query, available_tables)
        if not is_valid:
            st.error(f"❌ Query validation failed: {validation_msg}")
            # Return 4 values consistently - including empty analysis_results
            return f"Query validation failed: {validation_msg}", generated_query, pd.DataFrame(), {}
        
        # Get optimization suggestions
        suggestions = optimize_query_suggestions(generated_query, st.session_state.get('table_info', {}))
        if suggestions:
            with st.expander("💡 Query Optimization Suggestions"):
                for suggestion in suggestions:
                    st.write(suggestion)
        
        st.write("🔧 Step 3: Executing query...")
        
        # Initialize results_df to avoid UnboundLocalError
        results_df = pd.DataFrame()
        
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
                
                if generated_query.upper().strip().startswith('SELECT'):
                    columns = [description[0] for description in cursor.description]
                    query_results = cursor.fetchall()
                    
                    # Convert to DataFrame for better processing
                    if query_results:
                        results_df = pd.DataFrame(query_results, columns=columns)
                    else:
                        results_df = pd.DataFrame()
                else:
                    query_results = "Query executed successfully"
                    results_df = pd.DataFrame()
                
                conn.close()
                st.write("✅ Direct SQLite execution successful")
            
            except Exception as e2:
                st.error(f"❌ Both execution methods failed: {str(e2)[:200]}")
                # Return 4 values consistently
                return f"Execution Error: {e2}", generated_query, pd.DataFrame(), {}
        
        # Process results
        if isinstance(query_results, str):
            if "Error" in query_results:
                final_answer = f"❌ Query execution failed: {query_results}"
                results_df = pd.DataFrame()
            else:
                final_answer = query_results
                results_df = pd.DataFrame()
        else:
            try:
                # Convert results to DataFrame
                if isinstance(query_results, list) and query_results:
                    if isinstance(query_results[0], (tuple, list)):
                        # Try to get column names from the last cursor
                        try:
                            columns = [description[0] for description in cursor.description]
                        except:
                            columns = [f"Column_{i+1}" for i in range(len(query_results[0]))]
                        results_df = pd.DataFrame(query_results, columns=columns)
                    else:
                        results_df = pd.DataFrame(query_results)
                else:
                    results_df = pd.DataFrame()
                
                if not results_df.empty:
                    final_answer = f"📊 **Results for '{question}':**\n\n"
                    final_answer += results_df.to_markdown(index=False)
                else:
                    final_answer = "No results found for your query."
                    
            except Exception as e:
                final_answer = f"📊 **Results for '{question}':**\n\n{str(query_results)[:1000]}"
                results_df = pd.DataFrame()
        
        st.write("📈 Step 4: Advanced Analysis...")
        
        # Perform advanced analysis if we have meaningful data
        analysis_results = {}
        if not results_df.empty and len(results_df) > 1:
            
            # Generate visualization
            viz_config = generate_visualization(results_df, generated_query, question)
            if viz_config:
                st.write("📊 Creating visualization...")
                fig = create_visualization(results_df, viz_config)
                if fig:
                    analysis_results["visualization"] = fig
            
            # Statistical analysis
            if len(results_df.select_dtypes(include=[np.number]).columns) > 0:
                st.write("📊 Performing statistical analysis...")
                stats_analysis = perform_statistical_analysis(results_df)
                analysis_results["statistics"] = stats_analysis
                
                # Anomaly detection
                anomalies = detect_anomalies(results_df)
                if anomalies:
                    analysis_results["anomalies"] = anomalies
                
                # Clustering analysis
                if len(results_df) >= 10:
                    clustered_df = simple_clustering(results_df)
                    if clustered_df is not None:
                        analysis_results["clustering"] = clustered_df
        
        # Cache the result
        cache_query_result(generated_query, {
            "answer": final_answer,
            "dataframe": results_df,
            "analysis": analysis_results
        })
        
        # Add to history
        add_to_query_history(question, generated_query, final_answer)
        
        # Update conversation context
        st.session_state.conversation_context.append({
            "question": question,
            "query": generated_query,
            "result": final_answer[:500]  # Truncated for context
        })
        
        # Keep only recent context
        if len(st.session_state.conversation_context) > 10:
            st.session_state.conversation_context = st.session_state.conversation_context[-10:]
        
        st.write("✅ Enhanced processing complete!")
        
        # Always return exactly 4 values
        return final_answer, generated_query, results_df, analysis_results
        
    except Exception as e:
        error_msg = f"❌ Error in enhanced processing: {str(e)}"
        st.error(error_msg)
        traceback.print_exc()
        # Always return exactly 4 values, even in error cases
        return error_msg, "", pd.DataFrame(), {}
def validate_sql_query(query: str, available_tables: List[str]) -> Tuple[bool, str]:
    """Validate SQL query before execution"""
    # Clean and normalize the query first
    query_clean = query.strip()
    if not query_clean:
        return False, "Empty query provided"
    
    # Remove common prefixes that might interfere
    if query_clean.startswith("```sql"):
        query_clean = query_clean[6:].strip()
    elif query_clean.startswith("```"):
        query_clean = query_clean[3:].strip()
    
    # Remove trailing ```
    if query_clean.endswith("```"):
        query_clean = query_clean[:-3].strip()
    
    query_upper = query_clean.upper()
    
    # Basic syntax checks - be more flexible with whitespace and formatting
    query_start = query_upper.lstrip()  # Remove leading whitespace
    if not query_start.startswith(('SELECT', 'WITH')):
        return False, f"Only SELECT queries are allowed for security reasons. Query starts with: '{query_clean[:20]}...'"
    
    # Check for dangerous keywords
    dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            return False, f"Query contains potentially dangerous keyword: {keyword}"
    
    # Check if query is too complex (basic heuristic)
    if query_clean.count('SELECT') > 5:
        return False, "Query appears to be too complex. Please simplify."
    
    # Check for table existence - make this more lenient
    if available_tables:
        table_referenced = False
        for table in available_tables:
            if table.lower() in query_clean.lower():
                table_referenced = True
                break
        
        if not table_referenced:
            return False, f"Query should reference at least one of these tables: {', '.join(available_tables)}"
    
    return True, "Query validation passed"

def suggest_follow_up_questions(question: str, results_df: pd.DataFrame, analysis: Dict) -> List[str]:
    """Generate intelligent follow-up question suggestions"""
    suggestions = []
    
    if results_df.empty:
        return suggestions
    
    question_lower = question.lower()
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = results_df.select_dtypes(include=['object']).columns.tolist()
    
    # Suggest drill-down questions
    if 'top' in question_lower or 'highest' in question_lower:
        suggestions.append("What are the bottom 10 items?")
        suggestions.append("Show me the average values")
    
    if 'count' in question_lower or 'how many' in question_lower:
        suggestions.append("What are the details of these items?")
        suggestions.append("Show me the distribution by category")
    
    # Suggest comparison questions
    if len(categorical_cols) > 0:
        suggestions.append(f"Compare {numeric_cols[0] if numeric_cols else 'values'} by {categorical_cols[0]}")
    
    # Suggest time-based questions if date columns exist
    date_like_cols = [col for col in results_df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if date_like_cols:
        suggestions.append("Show me trends over time")
        suggestions.append("What are the monthly/yearly patterns?")
    
    # Suggest statistical questions
    if numeric_cols:
        suggestions.append("What are the outliers in this data?")
        suggestions.append("Show me correlation analysis")
    
    # Based on analysis results
    if analysis.get('anomalies'):
        suggestions.append("Show me details of the anomalous values")
    
    if analysis.get('strong_correlations'):
        suggestions.append("Explain the strongest correlations")
    
    return suggestions[:5]  # Limit to 5 suggestions

def create_dashboard_item(question: str, query: str, results_df: pd.DataFrame, viz_config: Dict = None):
    """Add item to dashboard"""
    dashboard_item = {
        "id": len(st.session_state.dashboard_items),
        "timestamp": datetime.now(),
        "question": question,
        "query": query,
        "data": results_df.to_dict('records') if not results_df.empty else [],
        "viz_config": viz_config,
        "title": question[:50] + "..." if len(question) > 50 else question
    }
    
    st.session_state.dashboard_items.append(dashboard_item)

def display_dashboard():
    """Display dashboard with saved items"""
    if not st.session_state.dashboard_items:
        st.info("No dashboard items yet. Add queries to your dashboard using the 'Add to Dashboard' button.")
        return
    
    st.header("📊 Your Data Dashboard")
    
    # Dashboard controls
    col1, col2 = st.columns([3, 1])
    with col1:
        layout = st.selectbox("Layout", ["Single Column", "Two Columns"], key="dashboard_layout")
    with col2:
        if st.button("Clear Dashboard", type="secondary"):
            st.session_state.dashboard_items = []
            st.rerun()
    
    # Display items
    if layout == "Two Columns":
        cols = st.columns(2)
        for i, item in enumerate(st.session_state.dashboard_items):
            with cols[i % 2]:
                display_dashboard_item(item)
    else:
        for item in st.session_state.dashboard_items:
            display_dashboard_item(item)

def display_dashboard_item(item: Dict):
    """Display individual dashboard item"""
    with st.container():
        st.subheader(item["title"])
        st.caption(f"Added: {item['timestamp'].strftime('%Y-%m-%d %H:%M')}")
        
        if item["data"]:
            df = pd.DataFrame(item["data"])
            
            # Show visualization if available
            if item["viz_config"]:
                fig = create_visualization(df, item["viz_config"])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            with st.expander("View Data"):
                st.dataframe(df, use_container_width=True)
            
            # Controls
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Refresh {item['id']}", key=f"refresh_{item['id']}"):
                    st.info("Refresh functionality would re-run the query")
            with col2:
                if st.button(f"Remove {item['id']}", key=f"remove_{item['id']}"):
                    st.session_state.dashboard_items = [
                        x for x in st.session_state.dashboard_items if x["id"] != item["id"]
                    ]
                    st.rerun()
        
        st.write("---")

# --- Query Templates ---

QUERY_TEMPLATES = {
    "Data Overview": [
        "Show me the first 10 rows",
        "How many records are in each table?",
        "What columns are available?",
        "Show me a summary of the data"
    ],
    "Aggregations": [
        "What is the total/sum of [column]?",
        "What is the average [column]?",
        "Show me the count by [category]",
        "What are the maximum and minimum values?"
    ],
    "Top/Bottom Analysis": [
        "Show me the top 10 [items] by [metric]",
        "What are the highest values?",
        "Which [category] has the most [items]?",
        "Show me the bottom performers"
    ],
    "Time Analysis": [
        "Show me trends over time",
        "What are the monthly totals?",
        "Compare this year vs last year",
        "Show me seasonal patterns"
    ],
    "Comparisons": [
        "Compare [A] vs [B]",
        "Show me differences between groups",
        "Which category performs better?",
        "Rank items by performance"
    ]
}

# --- Main Streamlit UI ---

# Page configuration
st.set_page_config(
    page_title="Enhanced NL-to-SQL Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Enhanced Natural Language to SQL Analyzer")
st.write("Upload your data files, ask questions, and get intelligent insights with advanced analytics!")

# Main navigation
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Query Interface", "📊 Dashboard", "📋 History", "📈 Data Quality"])

with tab1:
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
                with st.spinner("Creating enhanced database from CSV files..."):
                    try:
                        db_path, table_info, schema_description = create_database_from_csv_files(csv_files)
                        st.success(f"✅ Created database with {len(csv_files)} tables!")
                        
                        with st.expander("View Table Information"):
                            st.text(table_info)
                            
                        with st.expander("View Enhanced Schema Analysis"):
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
                with st.spinner("Creating enhanced database from Excel file..."):
                    try:
                        db_path, table_info, schema_description = create_database_from_excel(excel_file)
                        st.success("✅ Created database from Excel file!")
                        
                        with st.expander("View Table Information"):
                            st.text(table_info)
                            
                        with st.expander("View Enhanced Schema Analysis"):
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
                with st.spinner("Creating enhanced database from Excel file..."):
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
                            
                        with st.expander("View Enhanced Schema Analysis"):
                            st.text(schema_description)
                            
                    except Exception as e:
                        st.error(f"Error creating database: {e}")
        
        # Query Templates Section
        st.header("📝 Query Templates")
        selected_category = st.selectbox("Choose a category:", list(QUERY_TEMPLATES.keys()))
        
        if selected_category:
            st.write("**Example questions:**")
            for template in QUERY_TEMPLATES[selected_category]:
                if st.button(template, key=f"template_{template}", use_container_width=True):
                    st.session_state.template_question = template
        
        # Bookmarked Queries
        if st.session_state.bookmarked_queries:
            st.header("🔖 Bookmarked Queries")
            for i, bookmark in enumerate(st.session_state.bookmarked_queries):
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(bookmark["question"][:40] + "...", key=f"bookmark_{i}", use_container_width=True):
                        st.session_state.template_question = bookmark["question"]
                with col2:
                    if st.button("🗑️", key=f"del_bookmark_{i}"):
                        st.session_state.bookmarked_queries.pop(i)
                        st.rerun()

    # Initialize database connection in session state
    if db_path and db_path not in st.session_state.get('processed_files', set()):
        try:
            st.session_state.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
            st.session_state.table_info = table_info
            st.session_state.schema_description = schema_description
            
            if 'processed_files' not in st.session_state:
                st.session_state.processed_files = set()
            st.session_state.processed_files.add(db_path)
            
            st.success("🎉 Enhanced database ready! You can now ask questions with advanced analytics.")
            
            try:
                available_tables = st.session_state.db.get_usable_table_names()
                st.info(f"📋 Available tables: {', '.join(available_tables)}")
            except:
                pass
                
        except Exception as e:
            st.error(f"❌ Error connecting to database: {e}")
            if 'db' in st.session_state:
                del st.session_state.db

    # Main chat interface
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
if 'db' in st.session_state:
    # Check if there's a template question to use
    default_question = st.session_state.get('template_question', '')
    
    # If there's a template question, show it and process it automatically
    if default_question:
        st.info(f"Using template question: {default_question}")
        user_question = default_question
        del st.session_state.template_question
    else:
        # Use chat input without the unsupported value parameter
        user_question = st.chat_input("Ask a question about your data...")
    
    if user_question:
        st.session_state.history.add_user_message(user_question)
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            try:
                response, generated_query, results_df, analysis_results = process_question_enhanced(
                    user_question, 
                    st.session_state.db,
                    st.session_state.schema_description
                )
                
                st.markdown(response)
                st.session_state.history.add_ai_message(response)
                
                # Display analysis results
                if analysis_results:
                    
                    # Visualization
                    if "visualization" in analysis_results:
                        st.plotly_chart(analysis_results["visualization"], use_container_width=True)
                    
                    # Statistical Analysis
                    if "statistics" in analysis_results:
                        with st.expander("📊 Statistical Analysis"):
                            stats = analysis_results["statistics"]
                            
                            if "descriptive" in stats:
                                st.write("**Descriptive Statistics:**")
                                desc_df = pd.DataFrame(stats["descriptive"])
                                st.dataframe(desc_df, use_container_width=True)
                            
                            if "strong_correlations" in stats and stats["strong_correlations"]:
                                st.write("**Strong Correlations Found:**")
                                for corr in stats["strong_correlations"]:
                                    st.write(f"- {corr['col1']} ↔ {corr['col2']}: {corr['correlation']:.3f}")
                            
                            if "outliers" in stats and stats["outliers"]:
                                st.write("**Outliers Detected:**")
                                for col, count in stats["outliers"].items():
                                    st.write(f"- {col}: {count} outliers")
                    
                    # Anomaly Detection
                    if "anomalies" in analysis_results:
                        with st.expander("🚨 Anomaly Detection"):
                            for anomaly in analysis_results["anomalies"]:
                                st.warning(anomaly)
                    
                    # Clustering Results
                    if "clustering" in analysis_results:
                        with st.expander("🎯 Cluster Analysis"):
                            clustered_df = analysis_results["clustering"]
                            st.write(f"Data grouped into {clustered_df['Cluster'].nunique()} clusters")
                            
                            cluster_summary = clustered_df.groupby('Cluster').size()
                            st.bar_chart(cluster_summary)
                
                # Action buttons
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button("🔖 Bookmark Query"):
                        bookmark = {
                            "question": user_question,
                            "query": generated_query,
                            "timestamp": datetime.now()
                        }
                        st.session_state.bookmarked_queries.append(bookmark)
                        st.success("Query bookmarked!")
                
                with col2:
                    if st.button("📊 Add to Dashboard") and not results_df.empty:
                        viz_config = generate_visualization(results_df, generated_query, user_question)
                        create_dashboard_item(user_question, generated_query, results_df, viz_config)
                        st.success("Added to dashboard!")
    
        # Generate viz_config properly for dashboard storage
        
                
                with col3:
                    if st.button("📥 Export Results") and not results_df.empty:
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                with col4:
                    if st.button("🔄 Suggest Follow-ups"):
                        suggestions = suggest_follow_up_questions(user_question, results_df, analysis_results)
                        if suggestions:
                            st.write("**Suggested follow-up questions:**")
                            for suggestion in suggestions:
                                if st.button(suggestion, key=f"followup_{suggestion}"):
                                    st.session_state.template_question = suggestion
                                    st.rerun()
                
                # Show technical details
                with st.expander("🔍 Technical Details"):
                    st.subheader("Generated SQL Query")
                    st.code(generated_query, language="sql")
                    
                    st.subheader("Raw Query Results")
                    if not results_df.empty:
                        st.dataframe(results_df, use_container_width=True)
                    else:
                        st.text("No data returned")
                    
                    if analysis_results:
                        st.subheader("Analysis Metadata")
                        st.json({k: str(v)[:200] + "..." if len(str(v)) > 200 else str(v) 
                                for k, v in analysis_results.items() if k != "visualization"})

            except Exception as e:
                error_msg = f"❌ Enhanced processing failed: {str(e)}"
                st.error(error_msg)
                st.session_state.history.add_ai_message(error_msg)
                traceback.print_exc()
                
else:
    st.info("👆 Please upload your data files using the sidebar to get started!")

with tab2:
    display_dashboard()

with tab3:
    st.header("📋 Query History")
    
    if not st.session_state.query_history:
        st.info("No query history yet. Start asking questions to build your history.")
    else:
        # History controls
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("Search history:", placeholder="Type to filter queries...")
        with col2:
            if st.button("Clear History", type="secondary"):
                st.session_state.query_history = []
                st.success("History cleared!")
                st.rerun()
        
        # Filter history
        filtered_history = st.session_state.query_history
        if search_term:
            filtered_history = [h for h in st.session_state.query_history 
                              if search_term.lower() in h['question'].lower()]
        
        # Display history
        for i, hist_item in enumerate(filtered_history):
            with st.expander(f"{hist_item['timestamp'].strftime('%Y-%m-%d %H:%M')} - {hist_item['question'][:60]}..."):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write("**Question:**", hist_item['question'])
                    st.code(hist_item['query'], language="sql")
                    st.write("**Result Preview:**", hist_item['result_preview'])
                
                with col2:
                    if st.button("Re-run Query", key=f"rerun_{i}"):
                        st.session_state.template_question = hist_item['question']
                        st.switch_page("Query Interface")
                    
                    if st.button("Bookmark", key=f"bookmark_hist_{i}"):
                        bookmark = {
                            "question": hist_item['question'],
                            "query": hist_item['query'],
                            "timestamp": datetime.now()
                        }
                        st.session_state.bookmarked_queries.append(bookmark)
                        st.success("Bookmarked!")

with tab4:
    st.header("📈 Data Quality Report")
    
    if not st.session_state.data_quality_report:
        st.info("Upload data files to see quality analysis.")
    else:
        for table_name, columns_info in st.session_state.data_quality_report.items():
            st.subheader(f"Table: {table_name}")
            
            # Create quality summary
            total_cols = len(columns_info)
            high_null_cols = sum(1 for col_info in columns_info.values() if col_info['null_percentage'] > 20)
            duplicate_cols = sum(1 for col_info in columns_info.values() if col_info['duplicate_count'] > 0)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Columns", total_cols)
            with col2:
                st.metric("High Null Columns", high_null_cols)
            with col3:
                st.metric("Columns with Duplicates", duplicate_cols)
            with col4:
                avg_null = np.mean([col_info['null_percentage'] for col_info in columns_info.values()])
                st.metric("Avg Null %", f"{avg_null:.1f}%")
            
            # Detailed column analysis
            with st.expander(f"Detailed Analysis - {table_name}"):
                quality_data = []
                for col_name, col_info in columns_info.items():
                    quality_data.append({
                        "Column": col_name,
                        "Type": col_info['type'],
                        "Null %": f"{col_info['null_percentage']:.1f}%",
                        "Unique Values": col_info['unique_count'],
                        "Duplicates": col_info['duplicate_count'],
                        "Data Quality": "⚠️ High Nulls" if col_info['null_percentage'] > 20 else 
                                       "⚠️ Many Duplicates" if col_info['duplicate_count'] > col_info['total_count'] * 0.5 else "✅ Good"
                    })
                
                quality_df = pd.DataFrame(quality_data)
                st.dataframe(quality_df, use_container_width=True)
            
            st.write("---")

# Footer controls
st.write("---")
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    if st.button("🔄 Reset Conversation"):
        st.session_state.history = ChatMessageHistory()
        st.session_state.conversation_context = []
        st.success("Conversation reset!")

with col2:
    if st.button("🗑️ Clear All Cache"):
        st.session_state.query_cache = {}
        st.success("Cache cleared!")

with col3:
    cache_size = len(st.session_state.query_cache)
    st.caption(f"Cache: {cache_size} items")
