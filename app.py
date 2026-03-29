import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
import re


from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


if "messages" not in st.session_state:
    st.session_state.messages = [] 
if "df" not in st.session_state:
    st.session_state.df = None
if "data_summary" not in st.session_state:
    st.session_state.data_summary = None

if "charts" not in st.session_state:
    st.session_state.charts = []

def reset_application():
    st.rerun()

def get_data_summary(df):

   summary_dict = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'sample': df.head(5).to_string(),
        'stats': df.describe().to_string()
    }
   return summary_dict

def prepare_export_text():

    export_data = "--- DATA ANALYSIS REPORT ---\n\n"

    for msg in st.session_state.messages:
        role = msg['role'].upper()
        content = msg['content']
        export_data += f"{role}:\n{content}\n\n"
        export_data += "-" * 30 + "\n\n"
    return export_data

def display_sidebar():

    with st.sidebar:
        st.header("Data Upload")

        uploaded_file = st.file_uploader("upload a CSV", type = ["csv"])

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.success(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")

                with st.expander('Preview Data'):
                    st.dataframe(df.head())

                with st.expander('Data Metrics'):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Total Rows", df.shape[0])
                        st.metric("Total Columns", df.shape[1])
                    with col2:
                        st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024:.1f} KB")
                        st.metric("Missing Values", df.isnull().sum().sum())
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                st.info("Please make sure your file is a valid CSV format.")
        else:
            st.info("👆 Upload a CSV file to start analyzing!")
        
        try:
            available_models = [
                m.name for m in genai.list_models() 
                if 'generateContent' in m.supported_generation_methods
            ]
            selected_model = st.selectbox("Choose AI Model", available_models)
        except:
            selected_model = st.selectbox("Choose AI Model", ["models/gemini-1.5-flash"])


        if st.button("Reset Chat"):
            st.session_state.messages = []
            st.rerun()

        if st.session_state.messages:
            st.divider()
            st.subheader("Final Report")
            
            report_text = prepare_export_text()
            
            st.download_button(
                label=" Download Chat History",
                data=report_text,
                file_name="analysis_report.txt",
                mime="text/plain"
            )

        return uploaded_file, selected_model

    

def get_gemini_response(user_input, data_context, model_name):


    system_instruction = f"""
    You are a helpful data analyst assistant. 
    
    The user has uploaded a CSV file with the following information:
    {data_context}
    
    The data is loaded in a pandas DataFrame called `df`.
    
    Guidelines:
    - Answer the user's question clearly and concisely.
    - If the question requires analysis, write Python code using pandas, matplotlib, or seaborn.
    - For visualizations, always use plt.figure() before plotting and include plt.tight_layout().
    - Always validate data before operations (check for nulls, data types, etc.).
    - If you can't answer due to data limitations, explain why.
    - Keep responses focused on the data and question asked.
    - Summarize your findings, insights, and any relevant statistics or visual trends.
    - Focus on delivering the results and what they mean, not on how to get them.
    
    When writing code:
    - Import statements are already done (import pandas as pd, import matplotlib.pyplot as plt, import seaborn as sns).
    - The dataframe is available as 'df'.
    - For plots, use plt.figure(figsize=(10, 6)) for better display.
    - Always add titles and labels to plots.
    """
        
    model = genai.GenerativeModel(model_name = model_name, system_instruction=system_instruction)

    history = []
    for msg in st.session_state.messages:
        role = "model" if msg["role"] == "assistant" else "user"
        history.append({"role": role, "parts": [msg["content"]]})

    chat = model.start_chat(history=history)

    generation_config = genai.types.GenerationConfig(temperature= 0.1, max_output_tokens=1500)

    response = model.generate_content(user_input,generation_config=generation_config)

    return response.text


       
st.set_page_config(
    page_title = 'Ask your CSV',
    page_icon = '📊',
    layout= 'wide'
)

st.title('📊 Ask your CSV')
st.markdown('Upload your data and ask Questions in plain English')

file, model = display_sidebar()

if st.session_state.df is not None:

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input('Ask Questions About Your Data')

    if user_input:
        st.session_state.messages.append({'role': 'user', 'content': user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("analyzing data..."):

                try:
                    raw_summary = get_data_summary(st.session_state.df)

                    df = st.session_state.df

                    if len(df) > 100:
                        data_context = f"""
                        Dataset shape: {raw_summary['shape']}
                        Columns: {', '.join(raw_summary['columns'])}
                        Data types: {raw_summary['dtypes']}
                        Sample rows: {raw_summary['sample']}
                        Basic statistics: {raw_summary['stats']}
                        """
                    else:
                        data_context = f"Full dataset:\n{df.to_string()}"

                    response = get_gemini_response(user_input, data_context, model)

                    st.markdown(response)

                    if "```python" in response:
                        code_blocks = response.split("```python")
                        for i in range(1, len(code_blocks)):
                            code = code_blocks[i].split("```")[0].strip()

                            local_vars = {"df": df, "plt": plt, "sns": sns, "pd": pd,"st": st}

                            try:

                                exec(code.strip(), globals(), local_vars)

                                if plt.get_fignums():
                                    fig = plt.gcf()
                                    st.pyplot(fig)
                                    st.session_state.charts.append(fig)
                                    plt.clf()

                            except Exception as e:
                                st.error(f"Execution Error: {e}")
                    

                    st.session_state.messages.append({'role': 'assistant', 'content': response})

                except Exception as e:
                    st.error(f"Something went wrong: {e}")


else:
    col1, col2, col3 = st.columns([1,2,1])

    with col2:

        st.info("Please upload a CSV file to Start")
        st.markdown("###  Example questions you can ask:")
        st.markdown("""
        - What are the main trends in my data?
        - Show me a correlation matrix
        - Create a bar chart of the top 10 categories
        - What's the average value by month?
        - Are there any outliers in the price column?
        """)







