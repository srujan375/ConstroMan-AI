# httpserver.py
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from AI_Tools.PandasAI import PandasAI
from AI_Tools.Gemini import GeminiLLMAPI
import time
from Analysis.DashboardGenerator import DashboardPlotter
from io import BytesIO
import base64

# Initialize session states
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'dashboard_generated' not in st.session_state:
    st.session_state.dashboard_generated = False

if 'selected_column' not in st.session_state:
    st.session_state.selected_column = None

if 'embeddings_df' not in st.session_state:
    st.session_state.embeddings_df = None

GeminiLLMAPI.configure_api()

# Sidebar for file upload
st.sidebar.title('ConstroMan-AI: Your Data Analysis Assistant')
st.sidebar.header('Upload your Excel file')
uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

# Main interface
st.title('ConstroMan-AI: Your Data Analysis Assistant')
st.subheader("Get a bird's eye view on your Project")

@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    embeddings_df = GeminiLLMAPI.generate_embeddings(df)
    return df, embeddings_df

if uploaded_file is not None:
    df, st.session_state.embeddings_df = load_data(uploaded_file)
    st.subheader("Data Preview:")
    st.dataframe(df.head())

    if st.session_state:
        st.subheader('Basic Statistics')
        st.write(df.describe())

    # Chat Interface
    st.header('Ask Questions About Your Data')

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.write(message['content'])
            if message['role'] == 'assistant' and 'visualization' in message:
                if isinstance(message['visualization'], str) and message['visualization'].startswith("Unfortunately"):
                    st.write("No visualization available for this response.")
                elif message['visualization']:
                    st.image(message['visualization'])

    # User input for new question
    user_input = st.chat_input("Enter your question:")
    

    if user_input and st.session_state.embeddings_df is not None:

        # Store user message
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})

        # Display the user query
        with st.chat_message("user"):
            st.write(user_input)

        # Show a spinner during a process
        with st.spinner('Processing...'):

            response = GeminiLLMAPI.process_query(st.session_state.embeddings_df, user_input)
            response_data = json.loads(response)
            text_response = response_data.get('Answer')

            #  Use DashboardPlotter to create visualization
            visualization = None
            if 'Dashboard' in response_data:
                try:
                    plotter = DashboardPlotter(response_data)
                    fig = plotter.plot()
                    
                    # Convert plot to base64 string
                    buf = BytesIO()
                    fig.savefig(buf, format='png')
                    buf.seek(0)
                    img_str = base64.b64encode(buf.getvalue()).decode()
                    visualization = f"data:image/png;base64,{img_str}"
                    plt.close(fig)  # Close the figure to free up memory
                except Exception as e:
                    st.error(f"Error generating visualization: {str(e)}")
                    visualization = None


            # Display the answer and visualization
            with st.chat_message("assistant"):
                st.write(text_response)
                if visualization:
                    st.image(visualization)
                else:
                    st.write("")

        # Store bot response with visualization
        st.session_state.chat_history.append({
            'role': 'assistant', 
            'content': text_response,
            'visualization': visualization
        })
