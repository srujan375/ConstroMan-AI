# GeminiLLMAPI.py
import google.generativeai as genai
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from Analysis.data_preprocessor import DataPreprocessor
import numpy as np
import os

class GeminiLLMAPI:
    
    @staticmethod
    def configure_api():
        genai.configure(api_key= os.environ['GEMINI_API_KEY'])

    @staticmethod
    def get_ai_response(prompt, dataframe):
        full_prompt = f"Relevant data:\n{dataframe}\n\nQuestion: {prompt}\n\nAnswer:"
        
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config = {"response_mime_type" : "application/json", "max_output_tokens": 8196, "temperature": 0.2},
            system_instruction=f"""You are an expert and experienced civil engineer and data analyst, You will be provided with an excel file containing construction project management data on which you may be asked to analyze or simply be asked some question about that provided data, Your job is to choose the best method to analyze based on the kind of data provided and when answering also be ready to support your answer with some visulalization like charts and graphs, whatever you think is the best way to help the user understand your answer visually, Strictly do not provide code or other steps on how the data is to be analyzed, You are to do it for them and only display the end-result. Provide only a single object,
            If a visualization is not important then strictly leave the "Dashboard" key empty.
            The output is expected in the following JSON format:
            {{
                "Answer": "Answer to the user query",
                "Dashboard": {{
                        "Name": "Name of the visualization",
                        "Type": "LineChart | BarChart | PieChart | DonutChart | ScatterPlot | Histogram |",
                        "X-axis label": "Label of the X-axis",
                        "Y-axis label": "Label of the Y-axis",
                        "X-axis data": "Data for the X-axis",
                        "Y-axis data": "Data for the Y-axis",
                        "Labels": "Labels for the data (used in Pie and Donut charts)", 
                        "Values": "Range of values for the data (used in Pie and Donut charts)"
                    }}
            }}
            Only one chart per response is expected. If you have multiple charts, choose the most relevant one.
            **IMPORTANT**: 1) Always have all the necessary data for the X-axis and Y-axis data fields whenever required, Same for all the other fields present
                       2) Always have the data in the form of a list, even if it is a single value, like "X-axis data": ["value"], "Y-axis data": ["value"], etc Incase of a Pie-chart or a donut-chart, You can proivde a range of values instead that can be used.
                       3) Use proper notations and labels for the data fields, like "X-axis label": "Year", "Y-axis label": "Sales", etc.
                       4) The user is an business person and we are providing him with valuable insight so as to make a informed decision, So huge amounts of investments are at stake here so our answers should be accurate and to the point, Also the user is not a data analyst so the answers should be easy to understand and should be supported with visualizations to make it more clear to the user.
                       5) If the values for a particular axis are the same throughout then provide them as a single value
                       6) Number of labels must always match the number of values
                       7) Do not exceed more than 8 values for the X-axis data and Y-axis data fields, If you have more than 8 values then provide the most relevant 8 values
                       8) Having a Visualization at all times is not required, If you think that a visualization is not required then leave the "Dashboard" key empty. Only use it when the data can be better understood with a visualization."""
        )
        
        prompt = f"""The relevant data and the question is as follows: {full_prompt}"""
            
        response = model.generate_content(prompt)
        response_content = response.text
        
        
        return response_content
    
    @staticmethod
    def generate_embeddings(df):
        preprocessed_df = DataPreprocessor.preprocess(df)
        text_data = DataPreprocessor.prepare_for_embedding(preprocessed_df)
        
        embeddings_list = []
        for text in text_data:
            response = genai.embed_content(
                content=text,
                model="models/embedding-001",
                task_type="retrieval_document"
            )
            embedding = response['embedding']
            embeddings_list.append(embedding)
        
        embeddings_df = pd.DataFrame(embeddings_list)
        embeddings_df.columns = [f'embedding_{i}' for i in range(len(embeddings_df.columns))]
        return pd.concat([preprocessed_df.reset_index(drop=True), embeddings_df], axis=1)

    @staticmethod
    def find_relevant_data(query, embeddings_df, top_k=5):
        query_response = genai.embed_content(
            content=query,
            model="models/embedding-001",
            task_type="retrieval_query"
        )
        query_embedding = query_response['embedding']

        # Ensure embeddings_df is a DataFrame
        if not isinstance(embeddings_df, pd.DataFrame):
            raise ValueError("embeddings_df must be a pandas DataFrame")

        data_embeddings = embeddings_df[[col for col in embeddings_df.columns if col.startswith('embedding_')]].values
        
        # Calculate dot products
        dot_products = np.dot(data_embeddings, query_embedding)
        
        # Get indices of top_k highest dot products
        top_indices = np.argsort(dot_products)[-top_k:][::-1]

        relevant_data = embeddings_df.iloc[top_indices].drop([col for col in embeddings_df.columns if col.startswith('embedding_')], axis=1)
        return relevant_data.to_string()

    @staticmethod
    def process_query(embeddings_df, query):
        relevant_data = GeminiLLMAPI.find_relevant_data(query, embeddings_df)
        response = GeminiLLMAPI.get_ai_response(query, relevant_data)
        return response
    
