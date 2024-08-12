import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pandasai import SmartDataframe, Agent
from pandasai.llm import GoogleGemini
import os

class PandasAI:
    @staticmethod
    def get_llm_api():
        return GoogleGemini(api_key= os.environ['GEMINI_API_KEY'])

    @staticmethod
    def generate_visualization(dashboard_data, dataframe):

        llm = GoogleGemini(api_key= os.environ['GEMINI_API_KEY'])
        
        file_path = "exports/charts"
        
        agent = Agent(
            dfs=dataframe,
            description="""You will be provided data in the following JSON format, 
            "Visualization": [
            {
                "Title": "Title of the Chart",
                "Type": "Type of the visualization",
                "X-axis label": "Label of the X-axis",
                "Y-axis label": "Label of the Y-axis",
                "X-axis data": "Data for the X-axis",
                "Y-axis data": "Data for the Y-axis"
            }
            ]
            You are to look at the "Type" key and create the visualizations accordingly using the values provided and labels provided. Use the labels and legends provided to make it easy to understand the data. You are to provide the user with the best visualization that you think will help the user understand the data best. The Title of the chart or the visualization is also provided to you under the "Title" key. This should always be on top of the visualization. Use only the values provided in the "X-axis data" and "Y-axis data" keys to create the visualization. Do not use any other data from the dataset.
            **VERY IMPORTANT**- 1) Please make sure that no keys or data points overlap in the visualization. 
                                2) Provide a legend in each Visualization to help the user understand the data better. Make sure that it is in the top right corner of the visualization. Neatly arrange the legend so that it does not overlap with the visualization. Also properly show what each color represents in the legend.
                                3) If you are not able to create the visualization, please provide a message to the user that you are unable to create the visualization and provide a reason for the same. If you are able to create the visualization, please provide the visualization to the user.""",
            config={
                "llm": llm, 
                "save_charts": True,
                "save_charts_path": file_path
            }
        )
        response = agent.chat(
            f"""The data on which you will be working is as follows: {dashboard_data}"""
        )

        return response

    @staticmethod
    def process_ai_response(response_json, dataframe):
        try:
            response_data = json.loads(response_json)
            dashboard_data = response_data.get('Dashboard')
            if not dashboard_data:
                return None

            # Clean the data
            dataframe = dataframe.apply(lambda x: x.str.replace("'", "") if x.dtype == "object" else x)
            dataframe = dataframe.apply(lambda x: x.str.replace('"', "") if x.dtype == "object" else x)
            dataframe = dataframe.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.dtype != "object" else x)
            dataframe = dataframe.fillna(0)

            visualizations = []
            visualization = PandasAI.generate_visualization(dashboard_data, dataframe)
            visualizations.append(visualization)
            return visualizations

        except json.JSONDecodeError:
            return "Error: Invalid JSON response"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: {e}"