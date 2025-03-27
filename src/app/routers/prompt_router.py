# app/routers/prompt_router.py
import copy
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from typing import List
from app.repositories.prompt_repository import PromptRepository, PromptResponseRepository
from app.models.prompt import Prompt, PromptCreate
from app.instructions import get_query_instruction, get_graph_instruction, get_planner_instruction, get_planner_instruction_with_data
from io import BytesIO
from fastapi.responses import JSONResponse

import os
import re
import pandas as pd
from datetime import datetime

#Pandas AI Implementation
from pandasai import SmartDatalake
from pandasai import Agent, SmartDataframe

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent, create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI, OpenAI

from types import FrameType
from loguru import logger


from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
import pandas as pd
import numpy as np
import re
import json
from typing import Any, Dict, Union
from pydantic import BaseModel

import json
from datetime import datetime, date
from typing import Any
import pandas as pd
import numpy as np
from multiprocessing import Pool

from app.repositories.data_management_table_repository import DataManagementTableRepository

router = APIRouter(prefix="/prompts", tags=["Prompts"])

prompt_repository = PromptRepository()
prompt_response_repository = PromptResponseRepository()

@router.post("/", response_model=Prompt)
def create_prompt_route(prompt_create: Prompt):
    new_prompt = prompt_repository.create_prompt(prompt_create)
    return new_prompt

@router.get("/boards/{board_id}", response_model=List[Prompt])
def get_prompts_for_board_route(board_id: int):
    prompts = prompt_repository.get_prompts_for_board(board_id)
    return prompts

@router.get("/{prompt_id}", response_model=Prompt)
def get_prompt_route(prompt_id: int):
    prompt = prompt_repository.get_prompt(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return prompt

@router.put("/{prompt_id}", response_model=Prompt)
def update_prompt_route(prompt_id: int, prompt: Prompt):
    updated_prompt = prompt_repository.update_prompt(prompt_id, prompt)
    if not updated_prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return updated_prompt

@router.delete("/{prompt_id}", response_model=Prompt)
def delete_prompt_route(prompt_id: int):
    deleted_prompt = prompt_repository.delete_prompt(prompt_id)
    if not deleted_prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return deleted_prompt

@router.get("/{main_board_id}/{board_id}/prompts", response_model=List[Prompt])
async def get_prompts_for_board_in_main_board(main_board_id: int, board_id: int):
    prompts = prompt_repository.get_prompts_for_board_in_main_board(main_board_id, board_id)
    return prompts


def convert_table_to_dataframe(table):
    if "columns" in table and "data" in table:
        # Assuming that "columns" and "data" keys are present in the table
        columns = table["columns"]
        data = table["data"]

        # Create a Pandas DataFrame
        df = pd.DataFrame(data, columns=columns)

        return df
    else:
        # Handle the case where "columns" or "data" is missing
        print("Invalid table structure. Unable to convert to DataFrame.")
        return None
        
def create_csv_langchain_agent(input_text, data, llm):
        agent = create_pandas_dataframe_agent(
            llm, data, 
            verbose=True, 
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
            handle_parsing_errors=True, 
            number_of_head_rows=0
        )

        # Define the preinstruction for the query.
        instruction = get_query_instruction()

        prompt = instruction + input_text 
        response_content = agent.run(prompt)

        #Remove special character
        response_content = re.sub(r"```|python|json", "",response_content, 0, re.MULTILINE)
        
        try:
            response_content = eval(response_content)
        except Exception as ex:
            # Format output as per our structure
            if isinstance(response_content, int):
                response_content = {"message": [str(response_content)], "table": {}}
        return response_content


class RePromptService:
    def __init__(self, prompt_repository: PromptRepository, llm_service: ChatOpenAI):
        self.prompt_repository = prompt_repository
        self.llm_service = llm_service

    @staticmethod
    def extract_text_between_double_quotes(text):
        #['sample', 'multiple', 'strings']
        pattern = r'"([^"]*)"'
        matches = re.findall(pattern, text)
        return matches
    
    

    def run_re_prompt(self, input_text: str, board_id: str):
        try:
            combined_contents, dataframes_list, table_name_list = self.prompt_repository.get_file_download_links_by_board_id(board_id)

            input_text = get_planner_instruction_with_data(input_text)
            llm_output = self.llm_service.invoke(input_text)
            pattern = re.compile(r"[\s\S]*?Output:\s*.*", re.DOTALL)
            match = pattern.search(llm_output.content)
            
            if match:
                return match.group()
            
            return llm_output.content
        
        except Exception as e:
            # Handle exceptions and return appropriate HTTP error response
            raise HTTPException(status_code=500, detail=f"Error processing re-prompt: {str(e)}")


@router.post("/re_prompt")
async def run_re_prompt(input_text: str, board_id: str):
    re_prompt_service = RePromptService(prompt_repository, ChatOpenAI(temperature=0, model="gpt-3.5-turbo"))
    return re_prompt_service.run_re_prompt(input_text, board_id)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, pd.Period)):
            return str(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            if np.isnan(obj) or np.isinf(obj):  # Handle NaN and Inf
                return None  # Or use a string like "NaN"/"Infinity"
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)
    
    
def clean_invalid_values(obj):
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None  # Or "NaN"
    elif isinstance(obj, dict):
        return {k: clean_invalid_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_invalid_values(v) for v in obj]
    return obj



def generate_chart_json(df):
    # Ensure the first column is the index (Row Labels)
    df.set_index(df.columns[0], inplace=True)
    # Labels for the charts (index of the DataFrame)
    labels = df.index.tolist()
    # Categories (columns of the DataFrame)
    categories = df.columns.tolist()
    # Values for the bar and line charts
    values = df.values.tolist()
    # Values for the pie chart (using the last column)
    pie_values = df.iloc[:, -1].tolist()

    # JSON structure
    chart_data = {
        "charts": [
            {
                "chart_type": "bar",
                "data_format": {
                    "labels": labels,
                    "categories": categories,
                    "values": values,
                    "isStacked": True
                },
                "insight": [
                    "Bar chart showing the distribution of values across different labels."
                ]
            },
            {
                "chart_type": "pie",
                "data_format": {
                    "labels": labels,
                    "categories": [categories[-1]],  # The last column is used for the pie chart
                    "values": pie_values,
                    "isStacked": False
                },
                "insight": [
                    "Pie chart showing the proportion of the last column values across different labels."
                ]
            },
            {
                "chart_type": "line",
                "data_format": {
                    "labels": labels,
                    "categories": categories,
                    "values": values,
                    "isStacked": False
                },
                "insight": [
                    "Line chart showing the trend of values across different labels."
                ]
            }
        ]
    }

    # Convert to JSON
    #json_output = json.dumps(chart_data, indent=4)
    
    return chart_data


class ResponseContent(BaseModel):
    message: list[str] = []
    table: Dict[str, Union[list, dict]] = {}


def handle_response_content(response_content, input_text, llm):
    """
    Handle and format the response content based on the type of response.

    Args:
        response_content (Any): The response content returned by the agent.
        input_text (str): The input prompt text.
        llm (ChatOpenAI): The language model instance.

    Returns:
        ResponseContent: The formatted response content.
    """
    if isinstance(response_content, (int, float)):
        #To do convert this in sentence format
        return {"message":[str(response_content)]}

    elif isinstance(response_content, pd.DataFrame):
        response_content = {
            "message": [],
            "table": {
                "columns": response_content.columns.tolist(),
                "data": response_content.values.tolist()
            }
        }
    else:
        return {"message":[str(response_content)]}



def generate_graph_json(response_content: ResponseContent, llm) -> dict:
    """
    Generate the graph JSON output from the response content.

    Args:
        response_content (ResponseContent): The formatted response content.
        llm (ChatOpenAI): The language model instance.

    Returns:
        dict: The graph JSON output.
    """
    try:
        if "columns" in response_content["table"] and len(response_content["table"]['data']):
            llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
            graph_df = convert_table_to_dataframe(response_content["table"])
            graph_instruction = get_graph_instruction()
            graph_output = llm.invoke(graph_instruction + graph_df.to_markdown())
            graph_output = re.sub(r'\bfalse\b', 'False', re.sub(r'\btrue\b', 'True', graph_output.content, flags=re.IGNORECASE), flags=re.IGNORECASE)
            graph_output = re.sub(r"```|python|json", "", graph_output, 0, re.MULTILINE)
            graph_output_json = eval(graph_output)
            logger.info("Graph Generation Success")
            return graph_output_json
    except Exception as ex:
        logger.error(f"Graph generation failed: {ex}")
        return {}

class DataFrameProcessor:
    def __init__(self, llm_model: str) -> None:
        self.llm = ChatOpenAI(temperature=0, model=llm_model)

    def process_dataframe_response(self, response_content: pd.DataFrame) -> Dict[str, Union[str, Dict]]:
        return {
            "message": [],
            "table": {
                "columns": response_content.columns.tolist(),
                "data": response_content.values.tolist()
            }
        }

class PromptHandler:
    def __init__(self, llm_model: str):
        self.llm = ChatOpenAI(temperature=0, model=llm_model)
        self.dataframe_processor = DataFrameProcessor(llm_model=llm_model)



    def run(self, input_text: str, dataframes_list: List[pd.DataFrame], matched_query) -> Dict[str, Any]:
        agent = Agent(dataframes_list, config={"llm": self.llm, "enable_cache": True})

    # Check if matched_query has more than 2 items
        query_to_use = matched_query if len(matched_query) > 2 else agent.rephrase_query(input_text)

        response_content = agent.chat(query_to_use)
        return self.handle_response_content(agent, response_content, input_text)


    def handle_response_content(self, agent, response_content, input_text: str) -> Dict[str, Union[str, Dict]]:
        if isinstance(response_content, (int, float)):
            return {"message": [str(response_content)], "table": {}}
        elif isinstance(response_content, pd.DataFrame):
            return self.dataframe_processor.process_dataframe_response(response_content)
        elif isinstance(response_content, str):
            return {"message": [response_content], "table": {}}
        else:
            return {"message": ["Unexpected response type"], "table": {}}

        
class GraphGenerator:
    def __init__(self, llm_model: str):
        self.llm = ChatOpenAI(temperature=0, model=llm_model)

    async def generate_graphs(self, response_content: Dict[str, Any]) -> Dict[str, Any]:
        try:
            graph_df = convert_table_to_dataframe(response_content["table"])
            graph_instruction = get_graph_instruction()
            graph_output = self.llm.invoke(graph_instruction + graph_df.to_markdown())
            graph_output = re.sub(r'\bfalse\b', 'False', re.sub(r'\btrue\b', 'True', graph_output.content, flags=re.IGNORECASE), flags=re.IGNORECASE)
            graph_output = re.sub(r"```|python|json", "", graph_output, 0, re.MULTILINE)
            graph_output_json = eval(graph_output)
            return graph_output_json
        except Exception as ex:
            logger.error(f"Graph generation failed: {str(ex)}")
            return {}

class PromptFacade:
    def __init__(self):
        self.prompt_handler = PromptHandler(llm_model="gpt-4o")
        self.graph_generator = GraphGenerator(llm_model="gpt-4o")
        self.dataframe_processor = DataFrameProcessor(llm_model="gpt-4o")

    async def handle_prompt(self, input_text: str, board_id: str, user_name:str, use_cache: bool, matched_query:str) -> Dict[str, Any]:
        '''
        Author: Shashi Raj
        Date: 09-06-2024
        
        '''
        start_time = datetime.now()

        combined_contents, dataframes_list, table_name_list = prompt_repository.get_file_download_links_by_board_id(board_id)
        hash_key = prompt_response_repository.generate_hash_key(combined_contents, input_text)


        response_content = self.prompt_handler.run(input_text, dataframes_list, matched_query)

        if "columns" in response_content["table"] and len(response_content["table"]['data']):
            graph_output_json = await self.graph_generator.generate_graphs(response_content)
        else:
            graph_output_json = {}

        end_time = datetime.now() 
        result = self.create_response(start_time, end_time, board_id, input_text, response_content, graph_output_json)
        # Save the response to the Prompt_response table
        result["user_name"] = user_name        
        return result

    def create_response(self, start_time: datetime, end_time: datetime, board_id: str, input_text: str, 
                        response_content: Dict[str, Any], graph_output_json: Dict[str, Any]) -> Dict[str, Any]:  
        duration = end_time - start_time
        result = {
            "status_code": 200,
            "detail": "Prompt Run Successfully",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "board_id": board_id,
            "prompt_text": input_text,
            **response_content,
            **graph_output_json
        }
        
        return result

@router.post("/run_prompt_v2")
async def run_prompt_v2(input_text: str, board_id: str, mactched_query: str='', user_name:str = '', use_cache: bool = True):
    """
    API endpoint to run prompt, validate, generate graphs, and extract insights.
    """
    try:
        facade = PromptFacade()
        result = await facade.handle_prompt(input_text, board_id, user_name, use_cache, mactched_query)
        return JSONResponse(content=json.loads(json.dumps(clean_invalid_values(result), cls=CustomJSONEncoder)))
    except Exception as e:
        logger.error(f"Error in run_prompt_v2: {str(e)}")
        raise HTTPException(status_code=500, detail=(f"Internal Server Error :{str(e)}") )
    


@router.get("/csv-data/{board_id}")
async def get_csv_data(board_id: int, repo: DataManagementTableRepository = Depends()):
    return repo.get_csv_data_by_board_id(board_id)