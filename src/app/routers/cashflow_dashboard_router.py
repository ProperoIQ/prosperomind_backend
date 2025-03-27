from fastapi import APIRouter, HTTPException
from app.repositories.prompt_repository import PromptRepository
from app.repositories.profitability_dashboard_settings_repository import ProfitabilityDashboardSettingsRepository
from app.repositories.cashflow_dashboard_repository import *
from app.repositories.prompt_repository import PromptRepository, PromptResponseRepository
import pandas as pd
import warnings
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union, Dict
import logging
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
import json
import copy
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import re
from app.routers.prompt_router import *
from app.instructions import (
    get_query_instruction, get_graph_instruction, get_planner_instruction
)

from pandasai import SmartDatalake, Agent
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import (
    create_csv_agent, create_pandas_dataframe_agent
)
from langchain_openai import ChatOpenAI, OpenAI
from loguru import logger
from app.repositories.profitability_dashboard_repository import *
# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Define log format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Create stream handler to output logs to console
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

pd.options.display.float_format = '{:.2f}'.format
warnings.filterwarnings("ignore")

router = APIRouter(prefix="/cashflow-dashboard", tags=["Cash Flow Dashboard"])
prompt_repository = PromptRepository()
repo = ProfitabilityDashboardSettingsRepository()


@router.get("/cashflow-dashboard-overall/{board_id}", response_model=dict)
async def get_cashflow_variance_dashboard(board_id: int):
    try:
        start_time = datetime.now()
        settings_list = repo.get_profitability_dashboard_settings_by_board_id(board_id)
        start_date = settings_list.financial_year_start
        start_date = convert_to_standard_start_date(start_date)
        combined_contents, dataframes_list, table_name_list = prompt_repository.get_file_download_links_by_board_id(board_id)

        RevenueActual = dataframes_list[0]
        RevenueForecast = dataframes_list[1]
        ExpenseForecast = dataframes_list[2]
        ExpenseActual = dataframes_list[3]
        tax_rate = 0.2

        cashflow_report_df = generate_cashflow_variance_report(RevenueActual, RevenueForecast, ExpenseActual, ExpenseForecast, tax_rate, start_date)
        bar_chart = generate_cash_flow_operations_bar_chart(cashflow_report_df)
        cashflow_report_formatted = format_financial_dataframe(cashflow_report_df)
        cashflow_report_formatted = {"table": {"columns": cashflow_report_formatted.columns.tolist(), "data": cashflow_report_formatted.values.tolist()}}

        #Variance Table and line chart
        actual_monthly_data = generate_cashflow_report_monthly(RevenueActual, ExpenseActual, start_date)
        forecast_monthly_data = generate_cashflow_report_monthly(RevenueForecast, ExpenseForecast, start_date)
        budget_monthly_data = generate_cashflow_report_monthly(RevenueForecast, ExpenseForecast, start_date)
        combined_df = pd.concat([actual_monthly_data.iloc[-1], forecast_monthly_data.iloc[-1], budget_monthly_data.iloc[-1]], axis=1)
        combined_df.reset_index(inplace=True)
        combined_df.columns = ['Month_Year', 'Actual Cash Flow from Operations', 'Forecast Cash Flow from Operations', 'Budget Cash Flow from Operations']
        combined_df = combined_df.iloc[1:]
        combined_df.reset_index(drop=True, inplace=True)
        combined_df = combined_df.fillna(0)
        combined_df['Variance: Actual vs Forecast'] = combined_df['Forecast Cash Flow from Operations'] - combined_df['Actual Cash Flow from Operations']
        combined_df['Variance: Actual vs Budget'] = combined_df['Budget Cash Flow from Operations'] - combined_df['Actual Cash Flow from Operations']
        line_chart = generate_operating_profit_line_chart(combined_df, "Cash Flow from Operations")
        variance_output = format_numeric_dataframe(combined_df)
        variance_output_formatted = {"table": {"columns": variance_output.columns.tolist(), "data": variance_output.values.tolist(), "title":"Variance Table"}}
        
        #Add title
        cashflow_report_formatted['title'] = "Cash Flow Dashboard Overall"
        bar_chart['title'] = "Cash Flow Overall"
        line_chart['title'] = "Cash Flow From Operations for Actual , Budget and Forecast"
        variance_output_formatted['table']['title'] = "Variance Table"

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        return JSONResponse(content={
            "status_code":200,
            "status": "success",
            "processing_time_seconds": processing_time,
            "variance_table":variance_output_formatted["table"],
            ** cashflow_report_formatted,
            "charts":[bar_chart, line_chart],
            "message": "Data retrieved successfully",
            "meta_info":{
                "financial year start" : "October 1st, 2023",
                "financial year end" : "September 30th, 2024",
                "forecast period": "12 months",
                "Actual data start date":"October 1st, 2023",
                "Actual data end date":"December 30th, 2023"
        }
        })
    except Exception as e:
        logger.error(f"An error occurred while processing profitability dashboard for board_id {board_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/cashflow-dashboard-actual/{board_id}", response_model=dict)
async def get_cashflow_actual_dashboard(board_id: int):
    try:
        start_time = datetime.now()
        settings_list = repo.get_profitability_dashboard_settings_by_board_id(board_id)
        start_date = settings_list.financial_year_start
        start_date = convert_to_standard_start_date(start_date)
        combined_contents, dataframes_list, table_name_list = prompt_repository.get_file_download_links_by_board_id(board_id)

        RevenueActual = dataframes_list[0]
        RevenueForecast = dataframes_list[1]
        ExpenseForecast = dataframes_list[2]
        ExpenseActual = dataframes_list[3]
        tax_rate = 0.2
        
        cashflow_monthly_report = generate_cashflow_report_monthly(RevenueActual, ExpenseActual, start_date)
        bar_chart = generate_cash_flow_bar_chart_monthly(cashflow_monthly_report)
        cashflow_report_formatted = format_financial_dataframe(cashflow_monthly_report)
        cashflow_report_formatted = {"table": {"columns": cashflow_report_formatted.columns.tolist(), "data": cashflow_report_formatted.values.tolist()}}
        cashflow_report_formatted['title'] = "Cash Flow Dashboard Overall"

        #Add title
        cashflow_report_formatted['title'] = "Cash Flow Actual"
        bar_chart['title'] = "Cash Flow Actual"

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        return JSONResponse(content={
            "status_code":200,
            "status": "success",
            "processing_time_seconds": processing_time,
            ** cashflow_report_formatted,
            "charts":[bar_chart],
            "message": "Data retrieved successfully",
            "meta_info":{
                "financial year start" : "October 1st, 2023",
                "financial year end" : "September 30th, 2024",
                "forecast period": "12 months",
                "Actual data start date":"October 1st, 2023",
                "Actual data end date":"December 30th, 2023"
        }
        })
    except Exception as e:
        logger.error(f"An error occurred while processing profitability dashboard for board_id {board_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/cashflow-dashboard-forecast/{board_id}", response_model=dict)
async def get_cashflow_forecast_dashboard(board_id: int):
    try:
        start_time = datetime.now()

        settings_list = repo.get_profitability_dashboard_settings_by_board_id(board_id)
        start_date = settings_list.financial_year_start
        start_date = convert_to_standard_start_date(start_date)
        combined_contents, dataframes_list, table_name_list = prompt_repository.get_file_download_links_by_board_id(board_id)

        RevenueActual = dataframes_list[0]
        RevenueForecast = dataframes_list[1]
        ExpenseForecast = dataframes_list[2]
        ExpenseActual = dataframes_list[3]
        tax_rate = 0.2
        
        cashflow_monthly_report = generate_cashflow_report_monthly(RevenueForecast, ExpenseForecast, start_date)
        bar_chart = generate_cash_flow_bar_chart_monthly(cashflow_monthly_report)
        cashflow_report_formatted = format_financial_dataframe(cashflow_monthly_report)
        cashflow_report_formatted = {"table": {"columns": cashflow_report_formatted.columns.tolist(), "data": cashflow_report_formatted.values.tolist()}}

        #Add title
        cashflow_report_formatted['title'] = "Cash Flow Forecast"
        bar_chart['title'] = "Cash Flow Forecast"

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        return JSONResponse(content={
            "status_code":200,
            "status": "success",
            "processing_time_seconds": processing_time,
            **cashflow_report_formatted,
            "charts":[bar_chart],
            "message": "Data retrieved successfully",
            "meta_info":{
                "financial year start" : "October 1st, 2023",
                "financial year end" : "September 30th, 2024",
                "forecast period": "12 months",
                "Actual data start date":"October 1st, 2023",
                "Actual data end date":"December 30th, 2023"
            }
            })
    except Exception as e:
        logger.error(f"An error occurred while processing profitability dashboard for board_id {board_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/chat-integration")
async def chat_integration(
    board_id: int,
    input_text:str,
    name: str = None,
    use_cache: bool = True
):
    try:
        response_content = "Please review and modify the prompt with more specifics."
        start_time = datetime.now()
        graph_output_json = {}
        settings_list = repo.get_profitability_dashboard_settings_by_board_id(board_id)
        start_date = settings_list.financial_year_start
        start_date = convert_to_standard_start_date(start_date)
        combined_contents, dataframes_list, table_name_list = prompt_repository.get_file_download_links_by_board_id(board_id)
        
        hash_key = prompt_response_repository.generate_hash_key(combined_contents, input_text)

        # Check if the response is already present in the Prompt_response table
        existing_response = await prompt_response_repository.check_existing_response(hash_key)

        if existing_response and use_cache:
            # If response is already present, return the existing response
            logger.info("Using the existing response")
            return JSONResponse(content=existing_response[3])
        
        RevenueActual = dataframes_list[0]
        RevenueForecast = dataframes_list[1]
        ExpenseForecast = dataframes_list[2]
        ExpenseActual = dataframes_list[3]

        #1st table type for overall
        tax_rate = 0.2
        cashflow_report_df = generate_cashflow_variance_report(RevenueActual, RevenueForecast, ExpenseActual, ExpenseForecast, tax_rate, start_date)
        cashflow_report_df['Table_name'] = "Overall Data"
        
        #2nd Table Actual Monthly
        cashflow_actual_monthly_report = generate_cashflow_report_monthly(RevenueActual, ExpenseActual, start_date)
        cashflow_actual_monthly_report['Table_name'] = "Actual Monthly Data"

        #3rd Table Forecast Monthly    
        cashflow_forecast_monthly_report = generate_cashflow_report_monthly(RevenueForecast, ExpenseForecast, start_date)
        cashflow_forecast_monthly_report['Table_name'] = "Forecast Monthly Data"

        
        llm = ChatOpenAI(temperature=0, model="gpt-4")
        #Here we are defining the data
        dataframes_list = [cashflow_report_df, cashflow_actual_monthly_report, cashflow_forecast_monthly_report]
        # dl = SmartDatalake(dataframes_list, config={"llm": llm})
        agent = Agent(dataframes_list, config={"llm": llm, "verbose":True, "enable_cache": False, "max_retries":10})
        #input_text = get_planner_instruction(input_text)
        rephrased_query = agent.rephrase_query(input_text)
        response_content = agent.chat(rephrased_query)
        
        if isinstance(response_content, int) or isinstance(response_content, float):
            response_content = {"message": [str(int(response_content))], "table": {}}
        elif "Unfortunately" in response_content or ".png" in response_content or 'No data available for the given conditions' in response_content:
            try:
                #logger.info("Running Langchain CSV agent")
                #response_content = create_csv_langchain_agent(input_text, dataframes_list, llm)
                logger.info("Running Pandasai Agnet 2nd time with Planner")
                input_text = get_planner_instruction(input_text)
                rephrased_query = agent.rephrase_query(input_text)
                response_content = agent.chat(rephrased_query)
            except Exception as ex:
                logger.error(f"2nd Time After planning also : {ex}")
                response_content = {}
                response_content["message"] = "Please review and modify the prompt with more specifics."
        else:
            if isinstance(response_content, pd.DataFrame):
                response_content = response_content.fillna(0).round(2)
                response_content = convert_timestamps_to_strings(response_content)
                response_content = response_content = {"message": [], "table": {"columns": response_content.columns.tolist(), "data": response_content.values.tolist()}}
            else:
                response_content = {"message": [str(response_content)], "table": {}}
            # #Graph Processing from Here
            try:
                if "columns" in response_content["table"] and len(response_content["table"]['data']):
                    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
                    graph_df = convert_table_to_dataframe(response_content["table"])
                    graph_instruction = get_graph_instruction()
                    graph_output = llm.invoke(graph_instruction + graph_df.to_markdown())
                    graph_output = re.sub(r'\bfalse\b', 'False', re.sub(r'\btrue\b', 'True', graph_output.content, flags=re.IGNORECASE), flags=re.IGNORECASE)
                    #Remove special character
                    graph_output = re.sub(r"```|python|json", "",graph_output, 0, re.MULTILINE)
                    graph_output_json = eval(graph_output)
                    logger.info("Graph Generation Success")
                else:
                    graph_output_json = {}
            except Exception as ex:
                logger.error(f"Graph generation Fails {ex} ")
                graph_output_json = {}

        end_time = datetime.now()
        duration = end_time - start_time
            
        logger.info(f"Response content {response_content}")
        
        result = {
            "status_code": 200,
            "detail": "Prompt Run Successfully",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "board_id": board_id,
            "prompt_text":input_text,
            **response_content,
            **graph_output_json
        }
        logger.info(f"Result :{result}")

        try:
            # Save the response to the Prompt_response table
            if response_content["message"] != "Please review and modify the prompt with more specifics.":
                await prompt_response_repository.save_response_to_database(hash_key, result)
        except Exception as ex:
            logger.error(f"Save Response to database Fails: {ex}")
                        
        return JSONResponse(content=result)
    except Exception as ex:
        # Handle exceptions and return an error response if needed
        logger.error(f"Internal fails {ex}")
        return JSONResponse(content={"error": "Prompt Error","detail":response_content}, status_code=500)