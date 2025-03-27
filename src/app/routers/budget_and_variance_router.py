from fastapi import APIRouter
from fastapi.responses import JSONResponse
from datetime import datetime
import logging
import warnings

import pandas as pd

from app.models.forecast_response import ForecastResponse
from app.repositories.prompt_repository import PromptRepository
from app.repositories.profitability_dashboard_repository import *
from app.repositories.profitability_dashboard_settings_repository import (
    ProfitabilityDashboardSettingsRepository
)
from app.repositories.prompt_repository import PromptRepository, PromptResponseRepository
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

pd.options.display.float_format = '{:.2f}'.format
logging.basicConfig(level=logging.INFO)  # Set logging level to INFO
logger = logging.getLogger("Budget Variance")

# Ignore all warnings
warnings.filterwarnings("ignore")

router = APIRouter(prefix="/budget-variance", tags=["Budget Variance"])
prompt_repository = PromptRepository()
repo = ProfitabilityDashboardSettingsRepository()

@router.get("/profitability-dashboard-budget-monthly/{board_id}", response_model=dict)
async def get_profitability_dashboard_budget_monthly(board_id:int):
    #To do impement settings
    #This is not generic implmentation . THis is just for demo purpose
    board_id = 13
    settings_list = repo.get_profitability_dashboard_settings_by_board_id(board_id)
    start_date = settings_list.financial_year_start
    start_date = convert_to_standard_start_date(start_date)
    combined_contents, dataframes_list, table_name_list = prompt_repository.get_file_download_links_by_board_id(board_id)
    
    RevenueActual = dataframes_list[0]
    RevenueForecast = dataframes_list[1]
    ExpenseForecast = dataframes_list[2]
    ExpenseActual = dataframes_list[3]
    
    all_monthly_data = list()
    all_monthly_data.append(create_revenue_monthly(RevenueForecast, sale_value_column='Sale_Value', start_date=start_date))
    all_monthly_data.append(create_cogs_monthly(ExpenseForecast, start_date))
    all_monthly_data.append(create_other_operating_expenses_monthly(ExpenseForecast, start_date))
    all_monthly_data.append(create_other_administrative_expenses_monthly(ExpenseForecast, start_date))
    all_monthly_data.append(calculate_profitability_monthly(all_monthly_data))
    
    bar_chart = generate_bar_chart_monthly(all_monthly_data)
    all_monthly_data = process_payload_monthly(all_monthly_data)
    all_monthly_data = convert_to_json_serializable(all_monthly_data)
    
    #Add title for Bar Chart and Line Chart
    bar_chart['title'] = "Actual Operating Profit on Month Level"
    
    return JSONResponse(content={
        "Success": True,
        "hierarchy_table": {"data":all_monthly_data,
                            "title":"Actual Operating Profit on Month Level"},
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
    
@router.get("/profitability-dashboard-overall/{board_id}", response_model=dict)
async def get_profitability_dashboard_overall(board_id:int):
    #start_date='2023-10-01'
    board_id = 13
    settings_list = repo.get_profitability_dashboard_settings_by_board_id(board_id)
    start_date = settings_list.financial_year_start
    start_date = convert_to_standard_start_date(start_date)
    combined_contents, dataframes_list, table_name_list = prompt_repository.get_file_download_links_by_board_id(board_id)
    
    RevenueActual = dataframes_list[0]
    RevenueForecast = dataframes_list[1]
    ExpenseForecast = dataframes_list[2]
    ExpenseActual = dataframes_list[3]
    
    profitability_analysis_data = list()
    
    profitability_analysis_data.append(create_actual_vs_forecast_revenue(RevenueActual, RevenueForecast, start_date))
    profitability_analysis_data.append(create_actual_vs_forecast_cogs(ExpenseActual, ExpenseForecast, start_date))
    profitability_analysis_data.append(create_actual_vs_forecast_other_operating_expenses(ExpenseActual, ExpenseForecast, start_date))
    profitability_analysis_data.append(create_actual_vs_forecast_other_administrative_expenses(ExpenseActual, ExpenseForecast, start_date))
    profitability_analysis_data.append(create_actual_vs_forecast_operating_profit(profitability_analysis_data))
    
    for item in profitability_analysis_data:
        for subrow in item.get('subRows', []):
            subrow['Label'] = subrow.pop('Item', subrow.get('Label'))
            
    profitability_analysis_data = calculate_variance(profitability_analysis_data)

    bar_chart = generate_bar_chart(profitability_analysis_data)
    profitability_analysis_data = format_response(profitability_analysis_data)
    
    #Create Line chart 
    line_chart, variance_output = calculate_line_chart_and_variance(RevenueActual, RevenueForecast, ExpenseActual, ExpenseForecast, start_date)
    
    #Add title for Bar Chart and Line Chart
    bar_chart['title'] = "Actual vs Forecast vs Budget Revenue"
    line_chart['title'] = "Actual vs Forecast vs Budget Revenue"
    variance_output['table']['title'] = "Variance Table"

    return JSONResponse(content={
        "Success": True,
        "hierarchy_table": {"data":profitability_analysis_data,
                            "title":"Actual vs Forecast vs Budget Revenue"},
        #"bar_graph": bar_chart,
        #"line_chart": line_chart,
        "charts":[bar_chart, line_chart],
        "table" : variance_output["table"],
        "message": "Data retrieved successfully",
        "meta_info":{
            "financial year start" : "October 1st, 2023",
            "financial year end" : "September 30th, 2024",
            "forecast period": "12 months",
            "Actual data start date":"October 1st, 2023",
            "Actual data end date":"December 30th, 2023"
        }
    })