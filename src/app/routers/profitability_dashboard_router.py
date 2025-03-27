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
logger = logging.getLogger("Profitability-Dashboard")

# Ignore all warnings
warnings.filterwarnings("ignore")

router = APIRouter(prefix="/profitability-dashboard", tags=["Profitability Dashboard"])
prompt_repository = PromptRepository()
repo = ProfitabilityDashboardSettingsRepository()


@router.get("/profitability-dashboard-overall/{board_id}", response_model=dict)
async def get_profitability_dashboard_overall(board_id:int):
    #start_date='2023-10-01'
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
    bar_chart['title'] = "Actual vs Forecast vs Budget Operating Profit"
    line_chart['title'] = "Actual vs Forecast vs Budget Operating Profit"
    variance_output['table']['title'] = "Variance Table"
    
    return JSONResponse(content={
        "Success": True,
        "hierarchy_table": {"data":profitability_analysis_data,
                            "title":"Actual vs Forecast vs Budget Operating Profit"},
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
    
@router.get("/profitability-dashboard-actual-monthly/{board_id}", response_model=dict)
async def get_profitability_dashboard_actual_monthly(board_id:int):
    #To do impement settings
    settings_list = repo.get_profitability_dashboard_settings_by_board_id(board_id)
    start_date = settings_list.financial_year_start
    start_date = convert_to_standard_start_date(start_date)    
    combined_contents, dataframes_list, table_name_list = prompt_repository.get_file_download_links_by_board_id(board_id)
    
    RevenueActual = dataframes_list[0]
    RevenueForecast = dataframes_list[1]
    ExpenseForecast = dataframes_list[2]
    ExpenseActual = dataframes_list[3]
    
    all_monthly_data = list()
    all_monthly_data.append(create_revenue_monthly(RevenueActual, sale_value_column='Sale_Value', start_date=start_date))
    all_monthly_data.append(create_cogs_monthly(ExpenseActual, start_date))
    all_monthly_data.append(create_other_operating_expenses_monthly(ExpenseActual, start_date))
    all_monthly_data.append(create_other_administrative_expenses_monthly(ExpenseActual, start_date))
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
        #"bar_graph": bar_chart,
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
    
@router.get("/profitability-dashboard-forecast-monthly/{board_id}", response_model=dict)
async def get_profitability_dashboard_forecast_monthly(board_id:int):
    #To do implement settings
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
    bar_chart['title'] = "Forecast Operating Profit on Month Level"
    return JSONResponse(content={
        "Success": True,
        "hierarchy_table": {"data":all_monthly_data,
                            "title":"Forecast Operating Profit on Month Level"},
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
    
@router.get("/profitability-dashboard-budget-monthly/{board_id}", response_model=dict)
async def get_profitability_dashboard_budget_monthly(board_id:int):
    #To do impement settings
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
    bar_chart['title'] = "Budget Operating Profit on Month Level"
    return JSONResponse(content={
        "Success": True,
        "hierarchy_table": {"data":all_monthly_data,
                            "title":"Budget Operating Profit on Month Level"},
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
        profitability_analysis_data = pd.DataFrame(profitability_analysis_data)
        profitability_analysis_data = process_subrows(profitability_analysis_data)
        profitability_analysis_data['Table_name'] = "Overall Data"

        #2nd Table Actual Monthly
        all_monthly_data = list()
        all_monthly_data.append(create_revenue_monthly(RevenueActual, sale_value_column='Sale_Value', start_date=start_date))
        all_monthly_data.append(create_cogs_monthly(ExpenseActual, start_date))
        all_monthly_data.append(create_other_operating_expenses_monthly(ExpenseActual, start_date))
        all_monthly_data.append(create_other_administrative_expenses_monthly(ExpenseActual, start_date))
        all_monthly_data.append(calculate_profitability_monthly(all_monthly_data))
        actual_monthly_data = pd.DataFrame(all_monthly_data)    
        actual_monthly_data = process_subrows(actual_monthly_data)   
        actual_monthly_data['Table_name'] = "Actual Monthly Data"    
            
        #3rd Table Forecast Monthly    
        all_monthly_data = list()
        all_monthly_data.append(create_revenue_monthly(RevenueForecast, sale_value_column='Sale_Value', start_date=start_date))
        all_monthly_data.append(create_cogs_monthly(ExpenseForecast, start_date))
        all_monthly_data.append(create_other_operating_expenses_monthly(ExpenseForecast, start_date))
        all_monthly_data.append(create_other_administrative_expenses_monthly(ExpenseForecast, start_date))
        all_monthly_data.append(calculate_profitability_monthly(all_monthly_data))
        forecast_monthly_data = pd.DataFrame(all_monthly_data)    
        forecast_monthly_data = process_subrows(forecast_monthly_data)
        forecast_monthly_data['Table_name'] = "Forecast Monthly Data"
        
        #4th Table Budget table
        all_monthly_data = list()
        all_monthly_data.append(create_revenue_monthly(RevenueForecast, sale_value_column='Sale_Value', start_date=start_date))
        all_monthly_data.append(create_cogs_monthly(ExpenseForecast, start_date))
        all_monthly_data.append(create_other_operating_expenses_monthly(ExpenseForecast, start_date))
        all_monthly_data.append(create_other_administrative_expenses_monthly(ExpenseForecast, start_date))
        all_monthly_data.append(calculate_profitability_monthly(all_monthly_data))
        budget_monthly_data = pd.DataFrame(all_monthly_data)    
        budget_monthly_data = process_subrows(budget_monthly_data)
        budget_monthly_data['Table_name'] = "Budget Monthly Data"
        
        llm = ChatOpenAI(temperature=0, model="gpt-4")
        #Here we are defining the data
        dataframes_list = [profitability_analysis_data, actual_monthly_data, forecast_monthly_data, budget_monthly_data]
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