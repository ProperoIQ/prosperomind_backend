from fastapi import APIRouter, HTTPException
from pmdarima import auto_arima
from app.models.forecast_response import ForecastResponse
from app.repositories.prompt_repository import PromptRepository
from app.repositories.forecast_settings_repository import ForecastSettingsRepository
from app.repositories.forecast_response_repository import ForecastResponseRepository
from app.repositories.forecast_chat_response_repository import ForecastChatResponseRepository
from app.routers.prompt_router import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import warnings
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import logging
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
import json
import copy

from app.instructions import get_query_instruction, get_graph_instruction, get_planner_instruction
from io import BytesIO
from fastapi.responses import JSONResponse

import os
import re
import pandas as pd
from datetime import datetime

#Pandas AI Implementation
from pandasai import SmartDatalake
from pandasai import Agent

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent, create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI, OpenAI

from types import FrameType
from loguru import logger


pd.options.display.float_format = '{:.2f}'.format
logging.basicConfig(level=logging.INFO)  # Set logging level to INFO
logger = logging.getLogger("Forecast")

# Ignore all warnings
warnings.filterwarnings("ignore")

router = APIRouter(prefix="/forecast-response", tags=["Forecast Response"])
prompt_repository = PromptRepository()
forecast_settings_repo = ForecastSettingsRepository()
forecast_response_repo = ForecastResponseRepository()
forecast_chat_response_repo = ForecastChatResponseRepository()
class TableData(BaseModel):
    columns: List[str]
    data: List[List[Union[str, float, int]]]
    title: str

class CoefficientsJson(BaseModel):
    table: TableData

class PredictedDataJson(BaseModel):
    table: TableData
    
class ActualVSForecast(BaseModel):
    table: TableData
    
class DataFormat(BaseModel):
    labels: List[str]
    categories: List[str]
    values: List[List[Union[float, int]]]
    isStacked: bool

class ActualVsForecastChart(BaseModel):
    chart_type: str
    data_format: DataFormat
    title: str
    
class RequestBody(BaseModel):
    actual_vs_forecast: ActualVSForecast
    actual_vs_forecast_chart: ActualVsForecastChart
    weight_coefficients: CoefficientsJson
    independent_variable: PredictedDataJson
    abs_sum_coef: float
    label: str
    
# class RequestBody(BaseModel):
#     item_metadata: List[ItemMetaData]
#     hierarchy_table: List[any]
#     total_level_line_chart: ActualVsForecastChart
#     item_level_line_chart: ActualVsForecastChart
    
@router.get("/forecast_responses", response_model=List[ForecastResponse])
async def get_all_forecast_responses():
    try:
        forecast_responses = forecast_response_repo.get_all_forecast_responses()
        return forecast_responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forecast_responses", response_model=ForecastResponse)
async def create_forecast_response(forecast_response: ForecastResponse):
    try:
        created_forecast_response = forecast_response_repo.create_forecast_response(forecast_response)
        return created_forecast_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/forecast_responses/{board_id}/{forecast_response_id}", response_model=dict)
async def get_forecast_response(board_id:int, forecast_response_id: int):
    try:
        forecast_settings = forecast_settings_repo.get_forecast_settings_by_board_id(board_id)
        financial_year_start = forecast_settings.financial_year_start
        date_variable = forecast_settings.date_variable
        independent_variables = forecast_settings.independent_variables
        dependent_variable = forecast_settings.dependent_variable
        budget = forecast_settings.budget
        column_name = 'Forecast ' + dependent_variable    
        forecast_response = forecast_response_repo.get_forecast_response(forecast_response_id)
        false = False
        final_result = eval(forecast_response.output_response)
        output_response = generate_output_response(final_result, dependent_variable, column_name, budget)
        output_response = round_json(output_response)
        if not forecast_response:
            raise HTTPException(status_code=404, detail=f"Forecast response with id {forecast_response_id} not found")
        return {
            "board_id":forecast_response.board_id,
            "forecast_response_id":forecast_response.id,
            "name":forecast_response.name,
            "first_level_filter":forecast_response.first_level_filter,
            "second_level_filters":forecast_response.second_level_filter,
            "forecast_period":forecast_response.forecast_period,
            "output_response":output_response,
            "crated_at":forecast_response.created_at,
            "updated_at":forecast_response.updated_at
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/forecast_responses/{forecast_response_id}", response_model=ForecastResponse)
async def update_forecast_response(forecast_response_id: int, forecast_response: ForecastResponse):
    try:
        updated_forecast_response = forecast_response_repo.update_forecast_response(forecast_response_id, forecast_response)
        if not updated_forecast_response:
            raise HTTPException(status_code=404, detail=f"Forecast response with id {forecast_response_id} not found")
        return updated_forecast_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/forecast_responses/{forecast_response_id}", response_model=ForecastResponse)
async def delete_forecast_response(forecast_response_id: int):
    try:
        deleted_forecast_response = forecast_response_repo.delete_forecast_response(forecast_response_id)
        if not deleted_forecast_response:
            raise HTTPException(status_code=404, detail=f"Forecast response with id {forecast_response_id} not found")
        return deleted_forecast_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/forecast-response-board-id/{board_id}", response_model=dict)
async def get_forecast_response_by_board_id(board_id: int):
    ''' 
    Use this functionality only for CFO screen
    SELECT * FROM ForecastResponse WHERE board_id = :board_id and publish_to_cfo = TRUE ORDER BY updated_at DESC LIMIT 1;
    '''
    forecast_response = forecast_response_repo.get_forecast_response_by_board_id(board_id)
    if not forecast_response:
        raise HTTPException(status_code=404, detail="Forecast response not found")
    
    forecast_settings = forecast_settings_repo.get_forecast_settings_by_board_id(board_id)
    dependent_variable = forecast_settings.dependent_variable
    budget = forecast_settings.budget
    column_name = 'Forecast ' + dependent_variable   
    false = False
    final_result = eval(forecast_response.output_response)
    output_response = generate_output_response(final_result, dependent_variable, column_name, budget)
    output_response = round_json(output_response)
    return {
    "board_id":forecast_response.board_id,
    "name":forecast_response.name,
    "first_level_filter":forecast_response.first_level_filter,
    "second_level_filters":forecast_response.second_level_filter,
    "forecast_period":forecast_response.forecast_period,
    "output_response":output_response,
    "crated_at":forecast_response.created_at,
    "updated_at":forecast_response.updated_at
    }
    
@router.get("/forecast-response-board-id-consultant/{board_id}", response_model=list[dict])
async def get_forecast_response_by_board_id(board_id: int):
    
    forecast_response_list = forecast_response_repo.get_forecast_response_by_board_id_consultant(board_id)
    if not forecast_response_list:
        raise HTTPException(status_code=404, detail="Forecast response not found")
    
    forecast_settings = forecast_settings_repo.get_forecast_settings_by_board_id(board_id)
    dependent_variable = forecast_settings.dependent_variable
    budget = forecast_settings.budget
    column_name = 'Forecast ' + dependent_variable   
    false = False
    return_output = list()
    for forecast_response in forecast_response_list:
        try:
            final_result = forecast_response.output_response
            final_result = eval(final_result)
            output_response = generate_output_response(final_result, dependent_variable, column_name, budget)
            output_response = round_json(output_response)
            return_output.append({
            "board_id":forecast_response.board_id,
            "forecast_response_id":forecast_response.id,
            "name":forecast_response.name,
            "first_level_filter":forecast_response.first_level_filter,
            "second_level_filters":forecast_response.second_level_filter,
            "forecast_period":forecast_response.forecast_period,
            "output_response":output_response,
            "crated_at":forecast_response.created_at,
            "updated_at":forecast_response.updated_at
            })
        except Exception as ex:
            continue
    return return_output
    
    
@router.get("/forecast-response-publish-cfo/{forecast_response_id}", response_model=dict)
async def update_publish_to_cfo_by_forecast_id(forecast_response_id: int, cfo_publish:bool = False):
    #Only one response in any board id can approve for cfo screen dashboard
    #Or we can have a logic by which we always show the latest one
    #IN the backend i can implment the logic which always show the latest one
    forecast_response = forecast_response_repo.update_publish_to_cfo(forecast_response_id, cfo_publish)
    if not forecast_response:
        raise HTTPException(status_code=404, detail="Forecast response not found")
    return {
        "message":"Sucessfully updated the status"
    }

#To do , it is working with all id , may be need a fix to do at specific level
@router.get("/get_first_filter/{board_id}", response_model=dict)
def get_first_filter_category(board_id: int):
    """
    Retrieve the first filter category for forecast based on board ID.
    """
    forecast_category = ["OVERALL", "Bill Type"] #Item
    return {"forecast_first_category": forecast_category}

@router.get("/get_second_filter/{board_id}/{category}", response_model=dict)
def get_second_filter_category(board_id: int, category: str):
    """
    Retrieve the unique values for the specified category from concatenated data.
    """
    combined_contents, dataframes_list, table_name_list = prompt_repository.get_file_download_links_by_board_id(board_id)
    concatenated_df = pd.concat(dataframes_list, ignore_index=True)
    
    if category in ['OVERALL']:
        return {"forecast_second_category": "OVERALL"}
    # Check if the category exists in the concatenated DataFrame
    elif category not in concatenated_df.columns:
        return {"error": "Category not found in data"}
    
    unique_values = concatenated_df[category].unique().tolist()
    cleaned_list = list(filter(lambda x: not pd.isna(x), unique_values))
    return {"forecast_second_category": cleaned_list}

def normalize_data(X_train, normalization='minmax'):
    """
    Normalize the input features using different normalization techniques.

    Parameters:
        X_train (array-like): The input features for training.
        normalization (str): The type of normalization technique to use.
                             Possible values: 'standard', 'minmax', 'robust'.
                             Default is 'minmax'.

    Returns:
        X_train_normalized (array-like): The normalized input features.
    """
    scaler_map = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }
    scaler = scaler_map.get(normalization, MinMaxScaler())
    X_train_normalized = scaler.fit_transform(X_train)
    return X_train_normalized

def train_model_level2(X_train_level2, y_train_level2):
    """Train the level-2 linear regression model."""
    X_train_scaled = normalize_data(X_train_level2)
    model_level2 = LinearRegression()
    
    model_level2.fit(X_train_scaled, y_train_level2)
    return model_level2

def get_coefficients(model, features):
    """Extract coefficients from the trained model."""
    coef_dict = {'Intercept': model.intercept_}
    coef_dict.update({features[i]: model.coef_[i] for i in range(len(features))})
    prediction_coef = pd.DataFrame(list(coef_dict.items()), columns=['names', 'coef'])
    abs_sum_coef = prediction_coef['coef'].abs().sum()
    #prediction_coef['coef_scaled'] = prediction_coef['coef'] / abs_sum_coef if abs_sum_coef != 0 else 0 
    # Check if sum of absolute values of coefficients is zero to avoid division by zero error
    if abs_sum_coef != 0:
        prediction_coef['coef_scaled'] = prediction_coef['coef'] / abs_sum_coef
    else:
        prediction_coef['coef_scaled'] = 0 
    return prediction_coef, abs_sum_coef

def preprocess_data(df_new, date_variable, independent_variables, dependent_variable, mode='train'):
    """
    Preprocess the input data by resampling it monthly, extracting month and year features,
    and splitting it into input features and target variable.

    Parameters:
        df_new (DataFrame): The input dataframe.
        independent_variables (list): List of independent variable column names.
        dependent_variable (str): The dependent variable column name.
        mode (str, optional): Indicates whether it's training mode or not. Default is 'train'.

    Returns:
        X_train_level2 (DataFrame): Input features dataframe.
        y_train_level2 (DataFrame or None): Target variable dataframe if in training mode, else None.
    """
    # Make a copy of the input dataframe to avoid modifying the original dataframe
    df_imputed = df_new.copy()
    
    # Convert 'Sales_Invoice_Date' column to datetime format and set it as the index
    df_imputed[date_variable] = pd.to_datetime(df_imputed[date_variable])
    df_imputed.set_index(date_variable, inplace=True)

    # Resample the data monthly and aggregate based on the columns
    total_columns = independent_variables + [dependent_variable] if mode == 'train' else independent_variables
    grouped_data = pd.DataFrame()
    for column in total_columns:
        if column == 'SELLING_RATE':
            grouped_data[column] = df_imputed.resample('M')[column].mean()
        else:
            grouped_data[column] = df_imputed.resample('M')[column].sum()

    # Reset index and extract month and year features
    grouped_data.reset_index(inplace=True)
    grouped_data['Month'] = grouped_data[date_variable].dt.month
    grouped_data['Year'] = grouped_data[date_variable].dt.year
    
    # Select input features and target variable
    X_train_level2 = grouped_data[independent_variables + ['Month', 'Year']]
    y_train_level2 = grouped_data[dependent_variable] if mode == 'train' else None
    
    return X_train_level2, y_train_level2

def predict_variables(data: pd.DataFrame, date_variable: str, independent_variables: list, n_steps: int, forecast_start_date: str) -> pd.DataFrame:
    final_dataframe = pd.DataFrame()
    for column in independent_variables:
        temp_data = data[[date_variable, column]]
        #temp_data.drop_duplicates(keep=False, inplace=True)
        temp_data[date_variable] = pd.to_datetime(temp_data[date_variable])
        temp_data['YearMonth'] = temp_data[date_variable].dt.to_period('M')
        grouped_data = temp_data.groupby(['YearMonth']).agg({column: 'mean' if column == 'SELLING_RATE' else 'sum'}).reset_index()
        try:
            model = auto_arima(grouped_data[column], seasonal=True, m=12, n_jobs=-1)
            forecast = model.predict(n_periods=n_steps)
        except:
            forecast = [0] * n_steps
        forecast_dates = pd.date_range(start=forecast_start_date, periods=n_steps, freq='MS')
        forecast_df = pd.DataFrame({date_variable: forecast_dates, column: forecast})
        final_dataframe = pd.merge(final_dataframe, forecast_df, on=date_variable, how='left') if not final_dataframe.empty else forecast_df
    return final_dataframe

def prepare_actual_data(actual_data, pred_sale_value_df, date_column, independent_variable, dependent_variable):
    actual_data['YearMonth'] = pd.to_datetime(actual_data[date_column]).dt.strftime('%B-%Y')
    agg_dict = {independent_variable: 'sum', 
                #'SELLING_RATE': 'mean' if independent_variable != 'SELLING_RATE' else 'sum', 
                dependent_variable: 'sum'}
    actual_data_monthly = actual_data.groupby(['YearMonth']).agg(agg_dict).reset_index()
    pred_sale_value_df[dependent_variable] = actual_data_monthly[dependent_variable]
    pred_sale_value_df.fillna(0, inplace=True)
    return pred_sale_value_df, actual_data_monthly

def dataframe_to_json_format(df):
    # Extract column names
    columns = df.columns.tolist()
    
    # Extract data values as a list of lists
    data = df.values.tolist()
    
    # Create dictionary in the specified format
    json_format = {
        "table": {
            "columns": columns,
            "data": data
        }
    }
    
    return json_format

def generate_line_chart(df, dependent_column, catgory_column):
    """
    Generate line chart structure from the given dataframe.

    Parameters:
        df (pd.DataFrame): Input dataframe with columns "YearMonth", "Predicted_Sale_Value", "Actual Data".

    Returns:
        dict: Line chart structure.
    """
    # Prepare data for line chart
    labels = df["YearMonth"].tolist()
    categories = [catgory_column, dependent_column]
    values = df[categories].values.tolist()
    isStacked = False  # Change to True if you want stacked chart

    # Line chart structure
    line_chart = {
        "chart_type": "line",
        "data_format": {
            "labels": labels,
            "categories": categories,
            "values": values,
            "isStacked": isStacked
        }
    }

    return line_chart

def split_and_transform_dataframes(original_data):
    item_dfs = list()
    dfs = list()
    data = copy.deepcopy(original_data)

    for subRows_item in data:
        for item in subRows_item['subRows']:
            item_dfs.append(item)
        del subRows_item['subRows']
        dfs.append(subRows_item)
                
    total_dfs = pd.DataFrame(dfs)
    total_dfs.set_index('Label', inplace=True)
    total_dfs = total_dfs.T
    item_dfs = pd.DataFrame(item_dfs)
    item_dfs.set_index('Label', inplace=True)
    item_dfs = item_dfs.T

    return total_dfs, item_dfs

def generate_comparison_line_chart(df):
    # Extract label and category columns
    labels = df.index.tolist()  # Use index instead of 'Label' column
    categories = df.columns.tolist()
    values = df.values.tolist()

    # Line chart structure
    line_chart = {
        "chart_type": "line",
        "data_format": {
            "labels": labels,
            "categories": categories,
            "values": values,
            "isStacked": False  # Change to True if you want stacked chart
        }
    }

    return line_chart

def validate_request_body(request_body):
    if "coefficients_json" in request_body and "predicted_data_json" in request_body:
        coefficients_json = request_body["coefficients_json"]
        predicted_data_json = request_body["predicted_data_json"]

        if "table" in coefficients_json and "table" in predicted_data_json:
            coefficients_data = coefficients_json["table"]["data"]
            predicted_data_data = predicted_data_json["table"]["data"]

            if coefficients_data and predicted_data_data:
                return True
            else:
                return False
        else:
            return False
    else:
        return False
    
        

@router.post("/save_forecast")
async def save_forecast_data(
    payload: list[RequestBody],
    board_id: int,
    first_level_filter: str,
    forecast_period: int,
    forecast_name: str,
    second_level_filter: str = None): 
    # Serialize the payload to JSON
    # Create ForecastResponse instance
    forecast_response = ForecastResponse(
        board_id=board_id,
        name=forecast_name,
        first_level_filter=first_level_filter,
        second_level_filter=second_level_filter,
        forecast_period=forecast_period,
        output_response=json.dumps([item.dict() for item in payload])
    )
    saved_forecast_response = forecast_response_repo.create_forecast_response(forecast_response)

    return {"message": "Forecast data saved successfully."}

def round_json(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = round_json(value)
        return obj
    elif isinstance(obj, list):
        return [round_json(item) for item in obj]
    elif isinstance(obj, (int, float)):
        return round(obj, 2)
    else:
        return obj

def add_rupee_sign(obj):
    if isinstance(obj, dict):
        return {key: add_rupee_sign(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [add_rupee_sign(item) for item in obj]
    elif isinstance(obj, (int, float)):
        return '₹ ' + '{:,.2f}'.format(obj)
    else:
        return obj
    
def generate_forecast_dict(info, sale_value_column='Sale_Value'):
    final_df = pd.DataFrame()
    for each_item in info:
        data = each_item['actual_vs_forecast']
        # Extract column names and data
        columns = data['table']['columns']
        table_data = data['table']['data']
        df = pd.DataFrame(table_data, columns=columns)
        df['label'] = each_item['label']
        final_df = pd.concat([final_df, df], axis=0, ignore_index=True)
        
    final_df = final_df.round(2)
    result = final_df.groupby('YearMonth').agg({sale_value_column: 'sum'}).reset_index()
    result['YearMonth'] = pd.to_datetime(result['YearMonth'])  # Convert to datetime
    result.sort_values('YearMonth', inplace=True)
    result['YearMonth'] = result['YearMonth'].dt.strftime('%B-%Y')
    result.set_index('YearMonth', inplace=True)
    final_dict = {
        "Label": result.T.index[0]
    }

    for index, row in result.iterrows():
        final_dict[index] = row[sale_value_column]

    sub_rows = list()
    for label in final_df['label'].unique():
        label_df = final_df[final_df['label'] == label]
        result = label_df.groupby('YearMonth').agg({sale_value_column: 'sum'}).reset_index()
        result['YearMonth'] = pd.to_datetime(result['YearMonth'])  # Convert to datetime
        result.sort_values('YearMonth', inplace=True)
        result['YearMonth'] = result['YearMonth'].dt.strftime('%B-%Y')
        result.set_index('YearMonth', inplace=True)
        result = result.rename(columns={sale_value_column: label})

        output_dict = {
            "Label": result.T.index[0]
        }

        for index, row in result.iterrows():
            output_dict[index] = row[label]
        sub_rows.append(output_dict)
    final_dict['subRows'] = sub_rows
    return final_dict

def replace_zero_with_hyphen(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = replace_zero_with_hyphen(value)
        return obj
    elif isinstance(obj, list):
        return [replace_zero_with_hyphen(item) for item in obj]
    elif obj == 0:
        return '-'
    else:
        return obj

def replace_hyphen_with_zero(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "values":
                data[key] = [[0 if v == "-" else v for v in row] for row in value]
            else:
                data[key] = replace_hyphen_with_zero(value)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            data[i] = replace_hyphen_with_zero(item)
    return data

def generate_output_response(final_result, dependent_variable, column_name, budget):
    # Generate forecast dictionaries
    sale_value_dict = generate_forecast_dict(final_result, dependent_variable)
    forecast_dict = generate_forecast_dict(final_result, column_name)
    
    # Format dictionaries
    sale_value_dict_format = replace_zero_with_hyphen(sale_value_dict)
    sale_value_dict_format = add_rupee_sign(sale_value_dict_format) 
    forecast_dict_format = add_rupee_sign(forecast_dict)
    
    # Prepare output response
    output_response = {
        'item_metadata': final_result,
        "hierarchy_table": {"data":[sale_value_dict_format, forecast_dict_format],
                    "title":"Actual vs Forecast Revenue"},
    }
    
    # Split and transform dataframes
    out = [sale_value_dict, forecast_dict]
    total_dfs, item_dfs = split_and_transform_dataframes(out)
    
    # Generate comparison line charts
    total_dfs_line_chart = generate_comparison_line_chart(total_dfs)
    item_dfs_line_chart = generate_comparison_line_chart(item_dfs)
    
    #Add Title
    total_dfs_line_chart["title"] = "Actual VS Forecast line chart: Total level"
    item_dfs_line_chart["title"] = "Actual VS Forecast line chart: Item level"
    
    # Add budget variable to the total level line chart annotation
    total_dfs_line_chart["annotation"] = {
        "label": "Target",
        "value": budget,
    }
    
    # Update output response with line charts
    output_response["total_level_line_chart"] = total_dfs_line_chart
    output_response["item_level_line_chart"] = item_dfs_line_chart
    
    return output_response

@router.post("/run_forecast_hybrid_v2")
async def run_forecast_hybrid_v2(
    board_id: int,
    first_level_filter: str,
    forecast_period: int,
    second_level_filter: str = None,
    name: str = None,
    edit_forecast_run: bool = False,
    payload: list[RequestBody] = None
):
    forecast_settings = forecast_settings_repo.get_forecast_settings_by_board_id(board_id)
    financial_year_start = forecast_settings.financial_year_start
    date_variable = forecast_settings.date_variable
    independent_variables = forecast_settings.independent_variables
    dependent_variable = forecast_settings.dependent_variable
    budget = forecast_settings.budget    

    # Create Mapping for independet variable
    mapping = {
        "Invoice Date": "Sales_Invoice_Date",
        "Quantity": "QTY",
        "Discounted Quantity": "QTY_DISC",
        "Net Quantity": "Net_Qty",
        "Selling Rate": "SELLING_RATE"
    }

    # Reverse mapping
    reverse_mapping = {v: k for k, v in mapping.items()}

    logging.info(f"Getting information from Database for Board id {board_id}")
    
    # Step 1: Retrieve dataframes list and concatenate them
    combined_contents, dataframes_list, table_name_list = prompt_repository.get_file_download_links_by_board_id(board_id)
    all_data = pd.concat(dataframes_list, ignore_index=True)
    column_name = 'Forecast ' + dependent_variable
    # Step 2: Filter data based on the second level filter if provided
    if (second_level_filter is not None and first_level_filter != 'OVERALL') or (second_level_filter is None and first_level_filter == 'OVERALL'):
        #Code block for actual and concatenated_df data
        if second_level_filter is not None and first_level_filter != 'OVERALL':
            actual_data = all_data[(all_data[first_level_filter] == second_level_filter) & (all_data[date_variable] >= financial_year_start)]
            concatenated_df = all_data[(all_data[first_level_filter] == second_level_filter) & (all_data[date_variable] < financial_year_start)]
        elif second_level_filter is None and first_level_filter == 'OVERALL':
            actual_data = all_data[(all_data[date_variable] >= financial_year_start)]
            concatenated_df = all_data[(all_data[date_variable] < financial_year_start)]
            
        #Code block to use edit_forecast_run/default code flow
        if edit_forecast_run:
            coefficients = pd.DataFrame(payload[0].weight_coefficients.table.data, columns=payload[0].weight_coefficients.table.columns)
            predicted_data = pd.DataFrame(payload[0].independent_variable.table.data, columns=payload[0].independent_variable.table.columns)
            predicted_data.rename(columns=mapping, inplace=True)
            abs_sum_coef = payload[0].abs_sum_coef
            label = payload[0].label
            coefficients['coef'] = coefficients['coef_scaled'] * abs_sum_coef
            coefficients['names'] = coefficients['names'].map(mapping).fillna(coefficients['names'])

        else:
            logging.info("Filter data on first and second filter  levels")
            
            # Step 3: Generate predicted data
            predicted_data = predict_variables(concatenated_df, date_variable, independent_variables, forecast_period, financial_year_start)
            logging.info("Creating independent variable for the future forecast period")
            
            # Step 4: Preprocess training data
            X_train_level2, y_train_level2 = preprocess_data(concatenated_df, date_variable, independent_variables, dependent_variable)

            # Step 5: Train Model Level 2
            model_level2 = train_model_level2(X_train_level2, y_train_level2)

            # Step 6: Extract coefficients
            coefficients, abs_sum_coef = get_coefficients(model_level2, independent_variables + ['Month', 'Year'])

        # Step 7: Preprocess test data
        X_test_level2, _ = preprocess_data(predicted_data, date_variable, independent_variables, dependent_variable, mode='test')
        X_test_level2_df = pd.DataFrame(normalize_data(X_test_level2), columns=independent_variables + ['Month', 'Year'])

        # Step 8: Predict next month sale value
        coef_dict = coefficients.set_index('names')['coef'].to_dict()
        predicted_next_month_sale_value = []
        for _, row in X_test_level2_df.iterrows():
            out = coef_dict['Intercept']
            for col in independent_variables + ['Month', 'Year']:
                out += row[col] * coef_dict.get(col, 0)
            predicted_next_month_sale_value.append(out)

        # Step 8.1:We are updating logic here and getting output using decision tree model
        X_train_level2, y_train_level2 = preprocess_data(concatenated_df, date_variable, independent_variables, dependent_variable)
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train_level2, y_train_level2)
        predicted_next_month_sale_value = model.predict(X_test_level2)

        # Step 9: Prepare predicted sale value dataframe
        pred_sale_value_df = pd.DataFrame({'YearMonth': pd.to_datetime(predicted_data[date_variable]).dt.strftime('%B-%Y'), column_name: predicted_next_month_sale_value})
        pred_sale_value_df, _ = prepare_actual_data(actual_data, pred_sale_value_df, date_variable, independent_variables[0], dependent_variable)

        # Step 10: Convert dataframes to JSON format
        pred_sale_value_df = pred_sale_value_df.round(2)
        coefficients = coefficients.round(2)
        pred_sale_value_df_json = dataframe_to_json_format(pred_sale_value_df)
        coefficients['names'] = coefficients['names'].map(reverse_mapping).fillna(coefficients['names'])
        coefficients_json = dataframe_to_json_format(coefficients[['names','coef_scaled']])

        #Step 10.1 Format data to month-year for predicted_data
        #Step 10.2 Apply reverse mapping for columns
        predicted_data[date_variable] = pd.to_datetime(predicted_data[date_variable]).dt.strftime('%B-%Y')
        predicted_data.rename(columns=reverse_mapping, inplace=True)
        predicted_data = predicted_data.round(2)
        predicted_data_json = dataframe_to_json_format(predicted_data)
        predicted_data_json["table"]["editable_columns"] = ['Disc. Amt']
        
        # Step 11: Generate line chart
        line_chart = generate_line_chart(pred_sale_value_df, dependent_variable, column_name)

        #Step 12 Add title to table
        pred_sale_value_df_json["table"]["title"] = "Actual vs Forecast Data"
        line_chart["title"] = "Actual VS Forecast Line chart"
        coefficients_json["table"]["title"] = "Predicted Variable Coefficients"
        predicted_data_json["table"]["title"] = "Predicted Data"

        #Update label
        if first_level_filter == "OVERALL":
            label = "OVERALL"
        elif second_level_filter is not None:
            label = second_level_filter
            
        result_dict = {
        "actual_vs_forecast": pred_sale_value_df_json,
        "actual_vs_forecast_chart": line_chart,
        "weight_coefficients": coefficients_json,
        "independent_variable": predicted_data_json,
        "abs_sum_coef": abs_sum_coef,
        "label": label}
        logging.info(f"Forecast Done for {label}")
        final_result = list()
        final_result.append(result_dict)
        output_response = generate_output_response(final_result, dependent_variable, column_name, budget)
        return round_json(output_response)
    
    elif second_level_filter is None and first_level_filter in all_data.columns.tolist():
        column_name = 'Forecast ' + dependent_variable
        final_result = list()
        unique_values = all_data[first_level_filter].unique().tolist()
        cleaned_list = list(filter(lambda x: not pd.isna(x), unique_values))
        for index, label in enumerate(cleaned_list):
            actual_data = all_data[(all_data[first_level_filter] == label) & (all_data[date_variable] >= financial_year_start)]
            concatenated_df = all_data[(all_data[first_level_filter] == label) & (all_data[date_variable] < financial_year_start)]
            logging.info("Filter data on first and second filter  levels")
            
            # Code block to use edit_forecast_run/default code flow
            if edit_forecast_run:
                logging.info("Using Pre computed data with user edit to find the updated value")
                coefficients = pd.DataFrame(payload[index].weight_coefficients.table.data, columns=payload[index].weight_coefficients.table.columns)
                predicted_data = pd.DataFrame(payload[index].independent_variable.table.data, columns=payload[index].independent_variable.table.columns)
                predicted_data.rename(columns=mapping, inplace=True)
                abs_sum_coef = payload[index].abs_sum_coef
                label = payload[index].label
                coefficients['coef'] = coefficients['coef_scaled'] * abs_sum_coef
                coefficients['names'] = coefficients['names'].map(mapping).fillna(coefficients['names'])
            else:
                logging.info("Running full forecast flow")
                
                # Step 3: Generate predicted data
                predicted_data = predict_variables(concatenated_df, date_variable, independent_variables, forecast_period, financial_year_start)
                logging.info("Creating independent variable for the future forecast period")
                
                # Step 4: Preprocess training data
                X_train_level2, y_train_level2 = preprocess_data(concatenated_df, date_variable, independent_variables, dependent_variable)

                # Step 5: Train Model Level 2
                model_level2 = train_model_level2(X_train_level2, y_train_level2)

                # Step 6: Extract coefficients
                coefficients, abs_sum_coef = get_coefficients(model_level2, independent_variables + ['Month', 'Year'])

            # Step 7: Preprocess test data
            X_test_level2, _ = preprocess_data(predicted_data, date_variable, independent_variables, dependent_variable, mode='test')
            X_test_level2_df = pd.DataFrame(normalize_data(X_test_level2), columns=independent_variables + ['Month', 'Year'])

            # Step 8: Predict next month sale value
            coef_dict = coefficients.set_index('names')['coef'].to_dict()
            predicted_next_month_sale_value = []
            for _, row in X_test_level2_df.iterrows():
                out = coef_dict['Intercept']
                for col in independent_variables + ['Month', 'Year']:
                    out += row[col] * coef_dict.get(col, 0)
                predicted_next_month_sale_value.append(out)

            # Step 8.1:We are updating logic here and getting output using decision tree model
            X_train_level2, y_train_level2 = preprocess_data(concatenated_df, date_variable, independent_variables, dependent_variable)
            model = DecisionTreeRegressor(random_state=42)
            model.fit(X_train_level2, y_train_level2)
            predicted_next_month_sale_value = model.predict(X_test_level2)
                

            # Step 9: Prepare predicted sale value dataframe
            pred_sale_value_df = pd.DataFrame({'YearMonth': pd.to_datetime(predicted_data[date_variable]).dt.strftime('%B-%Y'), column_name: predicted_next_month_sale_value})
            pred_sale_value_df, _ = prepare_actual_data(actual_data, pred_sale_value_df, date_variable, independent_variables[0], dependent_variable)

            # Step 10: Convert dataframes to JSON format and round upto 2
            pred_sale_value_df = pred_sale_value_df.round(2)
            coefficients = coefficients.round(2)
            pred_sale_value_df_json = dataframe_to_json_format(pred_sale_value_df)
            coefficients['names'] = coefficients['names'].map(reverse_mapping).fillna(coefficients['names'])
            coefficients_json = dataframe_to_json_format(coefficients[['names','coef_scaled']])
            
            #Step 10.1 Format data to month-year for predicted_data
            #Step 10.2 Apply reverse mapping for columns
            predicted_data[date_variable] = pd.to_datetime(predicted_data[date_variable]).dt.strftime('%B-%Y')
            predicted_data.rename(columns=reverse_mapping, inplace=True)
            predicted_data = predicted_data.round(2)
            predicted_data_json = dataframe_to_json_format(predicted_data)
            predicted_data_json["table"]["editable_columns"] = ["Discounted Quantity", "Selling Rate"]
            # Step 11: Generate line chart
            line_chart = generate_line_chart(pred_sale_value_df, dependent_variable, column_name)
            
            #Step 12: Add title to table
            pred_sale_value_df_json["table"]["title"] = "Actual vs Forecast Data"
            line_chart["title"] = "Actual VS Forecast Line chart"
            coefficients_json["table"]["title"] = "Predicted Variable Coefficients"
            predicted_data_json["table"]["title"] = "Predicted Data"

            result_dict = {
            "actual_vs_forecast": pred_sale_value_df_json,
            "actual_vs_forecast_chart": line_chart,
            "weight_coefficients": coefficients_json,
            "independent_variable": predicted_data_json,
            "abs_sum_coef": abs_sum_coef,
            "label": label}
            logging.info(f"Forecast Done for {label}")
            
            final_result.append(result_dict)
        output_response = generate_output_response(final_result, dependent_variable, column_name, budget)
        return round_json(output_response)
    else:
        return {
            "Error": "Please Select the correct First Level Filter"
        }


@router.post("/chat_integration")
async def chat_integration(
    board_id: int,
    forecast_response_id: int,
    first_level_filter: str,
    forecast_period: int,
    input_text:str,
    second_level_filter: str = None,
    name: str = None,
):
    try:
        response_content = "Please review and modify the prompt with more specifics."
        start_time = datetime.now()
        
        forecast_settings = forecast_settings_repo.get_forecast_settings_by_board_id(board_id)
        financial_year_start = forecast_settings.financial_year_start
        date_variable = forecast_settings.date_variable
        independent_variables = forecast_settings.independent_variables
        dependent_variable = forecast_settings.dependent_variable
        budget = forecast_settings.budget    

        # Create Mapping for independent variable
        mapping = {
            "Invoice Date": "Sales_Invoice_Date",
            "Quantity": "QTY",
            "Discounted Quantity": "QTY_DISC",
            "Net Quantity": "Net_Qty",
            "Selling Rate": "SELLING_RATE"
        }

        # Reverse mapping
        reverse_mapping = {v: k for k, v in mapping.items()}

        logging.info(f"Getting information from Database for Board id {board_id}")
        
        # Step 1: Retrieve dataframes list and concatenate them
        combined_contents, dataframes_list, table_name_list = prompt_repository.get_file_download_links_by_board_id(board_id)
        all_data = pd.concat(dataframes_list, ignore_index=True)
        column_name = 'Forecast ' + dependent_variable
        
        
        # Generate hash key based on CSV file content and input text
        hash_key = forecast_chat_response_repo.generate_hash_key(combined_contents, input_text)

        # Check if the response is already present in the Prompt_response table
        existing_response = await forecast_chat_response_repo.check_existing_response(hash_key)

        if existing_response:
            # If response is already present, return the existing response
            return JSONResponse(content=existing_response.response_content)
        
            # Step 2: Filter data based on the second level filter if provided
        if (second_level_filter is not None and first_level_filter != 'OVERALL') or (second_level_filter is None and first_level_filter == 'OVERALL'):
            #Code block for actual and concatenated_df data
            if second_level_filter is not None and first_level_filter != 'OVERALL':
                actual_data = all_data[(all_data[first_level_filter] == second_level_filter) & (all_data[date_variable] >= financial_year_start)]
                concatenated_df = all_data[(all_data[first_level_filter] == second_level_filter) & (all_data[date_variable] < financial_year_start)]
            elif second_level_filter is None and first_level_filter == 'OVERALL':
                actual_data = all_data[(all_data[date_variable] >= financial_year_start)]
                concatenated_df = all_data[(all_data[date_variable] < financial_year_start)]
        elif second_level_filter is None and first_level_filter is not None:
            actual_data = all_data[(all_data[date_variable] >= financial_year_start)]
            concatenated_df = all_data[(all_data[date_variable] < financial_year_start)]
    
        forecast_response = forecast_response_repo.get_forecast_response(forecast_response_id)
        false = False
        final_result = eval(forecast_response.output_response)
        output_response = generate_output_response(final_result, dependent_variable, column_name, budget)
        output_response = round_json(output_response)
            
        # Initialize an empty list to store DataFrames
        coefficients_list = []
        predicted_data_list = []

        # Iterate over each element in 'payload'
        for element in output_response['item_metadata']:
            # Extracting coefficient DataFrame
            coefficients_data = element['weight_coefficients']['table']['data']
            coefficients_columns = element['weight_coefficients']['table']['columns']
            coefficients = pd.DataFrame(coefficients_data, columns=coefficients_columns)
            
            # Extracting predicted data DataFrame
            predicted_data_data = element['independent_variable']['table']['data']
            predicted_data_columns = element['independent_variable']['table']['columns']
            predicted_data = pd.DataFrame(predicted_data_data, columns=predicted_data_columns)
            
            # Renaming columns of predicted data using 'mapping'
            predicted_data.rename(columns=mapping, inplace=True)
            
            # Calculating 'abs_sum_coef' and 'label'
            abs_sum_coef = element["abs_sum_coef"]
            label = element["label"]
            
            # Performing necessary operations on coefficients DataFrame
            coefficients['coef'] = coefficients['coef_scaled'] * abs_sum_coef
            coefficients['names'] = coefficients['names'].map(mapping).fillna(coefficients['names'])
            
            coefficients['label'] = label
            predicted_data['label'] = label
            
            # Appending DataFrames to the respective lists
            coefficients_list.append(coefficients)
            predicted_data_list.append(predicted_data)
            
        coefficients = pd.concat(coefficients_list, ignore_index=True)
        predicted_data = pd.concat(predicted_data_list, ignore_index=True)
    
        llm = ChatOpenAI(temperature=0, model="gpt-4")
        #Here we are defining the data
        dataframes_list = [all_data, actual_data, predicted_data, coefficients]
        # dl = SmartDatalake(dataframes_list, config={"llm": llm})
        agent = Agent(dataframes_list, config={"llm": llm, "verbose":True, "enable_cache": False, "max_retries":10})
        #input_text = get_planner_instruction(input_text)
        rephrased_query = agent.rephrase_query(input_text)
        response_content = agent.chat(rephrased_query)
        graph_output_json = {}

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
            
        # Save the response to the Prompt_response table
        forecast_chat_response_repo.save_response_to_database(board_id, 
                                                              forecast_response_id, 
                                                              input_text,
                                                              first_level_filter, 
                                                              second_level_filter, 
                                                              forecast_period,
                                                              name, 
                                                              result, 
                                                              hash_key)
            
        return JSONResponse(content=result)
    except Exception as ex:
        # Handle exceptions and return an error response if needed
        logger.error(f"Internal fails {ex}")
        return JSONResponse(content={"error": "Prompt Error","detail":response_content}, status_code=500)
    
    