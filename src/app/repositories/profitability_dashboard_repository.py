import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
from typing import List, Dict, Tuple

def convert_to_standard_start_date(date_str):
    """
    Convert the given date string to the standard start date format.
    
    Args:
    - date_str (str): The date string in the format 'YYYY-MM-DDTHH:MM:SS.SSSZ'.
    
    Returns:
    - str: The standard start date string in the format 'YYYY-MM-DD'.
    """
    # Parse the string into a datetime object
    date_obj = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%fZ')

    # Format the datetime object into the standard start date format
    standard_start_date = date_obj.strftime('%Y-%m-%d')
    
    return standard_start_date

def  create_actual_vs_forecast_revenue(actual_data, forecast_data, start_date='2023-10-01'):
    actual_data['Month_Year'] = pd.to_datetime(actual_data['Month_Year'])
    filter_actual_data = actual_data[actual_data['Month_Year'] >= start_date]
    max_actual_date = filter_actual_data['Month_Year'].max()
    filter_actual_data = filter_actual_data.groupby('Item').agg({'Sale_Value': 'sum'}).reset_index()
    
    forecast_data['Month_Year'] = pd.to_datetime(forecast_data['Month_Year'])
    forecast_data = forecast_data[(forecast_data['Month_Year'] >= start_date) & (forecast_data['Month_Year'] <= max_actual_date)]
    forecast_data = forecast_data.groupby('Item').agg({'Sale_Value': 'sum'}).reset_index()
    
    #To do : This Budget should come from some datasource
    forecast_data['Budget'] = forecast_data['Sale_Value']
    
    total_forecast = forecast_data['Sale_Value'].sum()
    total_actual = filter_actual_data['Sale_Value'].sum()
    sub_rows_list = forecast_data.rename(columns={'Label': 'Item', 'Sale_Value': 'Forecast', 'Budget':'Budget'}).to_dict(orient='records')
    for sub_row in sub_rows_list:
        label = sub_row['Item']
        actual_value = filter_actual_data[filter_actual_data['Item'] == label]['Sale_Value'].iloc[0]
        sub_row['Actual'] = actual_value
    json_data = {
        'Label': 'Revenue',
        'Forecast': total_forecast,
        'Budget': total_forecast,
        'Actual': total_actual,
        'subRows': sub_rows_list
    }
    return json_data
    
#Step2 : COGS Calculation
#others_forecast_df = create_forecast_others(combined_df)
def create_actual_vs_forecast_cogs(actual_data, forecast_data, start_date='2023-10-01'):
    actual_data['Month_Year'] = pd.to_datetime(actual_data['Month_Year'])
    filter_actual_data = actual_data[actual_data['Month_Year'] >=  start_date]
    max_actual_date = filter_actual_data['Month_Year'].max()
    filter_actual_data = filter_actual_data[['packing_expenses','direct_labor','manufacturing_overheads','Raw_Materials']]
    filter_actual_data['cogs'] = filter_actual_data.sum(axis=1)
    
    forecast_data['Month_Year'] = pd.to_datetime(forecast_data['Month_Year'])
    forecast_data = forecast_data[(forecast_data['Month_Year'] >= start_date) & (forecast_data['Month_Year'] <= max_actual_date)]
    forecast_data = forecast_data[['packing_expenses','direct_labor','manufacturing_overheads','Raw_Materials']]
    forecast_data['cogs'] = forecast_data.sum(axis=1)

    total_actual = filter_actual_data['cogs'].sum()
    total_forecast = forecast_data['cogs'].sum()
    sub_rows_list = [{'Label':'Packing Expenses', 'Forecast':forecast_data['packing_expenses'].sum(), 'Budget':forecast_data['packing_expenses'].sum() , 'Actual':filter_actual_data['packing_expenses'].sum()},
                    {'Label':'Direct Labor', 'Forecast':forecast_data['direct_labor'].sum(),'Budget':forecast_data['direct_labor'].sum(), 'Actual':filter_actual_data['direct_labor'].sum()},
                    {'Label':'Manufacturing Overheads', 'Forecast':forecast_data['manufacturing_overheads'].sum(),'Budget':forecast_data['manufacturing_overheads'].sum(), 'Actual':filter_actual_data['manufacturing_overheads'].sum()},
                    {'Label':'Raw Materials', 'Forecast':forecast_data['Raw_Materials'].sum(), 'Budget':forecast_data['Raw_Materials'].sum(), 'Actual':filter_actual_data['Raw_Materials'].sum()}]
    json_data = {
        'Label': 'COGS',
        'Forecast': total_forecast,
        'Budget': total_forecast,
        'Actual': total_actual,
        'subRows': sub_rows_list
    }
    return json_data

#Step 3: Other Operating Expenses
def create_actual_vs_forecast_other_operating_expenses(actual_data, forecast_data, start_date='2023-10-01'):
    actual_data['Month_Year'] = pd.to_datetime(actual_data['Month_Year'])
    filter_actual_data = actual_data[actual_data['Month_Year'] >=  start_date]
    max_actual_date = filter_actual_data['Month_Year'].max()
    filter_actual_data = filter_actual_data[['distribution_expenses','marketing_expenses']]
    filter_actual_data['other_operating_expenses'] = filter_actual_data.sum(axis=1)
    
    forecast_data['Month_Year'] = pd.to_datetime(forecast_data['Month_Year'])
    forecast_data = forecast_data[(forecast_data['Month_Year'] >= start_date) & (forecast_data['Month_Year'] <= max_actual_date)]
    forecast_data = forecast_data[['distribution_expenses','marketing_expenses']]
    forecast_data['other_operating_expenses'] = forecast_data.sum(axis=1)

    total_actual = filter_actual_data['other_operating_expenses'].sum()
    total_forecast = forecast_data['other_operating_expenses'].sum()
    sub_rows_list = [{'Label':'Distribution Expenses', 'Forecast':forecast_data['distribution_expenses'].sum(), 'Budget': forecast_data['distribution_expenses'].sum(), 'Actual':filter_actual_data['distribution_expenses'].sum()},
                    {'Label':'Marketing Expenses', 'Forecast':forecast_data['marketing_expenses'].sum(),'Budget':forecast_data['marketing_expenses'].sum() , 'Actual':filter_actual_data['marketing_expenses'].sum()}]
    json_data = {
        'Label': 'Other Operating Expenses',
        'Forecast': total_forecast,
        'Budget': total_forecast,
        'Actual': total_actual,
        'subRows': sub_rows_list
    }
    return json_data

#Step 4: Other Administrative Expenses
def create_actual_vs_forecast_other_administrative_expenses(actual_data, forecast_data, start_date='2023-10-01'):
    actual_data['Month_Year'] = pd.to_datetime(actual_data['Month_Year'])
    filter_actual_data = actual_data[actual_data['Month_Year'] >=  start_date]
    max_actual_date = filter_actual_data['Month_Year'].max()
    filter_actual_data = filter_actual_data[['other_administrative_expenses']]
    
    forecast_data['Month_Year'] = pd.to_datetime(forecast_data['Month_Year'])
    forecast_data = forecast_data[(forecast_data['Month_Year'] >= start_date) & (forecast_data['Month_Year'] <= max_actual_date)]
    forecast_data = forecast_data[['other_administrative_expenses']]

    total_actual = filter_actual_data['other_administrative_expenses'].sum()
    total_forecast = forecast_data['other_administrative_expenses'].sum()
    sub_rows_list = []
    json_data = {
        'Label': 'Other Administrative Expenses',
        'Forecast': total_forecast,
        'Budget': total_forecast,
        'Actual': total_actual,
        'subRows': sub_rows_list
    }
    return json_data

#Step 5: Operating Profit
#operating_profit
def create_actual_vs_forecast_operating_profit(profitability_analysis_data):
    rows = []
    for entry in profitability_analysis_data:
        label = entry['Label']
        forecast = entry['Forecast']
        budget = entry['Budget']
        actual = entry['Actual']
        sub_rows = entry.get('subRows', [])
        for sub_row in sub_rows:
            row = {'Label': label}
            row.update(sub_row)
            row['Forecast'] = sub_row.get('Forecast', forecast)
            row['Budget'] = sub_row.get('Budget', budget)
            row['Actual'] = sub_row.get('Actual', actual)
            rows.append(row)
    all_data = pd.DataFrame(rows)
    # Separate revenue rows and other rows
    revenue_rows = all_data[all_data['Label'] == 'Revenue']
    other_rows = all_data[all_data['Label'] != 'Revenue']

    # Calculate total forecast and actual revenue
    total_forecast_revenue = revenue_rows['Forecast'].sum()
    total_budget_revenue = revenue_rows['Budget'].sum()
    total_actual_revenue = revenue_rows['Actual'].sum()

    # Calculate total forecast and actual expenses
    total_forecast_expenses = other_rows['Forecast'].sum()
    total_budget_expenses = other_rows['Budget'].sum()
    total_actual_expenses = other_rows['Actual'].sum()

    # Calculate operating profit for forecast and actual
    forecast_operating_profit = total_forecast_revenue - total_forecast_expenses
    budget_operating_profit = total_budget_revenue - total_budget_expenses
    actual_operating_profit = total_actual_revenue - total_actual_expenses
    json_data = {
        'Label': 'Operating Profit',
        'Forecast': forecast_operating_profit,
        'Budget': budget_operating_profit,
        'Actual': actual_operating_profit,
        'subRows': []
    }
    return json_data


def format_response(response):
    formatted_response = []
    for item in response:
        try:
            formatted_item = {}
            formatted_item['Label'] = item['Label']
            formatted_item['Forecast'] = '₹' + '{:,.2f}'.format(item['Forecast']) if item['Forecast'] != 0 else '-'
            formatted_item['Budget'] = '₹' + '{:,.2f}'.format(item['Budget']) if item['Budget'] != 0 else '-'
            formatted_item['Actual'] = '₹' + '{:,.2f}'.format(item['Actual']) if item['Actual'] != 0 else '-'
            formatted_item['Variance: Actual vs Forecast'] = '₹' + '{:,.2f}'.format(item['Variance: Actual vs Forecast']) if item['Variance: Actual vs Forecast'] != 0 else '-'
            formatted_item['Variance: Actual vs Budget'] = '₹' + '{:,.2f}'.format(item['Variance: Actual vs Budget']) if item['Variance: Actual vs Budget'] != 0 else '-'
            formatted_item['subRows'] = []
            for sub_item in item['subRows']:
                try:
                    formatted_sub_item = {}
                    formatted_sub_item['Label'] = sub_item['Label']
                    formatted_sub_item['Forecast'] = '₹' + '{:,.2f}'.format(sub_item['Forecast']) if sub_item['Forecast'] != 0 else '-'
                    formatted_sub_item['Budget'] = '₹' + '{:,.2f}'.format(sub_item['Budget']) if sub_item['Budget'] != 0 else '-'
                    formatted_sub_item['Actual'] = '₹' + '{:,.2f}'.format(sub_item['Actual']) if sub_item['Actual'] != 0 else '-'
                    formatted_item['Variance: Actual vs Forecast'] = '₹' + '{:,.2f}'.format(item['Variance: Actual vs Forecast']) if item['Variance: Actual vs Forecast'] != 0 else '-'
                    formatted_item['Variance: Actual vs Budget'] = '₹' + '{:,.2f}'.format(item['Variance: Actual vs Budget']) if item['Variance: Actual vs Budget'] != 0 else '-'
                    formatted_item['subRows'].append(formatted_sub_item)
                except Exception as ex:
                    formatted_sub_item = {}
                    formatted_sub_item['Item'] = sub_item['Item']
                    formatted_sub_item['Forecast'] = '₹' + '{:,.2f}'.format(sub_item['Forecast']) if sub_item['Forecast'] != 0 else '-'
                    formatted_sub_item['Budget'] = '₹' + '{:,.2f}'.format(sub_item['Budget']) if sub_item['Budget'] != 0 else '-'
                    formatted_sub_item['Actual'] = '₹' + '{:,.2f}'.format(sub_item['Actual']) if sub_item['Actual'] != 0 else '-'
                    formatted_item['Variance: Actual vs Forecast'] = '₹' + '{:,.2f}'.format(item['Variance: Actual vs Forecast']) if item['Variance: Actual vs Forecast'] != 0 else '-'
                    formatted_item['Variance: Actual vs Budget'] = '₹' + '{:,.2f}'.format(item['Variance: Actual vs Budget']) if item['Variance: Actual vs Budget'] != 0 else '-'
                    formatted_item['subRows'].append(formatted_sub_item)
                    
            formatted_response.append(formatted_item)
        except Exception as ex:
            continue
    return formatted_response

def generate_bar_chart(data):
    """
    Generate bar chart structure from the given data.

    Parameters:
        data (list): List of dictionaries representing the input data.

    Returns:
        dict: Bar chart structure.
    """
    # Initialize lists to store labels and values
    labels = []
    values_forecast = []
    values_budget = []
    values_actual = []

    # Extract data from input
    for item in data:
        if item['Label'] == 'Operating Profit':
            labels.append(item['Label'])
            values_forecast.append(round(item['Forecast']))
            values_budget.append(round(item['Budget']))
            values_actual.append(round(item['Actual']))

    # Bar chart structure
    bar_chart = {
        "chart_type": "bar",
        "data_format": {
            "labels": labels,
            "categories": ["Forecast", "Budget", "Actual"],
            "values": [values_forecast, values_budget, values_actual],
            "isStacked": False  # Change to True if you want stacked chart
        }
    }

    return bar_chart

def calculate_variance(profitability_analysis_data):
    """
    Calculates the variance for each item and its subRows in the profitability_analysis_data.
    
    Args:
    - profitability_analysis_data (list): A list of dictionaries representing profitability analysis data.

    Returns:
    - list: The profitability analysis data with variance calculated and added as 'Variance' key.
    """
    for item in profitability_analysis_data:
        item['Variance: Actual vs Forecast'] = item['Forecast'] - item['Actual']
        item['Variance: Actual vs Budget'] = item['Budget'] - item['Actual']
        if 'subRows' in item:
            for subitem in item['subRows']:
                subitem['Variance: Actual vs Forecast'] = subitem['Forecast'] - subitem['Actual']
                subitem['Variance: Actual vs Budget'] = subitem['Budget'] - subitem['Actual']
    return profitability_analysis_data

#################################################################################################################################
##Monthly Level Code #####################
################################################################################################################################
#Step 1: Calculate Revenue
def create_revenue_monthly(final_df, sale_value_column='Sale_Value', start_date='2023-10-01'):
    start_date = pd.to_datetime(start_date)
    final_df['Month_Year'] = pd.to_datetime(final_df['Month_Year'])
    final_df = final_df[final_df['Month_Year'] >= start_date]
    final_df['Month_Year'] = final_df['Month_Year'].dt.strftime('%B-%Y')
    if 'Label' not in final_df.columns:
        final_df['Label'] = final_df['Item']
    result = final_df.groupby('Month_Year').agg({sale_value_column: 'sum'}).reset_index()
    result['Month_Year'] = pd.to_datetime(result['Month_Year'])
    result.sort_values('Month_Year', inplace=True)
    result['Month_Year'] = result['Month_Year'].dt.strftime('%B-%Y')
    result.set_index('Month_Year', inplace=True)
    final_dict = {
        "Label": "Revenue"
    }
    for index, row in result.iterrows():
        final_dict[index] = row[sale_value_column]
    sub_rows = []
    for label in final_df['Label'].unique():
        label_df = final_df[final_df['Label'] == label]
        label_result = label_df.groupby('Month_Year').agg({sale_value_column: 'sum'}).reset_index()
        label_result['Month_Year'] = pd.to_datetime(label_result['Month_Year'])
        label_result.sort_values('Month_Year', inplace=True)
        label_result['Month_Year'] = label_result['Month_Year'].dt.strftime('%B-%Y')
        label_result.set_index('Month_Year', inplace=True)
        label_result = label_result.rename(columns={sale_value_column: label})
        
        label_output_dict = {"Label": label_result.T.index[0]}
        for idx, row in label_result.iterrows():
            label_output_dict[idx] = row[label]
        sub_rows.append(label_output_dict)
    
    # Add subRows to the main dictionary
    final_dict['subRows'] = sub_rows
    
    return final_dict

#Step2 : COGS Calculation
#others_forecast_df = create_forecast_others(combined_df)
def create_cogs_monthly(data, start_date='2023-10-01'):
    start_date = pd.to_datetime(start_date)
    data['Month_Year'] = pd.to_datetime(data['Month_Year'])
    data = data[data['Month_Year'] >= start_date]
    data['Month_Year'] = data['Month_Year'].dt.strftime('%B-%Y')
    data = data[['Month_Year','packing_expenses','direct_labor','manufacturing_overheads','Raw_Materials']]
    try:
        data['Month_Year'] = data['Month_Year'].dt.strftime('%B-%Y')
    except Exception as ex:
        pass   
    data['COGS'] = data.sum(axis=1)
    cogs_df = data[['Month_Year', 'COGS']]
    cogs_df.set_index('Month_Year', inplace=True)
    final_dict = {
        "Label": cogs_df.T.index[0]
    }
    for index, row in cogs_df.iterrows():
        final_dict[index] = row['COGS']
        
    # Create dictionaries for each subrow
    sub_rows = []
    for column in ['packing_expenses', 'direct_labor', 'manufacturing_overheads', 'Raw_Materials']:
        sub_row_data = data[['Month_Year', column]]
        sub_row_data.set_index('Month_Year', inplace=True)
        sub_row_dict = {
            "Label": column.replace('_', ' ').title()
        }
        for index, row in sub_row_data.iterrows():
            sub_row_dict[index] = row[column]
        sub_rows.append(sub_row_dict)

    # Add subrows to final_dict
    final_dict['subRows'] = sub_rows
    return final_dict

#Step 3: Other Operating Expenses
def create_other_operating_expenses_monthly(data, start_date='2023-10-01'):
    start_date = pd.to_datetime(start_date)
    data['Month_Year'] = pd.to_datetime(data['Month_Year'])
    data = data[data['Month_Year'] >= start_date]
    data['Month_Year'] = data['Month_Year'].dt.strftime('%B-%Y')
    data = data[['Month_Year','distribution_expenses','marketing_expenses']]
    try:
        data['Month_Year'] = data['Month_Year'].dt.strftime('%B-%Y')
    except Exception as ex:
        pass    
    data['Other Operating Expenses'] = data.sum(axis=1)
    cogs_df = data[['Month_Year', 'Other Operating Expenses']]
    cogs_df.set_index('Month_Year', inplace=True)
    final_dict = {
        "Label": cogs_df.T.index[0]
    }
    for index, row in cogs_df.iterrows():
        final_dict[index] = row['Other Operating Expenses']
    sub_rows = []
    for column in ['distribution_expenses','marketing_expenses']:
        sub_row_data = data[['Month_Year', column]]
        sub_row_data.set_index('Month_Year', inplace=True)
        sub_row_dict = {
            "Label": column.replace('_', ' ').title()
        }
        for index, row in sub_row_data.iterrows():
            sub_row_dict[index] = row[column]
        sub_rows.append(sub_row_dict)
    final_dict['subRows'] = sub_rows
    return final_dict

#Step 4: Other Administrative Expenses
def create_other_administrative_expenses_monthly(data, start_date='2023-10-01'):
    start_date = pd.to_datetime(start_date)
    data['Month_Year'] = pd.to_datetime(data['Month_Year'])
    data = data[data['Month_Year'] >= start_date]
    data['Month_Year'] = data['Month_Year'].dt.strftime('%B-%Y')
    data = data[['Month_Year','other_administrative_expenses']]
    try:
        data['Month_Year'] = data['Month_Year'].dt.strftime('%B-%Y')
    except Exception as ex:
        pass    
    data['Other Administrative Expenses'] = data.sum(axis=1)
    cogs_df = data[['Month_Year', 'Other Administrative Expenses']]
    cogs_df.set_index('Month_Year', inplace=True)
    final_dict = {
        "Label": cogs_df.T.index[0]
    }
    for index, row in cogs_df.iterrows():
        final_dict[index] = row['Other Administrative Expenses']
    return final_dict

#Step 5: Operating Profit
def calculate_profitability_monthly(data):
    # Extracting column names
    columns = ['Label']
    for item in data:
        columns.extend(item.keys())

    # Extracting unique months
    months = set()
    for item in data:
        if 'subRows' in item:
            for sub_item in item['subRows']:
                months.update(sub_item.keys())

    # Creating a dictionary to store the values
    values = {}
    for item in data:
        label = item['Label']
        values[label] = {}
        for month in months:
            if month in item:
                values[label][month] = item[month]
            elif 'subRows' in item:
                for sub_item in item['subRows']:
                    if sub_item['Label'] in columns:
                        values[label][sub_item['Label']] = sub_item[month]
    df = pd.DataFrame(values).T
    del df['Label']
    df_sorted = df.reindex(sorted(df.columns, key=lambda x: pd.to_datetime(x, format='%B-%Y')), axis=1)
    df_operating_profit = df_sorted[:1] - df_sorted[1:].sum()
    final_dict = {
        "Label": "Operating Profit"
    }
    for index, row in df_operating_profit.T.iterrows():
        final_dict[index] = row['Revenue']
    return final_dict

def format_decimal(value):
    if isinstance(value, float):
        return '₹{:.2f}'.format(value)
    elif isinstance(value, int):
        if value == 0:
            return '-'
        else:
            return '₹{:,}'.format(value).replace(",", "-")
    else:
        return value

def process_payload_monthly(payload):
    for item in payload:
        for key, value in item.items():
            if key != 'Label':
                item[key] = format_decimal(value)
            elif key == 'Label' and 'subRows' in item:
                process_payload_monthly(item['subRows'])
    return payload

def convert_to_json_serializable(data):
    if isinstance(data, np.int64):
        return int(data)
    elif isinstance(data, dict):
        return {k: convert_to_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_json_serializable(item) for item in data]
    else:
        return data
    
    
def generate_bar_chart_monthly(data):
    # Find the dictionary corresponding to 'Operating Profit'
    operating_profit_dict = next((item for item in data if item['Label'] == 'Operating Profit'), None)
    if operating_profit_dict is None:
        raise ValueError("Operating Profit data not found.")

    # Extract labels (months) and values
    labels = list(operating_profit_dict.keys())[1:]
    values = [int(operating_profit_dict[label]) for label in labels]

    # Bar chart structure
    bar_chart = {
        "chart_type": "bar",
        "data_format": {
            "labels": labels,
            "values": [values],
            "categories": ["Operating Profit"],
            "isStacked": False  # Change to True if you want stacked chart
        }
    }

    return bar_chart

def format_numeric_dataframe(df, currency_symbol="₹", numeric_format="{:.2f}", zero_placeholder="-"):
    """
    Format DataFrame by replacing 0 with a specified placeholder,
    adding a currency symbol to numeric values, and applying a numeric format.
    
    Parameters:
    df (DataFrame): Input DataFrame to be formatted.
    currency_symbol (str): Symbol to add to numeric values (default is "₹").
    numeric_format (str): Format string for numeric values (default is "{:.2f}").
    zero_placeholder (str): Placeholder for zero values (default is "-").
    
    Returns:
    DataFrame: Formatted DataFrame.
    """
    # Create a copy of the input DataFrame to avoid modifying the original DataFrame
    formatted_df = df.copy()
    
    # Iterate over each column in the DataFrame
    for col in formatted_df.columns:
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(formatted_df[col]):
            # Replace 0 with the specified placeholder and add currency symbol to numeric values
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{currency_symbol}{numeric_format.format(x)}" if x != 0 else zero_placeholder)
        else:
            # Replace 0 with the specified placeholder for non-numeric columns
            formatted_df[col] = formatted_df[col].replace(0, zero_placeholder)
    
    return formatted_df

def generate_operating_profit_line_chart(df, column_name="Operating Profit"):
    """
    Generate a line chart structure for operating profit data.

    Parameters:
        df (pd.DataFrame): Input dataframe with columns "Month_Year", "Actual Operating Profit",
                           "Forecast Operating Profit", "Budget Operating Profit",
                           "Variance: Actual vs Forecast", and "Variance: Actual vs Budget".
        column_name : This is predix name for the column
    Returns:
        dict: Line chart structure.
    """
    # Initialize variables
    cumulative_profit = 0
    actual_profit_values = []
    forecast_profit_values = []
    budget_profit_values = []
    labels = []
    actual_value = f"Actual {column_name}"
    forecast_value = f"Forecast {column_name}"
    budget_value = f"Budget {column_name}"

    # Iterate over the dataframe
    for index, row in df.iterrows():
        # Check if the actual profit is zero or negative
        if row[actual_value] != 0:
            # Calculate cumulative actual profit
            cumulative_profit += row[actual_value]
            actual_profit_values.append(int(cumulative_profit))
        else:
            actual_profit_values.append(0)

    # Calculate cumulative forecast and budget profits
    forecast_cumulative_profit = 0
    budget_cumulative_profit = 0
    for index, row in df.iterrows():
        forecast_cumulative_profit += row[forecast_value]
        budget_cumulative_profit += row[budget_value]
        forecast_profit_values.append(int(forecast_cumulative_profit))
        budget_profit_values.append(int(budget_cumulative_profit))
        labels.append(row['Month_Year'])
    
    # Line chart structure
    line_chart = {
        "chart_type": "line",
        "data_format": {
            "labels": labels,
            "categories": ["Actual Operating Profit", "Forecast Operating Profit", "Budget Operating Profit"],
            "values": [actual_profit_values, forecast_profit_values, budget_profit_values],
            "isStacked": False
        }
    }

    return line_chart


def calculate_profitability_analysis_data(revenue_actual, revenue_forecast, expense_actual, expense_forecast, start_date) -> List[Dict]:
    try:
        profitability_analysis_data = []
        profitability_analysis_data.append(create_actual_vs_forecast_revenue(revenue_actual, revenue_forecast, start_date))
        profitability_analysis_data.append(create_actual_vs_forecast_cogs(expense_actual, expense_forecast, start_date))
        profitability_analysis_data.append(create_actual_vs_forecast_other_operating_expenses(expense_actual, expense_forecast, start_date))
        profitability_analysis_data.append(create_actual_vs_forecast_other_administrative_expenses(expense_actual, expense_forecast, start_date))
        profitability_analysis_data.append(create_actual_vs_forecast_operating_profit(profitability_analysis_data))
        return profitability_analysis_data
    except Exception as e:
        logger.error(f"Error occurred in calculate_profitability_analysis_data: {str(e)}")
        raise

def format_profitability_analysis_data(profitability_analysis_data: List[Dict]) -> List[Dict]:
    try:
        for item in profitability_analysis_data:
            for subrow in item.get('subRows', []):
                subrow['Label'] = subrow.pop('Item', subrow.get('Label'))
        profitability_analysis_data = calculate_variance(profitability_analysis_data)
        return profitability_analysis_data
    except Exception as e:
        logger.error(f"Error occurred in format_profitability_analysis_data: {str(e)}")
        raise

def calculate_line_chart_and_variance(revenue_actual, revenue_forecast, expense_actual, expense_forecast, start_date) -> Tuple[Dict, Dict]:
    try:
        actual_monthly_data = create_monthly_data(revenue_actual, expense_actual, start_date)
        forecast_monthly_data = create_monthly_data(revenue_forecast, expense_forecast, start_date)
        budget_monthly_data = create_monthly_data(revenue_forecast, expense_forecast, start_date)

        combined_df = combine_monthly_data(actual_monthly_data, forecast_monthly_data, budget_monthly_data)
        line_chart = generate_operating_profit_line_chart(combined_df)
        variance_output = format_numeric_dataframe(combined_df)
        variance_output_formatted = {"table": {"columns": variance_output.columns.tolist(), "data": variance_output.values.tolist(), "title":"Variance Table"}}

        return line_chart, variance_output_formatted
    except Exception as e:
        logger.error(f"Error occurred in calculate_line_chart_and_variance: {str(e)}")
        raise

def create_monthly_data(revenue_data, expense_data, start_date):
    try:
        all_monthly_data = []
        all_monthly_data.append(create_revenue_monthly(revenue_data, sale_value_column='Sale_Value', start_date=start_date))
        all_monthly_data.append(create_cogs_monthly(expense_data, start_date))
        all_monthly_data.append(create_other_operating_expenses_monthly(expense_data, start_date))
        all_monthly_data.append(create_other_administrative_expenses_monthly(expense_data, start_date))
        all_monthly_data.append(calculate_profitability_monthly(all_monthly_data))
        monthly_data = pd.DataFrame(all_monthly_data)
        del monthly_data['subRows']
        return monthly_data
    except Exception as e:
        logger.error(f"Error occurred in create_monthly_data: {str(e)}")
        raise

def combine_monthly_data(actual_monthly_data, forecast_monthly_data, budget_monthly_data):
    try:
        combined_df = pd.concat([actual_monthly_data.iloc[-1], forecast_monthly_data.iloc[-1], budget_monthly_data.iloc[-1]], axis=1)
        combined_df.reset_index(inplace=True)
        combined_df.columns = ['Month_Year', 'Actual Operating Profit', 'Forecast Operating Profit', 'Budget Operating Profit']
        combined_df = combined_df.iloc[1:]
        combined_df.reset_index(drop=True, inplace=True)
        combined_df = combined_df.fillna(0)
        combined_df['Variance: Actual vs Forecast'] = combined_df['Forecast Operating Profit'] - combined_df['Actual Operating Profit']
        combined_df['Variance: Actual vs Budget'] = combined_df['Budget Operating Profit'] - combined_df['Actual Operating Profit']
        return combined_df
    except Exception as e:
        logger.error(f"Error occurred in combine_monthly_data: {str(e)}")
        raise
    
    
def process_subrows(profitability_analysis_data):
    # Extract subRows column values into a list and delete the subRows column
    sub_rows_list = profitability_analysis_data['subRows'].tolist()
    del profitability_analysis_data['subRows']
    
    # Iterate over each value in subRows_list, create a DataFrame, and concatenate it vertically to the original DataFrame
    for srow in sub_rows_list:
        try:
            df = pd.DataFrame(srow)
            profitability_analysis_data = pd.concat([profitability_analysis_data, df], ignore_index=True)
        except Exception as ex:
            continue
    return profitability_analysis_data