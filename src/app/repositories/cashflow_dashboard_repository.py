import pandas as pd
from app.repositories.profitability_dashboard_repository import *


def generate_cashflow_variance_report(RevenueActual, RevenueForecast, ExpenseActual, ExpenseForecast, tax_rate = 0.2, start_date='2023-10-01'):
    """
    Generate a cash flow variance report using actual and forecasted revenue and expense data.

    Args:
        RevenueActual (pandas.DataFrame): Actual revenue data.
        RevenueForecast (pandas.DataFrame): Forecasted revenue data.
        ExpenseActual (pandas.DataFrame): Actual expense data.
        ExpenseForecast (pandas.DataFrame): Forecasted expense data.
        tax_rate (float, optional): Tax rate to be applied. Default is 0.2 (20%).
        start_date(str, optional): Start date for the analysis in 'YYYY-MM' format. If not provided it will use the first available month from both
    Returns:
        pandas.DataFrame: Cash flow variance report with the following columns:
            - Label
            - Forecast
            - Actual
            - Variance

    The report includes the following rows:
        - Operating Profit
        - Tax
        - Operating Profit After Tax
        - Cash From Goods
        - Amount Paid to Vendors
        - Advance received from Customers
        - Advance Paid to Suppliers/Vendors
        - Cash Flow from Operations

    The 'Cash From Goods' row is calculated as 'Sale_Value' - 'Receivables' for both actual and forecasted data.
    The 'Amount Paid to Vendors' row is obtained from the 'Purchase_Paid_Amount' column in the expense data.
    The 'Cash Flow from Operations' row is calculated using the formula:
        Operating Profit After Tax + Cash From Goods + Advance received from Customers - (Amount Paid to Vendors + Advance Paid to Suppliers/Vendors)

    The 'Variance' column is calculated as 'Actual' - 'Forecast' for each row.
    """
    # Step 2: Initialize list to store profitability analysis data
    profitability_analysis_data = []

    # Step 3: Append actual vs forecast data for revenue and expenses to the list
    profitability_analysis_data.append(create_actual_vs_forecast_revenue(RevenueActual, RevenueForecast, start_date))
    profitability_analysis_data.append(create_actual_vs_forecast_cogs(ExpenseActual, ExpenseForecast, start_date))
    profitability_analysis_data.append(create_actual_vs_forecast_other_operating_expenses(ExpenseActual, ExpenseForecast, start_date))
    profitability_analysis_data.append(create_actual_vs_forecast_other_administrative_expenses(ExpenseActual, ExpenseForecast, start_date))
    profitability_analysis_data.append(create_actual_vs_forecast_operating_profit(profitability_analysis_data))

    # Step 4: Set display format for floating-point numbers
    pd.options.display.float_format = '{:.2f}'.format

    # Step 5: Extract data for operating profit
    operating_profit_data = [entry for entry in profitability_analysis_data if entry['Label'] == 'Operating Profit']
    operating_profit_data = operating_profit_data[0]

    # Step 6: Create DataFrame with operating profit data
    df_data = {
        'Label': [operating_profit_data['Label']],
        'Forecast': [operating_profit_data['Forecast']],
        'Budget': [operating_profit_data['Budget']],
        'Actual': [operating_profit_data['Actual']]
    }
    df = pd.DataFrame(df_data)
    df = df[['Label', 'Forecast', 'Budget', 'Actual']]

    # Step 7: Calculate tax for forecasted and actual operating profit
    tax_actual = df['Actual'] * tax_rate
    tax_forecast = df['Forecast'] * tax_rate
    tax_budget = df['Budget'] * tax_rate

    # Step 8: Calculate operating profit after tax for forecasted and actual
    operating_profit_after_tax_actual = df['Actual'] - tax_actual
    operating_profit_after_tax_forecast = df['Forecast'] - tax_forecast
    operating_profit_after_tax_budget = df['Budget'] - tax_budget

    # Step 9: Insert tax and operating profit after tax rows into the DataFrame
    df = df.append({'Label': 'Tax', 'Forecast': tax_forecast.values[0], 'Budget': tax_budget.values[0],'Actual': tax_actual.values[0]}, ignore_index=True)
    df = df.append({'Label': 'Operating Profit After Tax', 'Forecast': operating_profit_after_tax_forecast.values[0], 'Budget': operating_profit_after_tax_budget.values[0], 'Actual': operating_profit_after_tax_actual.values[0]}, ignore_index=True)

    # Step 10: Calculate realized sale value for revenue
    RevenueActual['Cash from Goods'] = RevenueActual['Receivables']
    RevenueForecast['Cash from Goods'] = RevenueForecast['Receivables']
    #To do: Add RevenueBudget also for now we are using RevenueForecast here
    revenue_actual_sale_value = RevenueActual['Cash from Goods'].sum()
    revenue_forecast_sale_value = RevenueForecast['Cash from Goods'].sum()
    revenue_budget_sale_value = RevenueForecast['Cash from Goods'].sum()
    
    # Step 11: Append realized sale value row to the DataFrame
    df = df.append({'Label': 'Cash From Goods', 'Forecast': revenue_forecast_sale_value,'Budget': revenue_budget_sale_value, 'Actual': revenue_actual_sale_value}, ignore_index=True)
    # for col in df.columns[1:]:
    #     df.at[3, col] = df.at[2, col] - df.at[3, col]
    
    # Step 12: Calculate amount paid to vendors
    df = df.append({'Label': 'Amount Paid to Vendors', 'Forecast': ExpenseForecast['Purchase_Paid_Amount'].sum(), 'Budget': ExpenseForecast['Purchase_Paid_Amount'].sum(), 'Actual': ExpenseActual['Purchase_Paid_Amount'].sum()}, ignore_index=True)
    #for col in df.columns[1:]:
    #    df.at[4, col] = df.at[4, col] + df.at[3, col]
    
    # Step 13: Append rows for advance received from customers and advance paid to suppliers/vendors
    advance_received_data = {'Label': ['Advance received from Customers'], **{col: [0] for col in df.columns if col != 'Label'}}
    df = df.append(pd.DataFrame(advance_received_data), ignore_index=True)

    advance_paid_data = {'Label': ['Advance Paid to Suppliers/Vendors'], **{col: [0] for col in df.columns if col != 'Label'}}
    df = df.append(pd.DataFrame(advance_paid_data), ignore_index=True)

    # Step 14: Calculate cash flow from operations
    operating_profit_after_tax_row = df.iloc[2, 1:].values
    realized_sale_value = df.iloc[3, 1:].values
    paid_amount = df.iloc[4, 1:].values
    advance_received_from_customer = df.iloc[5, 1:].values
    advance_paid_to_supplier = df.iloc[6, 1:].values

    cash_flow_from_operations_row =  (operating_profit_after_tax_row + paid_amount + advance_received_from_customer) - (advance_paid_to_supplier + realized_sale_value)
    df.loc[len(df)] = ['Cash Flow from Operations'] + list(cash_flow_from_operations_row)

    # Step 15: Calculate variance
    df['Variance:Actual VS Forecast'] = df['Actual'] - df['Forecast']
    df['Variance:Actual VS Budget'] = df['Actual'] - df['Budget']
    return df

def generate_cashflow_report_monthly(RevenueActual, ExpenseActual, start_date='2023-10-01', tax_rate = 0.2):
    """
    Generate a monthly cash flow report using actual revenue and expense data.

    Args:
        RevenueActual (pandas.DataFrame): Actual revenue data with columns 'Month_Year', 'Sale_Value', and 'Receivables'.
        ExpenseActual (pandas.DataFrame): Actual expense data with columns 'Month_Year' and 'Purchase_Paid_Amount'.
        start_date (str, optional): The start date for the report in the format 'YYYY-MM-DD'. Default is '2023-10-01'.
        tax_rate (float, optional): Tax rate to be applied. Default is 0.2 (20%).

    Returns:
        pandas.DataFrame: Monthly cash flow report with the following columns:
            - Label
            - [Month-Year columns for each month from start_date]

    The report includes the following rows:
        - Revenue
        - Cost of Goods Sold (COGS)
        - Other Operating Expenses
        - Other Administrative Expenses
        - Operating Profit
        - Tax
        - Operating Profit After Tax
        - Cash From Goods
        - Purchase Paid Amount
        - Advance received from Customers
        - Advance Paid to Suppliers/Vendors
        - Cash Flow from Operations

    The 'Cash From Goods' row is calculated as 'Sale_Value' - 'Receivables' for the revenue data.
    The 'Purchase Paid Amount' row is obtained from the 'Purchase_Paid_Amount' column in the expense data.
    The 'Cash Flow from Operations' row is calculated using the formula:
        Operating Profit After Tax + Cash From Goods + Advance received from Customers - (Purchase Paid Amount + Advance Paid to Suppliers/Vendors)

    The report includes data from the start_date onwards, and the columns are labeled with 'Month-Year' values.
    """
    all_monthly_data = []
    all_monthly_data.append(create_revenue_monthly(RevenueActual, sale_value_column='Sale_Value', start_date=start_date))
    all_monthly_data.append(create_cogs_monthly(ExpenseActual, start_date))
    all_monthly_data.append(create_other_operating_expenses_monthly(ExpenseActual, start_date))
    all_monthly_data.append(create_other_administrative_expenses_monthly(ExpenseActual, start_date))
    all_monthly_data.append(calculate_profitability_monthly(all_monthly_data))
    operating_profit_data = [entry for entry in all_monthly_data if entry['Label'] == 'Operating Profit']
    df = pd.DataFrame(operating_profit_data)
    tax_values = df.iloc[0, 1:] * tax_rate
    df.loc[1] = ['Tax'] + tax_values.tolist()
    operating_profit_values = df.iloc[0, 1:] - df.iloc[1, 1:]
    df.loc[2] = ['Operating Profit After Tax'] + operating_profit_values.tolist()

    #Operating profit After Taxv            100 - 30(Receive in future) (70)
    #Paybale 70 + (30 I have to pay in future) 100
    #df['Operating Profit After Tax'] -  RevenueActual['Receivables']
    '''
    Step 1. OPerating Profit After Tax
    Step 2. We are going to receive some amount in future for example 30. But at the momemnet we do not have this moment with us
    df['Operating Profit After Tax'] -  RevenueActual['Receivables']
    Step 3. We need to pay to vendor the reciable cusomters . This amount will be added as we have this amount with us
    df['Operating Profit After Tax'] +  ExpenseActual['Purchase_Paid_Amount']
    '''
    #Amount receive in 
    
    RevenueActual['Cash from Goods'] = RevenueActual['Receivables']
    # Group by 'Month_Year' and sum 'Cash from Goods'
    cash_from_goods = RevenueActual[['Month_Year', 'Cash from Goods']].groupby('Month_Year').sum()
    cash_from_goods = cash_from_goods[cash_from_goods.index >= start_date]
    cash_from_goods.index = cash_from_goods.index.strftime('%B-%Y')
    cash_from_goods = cash_from_goods.T.reset_index()
    cash_from_goods = cash_from_goods.rename(columns={'index': 'Label'})
    df = df.append(cash_from_goods, ignore_index=True)
    # for col in df.columns[1:]:
    #     df.at[3, col] = df.at[2, col] - df.at[3, col]
    
    # Amount paid to vendors
    amount_paid_to_vendors = ExpenseActual[ExpenseActual['Month_Year'] >= start_date]
    amount_paid_to_vendors = amount_paid_to_vendors[['Month_Year', 'Purchase_Paid_Amount']]
    amount_paid_to_vendors['Month_Year'] = amount_paid_to_vendors['Month_Year'].dt.strftime("%B-%Y")
    amount_paid_to_vendors.reset_index(inplace=True)
    del amount_paid_to_vendors['index']
    amount_paid_to_vendors = amount_paid_to_vendors.rename(columns={'Month_Year': 'Label'})
    amount_paid_to_vendors.set_index(["Label"], inplace=True)
    amount_paid_to_vendors = amount_paid_to_vendors.T
    df = df.append(amount_paid_to_vendors, ignore_index=True)
    df['Label'].fillna('Purchase Paid Amount', inplace=True)
    # for col in df.columns[1:]:
    #     df.at[4, col] = df.at[4, col] + df.at[3, col]
    
    # Other calculations
    # Step 13: Append rows for advance received from customers and advance paid to suppliers/vendors
    advance_received_data = {'Label': ['Advance received from Customers'], **{col: [0] for col in df.columns if col != 'Label'}}
    df = df.append(pd.DataFrame(advance_received_data), ignore_index=True)
    advance_paid_data = {'Label': ['Advance Paid to Suppliers/Vendors'], **{col: [0] for col in df.columns if col != 'Label'}}
    df = df.append(pd.DataFrame(advance_paid_data), ignore_index=True)

    # Step 14: Calculate cash flow from operations
    operating_profit_after_tax_row = df.iloc[2, 1:].values
    realized_sale_value = df.iloc[3, 1:].values
    paid_amount = df.iloc[4, 1:].values
    advance_received_from_customer = df.iloc[5, 1:].values
    advance_paid_to_supplier = df.iloc[6, 1:].values
    cash_flow_from_operations_row =  (operating_profit_after_tax_row + paid_amount + advance_received_from_customer) - (advance_paid_to_supplier + realized_sale_value)
    df.loc[len(df)] = ['Cash Flow from Operations'] + list(cash_flow_from_operations_row)

    return df


def format_financial_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats the financial DataFrame by rounding numbers to two decimal places,
    replacing 0 with '-', adding a rupee sign prefix to all non-zero numbers,
    and adding a '-' sign before the rupee sign if the value is negative.

    Args:
    df (pandas.DataFrame): The DataFrame containing financial data.

    Returns:
    pandas.DataFrame: The formatted DataFrame.
    """
    try:
        # Round all numbers to two decimal places
        df = df.round(2)

        # Add rupee sign to non-zero numeric values
        for col in df.columns:
            if df[col].dtype == 'float64':  # Check if the column contains numeric data
                df[col] = df[col].apply(lambda x: f"{'-₹' if x < 0 else '₹'}{abs(x):,.2f}")
        df = df.replace(0.00, "-")  # Replace 0 with '-'

        return df
    except Exception as e:
        # Log the error
        print(f"Error occurred while formatting financial DataFrame: {e}")
        # Return the original DataFrame in case of error
        return df


def generate_cash_flow_operations_bar_chart(df):
    """
    Generate bar chart structure for cash flow operations from the given DataFrame.

    Parameters:
        df (DataFrame): DataFrame containing the cash flow operations data.

    Returns:
        dict: Bar chart structure.
    """
    # Filter the DataFrame for Cash Flow from Operations
    cash_flow_df = df[df['Label'] == 'Cash Flow from Operations']

    # Extract values for Forecast and Actual
    forecast = cash_flow_df['Forecast'].values[0]
    budget = cash_flow_df['Budget'].values[0]
    actual = cash_flow_df['Actual'].values[0]

    # Bar chart structure
    bar_chart = {
        "chart_type": "bar",
        "data_format": {
            "labels": ["Cash Flow from Operations"],
            "categories": ["Forecast", 'Budget', "Actual"],
            "values": [[round(forecast, 2)], [round(budget, 2)], [round(actual, 2)]],
            "isStacked": False  # Change to True if you want stacked chart
        }
    }

    return bar_chart

def generate_cash_flow_bar_chart_monthly(df):
    # Find the dictionary corresponding to 'Operating Profit'
    cash_flow_df = df[df['Label'] == 'Cash Flow from Operations']

    # Extract values for Forecast and Actual
    values = cash_flow_df.iloc[:, 1:].values.round(2).tolist()[0]   # Extract all values except the first column (Label)
    labels = cash_flow_df.columns[1:].tolist()  # Extract column labels except the first one (Label)

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