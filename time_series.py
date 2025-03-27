# import streamlit as st
# import pandas as pd
# from autots import AutoTS
# import matplotlib.pyplot as plt

# def load_data(file, date_col, value_col):
#     df = pd.read_csv(file)
#     df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m')  # Adjust the format based on your actual data
#     return df

# def plot_data_forecast(df, prediction, date_col, value_col):
#     plt.figure(figsize=(10, 6))

#     # Plot the actual data
#     plt.plot(df[date_col], df[value_col], label='Actual Data', marker='o')

#     # Plot the forecasted data
#     plt.plot(prediction.forecast.index, prediction.forecast, label='Forecasted Data', marker='o')

#     plt.title('Train Data and Forecast Plot')
#     plt.xlabel(date_col)
#     plt.ylabel(value_col)
#     plt.legend()
#     plt.grid(True)
#     st.pyplot(plt)

# def main():
#     st.title("Time Series Forecasting with AutoTS and Streamlit")

#     # File upload
#     uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
#     if uploaded_file is not None:
#         # User input for column names and forecast length
#         date_col = st.text_input("Enter the date column name:")
#         value_col = st.text_input("Enter the value column name:")
#         forecast_length = st.number_input("Enter the forecast length:", min_value=1, value=3)

#         # Load data
#         try:
#             df = load_data(uploaded_file, date_col, value_col)
#             st.subheader("Loaded Data:")
#             st.write(df)
#         except Exception as e:
#             st.error(f"Error loading data: {e}")
#             return

#         # Time series forecasting
#         model = AutoTS(
#             forecast_length=forecast_length,
#             frequency='infer',
#             prediction_interval=0.9,
#             ensemble='auto',
#             model_list="fast",
#             transformer_list="fast",
#             drop_most_recent=1,
#             max_generations=4,
#             num_validations=2,
#             validation_method="backwards"
#         )

#         try:
#             model = model.fit(
#                 df,
#                 date_col=date_col,
#                 value_col=value_col,
#             )
#             st.subheader("Model fitted successfully.")
#         except Exception as e:
#             st.error(f"Error fitting the model: {e}")
#             return

#         # Forecasting results
#         prediction = model.predict()

#         # Display results
#         st.subheader("Forecasted Data:")
#         st.write(prediction.forecast)

#         # Plotting data and forecast
#         st.subheader("Train Data and Forecast Plot:")
#         plot_data_forecast(df, prediction, date_col, value_col)

#         # Print the details of the best model
#         st.subheader("Model Details:")
#         st.write(model)

#         # Accuracy of all tried model results
#         st.subheader("Model Results:")
#         st.write(model.results())

# if __name__ == "__main__":
#     main()


from typing import List, Optional, Dict, Any, Tuple

def convert_to_tree_structure(data: List[Tuple[int, str, Optional[int], str, Optional[int], str]]) -> List[Dict[str, Any]]:
    tree = {}

    for item in data:
        main_board_id, main_board_name, bcf_id, bcf_name, board_id, board_name = item

        if main_board_id not in tree:
            tree[main_board_id] = {
                "name": main_board_name,
                "is_selected": False,
                "bcfs": {}
            }

        if bcf_id is not None and bcf_id not in tree[main_board_id]["bcfs"]:
            tree[main_board_id]["bcfs"][bcf_id] = {
                "name": bcf_name,
                "is_selected": False,
                "boards": {}
            }

        if board_id is not None and board_id not in tree[main_board_id]["bcfs"][bcf_id]["boards"]:
            tree[main_board_id]["bcfs"][bcf_id]["boards"][board_id] = {
                "name": board_name,
                "is_selected": False
            }

    return list(tree.values())

# Example data
data = [
    (1, 'Analysis', 2, 'BCF Board2', None, None),
    (1, 'Analysis', 1, 'BCF Board', 2, 'Board2'),
    (1, 'Analysis', 1, 'BCF Board', 1, 'Board1'),
    (3, 'Updated Testing Analysis', None, None, None, None)
]

tree_structure = convert_to_tree_structure(data)
print(tree_structure)
