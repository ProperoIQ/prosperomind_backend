from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel

class ForecastSettings(BaseModel):
    id: Optional[int] = None
    board_id: int
    financial_year_start: str
    financial_year_end: str
    date_variable: str
    independent_variables: List[str]
    dependent_variable: str
    forecast_length: Optional[int] = None
    budget: Optional[int] = None
    model_level_1: Optional[str] = None  # Selected forecasting model (e.g., "ARIMA", "Prophet", "RandomForest")
    model_level_2: Optional[str] = None  # Selected forecasting model (e.g., "Linear Regression", "VAR", "Polynomial Regression")
    prediction_interval: Optional[float] = None  # Confidence level for prediction interval (e.g., 0.95 for 95% confidence)
    tuning_parameters: Optional[dict] = None  # Parameters for tuning the forecasting model
    preprocessing_options: Optional[dict] = None  # Options for preprocessing the data before forecasting
    automatic_forecasting: Optional[bool] = None  # Whether to use automatic model selection
    evaluation_metrics: Optional[List[str]] = None  # Evaluation metrics to assess forecast accuracy
    forecast_horizon: Optional[int] = None  # Time horizon for forecasting
    visualization_options: Optional[dict] = None  # Options for generating visualizations of the forecasts
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


