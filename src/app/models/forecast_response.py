# app/models/forecast_response.py
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel

class ForecastResponse(BaseModel):
    id: Optional[int] = None
    board_id: int #Logic is forecast can also ve a board and this will be connected with forecastResponse with all the variation
    name: Optional[str] = 'Forecast Response'
    first_level_filter: str
    second_level_filter: Optional[str] = None
    financial_year_start: Optional[str] = None
    forecast_period: int
    output_response:  Optional[str] = None
    publish_to_cfo: Optional[bool] = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


