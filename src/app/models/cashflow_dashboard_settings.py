# app/models/cashflow_dashborad_settings.py
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel

class CashFlowDashboardSettings(BaseModel):
    id: Optional[int] = None
    board_id: int #Logic is forecast can also ve a board and this will be connected with forecastResponse with all the variation
    name: Optional[str] = 'Cash Flow Settings'
    tax: float
    financial_year_start: Optional[str] = None
    financial_year_end: Optional[str] = None
    forecast_period: Optional[int] = None
    output_response:  Optional[str] = None
    publish_to_cfo: Optional[bool] = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

