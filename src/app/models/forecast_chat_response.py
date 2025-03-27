# app/models/forecast_chat_response.py

from datetime import datetime
from typing import Optional
from pydantic import BaseModel

class ForecastChatResponse(BaseModel):
    id: Optional[int] = None
    board_id: int
    forecast_response_id: int
    input_text: str
    first_level_filter: Optional[str] = None
    second_level_filter: Optional[str] = None
    forecast_period: Optional[int] = 0
    hash_key: Optional[str] = None
    name: Optional[str] = None
    response_content: Optional[dict] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


