# app/models/other_parameter_settings.py
from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel

class OtherParameterSettings(BaseModel):
    id: Optional[Any] = None
    board_id: int
    configuration_details: Optional[str] = None
    name: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
