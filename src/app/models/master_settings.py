# app/models/master_settings.py
from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel

class MasterSettings(BaseModel):
    id: Optional[Any] = None
    board_id: int
    configuration_details: str
    name: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
