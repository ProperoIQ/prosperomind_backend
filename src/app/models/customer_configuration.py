# app/models/customer_configuration.py

from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel

class CustomerConfiguration(BaseModel):
    id: Optional[Any] = None
    user_id: Optional[Any]
    configuration: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    client_number: Optional[str] = None
    customer_number: Optional[str] = None
    