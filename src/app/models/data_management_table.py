# app/models/data_management_table.py
from datetime import datetime
from typing import Optional, Any, Text
from pydantic import BaseModel, Json

class DataManagementTable(BaseModel):
    id: Optional[Any] = None
    board_id: Optional[Any]
    table_name: str
    table_description: Optional[Text]
    table_column_type_detail: Optional[Text]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class TableStatus(BaseModel):
    id: Optional[int] = None
    data_management_table_id: Optional[int] = None
    month_year: str
    approved: bool
    filename: str
    file_download_link: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    

