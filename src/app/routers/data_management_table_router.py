from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form, Query
from sqlalchemy.orm import Session
from app.dependencies import get_db
from app.repositories.data_management_table_repository import DataManagementTableRepository, TableStatusRepository
from app.repositories.ai_documentation_repository import AiDocumentationRepository
from app.models.data_management_table import DataManagementTable, TableStatus
from app.models.ai_documentation import AiDocumentation
from typing import List, Optional
from io import BytesIO
import pandas as pd
from google.cloud import storage
from datetime import datetime, timedelta

from fastapi import HTTPException, Depends, Path
from fastapi.responses import FileResponse

from langchain_openai import ChatOpenAI, OpenAI
from app.instructions import get_ai_documentation_instruction
import re
import json

import os
import asyncio

from pydantic import BaseModel

from app.routers.get_details import fetch_all_reports

storage_client = storage.Client()
bucket_name = "datamanagementtable"


router = APIRouter(prefix="/data-management-table", tags=["Data Management Tables"])

@router.post("/create", response_model=DataManagementTable)
async def create_data_management_table(data_management_table: DataManagementTable):
    repository = DataManagementTableRepository()
    return repository.create_data_management_table(data_management_table)

@router.get("/all", response_model=List[DataManagementTable])
async def get_all_data_management_tables():
    repository = DataManagementTableRepository()
    return repository.get_data_management_tables()

@router.get("/get_all_tables_with_files", response_model=List[dict])
async def get_all_data_management_tables():
    repository = DataManagementTableRepository()
    data_management_tables = repository.get_data_management_tables()
    repository = TableStatusRepository()
    result = []
    for data_table in data_management_tables:
        data_dict = {
            "id": data_table.id,
            "board_id": data_table.board_id,
            "table_name": data_table.table_name,
            "table_description": data_table.table_description,
            "table_column_type_detail": data_table.table_column_type_detail,
            "created_at": data_table.created_at,
            "updated_at": data_table.updated_at,
            "files": []
        }

        # Assuming each data table has a corresponding list of TableStatus instances
        table_statuses =  repository.get_table_statuses_for_data_table(data_table.id)
        for status in table_statuses:
            file_dict = {
                "id": status.id,
                "month_year": status.month_year,
                "approved": status.approved,
                "filename": status.filename,
                "file_download_link": status.file_download_link,
                "created_at": status.created_at,
                "updated_at": status.updated_at
            }
            data_dict["files"].append(file_dict)

        result.append(data_dict)

    return result

@router.get("/get_all_tables_with_files_by_board/{board_id}", response_model=List[dict])
async def get_all_data_management_tables_by_board(board_id: int):
    repository = DataManagementTableRepository()
    data_management_tables = repository.get_data_management_table(board_id)

    if not data_management_tables:
        raise HTTPException(status_code=404, detail="No data tables found for the given board ID.")

    table_status_repository = TableStatusRepository()
    result = []

    for data_table in data_management_tables:
        data_dict = {
            "id": data_table["id"],  # Use dictionary instead of object
            "board_id": data_table["board_id"],
            "table_name": data_table["table_name"],
            "table_description": data_table["table_description"],
            "table_column_type_detail": data_table["table_column_type_detail"],
            "created_at": data_table["created_at"],
            "updated_at": data_table["updated_at"],
            "files": []
        }

        # Fetch table statuses (files) for the given data table
        table_statuses = table_status_repository.get_table_statuses_for_data_table(data_table["id"])
        for status in table_statuses:
            file_dict = {
                "id": status.id,
                "month_year": status.month_year,
                "approved": status.approved,
                "filename": status.filename,
                "file_download_link": status.file_download_link,
                "created_at": status.created_at,
                "updated_at": status.updated_at
            }
            data_dict["files"].append(file_dict)

        result.append(data_dict)

    return result


#To do Download files API


@router.get("/{table_id}", response_model=DataManagementTable)
async def get_data_management_table(table_id: int):
    repository = DataManagementTableRepository()
    return repository.get_data_management_table(table_id)

@router.put("/{table_id}", response_model=DataManagementTable)
async def update_data_management_table(table_id: int, data_management_table: DataManagementTable):
    repository = DataManagementTableRepository()
    return repository.update_data_management_table(table_id, data_management_table)

@router.delete("/{table_id}", response_model=DataManagementTable)
async def delete_data_management_table(table_id: int):
    repository = DataManagementTableRepository()
    return repository.delete_data_management_table(table_id)

@router.get("/status/all", response_model=List[TableStatus])
async def get_all_table_status():
    repository = TableStatusRepository()
    return repository.get_all_table_status()

@router.get("/status/{table_id}", response_model=Optional[TableStatus])
async def get_table_status_by_id(table_id: int):
    repository = TableStatusRepository()
    return repository.get_table_status_by_id(table_id)

@router.put("/status/approve/{table_id}", response_model=TableStatus)
async def update_approval_status(table_id: int, new_approval_status: bool):
    repository = TableStatusRepository()
    return repository.update_approval_status(table_id, new_approval_status)

@router.post("/status/upload/{data_management_table_id}", response_model=TableStatus)
async def upload_file_to_table_status(
    data_management_table_id: int, 
    month_year: str = Form(...),
    file: UploadFile = File(...)
):
    status_repository = TableStatusRepository()
   

    # Read and process the uploaded file
    contents = await file.read()
    buffer = BytesIO(contents)
    df = pd.read_csv(buffer)
    buffer.close()
    file.file.close()

    # Check if the data for the specified table is already approved
    if status_repository.is_month_data_approved(data_management_table_id, month_year):
        raise HTTPException(status_code=400, detail=f"Data for table {data_management_table_id} and month {month_year} is already approved.")

    # Save new TableStatus to the database
    new_table_status = TableStatus(
        data_management_table_id=data_management_table_id,
        month_year=month_year,
        approved=False,
        filename=file.filename,
        file_download_link="",
        created_at=None,
        updated_at=None
    )
    updated_table_status = status_repository.upload_file_table_status(df, new_table_status)

    # Generate AI documentation
    

    return updated_table_status



@router.delete("/status/delete/{table_id}", response_model=TableStatus)
async def delete_table_status(table_id: int):
    repository = TableStatusRepository()
    deleted_status = repository.delete_table_status(table_id)
    if not deleted_status:
        raise HTTPException(status_code=404, detail=f"TableStatus with id {table_id} not found")
    return deleted_status



@router.post("/get-file-url")
async def get_file_url(file_path: str = Query(..., alias="filePath", description="The path to the file in the bucket")):
    # Debugging - Check what file_path is received
    print(f"Received file_path: {file_path}, Type: {type(file_path)}")  
    
    try:
        # Ensure file_path does not include `gs://bucket-name/`
        if file_path.startswith(f"gs://{bucket_name}/"):
            file_path = file_path[len(f"gs://{bucket_name}/"):]

        print(f"Processed file_path: {file_path}")  # Debugging

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)

        # Generate a signed URL that expires in 15 minutes
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=15),
            method="GET"
        )

        return {"url": url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate signed URL: {str(e)}")
    
    
    
@router.put("/status/update/{data_management_table_id}", response_model=TableStatus)
async def update_file_in_table_status(
    data_management_table_id: int,
    month_year: str = Form(...),
    file: UploadFile = File(...)
):
    status_repository = TableStatusRepository()
    ai_documentation_repository = AiDocumentationRepository()

    # Check if the data for the specified table is already approved
    if status_repository.is_month_data_approved(data_management_table_id, month_year):
        raise HTTPException(status_code=400, detail=f"Data for table {data_management_table_id} and month {month_year} is already approved and cannot be updated.")

    # Read and process the uploaded file
    contents = await file.read()
    buffer = BytesIO(contents)
    df = pd.read_csv(buffer)
    buffer.close()
    file.file.close()

    # Update TableStatus in the database
    existing_table_status = status_repository.get_table_status(data_management_table_id, month_year)
    if not existing_table_status:
        raise HTTPException(status_code=404, detail="TableStatus not found.")
    
    existing_table_status.filename = file.filename
    existing_table_status.file_download_link = ""  # Update with actual file storage link
    existing_table_status.updated_at = datetime.utcnow()
    
    updated_table_status = status_repository.update_table_status(existing_table_status)

    # Regenerate AI documentation
    board_id = status_repository.get_board_id_for_table_status_id(data_management_table_id)
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    ai_documentation_instruction = get_ai_documentation_instruction()
    config = llm.invoke(ai_documentation_instruction + df.head(2).to_markdown())
    config_output = re.sub(r'\bfalse\b', 'False', re.sub(r'\btrue\b', 'True', config.content, flags=re.IGNORECASE), flags=re.IGNORECASE)
    config_output = re.sub(r"```|python|json", "", config_output, 0, re.MULTILINE)
    config_dict = eval(config_output)["configuration_details"]
    config_str = json.dumps(config_dict, indent=2)
    
    ai_documentation = AiDocumentation(
        board_id=board_id,
        configuration_details=config_str,
        name=file.filename
    )
    ai_documentation_repository.update_ai_documentation_for_board(board_id, ai_documentation)

    return updated_table_status




FILE_TABLE_MAPPING = {
    "invoice_aging_details_smcs.csv": 47,
    "customer_balance_summary_details_smcs.csv": 46,
    "invoice_aging_details_nvb.csv": 45,
    "customer_balance_summary_details_nvb.csv": 44,
}

CSV_DIR = "csvdata"
MONTH_YEAR = "2025-03"

class UpdateResponse(BaseModel):
    message: str
    details: Optional[dict] = None

async def update_file_in_table_status_internal(data_management_table_id: int, file_path: str, month_year: str):
    """Internal function to update table status. If not found, create a new entry."""
    status_repository = TableStatusRepository()

    # Validate data_management_table_id
    if not isinstance(data_management_table_id, int):
        return {"status": "failed", "error": "Invalid data_management_table_id type."}

    # Check if month data is already approved
    if status_repository.is_month_data_approved(data_management_table_id, month_year):
        return {
            "status": "failed",
            "error": f"Data for table {data_management_table_id} and month {month_year} is already approved."
        }

    # Read and process the file
    try:
        with open(file_path, "rb") as f:
            contents = f.read()

        df = pd.read_csv(BytesIO(contents))
    except Exception as e:
        return {"status": "failed", "error": f"Error reading file: {str(e)}"}

    # Fetch TableStatus from DB
    existing_table_status = status_repository.get_table_status_by_id(data_management_table_id)

    if not existing_table_status:
        # If TableStatus doesn't exist, create it
        new_table_status = TableStatus(
            data_management_table_id=data_management_table_id,
            month_year=month_year,
            approved=False,
            filename=os.path.basename(file_path),
            file_download_link="",  # Update with actual storage link after upload
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        new_table_status = status_repository.upload_file_table_status(df, new_table_status)

        if not new_table_status:
            return {"status": "failed", "error": f"Failed to create TableStatus for ID {data_management_table_id}."}

        return {"status": "success", "table_id": data_management_table_id}

    # If TableStatus exists, update it
    existing_table_status.filename = os.path.basename(file_path)
    existing_table_status.file_download_link = ""  # Update with actual storage link
    existing_table_status.updated_at = datetime.utcnow()

    updated_table_status = status_repository.update_table_status( existing_table_status, df)

    return {"status": "success", "table_id": data_management_table_id}



@router.post("/update-files", response_model=UpdateResponse)
async def update_files():
    results = {}

    async def process_file(filename, table_id):
        file_path = os.path.join(CSV_DIR, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return {filename: {"status": "failed", "error": f"File {filename} not found in {CSV_DIR}."}}

        return {filename: await update_file_in_table_status_internal(table_id, file_path, MONTH_YEAR)}

    # Process all files concurrently
    tasks = [process_file(filename, table_id) for filename, table_id in FILE_TABLE_MAPPING.items()]
    task_results = await asyncio.gather(*tasks)

    # Merge results
    for result in task_results:
        results.update(result)

    if all(result["status"] == "success" for result in results.values()):
        return UpdateResponse(message="All files updated successfully.", details=results)
    else:
        raise HTTPException(status_code=400, detail={"message": "Some files failed to update.", "details": results})

