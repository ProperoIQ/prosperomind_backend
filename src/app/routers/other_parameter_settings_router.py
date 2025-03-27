# app/routers/other_parameter_settings_router.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.repositories.other_parameter_settings_repository import OtherParameterSettingsRepository
from app.models.other_parameter_settings import OtherParameterSettings

router = APIRouter(prefix="/other-parameter-settings", tags=["Other Parameter Settings"])

other_parameter_settings_repository = OtherParameterSettingsRepository()

@router.post("/", response_model=OtherParameterSettings)
async def create_other_parameter_settings(other_parameter_settings: OtherParameterSettings):
    try:
        # Attempt to create Other Parameter Settings
        created_settings = other_parameter_settings_repository.create_other_parameter_settings(other_parameter_settings)
        return created_settings
    except Exception as e:
        # Check if the error is due to a duplicate key violation
        if "duplicate key value violates unique constraint" in str(e):
            raise HTTPException(status_code=400, detail="Other Parameter Settings with this name already exist")
        else:
            # Re-raise the exception if it's not related to duplicate key violation
            raise HTTPException(status_code=500, detail="Other Parameter Settings API failed to process")

@router.get("/", response_model=List[OtherParameterSettings])
async def get_all_other_parameter_settings():
    all_settings = other_parameter_settings_repository.get_all_other_parameter_settings()
    return all_settings

@router.get("/{settings_id}", response_model=OtherParameterSettings)
async def get_other_parameter_settings(settings_id: int):
    settings = other_parameter_settings_repository.get_other_parameter_settings(settings_id)
    if not settings:
        raise HTTPException(status_code=404, detail="Other Parameter Settings not found")
    return settings

@router.put("/{settings_id}", response_model=OtherParameterSettings)
async def update_other_parameter_settings(settings_id: int, other_parameter_settings: OtherParameterSettings):
    updated_settings = other_parameter_settings_repository.update_other_parameter_settings(settings_id, other_parameter_settings)
    if not updated_settings:
        raise HTTPException(status_code=404, detail="Other Parameter Settings not found")
    return updated_settings

@router.delete("/{settings_id}", response_model=dict)
async def delete_other_parameter_settings(settings_id: int):
    deleted_settings = other_parameter_settings_repository.delete_other_parameter_settings(settings_id)
    if not deleted_settings:
        raise HTTPException(status_code=404, detail="Other Parameter Settings not found")
    response_data = {"status_code": 200, "detail": "Other Parameter Settings deleted successfully"}
    return response_data
