from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.models.forecast_settings import ForecastSettings
from app.repositories.forecast_settings_repository import ForecastSettingsRepository

router = APIRouter()
repo = ForecastSettingsRepository()

@router.post("/forecast_settings/", response_model=ForecastSettings)
async def create_forecast_settings(forecast_settings: ForecastSettings) -> ForecastSettings:
    return repo.create_forecast_settings(forecast_settings)

@router.get("/forecast_settings/{forecast_settings_id}", response_model=ForecastSettings)
async def get_forecast_settings(forecast_settings_id: int) -> ForecastSettings:
    forecast_settings = repo.get_forecast_settings(forecast_settings_id)
    if not forecast_settings:
        raise HTTPException(status_code=404, detail="Forecast settings not found")
    return forecast_settings

@router.put("/forecast_settings/{forecast_settings_id}", response_model=ForecastSettings)
async def update_forecast_settings(forecast_settings_id: int, forecast_settings: ForecastSettings) -> ForecastSettings:
    existing_forecast_settings = repo.get_forecast_settings(forecast_settings_id)
    if not existing_forecast_settings:
        raise HTTPException(status_code=404, detail="Forecast settings not found")
    return repo.update_forecast_settings(forecast_settings_id, forecast_settings)

@router.delete("/forecast_settings/{forecast_settings_id}", response_model=ForecastSettings)
async def delete_forecast_settings(forecast_settings_id: int) -> ForecastSettings:
    existing_forecast_settings = repo.get_forecast_settings(forecast_settings_id)
    if not existing_forecast_settings:
        raise HTTPException(status_code=404, detail="Forecast settings not found")
    return repo.delete_forecast_settings(forecast_settings_id)

@router.get("/forecast_settings/board/{board_id}", response_model=ForecastSettings)
async def get_forecast_settings_by_board_id(board_id: int) -> ForecastSettings:
    return repo.get_forecast_settings_by_board_id(board_id)
