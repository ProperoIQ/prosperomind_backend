from fastapi import APIRouter, HTTPException
from typing import List
from app.repositories.forecast_chat_response_repository import ForecastChatResponseRepository
from app.models.forecast_chat_response import ForecastChatResponse

router = APIRouter(prefix="/forecast-chat-response", tags=["Forecast Chat Response"])
repository = ForecastChatResponseRepository()

@router.get("/id/{response_id}", response_model=ForecastChatResponse)
async def get_forecast_chat_response(response_id: int):
    response = await repository.get_forecast_chat_response(response_id)
    if not response:
        raise HTTPException(status_code=404, detail="Forecast chat response not found")
    return response

@router.put("/id/{response_id}", response_model=ForecastChatResponse)
async def update_forecast_chat_response(response_id: int, forecast_chat_response: ForecastChatResponse):
    updated_response = await repository.update_response_in_database(response_id, forecast_chat_response.dict())
    if not updated_response:
        raise HTTPException(status_code=404, detail="Forecast chat response not found")
    return updated_response

@router.delete("/id/{response_id}", response_model=ForecastChatResponse)
async def delete_forecast_chat_response(response_id: int):
    deleted_response = await repository.delete_response_from_database(response_id)
    if not deleted_response:
        raise HTTPException(status_code=404, detail="Forecast chat response not found")
    return deleted_response

@router.get("/board/{board_id}", response_model=List[ForecastChatResponse])
async def get_forecast_chat_responses_for_board(board_id: int):
    responses = await repository.get_responses_for_board(board_id)
    return responses

@router.get("/forecast_response_id/{forecast_response_id}", response_model=List[ForecastChatResponse])
async def get_forecast_chat_responses_for_forecast(forecast_response_id: int):
    responses = await repository.get_responses_for_using_forecast_id(forecast_response_id)
    return responses
