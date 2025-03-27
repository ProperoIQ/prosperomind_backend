# app/routers/bcf_router.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.repositories.bcf_repository import BCFRepository
from app.models.bcf import BCF

router = APIRouter(prefix="/bcf", tags=["BCF"])

bcf_repository = BCFRepository()

@router.post("/", response_model=BCF)
async def create_bcf(bcf: BCF):
    created_bcf = bcf_repository.create_bcf(bcf)
    return created_bcf

@router.get("/", response_model=List[BCF])
async def get_bcfs():
    bcfs = bcf_repository.get_bcfs()
    return bcfs

@router.get("/{bcf_id}", response_model=BCF)
async def get_bcf(bcf_id: int):
    bcf = bcf_repository.get_bcf(bcf_id)
    if not bcf:
        raise HTTPException(status_code=404, detail="BCF not found")
    return bcf

@router.put("/{bcf_id}", response_model=BCF)
async def update_bcf(bcf_id: int, bcf: BCF):
    updated_bcf = bcf_repository.update_bcf(bcf_id, bcf)
    if not updated_bcf:
        raise HTTPException(status_code=404, detail="BCF not found")
    return updated_bcf

@router.delete("/{bcf_id}", response_model=dict)
async def delete_bcf(bcf_id: int):
    deleted_bcf = bcf_repository.delete_bcf(bcf_id)
    if not deleted_bcf:
        raise HTTPException(status_code=404, detail="BCF not found")
    response_data = {"status_code": 200, "detail": "BCF deleted successfully"}
    return response_data

@router.get("/{main_board_id}/bcf", response_model=List[BCF])
async def get_bcf_for_main_boards(main_board_id):
    bcfs = bcf_repository.get_bcf_for_main_boards(main_board_id)
    return bcfs