# app/routers/kpi_definition_router.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.repositories.kpi_definition_repository import KPIDefinitionRepository
from app.models.kpi_definition import KPIDefinition

router = APIRouter(prefix="/kpi-definition", tags=["KPI Definition"])

kpi_definition_repository = KPIDefinitionRepository()

@router.post("/", response_model=KPIDefinition)
async def create_kpi_definition(kpi_definition: KPIDefinition):
    try:
        # Attempt to create KPI Definition
        created_definition = kpi_definition_repository.create_kpi_definition(kpi_definition)
        return created_definition
    except Exception as e:
        # Check if the error is due to a duplicate key violation
        if "duplicate key value violates unique constraint" in str(e):
            raise HTTPException(status_code=400, detail="KPI Definition with this name already exists")
        else:
            # Re-raise the exception if it's not related to duplicate key violation
            raise HTTPException(status_code=500, detail="KPI Definition API failed to process")

@router.get("/", response_model=List[KPIDefinition])
async def get_all_kpi_definitions():
    all_definitions = kpi_definition_repository.get_all_kpi_definitions()
    return all_definitions

@router.get("/{definition_id}", response_model=KPIDefinition)
async def get_kpi_definition(definition_id: int):
    definition = kpi_definition_repository.get_kpi_definition(definition_id)
    if not definition:
        raise HTTPException(status_code=404, detail="KPI Definition not found")
    return definition

@router.put("/{definition_id}", response_model=KPIDefinition)
async def update_kpi_definition(definition_id: int, kpi_definition: KPIDefinition):
    updated_definition = kpi_definition_repository.update_kpi_definition(definition_id, kpi_definition)
    if not updated_definition:
        raise HTTPException(status_code=404, detail="KPI Definition not found")
    return updated_definition

@router.delete("/{definition_id}", response_model=dict)
async def delete_kpi_definition(definition_id: int):
    deleted_definition = kpi_definition_repository.delete_kpi_definition(definition_id)
    if not deleted_definition:
        raise HTTPException(status_code=404, detail="KPI Definition not found")
    response_data = {"status_code": 200, "detail": "KPI Definition deleted successfully"}
    return response_data
