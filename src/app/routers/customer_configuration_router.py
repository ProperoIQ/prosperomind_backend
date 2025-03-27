# app/routers/customer_configuration_router.py

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from app.repositories.customer_configuration_repository import CustomerConfigurationRepository
from app.models.customer_configuration import CustomerConfiguration

router = APIRouter(prefix="/customer-configurations", tags=["Customer Configurations"])

customer_configuration_repository = CustomerConfigurationRepository()

@router.post("/", response_model=CustomerConfiguration)
async def create_customer_configuration(customer_configuration: CustomerConfiguration):
    created_configuration = customer_configuration_repository.create_customer_configuration(customer_configuration)
    return created_configuration

@router.get("/", response_model=List[CustomerConfiguration])
async def get_customer_configurations():
    configurations = customer_configuration_repository.get_customer_configurations()
    return configurations

@router.get("/{configuration_id}", response_model=CustomerConfiguration)
async def get_customer_configuration(configuration_id: int):
    configuration = customer_configuration_repository.get_customer_configuration(configuration_id)
    if not configuration:
        raise HTTPException(status_code=404, detail="Customer Configuration not found")
    return configuration

@router.put("/{configuration_id}", response_model=CustomerConfiguration)
async def update_customer_configuration(configuration_id: int, customer_configuration: CustomerConfiguration):
    #updated_configuration = customer_configuration_repository.update_customer_configuration(configuration_id, customer_configuration)
    deleted_configuration = customer_configuration_repository.delete_customer_configuration(configuration_id)
    updated_configuration = customer_configuration_repository.create_customer_configuration(customer_configuration)    
    if not updated_configuration:
        raise HTTPException(status_code=404, detail="Customer Configuration not found")
    return updated_configuration

@router.delete("/{configuration_id}", response_model=dict)
async def delete_customer_configuration(configuration_id: int):
    deleted_configuration = customer_configuration_repository.delete_customer_configuration(configuration_id)
    if not deleted_configuration:
        raise HTTPException(status_code=404, detail="Customer Configuration not found")
    response_data = {"status_code": 200, "detail": "Customer Configuration deleted successfully"}
    return response_data
