# app/repositories/customer_configuration_repository.py

from typing import Any
from sqlalchemy import text
from app.repositories.base_repository import BaseRepository
from app.models.customer_configuration import CustomerConfiguration

class CustomerConfigurationRepository(BaseRepository):
    def __init__(self):
        super().__init__('customer_configuration')

    def create_customer_configuration(self, customer_configuration: CustomerConfiguration) -> Any:
        query = text("""
            INSERT INTO customer_configuration (user_id, configuration, created_at, updated_at, client_number, customer_number)
            VALUES (:user_id, :configuration, :created_at, :updated_at, :client_number, :customer_number)
            RETURNING id, user_id, configuration, created_at, updated_at, client_number, customer_number;
        """)

        values = {
            "user_id": customer_configuration.user_id,
            "configuration": customer_configuration.configuration,
            "created_at": customer_configuration.created_at,
            "updated_at": customer_configuration.updated_at,
            "client_number": customer_configuration.client_number,
            "customer_number": customer_configuration.customer_number
        }

        configuration_data_tuple = self.execute_query(query, values)
        configuration_instance = CustomerConfiguration(**dict(zip(CustomerConfiguration.__annotations__, configuration_data_tuple)))
        return configuration_instance

    def get_customer_configurations(self) -> Any:
        query = text("""
            SELECT * FROM customer_configuration;
        """)

        configuration_data_list = self.execute_query_all(query)
        configuration_dict = [CustomerConfiguration(**dict(zip(CustomerConfiguration.__annotations__, configuration_data))) for configuration_data in configuration_data_list]
        return configuration_dict

    def get_customer_configuration(self, configuration_id: int) -> Any:
        query = text("""
            SELECT * FROM customer_configuration WHERE id = :configuration_id;
        """)

        values = {"configuration_id": configuration_id}

        configuration_data_tuple = self.execute_query(query, values)
        configuration_instance = CustomerConfiguration(**dict(zip(CustomerConfiguration.__annotations__, configuration_data_tuple)))
        return configuration_instance

    def update_customer_configuration(self, configuration_id: int, customer_configuration: CustomerConfiguration) -> Any:
        query = text("""
            UPDATE customer_configuration
            SET configuration = :configuration, updated_at = :updated_at,
                client_number = :client_number, customer_number = :customer_number
            WHERE id = :configuration_id
            RETURNING id, user_id, configuration, created_at, updated_at, client_number, customer_number;
        """)

        values = {
            "configuration": customer_configuration.configuration,
            "updated_at": customer_configuration.updated_at,
            "client_number": customer_configuration.client_number,
            "customer_number": customer_configuration.customer_number,
            "configuration_id": configuration_id
        }

        configuration_data_tuple = self.execute_query(query, values)
        configuration_instance = CustomerConfiguration(**dict(zip(CustomerConfiguration.__annotations__, configuration_data_tuple)))
        return configuration_instance

    def delete_customer_configuration(self, configuration_id: int) -> Any:
        query = text("""
            DELETE FROM customer_configuration WHERE id = :configuration_id
            RETURNING id, user_id, configuration, created_at, updated_at, client_number, customer_number;
        """)

        values = {"configuration_id": configuration_id}

        configuration_data_tuple = self.execute_query(query, values)
        configuration_instance = CustomerConfiguration(**dict(zip(CustomerConfiguration.__annotations__, configuration_data_tuple)))
        return configuration_instance
