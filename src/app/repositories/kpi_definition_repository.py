# app/repositories/kpi_definition_repository.py
from typing import Any
from sqlalchemy import text
from app.repositories.base_repository import BaseRepository
from app.models.kpi_definition import KPIDefinition

class KPIDefinitionRepository(BaseRepository):
    def __init__(self):
        super().__init__('KPIDefinition')
        create_table_query = text("""
            CREATE TABLE IF NOT EXISTS KPIDefinition (
                id SERIAL PRIMARY KEY,
                board_id INT REFERENCES Boards(id),
                configuration_details TEXT,
                name VARCHAR(255) UNIQUE,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            );
        """)
        self.create_table(create_table_query)

    def create_kpi_definition(self, kpi_definition: KPIDefinition) -> Any:
        query = text("""
            INSERT INTO KPIDefinition (board_id, configuration_details, name, created_at, updated_at)
            VALUES (:board_id, :configuration_details, :name, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING id, board_id, configuration_details, name, created_at, updated_at;
        """)

        values = {
            "board_id": kpi_definition.board_id,
            "configuration_details": kpi_definition.configuration_details,
            "name": kpi_definition.name,
        }

        kpi_definition_data_tuple = self.execute_query(query, values)
        kpi_definition_instance = KPIDefinition(**dict(zip(KPIDefinition.__annotations__, kpi_definition_data_tuple)))
        return kpi_definition_instance

    def get_all_kpi_definitions(self) -> Any:
        query = text("""
            SELECT * FROM KPIDefinition;
        """)

        kpi_definition_data_list = self.execute_query_all(query)
        kpi_definition_list = []

        for definition_data in kpi_definition_data_list:
            kpi_definition_instance = KPIDefinition(**dict(zip(KPIDefinition.__annotations__, definition_data)))
            kpi_definition_instance.configuration_details = eval(kpi_definition_instance.configuration_details)
            kpi_definition_list.append(kpi_definition_instance)

        return kpi_definition_list

    def get_kpi_definition(self, definition_id: int) -> Any:
        query = text("""
            SELECT * FROM KPIDefinition WHERE id = :definition_id;
        """)

        values = {"definition_id": definition_id}

        kpi_definition_data_tuple = self.execute_query(query, values)
        if kpi_definition_data_tuple is None:
            return None
        kpi_definition_instance = KPIDefinition(**dict(zip(KPIDefinition.__annotations__, kpi_definition_data_tuple)))
        kpi_definition_instance.configuration_details = eval(kpi_definition_instance.configuration_details)
        return kpi_definition_instance

    def update_kpi_definition(self, definition_id: int, kpi_definition: KPIDefinition) -> Any:
        query = text("""
            UPDATE KPIDefinition
            SET board_id = :board_id, configuration_details = :configuration_details, name = :name, updated_at = CURRENT_TIMESTAMP
            WHERE id = :definition_id
            RETURNING id, board_id, configuration_details, name, created_at, updated_at;
        """)

        values = {
            "board_id": kpi_definition.board_id,
            "configuration_details": kpi_definition.configuration_details,
            "name": kpi_definition.name,
            "definition_id": definition_id
        }

        kpi_definition_data_tuple = self.execute_query(query, values)
        if kpi_definition_data_tuple is None:
            return kpi_definition_data_tuple
        kpi_definition_instance = KPIDefinition(**dict(zip(KPIDefinition.__annotations__, kpi_definition_data_tuple)))
        return kpi_definition_instance

    def delete_kpi_definition(self, definition_id: int) -> Any:
        query = text("""
            DELETE FROM KPIDefinition WHERE id = :definition_id
            RETURNING id, board_id, configuration_details, name, created_at, updated_at;
        """)

        values = {"definition_id": definition_id}

        kpi_definition_data_tuple = self.execute_query(query, values)
        if kpi_definition_data_tuple is None:
            return kpi_definition_data_tuple
        kpi_definition_instance = KPIDefinition(**dict(zip(KPIDefinition.__annotations__, kpi_definition_data_tuple)))
        return kpi_definition_instance
