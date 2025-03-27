# app/repositories/other_parameter_settings_repository.py
from typing import Any
from sqlalchemy import text
from app.repositories.base_repository import BaseRepository
from app.models.other_parameter_settings import OtherParameterSettings

class OtherParameterSettingsRepository(BaseRepository):
    def __init__(self):
        super().__init__('OtherParameterSettings')
        create_table_query = text("""
            CREATE TABLE IF NOT EXISTS OtherParameterSettings (
                id SERIAL PRIMARY KEY,
                board_id INT REFERENCES Boards(id),
                configuration_details TEXT,
                name VARCHAR(255),
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            );
        """)
        self.create_table(create_table_query)

    def create_other_parameter_settings(self, other_parameter_settings: OtherParameterSettings) -> Any:
        query = text("""
            INSERT INTO OtherParameterSettings (board_id, configuration_details, name, created_at, updated_at)
            VALUES (:board_id, :configuration_details, :name, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING id, board_id, configuration_details, name, created_at, updated_at;
        """)

        values = {
            "board_id": other_parameter_settings.board_id,
            "configuration_details": other_parameter_settings.configuration_details,
            "name": other_parameter_settings.name,
        }

        other_parameter_settings_data_tuple = self.execute_query(query, values)
        other_parameter_settings_instance = OtherParameterSettings(**dict(zip(OtherParameterSettings.__annotations__, other_parameter_settings_data_tuple)))
        return other_parameter_settings_instance

    def get_all_other_parameter_settings(self) -> Any:
        query = text("""
            SELECT * FROM OtherParameterSettings;
        """)

        other_parameter_settings_data_list = self.execute_query_all(query)
        other_parameter_settings_list = []

        for settings_data in other_parameter_settings_data_list:
            other_parameter_settings_instance = OtherParameterSettings(**dict(zip(OtherParameterSettings.__annotations__, settings_data)))
            other_parameter_settings_instance.configuration_details = eval(other_parameter_settings_instance.configuration_details)
            other_parameter_settings_list.append(other_parameter_settings_instance)

        return other_parameter_settings_list

    def get_other_parameter_settings(self, settings_id: int) -> Any:
        query = text("""
            SELECT * FROM OtherParameterSettings WHERE id = :settings_id;
        """)

        values = {"settings_id": settings_id}

        other_parameter_settings_data_tuple = self.execute_query(query, values)
        if other_parameter_settings_data_tuple is None:
            return None
        other_parameter_settings_instance = OtherParameterSettings(**dict(zip(OtherParameterSettings.__annotations__, other_parameter_settings_data_tuple)))
        other_parameter_settings_instance.configuration_details = eval(other_parameter_settings_instance.configuration_details)
        return other_parameter_settings_instance

    def update_other_parameter_settings(self, settings_id: int, other_parameter_settings: OtherParameterSettings) -> Any:
        query = text("""
            UPDATE OtherParameterSettings
            SET board_id = :board_id, configuration_details = :configuration_details, name = :name, updated_at = CURRENT_TIMESTAMP
            WHERE id = :settings_id
            RETURNING id, board_id, configuration_details, name, created_at, updated_at;
        """)

        values = {
            "board_id": other_parameter_settings.board_id,
            "configuration_details": other_parameter_settings.configuration_details,
            "name": other_parameter_settings.name,
            "settings_id": settings_id
        }

        other_parameter_settings_data_tuple = self.execute_query(query, values)
        if other_parameter_settings_data_tuple is None:
            return other_parameter_settings_data_tuple
        other_parameter_settings_instance = OtherParameterSettings(**dict(zip(OtherParameterSettings.__annotations__, other_parameter_settings_data_tuple)))
        return other_parameter_settings_instance

    def delete_other_parameter_settings(self, settings_id: int) -> Any:
        query = text("""
            DELETE FROM OtherParameterSettings WHERE id = :settings_id
            RETURNING id, board_id, configuration_details, name, created_at, updated_at;
        """)

        values = {"settings_id": settings_id}

        other_parameter_settings_data_tuple = self.execute_query(query, values)
        if other_parameter_settings_data_tuple is None:
            return other_parameter_settings_data_tuple
        other_parameter_settings_instance = OtherParameterSettings(**dict(zip(OtherParameterSettings.__annotations__, other_parameter_settings_data_tuple)))
        return other_parameter_settings_instance
