# app/repositories/master_settings_repository.py
from typing import Any
from sqlalchemy import text
from app.repositories.base_repository import BaseRepository
from app.models.master_settings import MasterSettings

class MasterSettingsRepository(BaseRepository):
    def __init__(self):
        super().__init__('MasterSettings')
        create_table_query = text("""
            CREATE TABLE IF NOT EXISTS MasterSettings (
                id SERIAL PRIMARY KEY,
                board_id INT REFERENCES Boards(id),
                configuration_details TEXT,
                name VARCHAR(255),
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            );
        """)
        self.create_table(create_table_query)

    def create_master_settings(self, master_settings: MasterSettings) -> Any:
        query = text("""
            INSERT INTO MasterSettings (board_id, configuration_details, name, created_at, updated_at)
            VALUES (:board_id, :configuration_details, :name, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING id, board_id, configuration_details, name, created_at, updated_at;
        """)

        values = {
            "board_id": master_settings.board_id,
            "configuration_details": master_settings.configuration_details,
            "name": master_settings.name,
        }

        master_settings_data_tuple = self.execute_query(query, values)
        master_settings_instance = MasterSettings(**dict(zip(MasterSettings.__annotations__, master_settings_data_tuple)))
        return master_settings_instance

    def get_all_master_settings(self) -> Any:
        query = text("""
            SELECT * FROM MasterSettings;
        """)

        master_settings_data_list = self.execute_query_all(query)
        master_settings_list = []

        for settings_data in master_settings_data_list:
            master_settings_instance = MasterSettings(**dict(zip(MasterSettings.__annotations__, settings_data)))
            master_settings_instance.configuration_details = eval(master_settings_instance.configuration_details)
            master_settings_list.append(master_settings_instance)

        return master_settings_list

    def get_master_settings(self, settings_id: int) -> Any:
        query = text("""
            SELECT * FROM MasterSettings WHERE id = :settings_id;
        """)

        values = {"settings_id": settings_id}

        master_settings_data_tuple = self.execute_query(query, values)
        if master_settings_data_tuple is None:
            return None
        master_settings_instance = MasterSettings(**dict(zip(MasterSettings.__annotations__, master_settings_data_tuple)))
        master_settings_instance.configuration_details = eval(master_settings_instance.configuration_details)
        return master_settings_instance

    def update_master_settings(self, settings_id: int, master_settings: MasterSettings) -> Any:
        query = text("""
            UPDATE MasterSettings
            SET board_id = :board_id, configuration_details = :configuration_details,
                name = :name, updated_at = CURRENT_TIMESTAMP
            WHERE id = :settings_id
            RETURNING id, board_id, configuration_details, name, created_at, updated_at;
        """)

        values = {
            "board_id": master_settings.board_id,
            "configuration_details": master_settings.configuration_details,
            "name": master_settings.name,
            "settings_id": settings_id
        }

        master_settings_data_tuple = self.execute_query(query, values)
        if master_settings_data_tuple is None:
            return master_settings_data_tuple
        master_settings_instance = MasterSettings(**dict(zip(MasterSettings.__annotations__, master_settings_data_tuple)))
        return master_settings_instance

    def delete_master_settings(self, settings_id: int) -> Any:
        query = text("""
            DELETE FROM MasterSettings WHERE id = :settings_id
            RETURNING id, board_id, configuration_details, name, created_at, updated_at;
        """)

        values = {"settings_id": settings_id}

        master_settings_data_tuple = self.execute_query(query, values)
        if master_settings_data_tuple is None:
            return master_settings_data_tuple
        master_settings_instance = MasterSettings(**dict(zip(MasterSettings.__annotations__, master_settings_data_tuple)))
        return master_settings_instance
