# app/repositories/bcf_repository.py
from typing import Any
from sqlalchemy import text
from app.repositories.base_repository import BaseRepository
from app.models.bcf import BCF
from app.models.main_board import  MainBoard

class BCFRepository(BaseRepository):
    def __init__(self):
        super().__init__('BCF')
        create_table_query = text("""
                                  CREATE TABLE IF NOT EXISTS BCF (
                                    id SERIAL PRIMARY KEY,
                                    main_board_id INT REFERENCES MainBoard(id),
                                    name VARCHAR(255),
                                    created_at TIMESTAMP,
                                    updated_at TIMESTAMP
                                );
                                  
        """)
        self.create_table(create_table_query)
        
    def create_bcf(self, bcf: BCF) -> Any:
        query = text("""
            INSERT INTO BCF (main_board_id, name, created_at, updated_at)
            VALUES (:main_board_id, :name, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING id, main_board_id, name, created_at, updated_at;
        """)

        values = {
            "main_board_id": bcf.main_board_id,
            "name": bcf.name,
        }

        bcf_data_tuple = self.execute_query(query, values)
        bcf_instance = BCF(**dict(zip(BCF.__annotations__, bcf_data_tuple)))
        return bcf_instance

    def get_bcfs(self) -> Any:
        query = text("""
            SELECT * FROM BCF;
        """)

        bcf_data_list = self.execute_query_all(query)
        bcf_dict = [BCF(**dict(zip(BCF.__annotations__, bcf_data))) for bcf_data in bcf_data_list]
        return bcf_dict

    def get_bcf(self, bcf_id: int) -> Any:
        query = text("""
            SELECT * FROM BCF WHERE id = :bcf_id;
        """)

        values = {"bcf_id": bcf_id}

        bcf_data_tuple = self.execute_query(query, values)
        bcf_instance = BCF(**dict(zip(BCF.__annotations__, bcf_data_tuple)))
        return bcf_instance

    def update_bcf(self, bcf_id: int, bcf: BCF) -> Any:
        query = text("""
            UPDATE BCF
            SET main_board_id = :main_board_id, name = :name, updated_at = CURRENT_TIMESTAMP
            WHERE id = :bcf_id
            RETURNING id, main_board_id, name, created_at, updated_at;
        """)

        values = {
            "main_board_id": bcf.main_board_id,
            "name": bcf.name,
            "bcf_id": bcf_id
        }

        bcf_data_tuple = self.execute_query(query, values)
        bcf_instance = BCF(**dict(zip(BCF.__annotations__, bcf_data_tuple)))
        return bcf_instance

    def delete_bcf(self, bcf_id: int) -> Any:
        query = text("""
            DELETE FROM BCF WHERE id = :bcf_id
            RETURNING id, main_board_id, name, created_at, updated_at;
        """)

        values = {"bcf_id": bcf_id}

        bcf_data_tuple = self.execute_query(query, values)
        bcf_instance = BCF(**dict(zip(BCF.__annotations__, bcf_data_tuple)))
        return bcf_instance
    
    def get_bcf_for_main_boards(self, main_board_id: int) -> Any:
        query = text("""
            SELECT BCF.*
            FROM BCF
            JOIN MainBoard ON BCF.main_board_id = MainBoard.id
            WHERE MainBoard.id = :main_board_id;
        """)

        values = {"main_board_id": main_board_id}

        bcf_data_list = self.execute_query_all(query, values)
        if bcf_data_list is None:
            return None
        bcf_dict = [BCF(**dict(zip(BCF.__annotations__, bcf_data))) for bcf_data in bcf_data_list]
        return bcf_dict