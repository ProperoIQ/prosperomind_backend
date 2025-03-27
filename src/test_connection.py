# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START cloud_sql_postgres_sqlalchemy_connect_unix]
import os

from google.cloud.sql.connector import Connector, IPTypes
import pg8000

import sqlalchemy


def connect_unix_socket() -> sqlalchemy.engine.base.Engine:
    """Initializes a Unix socket connection pool for a Cloud SQL instance of Postgres."""
    # Note: Saving credentials in environment variables is convenient, but not
    # secure - consider a more secure solution such as
    # Cloud Secret Manager (https://cloud.google.com/secret-manager) to help
    # keep secrets safe.
    db_user = 'postgres'  # e.g. 'my-database-user'
    db_pass = 'a55zWlt:CYtAi|FB(|jJSpRA90N}'  # e.g. 'my-database-password'
    db_name = 'fingpt-db-dev'  # e.g. 'my-database'
    unix_socket_path = 'independent-way-410316:us-central1:collaborativedocsdb-test'   # e.g. '/cloudsql/project:region:instance'
    ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC

    # initialize Cloud SQL Python Connector object
    connector = Connector()

    def getconn() -> pg8000.dbapi.Connection:
        conn: pg8000.dbapi.Connection = connector.connect(
            unix_socket_path,
            "pg8000",
            user=db_user,
            password=db_pass,
            db=db_name,
            ip_type=ip_type
        )
        return conn

    # The Cloud SQL Python Connector can be used with SQLAlchemy
    # using the 'creator' argument to 'create_engine'
    pool = sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=getconn,
        # [START_EXCLUDE]
        # Pool size is the maximum number of permanent connections to keep.
        pool_size=5,
        # Temporarily exceeds the set pool_size if no connections are available.
        max_overflow=2,
        # The total number of concurrent connections for your application will be
        # a total of pool_size and max_overflow.
        # 'pool_timeout' is the maximum number of seconds to wait when retrieving a
        # new connection from the pool. After the specified amount of time, an
        # exception will be thrown.
        pool_timeout=30,  # 30 seconds
        # 'pool_recycle' is the maximum number of seconds a connection can persist.
        # Connections that live longer than the specified amount of time will be
        # re-established
        pool_recycle=1800,  # 30 minutes
        # [END_EXCLUDE]
    )
    return pool
# Import additional SQLAlchemy components
from sqlalchemy import inspect

def show_tables(engine):
    """Show tables in the connected database."""
    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    print("Tables in the database:")
    for table_name in table_names:
        print(table_name)

from sqlalchemy import text

def create_tables(engine):
    with engine.connect() as conn:
        # Define the table creation SQL statements
        # Replace the following data with your actual user data
        # new_user_data = {
        #     "name": "John Doe",
        #     "email": "john@example.com",
        #     "password": "secure_password",
        #     "client_number": "001",
        #     "customer_number": "12345",
        # }

        # # Insert the new user into the Users table
        # conn.execute(
        #     """
        #     INSERT INTO users (name, email, password, client_number, customer_number)
        #     VALUES (%s, %s, %s, %s, %s)
        #     RETURNING id, name, email, client_number, customer_number;
        #     """,
        #     (
        #         new_user_data["name"],
        #         new_user_data["email"],
        #         new_user_data["password"],
        #         new_user_data["client_number"],
        #         new_user_data["customer_number"],
        #     ),
        # )

        # # Fetch the inserted user's information
        # inserted_user = conn.fetchone()

        # # Print the inserted user's information
        # print("User inserted successfully:")
        # print(dict(zip(["id", "name", "email", "client_number", "customer_number"], inserted_user)))


        create_users_table = text("""
            select * from users 
        """)

        create_boards_table = text("""
            CREATE TABLE IF NOT EXISTS boards (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                name VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                client_number VARCHAR(3),
                customer_number VARCHAR(10)
            );
        """)

        create_documents_table = text("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                board_id INTEGER REFERENCES boards(id),
                name VARCHAR(255),
                size INTEGER,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                client_number VARCHAR(3),
                customer_number VARCHAR(10)
            );
        """)

        create_prompts_table = text("""
            CREATE TABLE IF NOT EXISTS prompts (
                id SERIAL PRIMARY KEY,
                board_id INTEGER REFERENCES boards(id),
                prompt_text TEXT,
                prompt_out TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                client_number VARCHAR(3),
                customer_number VARCHAR(10)
            );
        """)

        create_data_management_table = text("""
            CREATE TABLE IF NOT EXISTS data_management_table (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                table_name VARCHAR(255),
                table_description VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                client_number VARCHAR(3),
                customer_number VARCHAR(10)
            );
        """)

        #   SELECT table_name
        #                                     FROM
        #                                         DataManagementTable
        #                                     INNER JOIN
        #                                         TableStatus ON DataManagementTable.id = TableStatus.data_management_table_id
                                  
        create_customer_configuration = text("""
                                SELECT
                                    mb.id AS main_board_id,
                                    mb.name AS main_board_name,
                                    bcf.id AS bcf_id,
                                    bcf.name AS bcf_name,
                                    b.id AS board_id,
                                    b.name AS board_name
                                FROM
                                    MainBoard mb
                                LEFT JOIN
                                    BCF bcf ON mb.id = bcf.main_board_id
                                LEFT JOIN
                                    Boards b ON bcf.id = b.bcf_id;""")
        # Execute table creation SQL statements
        out = conn.execute(create_customer_configuration)
        #conn.execute(create_boards_table)
        #conn.execute(create_documents_table)
        #conn.execute(create_prompts_table)
        print(out.fetchall())
        # Commit the changes
        conn.commit()

def show_tables(engine):
    """Show tables in the connected database."""
    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    print("Tables in the database:")
    for table_name in table_names:
        print(table_name)

if __name__ == "__main__":
    #engine = connect_unix_socket()
    from app.database import get_database_connection
    engine = get_database_connection()
    
    # Example: Create tables in the connected database
    create_tables(engine)

    # Example: Show tables in the connected database
    #show_tables(engine)
    
    # Close the engine (optional)
    engine.dispose()

# if __name__ == "__main__":
#     engine = connect_unix_socket()
    
#     # Example: Show tables in the connected database
#     show_tables(engine)
    
#     # Close the engine (optional)
#     engine.dispose()
