from fastapi import FastAPI, HTTPException, Query
from databases import Database

app = FastAPI()

@app.on_event("startup")
async def startup_db_client():
    # Connect to databases on startup
    await source_db.connect()
    await destination_db.connect()


@app.on_event("shutdown")
async def shutdown_db_client():
    # Disconnect from databases on shutdown
    await source_db.disconnect()
    await destination_db.disconnect()


@app.post("/copy-all-tables/")
async def copy_all_tables(
    source_db_url: str = Query(..., description="Source database URL"),
    destination_db_url: str = Query(..., description="Destination database URL"),
):
    try:
        # Create database instances
        source_db = Database(source_db_url)
        destination_db = Database(destination_db_url)

        # Connect to databases
        await source_db.connect()
        await destination_db.connect()

        # Get the list of table names from the source database
        query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        tables = await source_db.fetch_all(query)

        # Copy each table to the destination database
        for table in tables:
            table_name = table["table_name"]
            copy_query = f"INSERT INTO {table_name} SELECT * FROM {table_name}"
            await destination_db.execute(copy_query)

        return {"message": "All tables copied successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Disconnect from databases
        await source_db.disconnect()
        await destination_db.disconnect()
