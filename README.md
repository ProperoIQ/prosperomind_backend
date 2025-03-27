# FastAPI Project

This is a FastAPI project that implements an API for managing boards and prompts.

## Code Structure

The project is structured as follows:

- `app`: Contains the main application files.
  - `__init__.py`: Initializes the FastAPI app.
  - `database.py`: Configures the database connection.
  - `dependencies.py`: Defines dependency functions.
  - `logging_config.ini`: Configuration file for logging.
  - `main.py`: Main application entry point.
  - `routers`: Contains API routers.
    - `__init__.py`: Initializes the routers.
    - `board_router.py`: Handles routes related to boards.
    - `prompt_router.py`: Handles routes related to prompts.
    - `user_router.py`: Handles routes related to users.
  - `repositories`: Contains database interaction logic.
    - `__init__.py`: Initializes the repositories.
    - `base_repository.py`: Base repository with common functions.
    - `board_repository.py`: Repository for board-related operations.
    - `prompt_repository.py`: Repository for prompt-related operations.
    - `user_repository.py`: Repository for user-related operations.
  - `models`: Contains Pydantic models.
    - `__init__.py`: Initializes the models package.
    - `board.py`: Pydantic model for the board.
    - `prompt.py`: Pydantic model for the prompt.
    - `user.py`: Pydantic model for the user.
- `config.ini`: Configuration file for database connection.
- `requirements.txt`: Lists project dependencies.

## Getting Started

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Configure the database connection in `config.ini`.
4. Run these on terminal
export OPENAI_API_KEY="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"
export GOOGLE_APPLICATION_CREDENTIALS="/home/hp/Desktop/Work/FinGPT/src/independent-way-410316-5de888869e79.json"
5. Run the FastAPI app: `uvicorn main:app --reload`.

## Usage

- Visit the FastAPI documentation at `http://127.0.0.1:8000/docs` for interactive API documentation.


## For demo

export OPENAI_API_KEY="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"
export GOOGLE_APPLICATION_CREDENTIALS="/home/hp/Desktop/Work/FinGPT/src/independent-way-410316-5de888869e79.json"


