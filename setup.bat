@echo off
REM Create virtual environment
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate

REM Install dependencies
pip install -r requirements.txt

REM Create .env file if it doesn't exist
if not exist .env (
    copy .env.example .env
    echo Created .env file. Please add your OpenAI API key to .env
) 