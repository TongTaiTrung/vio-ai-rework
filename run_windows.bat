@echo off
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
pause