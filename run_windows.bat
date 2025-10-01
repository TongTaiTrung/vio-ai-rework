@echo off
python3.10 -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
python3.10 -m streamlit run app.py
pause