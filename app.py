import streamlit as st
import pandas as pd
from utils import results_display
from UIs import *

if __name__ == '__main__':
    sidebar.init()
    dashboard.init()
    results_display.init()
    
    print(sidebar.studentCode)