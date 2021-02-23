from subprocess import run
from time import sleep
import time
import pandas as pd
import math
import sqlite3
import datetime 
import beach_entry_3m_7_25

restart_timer = 5.0
def start_script():
    try:
        print ('Script is Running')
        # Make sure 'python' command is available
        #run("python 3.8 "+file_path, check=True) 
        beach_entry_3m_7_25.main()
    except Exception as e:
        print (e)
        print ('Script crashed. Restarting now.')
        # Script crashed, lets restart it!
        handle_crash()

def handle_crash():
    sleep(restart_timer)  # Restarts the script after 2 seconds
    start_script()

start_script()
