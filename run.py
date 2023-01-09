import os 
import sys
env_path = os.path.dirname(__file__)
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.run_webcam import run_webcam

def run():
    return run_webcam()