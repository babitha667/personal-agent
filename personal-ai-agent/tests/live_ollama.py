"""Compatibility entry point: run the current generic planner against local Ollama."""
import os
os.environ['LLM_PROVIDER']='ollama'
from live_agent import main

if __name__=='__main__':
    main()
