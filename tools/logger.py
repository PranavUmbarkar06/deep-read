from datetime import datetime

def log(action,content):
    """
    Appends a log entry to the 'logs.txt' file with a timestamp.

    Args:
        action (str): The action being logged.  
        content (str): The content to be logged.
    """
    with open('logs.txt', 'a',encoding='utf-8',errors='replace') as f:
        f.write(f"{datetime.now()} - {action}:  {content}\n")