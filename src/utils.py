import logging
import hashlib
import sys

# Setting up a simple logger to track what's happening
# This will print to the console so we can see errors or info
def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Check if we already have handlers to avoid duplicate logs
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

# Helper function to create a unique ID for a piece of text
# We use MD5 because it's fast and good enough for checking duplicates
def compute_md5_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# Simple check to see if text is "garbage" or not useful
# If it's too short or has weird characters, we probably want to skip it
def is_valid_text(text, min_length=50):
    if not text:
        return False
        
    # Check 1: Length
    if len(text.strip()) < min_length:
        return False
        
    # Check 2: Alphanumeric density (avoid OCR noise like ".,,;..'")
    # We count valid letters/numbers
    alnum_count = sum(c.isalnum() for c in text)
    if alnum_count / len(text) < 0.3: # If less than 30% is readable
        return False
        
    return True
