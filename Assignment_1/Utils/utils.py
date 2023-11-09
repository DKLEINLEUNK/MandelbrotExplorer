import logging
import time

# File and Directory Management: Functions to create directories for saving plots, check if a file already exists, or handle file paths.

def ensure_directory_exists(path):
    """Ensure that a directory exists, and if not, create it."""
    if not os.path.exists(path):
        os.makedirs(path)

# Data Handling: Functions to handle the input data (i.e. data processing) or parameters (e.g. data validation) to ensure they meet certain criteria.

def normalize_data(data):
    """Normalize data to a certain range or format."""
    # Implementation depends on the data format and requirements
    ...

def validate_positive_integer(value, name="Value"):
    """Ensure a value is a positive integer."""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer. Got {value}.")

# Formatting and Conversions: Functions to format data for plotting or convert between different data types or structures.

def format_plot_title(title):
    """Format a title string for a plot."""
    return title.replace("_", " ").title()

# Logging and Debugging: Functions to log information, warnings, or errors, or to assist with debugging.

def configure_logging(level=logging.INFO):
    """Configure logging for the project."""
    logging.basicConfig(level=level)

def timing_decorator(func):
    """A decorator to measure the execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} executed in {end_time - start_time:.2f}s")
        return result
    return wrapper

def handle_error(e):
    """Handle an error, use standardized logging."""
    logging.error(f"An error occurred: {str(e)}")
