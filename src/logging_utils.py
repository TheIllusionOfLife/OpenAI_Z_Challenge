"""
Logging and error handling utilities for the archaeological site discovery system.

This module provides comprehensive logging setup, error handling, and monitoring
capabilities for the entire project.
"""

import logging
import logging.handlers
import sys
import traceback
import functools
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import json
import warnings
from contextlib import contextmanager

from .config import config


class CustomFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    # Color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors

    def format(self, record):
        if self.use_colors and record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )
            record.msg = (
                f"{self.COLORS[record.levelname.strip()]}{record.msg}{self.RESET}"
            )

        # Add extra context for archaeological operations
        if hasattr(record, "operation"):
            record.msg = f"[{record.operation}] {record.msg}"

        return super().format(record)


class ArchaeologyLogger:
    """Enhanced logger for archaeological analysis operations."""

    def __init__(self, name: str, level: str = None):
        """Initialize archaeology-specific logger."""
        self.logger = logging.getLogger(name)
        self.level = level or config.logging.level
        self.setup_logger()

    def setup_logger(self):
        """Setup logger with handlers and formatters."""
        self.logger.setLevel(getattr(logging, self.level.upper()))

        # Clear existing handlers
        self.logger.handlers.clear()

        # File handler
        if config.logging.file_handler:
            log_file = config.data_paths.logs_dir / f"{self.logger.name}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=config.logging.max_file_size,
                backupCount=config.logging.backup_count,
            )
            file_formatter = logging.Formatter(config.logging.format)
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)

        # Console handler
        if config.logging.console_handler:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = CustomFormatter(use_colors=True)
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(getattr(logging, self.level.upper()))
            self.logger.addHandler(console_handler)

    def log_operation(self, operation: str, level: str = "INFO"):
        """Decorator for logging operation start/end."""

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                self.logger.info(
                    f"Starting {operation}", extra={"operation": operation}
                )

                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    self.logger.info(
                        f"Completed {operation} in {duration:.2f}s",
                        extra={"operation": operation},
                    )
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self.logger.error(
                        f"Failed {operation} after {duration:.2f}s: {str(e)}",
                        extra={"operation": operation},
                    )
                    raise

            return wrapper

        return decorator

    def log_data_processing(self, data_type: str, shape: tuple, processing_type: str):
        """Log data processing information."""
        self.logger.info(
            f"Processing {data_type} data: shape={shape}, type={processing_type}",
            extra={"operation": "data_processing"},
        )

    def log_site_discovery(self, coordinates: tuple, confidence: float, features: list):
        """Log archaeological site discovery."""
        self.logger.info(
            f"Site discovered at {coordinates} with confidence {confidence:.3f} "
            f"and features: {', '.join(features)}",
            extra={"operation": "site_discovery"},
        )

    def log_openai_request(self, model: str, tokens: int, task: str):
        """Log OpenAI API request."""
        self.logger.debug(
            f"OpenAI request: model={model}, tokens={tokens}, task={task}",
            extra={"operation": "openai_api"},
        )

    def log_validation_result(self, site_id: str, is_valid: bool, confidence: float):
        """Log site validation result."""
        status = "VALID" if is_valid else "INVALID"
        self.logger.info(
            f"Site {site_id} validation: {status} (confidence: {confidence:.3f})",
            extra={"operation": "validation"},
        )


class ErrorHandler:
    """Centralized error handling for the application."""

    def __init__(self, logger: Optional[ArchaeologyLogger] = None):
        """Initialize error handler."""
        self.logger = logger or ArchaeologyLogger("error_handler")
        self.error_counts = {}
        self.critical_errors = []

    def handle_data_error(
        self, error: Exception, data_type: str, file_path: str = None
    ):
        """Handle data loading/processing errors."""
        error_msg = f"Data error in {data_type}"
        if file_path:
            error_msg += f" (file: {file_path})"
        error_msg += f": {str(error)}"

        self.logger.logger.error(error_msg, extra={"operation": "data_error"})
        self._track_error("data_error", error_msg)

        # Suggest recovery actions
        if "FileNotFoundError" in str(type(error)):
            self.logger.logger.info(
                "Recovery suggestion: Check file path and ensure data exists"
            )
        elif "PermissionError" in str(type(error)):
            self.logger.logger.info("Recovery suggestion: Check file permissions")
        elif "MemoryError" in str(type(error)):
            self.logger.logger.info(
                "Recovery suggestion: Process data in smaller chunks"
            )

    def handle_api_error(self, error: Exception, api_name: str, retry_count: int = 0):
        """Handle API-related errors."""
        error_msg = f"API error in {api_name} (attempt {retry_count + 1}): {str(error)}"

        if retry_count < 3:
            self.logger.logger.warning(error_msg, extra={"operation": "api_error"})
        else:
            self.logger.logger.error(error_msg, extra={"operation": "api_error"})
            self._track_error("api_error", error_msg)

        # Suggest recovery actions based on error type
        if "rate limit" in str(error).lower():
            self.logger.logger.info(
                "Recovery suggestion: Implement exponential backoff"
            )
        elif "authentication" in str(error).lower():
            self.logger.logger.info("Recovery suggestion: Check API key configuration")
        elif "timeout" in str(error).lower():
            self.logger.logger.info("Recovery suggestion: Increase timeout value")

    def handle_processing_error(
        self, error: Exception, operation: str, data_info: Dict[str, Any] = None
    ):
        """Handle data processing errors."""
        error_msg = f"Processing error in {operation}: {str(error)}"
        if data_info:
            error_msg += f" (data info: {data_info})"

        self.logger.logger.error(error_msg, extra={"operation": "processing_error"})
        self._track_error("processing_error", error_msg)

        # Log stack trace for debugging
        self.logger.logger.debug(f"Stack trace:\n{traceback.format_exc()}")

    def handle_critical_error(self, error: Exception, context: str):
        """Handle critical errors that might stop execution."""
        error_msg = f"CRITICAL ERROR in {context}: {str(error)}"
        self.logger.logger.critical(error_msg, extra={"operation": "critical_error"})

        # Store critical error for reporting
        critical_info = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "error": str(error),
            "type": str(type(error).__name__),
            "traceback": traceback.format_exc(),
        }
        self.critical_errors.append(critical_info)

        # Save critical error to file
        self._save_critical_error(critical_info)

    def _track_error(self, error_type: str, message: str):
        """Track error occurrences."""
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1

        # Alert if too many errors of same type
        if self.error_counts[error_type] > 10:
            self.logger.logger.warning(
                f"High frequency of {error_type} errors: {self.error_counts[error_type]} occurrences"
            )

    def _save_critical_error(self, error_info: Dict[str, Any]):
        """Save critical error to file for analysis."""
        try:
            error_file = config.data_paths.logs_dir / "critical_errors.json"

            # Load existing errors
            existing_errors = []
            if error_file.exists():
                with open(error_file, "r") as f:
                    existing_errors = json.load(f)

            # Add new error
            existing_errors.append(error_info)

            # Keep only last 100 critical errors
            existing_errors = existing_errors[-100:]

            # Save back to file
            with open(error_file, "w") as f:
                json.dump(existing_errors, f, indent=2)

        except Exception as e:
            self.logger.logger.error(f"Failed to save critical error: {e}")

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered."""
        return {
            "error_counts": self.error_counts,
            "critical_errors_count": len(self.critical_errors),
            "last_critical_error": (
                self.critical_errors[-1] if self.critical_errors else None
            ),
        }


def setup_project_logging():
    """Setup logging for the entire project."""
    # Ensure log directory exists
    config.data_paths.logs_dir.mkdir(parents=True, exist_ok=True)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove default handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Setup main project log file
    main_log_file = config.data_paths.logs_dir / "main.log"
    file_handler = logging.handlers.RotatingFileHandler(
        main_log_file,
        maxBytes=config.logging.max_file_size,
        backupCount=config.logging.backup_count,
    )
    file_formatter = logging.Formatter(config.logging.format)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    # Setup console handler for root logger
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = CustomFormatter(use_colors=True)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(getattr(logging, config.logging.level.upper()))
    root_logger.addHandler(console_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)

    # Handle warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)


@contextmanager
def log_execution_time(logger: ArchaeologyLogger, operation: str):
    """Context manager to log execution time."""
    start_time = time.time()
    logger.logger.info(f"Starting {operation}", extra={"operation": operation})

    try:
        yield
        duration = time.time() - start_time
        logger.logger.info(
            f"Completed {operation} in {duration:.2f}s", extra={"operation": operation}
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.logger.error(
            f"Failed {operation} after {duration:.2f}s: {str(e)}",
            extra={"operation": operation},
        )
        raise


def log_function_calls(logger: Optional[ArchaeologyLogger] = None):
    """Decorator to log function calls with parameters and results."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = logger or ArchaeologyLogger(func.__module__)

            # Log function call
            func_logger.logger.debug(
                f"Calling {func.__name__} with args={len(args)}, kwargs={list(kwargs.keys())}",
                extra={"operation": "function_call"},
            )

            try:
                result = func(*args, **kwargs)
                func_logger.logger.debug(
                    f"Function {func.__name__} completed successfully",
                    extra={"operation": "function_call"},
                )
                return result
            except Exception as e:
                func_logger.logger.error(
                    f"Function {func.__name__} failed: {str(e)}",
                    extra={"operation": "function_call"},
                )
                raise

        return wrapper

    return decorator


def create_progress_logger(total_items: int, operation: str) -> Callable:
    """Create a progress logging function."""
    logger = ArchaeologyLogger("progress")

    def log_progress(current: int, message: str = ""):
        percentage = (current / total_items) * 100
        logger.logger.info(
            f"{operation} progress: {current}/{total_items} ({percentage:.1f}%) {message}",
            extra={"operation": "progress"},
        )

    return log_progress


# Global error handler instance
error_handler = ErrorHandler()

# Setup logging when module is imported
setup_project_logging()
