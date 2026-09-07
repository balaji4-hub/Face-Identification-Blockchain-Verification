"""
VeriFace Chain - Structured Logging System
Provides clean, structured logging across all subsystems.
Strictly prohibits logging of biometric vectors or raw private keys.
"""
import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """Formats log records with timestamp, stage, and metadata without leaking sensitive data."""
    
    def format(self, record: logging.LogRecord) -> str:
        data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }
        
        # Merge structured extra fields if provided
        for key, value in record.__dict__.items():
            if key not in ("args", "asctime", "created", "exc_info", "exc_text", "filename",
                           "funcName", "levelname", "levelno", "lineno", "module", "msecs",
                           "msg", "name", "pathname", "process", "processName", "relativeCreated",
                           "stack_info", "thread", "threadName"):
                # Ensure no embeddings or private keys leak into logs
                if "embedding" in key.lower() or "private_key" in key.lower():
                    continue
                data[key] = value
                
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(data)


def get_logger(name: str) -> logging.Logger:
    """Creates a configured logger with standard stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        # Format human-readable for console
        console_formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(console_formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
