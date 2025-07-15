#!/usr/bin/env python3
"""
Logging configuration using Rich library for NewsXP.ai
"""

import logging
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn


# Global console instance for consistent styling
console = Console()


def setup_logging(level: str = "INFO", 
                 quiet_mode: bool = False,
                 show_progress: bool = True) -> None:
    """
    Set up clean, developer-friendly logging using Rich.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        quiet_mode: If True, reduce noise from third-party libraries
        show_progress: If True, enable rich formatting (always True with Rich)
    """
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set up Rich handler with custom formatting
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
        log_time_format="[%H:%M:%S]"
    )
    
    # Custom format for cleaner output
    rich_handler.setFormatter(logging.Formatter(
        fmt="%(message)s",
        datefmt="[%X]"
    ))
    
    root_logger.addHandler(rich_handler)
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Quiet mode: reduce third-party library noise
    if quiet_mode:
        noisy_libraries = [
            'httpx', 'groq', 'groq._base_client', 'urllib3', 'aiohttp',
            'httpcore', 'requests', 'asyncio'
        ]
        for lib in noisy_libraries:
            logging.getLogger(lib).setLevel(logging.ERROR)
    
    # Set specific log levels for our modules to reduce noise
    if level.upper() != "DEBUG":
        # Reduce initialization noise
        logging.getLogger('backend.orchestrator').setLevel(logging.WARNING)
        logging.getLogger('backend.processors').setLevel(logging.WARNING)
        logging.getLogger('backend.collectors').setLevel(logging.WARNING)
        logging.getLogger('shared.config').setLevel(logging.WARNING)
        
        # Only show important bulk agent messages
        logging.getLogger('backend.processors.bulk_agent').setLevel(logging.WARNING)


def create_progress_logger(name: str) -> logging.Logger:
    """Create a logger that can show progress updates using Rich."""
    logger = logging.getLogger(name)
    
    # Use a simple global progress tracker instead of logger attributes
    _progress_instances = {}
    
    def log_progress(message: str, progress: float):
        """Log a progress update (0-100) using Rich progress bar."""
        logger_id = id(logger)
        
        if logger_id not in _progress_instances:
            progress_bar = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console
            )
            progress_bar.start()
            task = progress_bar.add_task(message, total=100)
            _progress_instances[logger_id] = (progress_bar, task)
        
        progress_bar, task = _progress_instances[logger_id]
        progress_bar.update(task, completed=progress)
        
        if progress >= 100:
            progress_bar.stop()
            del _progress_instances[logger_id]
    
    def end_progress():
        """Signal that progress logging is complete."""
        logger_id = id(logger)
        if logger_id in _progress_instances:
            progress_bar, _ = _progress_instances[logger_id]
            progress_bar.stop()
            del _progress_instances[logger_id]
    
    # Add methods to logger instance
    setattr(logger, 'log_progress', log_progress)
    setattr(logger, 'end_progress', end_progress)
    return logger


def get_clean_logger(name: str) -> logging.Logger:
    """Get a logger with Rich formatting."""
    return logging.getLogger(name)


# Simplified helper functions using Rich styling
def log_step(logger: logging.Logger, step: str, details: str = ""):
    """Log a major pipeline step with Rich styling."""
    if details:
        logger.info(f"[green]✓[/green] [bold]{step}[/bold]: {details}")
    else:
        logger.info(f"[green]✓[/green] [bold]{step}[/bold]")


def log_progress_summary(logger: logging.Logger, operation: str, 
                        current: int, total: int, elapsed_time: Optional[float] = None):
    """Log a concise progress summary with Rich styling."""
    percentage = (current / total * 100) if total > 0 else 0
    time_info = f" ([dim]{elapsed_time:.1f}s[/dim])" if elapsed_time else ""
    logger.info(f"[blue]▸[/blue] {operation}: [bold]{current}/{total}[/bold] ([yellow]{percentage:.1f}%[/yellow]){time_info}")


def log_result(logger: logging.Logger, operation: str, 
               input_count: int, output_count: int, 
               duration: Optional[float] = None):
    """Log operation results with Rich styling."""
    rate = f" ([green]{output_count/input_count*100:.1f}% pass rate[/green])" if input_count > 0 else ""
    time_info = f" in [dim]{duration:.1f}s[/dim]" if duration else ""
    logger.info(f"[green]✓[/green] [bold]{operation}[/bold]: {input_count} → {output_count}{rate}{time_info}")


def log_warning(logger: logging.Logger, message: str):
    """Log a warning with Rich styling."""
    logger.warning(f"[yellow]⚠[/yellow] {message}")


def log_error(logger: logging.Logger, message: str):
    """Log an error with Rich styling."""
    logger.error(f"[red]✗[/red] {message}")
