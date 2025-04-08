import jax
import jax.numpy as jnp
import functools
import time
import gc
import logging
import os
from typing import Callable, Dict, Any, Optional, TypeVar, List, Tuple
from memory_profiler import memory_usage
import threading
import psutil

# Setup a default logger with NullHandler
logger = logging.getLogger('kreg')
logger.addHandler(logging.NullHandler())

F = TypeVar('F', bound=Callable[..., Any])

def cartesian_prod(x: jax.Array, y: jax.Array):
    """
    Computes Cartesian product of two arrays x,y
    """
    a, b = jnp.meshgrid(y, x)
    full_X = jnp.vstack([b.flatten(), a.flatten()]).T
    return full_X

# Memory tracking variables
_memory_usage_data = {
    'peak_memory': 0,
    'current_memory': 0,
    'absolute_peak_memory': 0,
    'function_usage': {},
    'is_enabled': False,
    'memory_monitor_running': False,
    'monitor_thread': None,
    'monitor_interval': 0.1  # seconds
}

def start_memory_monitor(interval=0.1):
    """
    Start a background thread to continuously monitor process memory usage.
    
    Parameters
    ----------
    interval : float, default=0.1
        Interval in seconds between memory measurements
    
    Returns
    -------
    None
    """
    global _memory_usage_data
    
    if _memory_usage_data['memory_monitor_running']:
        logger.warning("Memory monitor is already running")
        return
    
    _memory_usage_data['monitor_interval'] = interval
    _memory_usage_data['memory_monitor_running'] = True
    
    def monitor_memory():
        process = psutil.Process(os.getpid())
        while _memory_usage_data['memory_monitor_running']:
            try:
                # Get current memory usage in MB
                memory_info = process.memory_info()
                current_memory = memory_info.rss / (1024 * 1024)
                
                # Update peak memory
                if current_memory > _memory_usage_data['absolute_peak_memory']:
                    _memory_usage_data['absolute_peak_memory'] = current_memory
                    logger.debug(f"New peak memory: {current_memory:.2f} MB")
                
                time.sleep(_memory_usage_data['monitor_interval'])
            except Exception as e:
                logger.error(f"Error in memory monitor: {e}")
                break
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
    monitor_thread.start()
    _memory_usage_data['monitor_thread'] = monitor_thread
    logger.info(f"Started continuous memory monitoring (interval: {interval}s)")

def stop_memory_monitor():
    """
    Stop the background memory monitoring thread.
    
    Returns
    -------
    None
    """
    global _memory_usage_data
    
    if not _memory_usage_data['memory_monitor_running']:
        logger.warning("Memory monitor is not running")
        return
    
    _memory_usage_data['memory_monitor_running'] = False
    
    # Wait for thread to terminate
    if _memory_usage_data['monitor_thread'] is not None:
        _memory_usage_data['monitor_thread'].join(timeout=1.0)
        _memory_usage_data['monitor_thread'] = None
    
    logger.info(f"Stopped memory monitoring. Absolute peak memory: {_memory_usage_data['absolute_peak_memory']:.2f} MB")

def setup_memory_tracking(enabled=False, log_level=logging.INFO, continuous_monitoring=True, monitor_interval=0.1):
    """
    Configure memory tracking for the kreg package.
    
    Parameters
    ----------
    enabled : bool, default=False
        Whether to enable memory tracking
    log_level : int, default=logging.INFO
        Logging level to use for memory tracking
    continuous_monitoring : bool, default=True
        Whether to start a background thread for continuous memory monitoring
    monitor_interval : float, default=0.1
        Interval in seconds between memory measurements when continuous monitoring is enabled
        
    Returns
    -------
    None
    """
    global _memory_usage_data
    
    # First stop monitoring if it's running
    if _memory_usage_data.get('memory_monitor_running', False):
        stop_memory_monitor()
    
    # Reset tracking data
    _memory_usage_data = {
        'peak_memory': 0,
        'current_memory': 0,
        'absolute_peak_memory': 0,
        'function_usage': {},
        'is_enabled': enabled,
        'memory_monitor_running': False,
        'monitor_thread': None,
        'monitor_interval': monitor_interval
    }
    
    # Configure logger if enabled
    if enabled:
        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create a console handler and set its level and formatter
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        
        # Get the kreg logger and set its level
        logger.setLevel(log_level)
        
        # Check if the logger already has handlers to avoid duplicates
        if not logger.handlers or all(isinstance(h, logging.NullHandler) for h in logger.handlers):
            logger.handlers = []
            logger.addHandler(console_handler)
        
        try:
            # Import memory_profiler only if needed
            import memory_profiler
            logger.info("Memory tracking enabled. Using memory_profiler version: %s", 
                       memory_profiler.__version__)
            
            # Start continuous monitoring if requested
            if continuous_monitoring:
                try:
                    import psutil
                    start_memory_monitor(interval=monitor_interval)
                except ImportError:
                    logger.warning("psutil package not found. Continuous memory monitoring disabled. Install with: pip install psutil")
        except ImportError:
            logger.warning("memory_profiler package not found. Function memory tracking disabled. Install with: pip install memory_profiler")
            _memory_usage_data['is_enabled'] = False

def get_memory_usage_summary() -> Dict[str, Any]:
    """
    Get a summary of memory usage statistics.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing memory usage statistics
    """
    return {
        'peak_memory_mb': _memory_usage_data['peak_memory'],
        'absolute_peak_memory_mb': _memory_usage_data['absolute_peak_memory'],
        'function_usage': {
            func_name: {
                'peak_memory_mb': stats['peak_memory'],
                'avg_memory_mb': stats['total_memory'] / max(1, stats['calls']),
                'calls': stats['calls'],
                'total_time_sec': stats['total_time']
            }
            for func_name, stats in _memory_usage_data['function_usage'].items()
        }
    }

def memory_profiled(func: F) -> F:
    """
    Decorator to profile memory usage of a function.
    
    Parameters
    ----------
    func : callable
        Function to profile
        
    Returns
    -------
    callable
        Wrapped function with memory profiling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not _memory_usage_data['is_enabled']:
            return func(*args, **kwargs)
        
        # Get function name
        func_name = f"{func.__module__}.{func.__qualname__}"
        
        # Initialize function stats if not already tracked
        if func_name not in _memory_usage_data['function_usage']:
            _memory_usage_data['function_usage'][func_name] = {
                'peak_memory': 0,
                'total_memory': 0,
                'calls': 0,
                'total_time': 0
            }
        
        # Force garbage collection to get a more accurate memory measurement
        gc.collect()
        
        try:
            # Import here to avoid issues if package is not installed
            
            # Measure memory usage
            start_time = time.time()
            mem_usage, result = memory_usage(
                (func, args, kwargs), 
                retval=True,
                interval=0.1,
                timeout=None,
                max_iterations=None,
                include_children=True,
                multiprocess=True
            )
            elapsed_time = time.time() - start_time
            
            # Calculate memory metrics
            baseline = mem_usage[0] if mem_usage else 0
            max_mem = max(mem_usage) if mem_usage else 0
            mem_used = max_mem - baseline
            
            # Update function statistics
            stats = _memory_usage_data['function_usage'][func_name]
            stats['peak_memory'] = max(stats['peak_memory'], mem_used)
            stats['total_memory'] += mem_used
            stats['calls'] += 1
            stats['total_time'] += elapsed_time
            
            # Update global peak memory
            _memory_usage_data['peak_memory'] = max(_memory_usage_data['peak_memory'], max_mem)
            
            # Log the memory usage
            logger.debug(
                "%s used %.2f MB memory (peak: %.2f MB) in %.2f seconds",
                func_name, mem_used, max_mem, elapsed_time
            )
            
            return result
            
        except ImportError:
            logger.warning("memory_profiler package not available, running function without profiling")
            return func(*args, **kwargs)
    
    return wrapper

def log_memory_stats():
    """
    Log memory usage statistics to the logger.
    """
    if not _memory_usage_data['is_enabled']:
        logger.info("Memory tracking is not enabled. Call setup_memory_tracking(enabled=True) to enable.")
        return
    
    summary = get_memory_usage_summary()
    
    logger.info("Memory Usage Summary:")
    logger.info("Function-level peak memory: %.2f MB", summary['peak_memory_mb'])
    logger.info("Absolute peak memory usage: %.2f MB", summary['absolute_peak_memory_mb'])
    logger.info("Function memory usage:")
    
    # Sort functions by peak memory usage
    sorted_funcs = sorted(
        summary['function_usage'].items(),
        key=lambda x: x[1]['peak_memory_mb'],
        reverse=True
    )
    
    for func_name, stats in sorted_funcs:
        logger.info(
            "  %s: peak=%.2f MB, avg=%.2f MB, calls=%d, total_time=%.2f sec",
            func_name,
            stats['peak_memory_mb'],
            stats['avg_memory_mb'],
            stats['calls'],
            stats['total_time_sec']
        )

def log_memory_every(seconds=10):
    """Log current memory usage every N seconds"""
    threading.Timer(seconds, log_memory_every, [seconds]).start()
    summary = get_memory_usage_summary()
    logger.info(f"Periodic memory check: {summary['absolute_peak_memory_mb']:.2f} MB")
