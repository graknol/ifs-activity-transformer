"""Background scheduler for automatic disaster recovery backups."""
import os
import json
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class BackupScheduler:
    """Scheduler for automatic disaster recovery backups."""
    
    CONFIG_FILE = "data/backup_schedule_config.json"
    
    def __init__(self, backup_service=None):
        """Initialize the backup scheduler.
        
        Args:
            backup_service: The backup service instance to use for creating backups
        """
        self.backup_service = backup_service
        self._scheduler_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._enabled = False
        self._interval_hours = 24  # Default: daily backups
        self._last_backup_time: Optional[datetime] = None
        self._next_backup_time: Optional[datetime] = None
        self._backup_callback: Optional[Callable] = None
        self._lock = threading.Lock()
        
        # Load saved configuration
        self._load_config()
    
    def _load_config(self):
        """Load scheduler configuration from file."""
        try:
            if os.path.exists(self.CONFIG_FILE):
                with open(self.CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    self._enabled = config.get('enabled', False)
                    self._interval_hours = config.get('interval_hours', 24)
                    
                    last_backup = config.get('last_backup_time')
                    if last_backup:
                        self._last_backup_time = datetime.fromisoformat(last_backup)
                    
                    logger.info(f"Loaded backup schedule config: enabled={self._enabled}, interval={self._interval_hours}h")
        except Exception as e:
            logger.error(f"Error loading backup schedule config: {e}")
    
    def _save_config(self):
        """Save scheduler configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.CONFIG_FILE), exist_ok=True)
            config = {
                'enabled': self._enabled,
                'interval_hours': self._interval_hours,
                'last_backup_time': self._last_backup_time.isoformat() if self._last_backup_time else None,
                'updated_at': datetime.now().isoformat()
            }
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info("Saved backup schedule config")
        except Exception as e:
            logger.error(f"Error saving backup schedule config: {e}")
    
    def set_backup_service(self, backup_service):
        """Set the backup service to use for creating backups."""
        self.backup_service = backup_service
    
    def set_backup_callback(self, callback: Callable):
        """Set a callback function to be called after each backup.
        
        Args:
            callback: Function that takes (success: bool, result: dict) as arguments
        """
        self._backup_callback = callback
    
    def configure(self, enabled: bool, interval_hours: int = 24) -> dict:
        """Configure the automatic backup schedule.
        
        Args:
            enabled: Whether automatic backups are enabled
            interval_hours: Hours between backups (1, 6, 12, 24, 48, 168)
            
        Returns:
            Configuration status dictionary
        """
        with self._lock:
            self._enabled = enabled
            self._interval_hours = max(1, min(168, interval_hours))  # Clamp 1-168 hours
            
            if enabled:
                self._calculate_next_backup_time()
                if not self._scheduler_thread or not self._scheduler_thread.is_alive():
                    self._start_scheduler()
            else:
                self._stop_scheduler()
                self._next_backup_time = None
            
            self._save_config()
            
            return self.get_status()
    
    def _calculate_next_backup_time(self):
        """Calculate the next backup time based on last backup and interval."""
        if self._last_backup_time:
            self._next_backup_time = self._last_backup_time + timedelta(hours=self._interval_hours)
            # If next backup is in the past, schedule it soon
            if self._next_backup_time < datetime.now():
                self._next_backup_time = datetime.now() + timedelta(minutes=5)
        else:
            # No previous backup, schedule one soon
            self._next_backup_time = datetime.now() + timedelta(minutes=5)
    
    def _start_scheduler(self):
        """Start the background scheduler thread."""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        logger.info("Backup scheduler started")
    
    def _stop_scheduler(self):
        """Stop the background scheduler thread."""
        self._stop_event.set()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        logger.info("Backup scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop that runs in background."""
        logger.info("Backup scheduler loop started")
        
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    if not self._enabled:
                        break
                    
                    if self._next_backup_time and datetime.now() >= self._next_backup_time:
                        logger.info("Scheduled backup time reached, starting backup...")
                        self._perform_backup()
                
                # Check every minute
                self._stop_event.wait(60)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                self._stop_event.wait(300)  # Wait 5 minutes on error
        
        logger.info("Backup scheduler loop ended")
    
    def _perform_backup(self):
        """Perform the scheduled backup."""
        if not self.backup_service:
            logger.error("No backup service configured, skipping backup")
            return
        
        try:
            description = f"Automatic scheduled backup ({self._interval_hours}h interval)"
            result = self.backup_service.create_full_backup(description=description)
            
            self._last_backup_time = datetime.now()
            self._calculate_next_backup_time()
            self._save_config()
            
            success = result.get('status') == 'success'
            logger.info(f"Scheduled backup completed: success={success}")
            
            if self._backup_callback:
                try:
                    self._backup_callback(success, result)
                except Exception as e:
                    logger.error(f"Backup callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Scheduled backup failed: {e}")
            if self._backup_callback:
                try:
                    self._backup_callback(False, {'error': str(e)})
                except Exception as cb_e:
                    logger.error(f"Backup callback error: {cb_e}")
    
    def trigger_backup_now(self, description: str = None) -> dict:
        """Trigger an immediate backup.
        
        Args:
            description: Optional description for the backup
            
        Returns:
            Backup result dictionary
        """
        if not self.backup_service:
            return {'status': 'error', 'error': 'Backup service not configured'}
        
        try:
            desc = description or "Manual backup triggered from scheduler"
            result = self.backup_service.create_full_backup(description=desc)
            
            with self._lock:
                self._last_backup_time = datetime.now()
                if self._enabled:
                    self._calculate_next_backup_time()
                self._save_config()
            
            return result
        except Exception as e:
            logger.error(f"Manual backup failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_status(self) -> dict:
        """Get the current scheduler status.
        
        Returns:
            Status dictionary with schedule information
        """
        with self._lock:
            return {
                'enabled': self._enabled,
                'interval_hours': self._interval_hours,
                'interval_display': self._get_interval_display(),
                'last_backup_time': self._last_backup_time.isoformat() if self._last_backup_time else None,
                'last_backup_display': self._format_time_ago(self._last_backup_time) if self._last_backup_time else 'Never',
                'next_backup_time': self._next_backup_time.isoformat() if self._next_backup_time else None,
                'next_backup_display': self._format_time_until(self._next_backup_time) if self._next_backup_time else 'Not scheduled',
                'scheduler_running': self._scheduler_thread is not None and self._scheduler_thread.is_alive()
            }
    
    def _get_interval_display(self) -> str:
        """Get human-readable interval display."""
        if self._interval_hours == 1:
            return "Every hour"
        elif self._interval_hours == 6:
            return "Every 6 hours"
        elif self._interval_hours == 12:
            return "Every 12 hours"
        elif self._interval_hours == 24:
            return "Daily"
        elif self._interval_hours == 48:
            return "Every 2 days"
        elif self._interval_hours == 168:
            return "Weekly"
        else:
            return f"Every {self._interval_hours} hours"
    
    def _format_time_ago(self, dt: datetime) -> str:
        """Format a datetime as 'X ago' string."""
        if not dt:
            return "Never"
        
        delta = datetime.now() - dt
        seconds = delta.total_seconds()
        
        if seconds < 60:
            return "Just now"
        elif seconds < 3600:
            mins = int(seconds / 60)
            return f"{mins} minute{'s' if mins != 1 else ''} ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        else:
            days = int(seconds / 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"
    
    def _format_time_until(self, dt: datetime) -> str:
        """Format a datetime as 'in X' string."""
        if not dt:
            return "Not scheduled"
        
        delta = dt - datetime.now()
        seconds = delta.total_seconds()
        
        if seconds <= 0:
            return "Imminent"
        elif seconds < 60:
            return "In less than a minute"
        elif seconds < 3600:
            mins = int(seconds / 60)
            return f"In {mins} minute{'s' if mins != 1 else ''}"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"In {hours} hour{'s' if hours != 1 else ''}"
        else:
            days = int(seconds / 86400)
            return f"In {days} day{'s' if days != 1 else ''}"
    
    def start_if_enabled(self):
        """Start the scheduler if automatic backups are enabled.
        
        Should be called when the app starts.
        """
        if self._enabled:
            self._calculate_next_backup_time()
            self._start_scheduler()
            logger.info(f"Backup scheduler auto-started, next backup: {self._next_backup_time}")
    
    def shutdown(self):
        """Shut down the scheduler gracefully."""
        self._stop_scheduler()


# Global scheduler instance
_backup_scheduler: Optional[BackupScheduler] = None


def get_backup_scheduler() -> BackupScheduler:
    """Get or create the global backup scheduler instance."""
    global _backup_scheduler
    if _backup_scheduler is None:
        _backup_scheduler = BackupScheduler()
    return _backup_scheduler


def init_backup_scheduler(backup_service=None) -> BackupScheduler:
    """Initialize and start the backup scheduler.
    
    Args:
        backup_service: The backup service to use for backups
        
    Returns:
        The initialized scheduler instance
    """
    scheduler = get_backup_scheduler()
    if backup_service:
        scheduler.set_backup_service(backup_service)
    scheduler.start_if_enabled()
    return scheduler
