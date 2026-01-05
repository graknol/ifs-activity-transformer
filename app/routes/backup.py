"""Backup management API routes."""
from flask import request
from app.utils import ResponseBuilder, DataManager, FileValidator


def get_service(service_type: str):
    """Helper to get service from container."""
    from flask import current_app
    return current_app.container.resolve(service_type)


def register_backup_routes(app):
    """Register backup routes."""
    
    @app.route('/api/backup/status', methods=['GET'])
    def get_backup_status():
        """Get backup service status and statistics."""
        try:
            backup_service = get_service('backup_service')
            stats = backup_service.get_backup_statistics()
            local_files = backup_service.get_local_files_status()
            
            return ResponseBuilder.success(
                'Backup status retrieved',
                {
                    'backup_enabled': backup_service.is_enabled(),
                    'statistics': stats,
                    'local_files': local_files
                }
            )
        except Exception as e:
            return ResponseBuilder.from_exception(e, "Get backup status")
    
    @app.route('/api/backup/create', methods=['POST'])
    def create_manual_backup():
        """Create a manual backup of annotations."""
        try:
            # Check if annotations exist
            exists, error_msg = FileValidator.check_file_exists(
                DataManager.ANNOTATIONS_PATH,
                'annotations to backup'
            )
            if not exists:
                return ResponseBuilder.error(error_msg, 404)
            
            data = request.get_json()
            backup_type = data.get('type', 'manual')
            milestone_name = data.get('milestone_name', '')
            description = data.get('description', '')
            
            # Load annotations
            annotations_df = DataManager.load_annotations()
            backup_service = get_service('backup_service')
            
            # Create backup based on type
            if backup_type == 'milestone' and milestone_name:
                blob_name = backup_service.create_milestone_backup(
                    annotations_df,
                    milestone_name,
                    description
                )
            else:
                blob_name = backup_service.create_snapshot(
                    annotations_df,
                    metadata={'created_by': 'user'},
                    snapshot_type='manual'
                )
            
            if blob_name:
                return ResponseBuilder.success(
                    'Backup created successfully',
                    {'blob_name': blob_name}
                )
            else:
                return ResponseBuilder.error('Backup failed or not enabled', 500)
                
        except Exception as e:
            return ResponseBuilder.from_exception(e, "Create backup")
    
    @app.route('/api/backup/list', methods=['GET'])
    def list_backups():
        """List available backups."""
        try:
            backup_type = request.args.get('type', None)
            limit = int(request.args.get('limit', 50))
            
            backup_service = get_service('backup_service')
            backups = backup_service.list_backups(backup_type, limit)
            
            return ResponseBuilder.success('Backups listed', {'backups': backups})
            
        except Exception as e:
            return ResponseBuilder.from_exception(e, "List backups")
    
    @app.route('/api/backup/restore', methods=['POST'])
    def restore_backup():
        """Restore annotations from a backup."""
        try:
            data = request.get_json()
            blob_name = data.get('blob_name')
            
            if not blob_name:
                return ResponseBuilder.error('No blob name provided', 400)
            
            backup_service = get_service('backup_service')
            restore_path = 'data/annotations_restored.csv'
            
            success = backup_service.restore_snapshot(blob_name, restore_path)
            
            if success:
                return ResponseBuilder.success(f'Backup restored to {restore_path}')
            else:
                return ResponseBuilder.error('Restore failed', 500)
                
        except Exception as e:
            return ResponseBuilder.from_exception(e, "Restore backup")
    
    # =========================================================================
    # DISASTER RECOVERY ENDPOINTS
    # =========================================================================
    
    @app.route('/api/backup/disaster-recovery/create', methods=['POST'])
    def create_disaster_recovery_backup():
        """
        Create a full disaster recovery backup.
        
        This backs up ALL critical files:
        - Training data (training_data.csv)
        - Bootstrap samples and annotations
        - Configuration files
        - Model checkpoints
        - Training history
        """
        try:
            data = request.get_json() or {}
            description = data.get('description', '')
            
            backup_service = get_service('backup_service')
            
            if not backup_service.is_enabled():
                return ResponseBuilder.error(
                    'Azure backup not enabled. Set AZURE_STORAGE_CONNECTION_STRING environment variable.',
                    400
                )
            
            result = backup_service.create_full_backup(description)
            
            if result['success']:
                return ResponseBuilder.success(
                    f"Disaster recovery backup created: {len(result['files_backed_up'])} files",
                    result
                )
            else:
                return ResponseBuilder.error(
                    f"Backup partially failed: {len(result['files_failed'])} files failed",
                    500,
                    result
                )
                
        except Exception as e:
            return ResponseBuilder.from_exception(e, "Create DR backup")
    
    @app.route('/api/backup/disaster-recovery/list', methods=['GET'])
    def list_disaster_recovery_backups():
        """List all disaster recovery backups."""
        try:
            backup_service = get_service('backup_service')
            backups = backup_service.list_disaster_recovery_backups()
            
            return ResponseBuilder.success(
                f'Found {len(backups)} disaster recovery backups',
                {'backups': backups}
            )
            
        except Exception as e:
            return ResponseBuilder.from_exception(e, "List DR backups")
    
    @app.route('/api/backup/disaster-recovery/restore', methods=['POST'])
    def restore_disaster_recovery_backup():
        """
        Restore all files from a disaster recovery backup.
        
        WARNING: This will overwrite existing local files!
        """
        try:
            data = request.get_json()
            backup_folder = data.get('backup_folder')
            
            if not backup_folder:
                return ResponseBuilder.error('No backup_folder provided', 400)
            
            backup_service = get_service('backup_service')
            result = backup_service.restore_full_backup(backup_folder)
            
            if result['success']:
                return ResponseBuilder.success(
                    f"Restored {len(result['files_restored'])} files",
                    result
                )
            else:
                return ResponseBuilder.error(
                    f"Restore failed: {result.get('error', 'Unknown error')}",
                    500,
                    result
                )
                
        except Exception as e:
            return ResponseBuilder.from_exception(e, "Restore DR backup")
    
    @app.route('/api/backup/local-files', methods=['GET'])
    def get_local_files_status():
        """Get status of all critical local files."""
        try:
            backup_service = get_service('backup_service')
            status = backup_service.get_local_files_status()
            
            return ResponseBuilder.success('Local files status', status)
            
        except Exception as e:
            return ResponseBuilder.from_exception(e, "Get local files")
    
    # =========================================================================
    # AUTOMATIC BACKUP SCHEDULER ENDPOINTS
    # =========================================================================
    
    @app.route('/api/backup/schedule/status', methods=['GET'])
    def get_backup_schedule_status():
        """Get the current backup schedule status."""
        try:
            from app.backup_scheduler import get_backup_scheduler
            
            scheduler = get_backup_scheduler()
            # Ensure scheduler has backup service
            backup_service = get_service('backup_service')
            scheduler.set_backup_service(backup_service)
            
            status = scheduler.get_status()
            
            return ResponseBuilder.success('Backup schedule status', status)
            
        except Exception as e:
            return ResponseBuilder.from_exception(e, "Get schedule status")
    
    @app.route('/api/backup/schedule/configure', methods=['POST'])
    def configure_backup_schedule():
        """Configure the automatic backup schedule.
        
        Request body:
        {
            "enabled": true/false,
            "interval_hours": 1|6|12|24|48|168
        }
        """
        try:
            from app.backup_scheduler import get_backup_scheduler
            
            data = request.get_json() or {}
            enabled = data.get('enabled', False)
            interval_hours = data.get('interval_hours', 24)
            
            # Validate interval
            valid_intervals = [1, 6, 12, 24, 48, 168]
            if interval_hours not in valid_intervals:
                return ResponseBuilder.error(
                    f'Invalid interval. Must be one of: {valid_intervals}',
                    400
                )
            
            scheduler = get_backup_scheduler()
            # Ensure scheduler has backup service
            backup_service = get_service('backup_service')
            scheduler.set_backup_service(backup_service)
            
            # Check if backup service is enabled before enabling scheduler
            if enabled and not backup_service.is_enabled():
                return ResponseBuilder.error(
                    'Cannot enable automatic backups: Azure backup not configured. '
                    'Set AZURE_STORAGE_CONNECTION_STRING environment variable.',
                    400
                )
            
            status = scheduler.configure(enabled, interval_hours)
            
            action = 'enabled' if enabled else 'disabled'
            return ResponseBuilder.success(
                f'Automatic backups {action}',
                status
            )
            
        except Exception as e:
            return ResponseBuilder.from_exception(e, "Configure schedule")
    
    @app.route('/api/backup/schedule/trigger', methods=['POST'])
    def trigger_immediate_backup():
        """Trigger an immediate backup (outside of schedule)."""
        try:
            from app.backup_scheduler import get_backup_scheduler
            
            data = request.get_json() or {}
            description = data.get('description', 'Manual trigger from UI')
            
            scheduler = get_backup_scheduler()
            backup_service = get_service('backup_service')
            scheduler.set_backup_service(backup_service)
            
            if not backup_service.is_enabled():
                return ResponseBuilder.error(
                    'Azure backup not enabled. Set AZURE_STORAGE_CONNECTION_STRING environment variable.',
                    400
                )
            
            result = scheduler.trigger_backup_now(description)
            
            if result.get('success') or result.get('status') == 'success':
                return ResponseBuilder.success(
                    f"Backup created: {len(result.get('files_backed_up', []))} files",
                    result
                )
            else:
                return ResponseBuilder.error(
                    result.get('error', 'Backup failed'),
                    500,
                    result
                )
                
        except Exception as e:
            return ResponseBuilder.from_exception(e, "Trigger backup")
