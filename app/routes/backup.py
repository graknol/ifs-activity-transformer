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
            
            return ResponseBuilder.success(
                'Backup status retrieved',
                {
                    'backup_enabled': backup_service.is_enabled(),
                    'statistics': stats
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
