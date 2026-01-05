"""
Azure Blob Storage backup service for annotation data.

This module provides automatic and manual backup of valuable training annotations
to Azure Blob Storage, ensuring data is never lost.

Disaster Recovery Features:
- Full backup of all data files (training_data.csv, bootstrap_annotated.csv, etc.)
- Model checkpoint backups
- Configuration file backups
- One-click restore capability
"""
import os
import json
import glob
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import AzureError
import logging

logger = logging.getLogger(__name__)


# Files critical for disaster recovery
CRITICAL_DATA_FILES = [
    'data/training_data.csv',
    'data/bootstrap_sample.csv',
    'data/bootstrap_annotated.csv',
    'data/bootstrap_annotated.json',
    'data/annotations.csv',
    'data/training_history.json',
    'data/pending_annotations.json',
]

CRITICAL_CONFIG_FILES = [
    'app/discipline_codes.json',
]

MODEL_DIRECTORIES = [
    'models/',
]


class AzureBlobBackupService:
    """
    Service for backing up annotation data to Azure Blob Storage.
    
    Provides automatic snapshots of annotations, manual backups,
    and restoration capabilities to prevent data loss.
    """
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        container_name: str = 'ifs-annotations'
    ):
        """
        Initialize Azure Blob Storage backup service.
        
        Args:
            connection_string: Azure Storage connection string (from env if None)
            container_name: Name of the blob container for backups
        """
        self.connection_string = connection_string or os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        self.container_name = container_name
        self.blob_service_client = None
        self.container_client = None
        
        if self.connection_string:
            try:
                self.blob_service_client = BlobServiceClient.from_connection_string(
                    self.connection_string
                )
                self.container_client = self.blob_service_client.get_container_client(
                    self.container_name
                )
                # Create container if it doesn't exist
                try:
                    self.container_client.create_container()
                    logger.info(f"Created container: {self.container_name}")
                except Exception:
                    # Container already exists
                    pass
            except AzureError as e:
                logger.error(f"Failed to initialize Azure Blob Storage: {e}")
                self.blob_service_client = None
        else:
            logger.warning("Azure Storage connection string not configured. Backups disabled.")
    
    def is_enabled(self) -> bool:
        """Check if Azure backup is enabled."""
        return self.blob_service_client is not None
    
    def create_snapshot(
        self,
        annotations_df: pd.DataFrame,
        metadata: Optional[Dict] = None,
        snapshot_type: str = 'auto'
    ) -> Optional[str]:
        """
        Create a snapshot of annotation data in Azure Blob Storage.
        
        Args:
            annotations_df: DataFrame with annotation data
            metadata: Optional metadata to include in snapshot
            snapshot_type: Type of snapshot ('auto', 'manual', 'milestone')
            
        Returns:
            Blob name if successful, None if failed
        """
        if not self.is_enabled():
            logger.warning("Azure backup not enabled. Skipping snapshot.")
            return None
        
        try:
            # Generate blob name with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            blob_name = f"annotations/{snapshot_type}_{timestamp}.json"
            
            # Prepare data for upload
            snapshot_data = {
                'timestamp': datetime.now().isoformat(),
                'snapshot_type': snapshot_type,
                'metadata': metadata or {},
                'statistics': {
                    'total_annotations': len(annotations_df),
                    'labeled_count': annotations_df['label'].notna().sum() if 'label' in annotations_df else 0
                },
                'annotations': annotations_df.to_dict('records')
            }
            
            # Convert to JSON
            json_data = json.dumps(snapshot_data, indent=2, default=str)
            
            # Upload to Azure
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            blob_client.upload_blob(json_data, overwrite=True)
            
            logger.info(f"Created snapshot: {blob_name}")
            return blob_name
            
        except AzureError as e:
            logger.error(f"Failed to create snapshot: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error creating snapshot: {e}")
            return None
    
    def backup_csv_file(
        self,
        file_path: str,
        backup_type: str = 'data'
    ) -> Optional[str]:
        """
        Backup a CSV file to Azure Blob Storage.
        
        Args:
            file_path: Path to the CSV file
            backup_type: Type of backup ('data', 'annotations', 'training')
            
        Returns:
            Blob name if successful, None if failed
        """
        if not self.is_enabled():
            return None
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = os.path.basename(file_path)
            blob_name = f"{backup_type}/{timestamp}_{file_name}"
            
            # Upload file
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            with open(file_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
            
            logger.info(f"Backed up file: {blob_name}")
            return blob_name
            
        except AzureError as e:
            logger.error(f"Failed to backup file: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error backing up file: {e}")
            return None
    
    def list_backups(
        self,
        backup_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        List available backups in Azure Blob Storage.
        
        Args:
            backup_type: Filter by backup type (annotations, data, training)
            limit: Maximum number of backups to return
            
        Returns:
            List of backup metadata dictionaries
        """
        if not self.is_enabled():
            return []
        
        try:
            backups = []
            prefix = f"{backup_type}/" if backup_type else ""
            
            blob_list = self.container_client.list_blobs(name_starts_with=prefix)
            
            for blob in blob_list:
                backups.append({
                    'name': blob.name,
                    'size': blob.size,
                    'created': blob.creation_time.isoformat() if blob.creation_time else None,
                    'last_modified': blob.last_modified.isoformat() if blob.last_modified else None
                })
                
                if len(backups) >= limit:
                    break
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x['created'] or '', reverse=True)
            
            return backups
            
        except AzureError as e:
            logger.error(f"Failed to list backups: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error listing backups: {e}")
            return []
    
    def restore_snapshot(
        self,
        blob_name: str,
        restore_path: str
    ) -> bool:
        """
        Restore annotation data from a snapshot.
        
        Args:
            blob_name: Name of the blob to restore
            restore_path: Path where to restore the data
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_enabled():
            return False
        
        try:
            # Download blob
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            blob_data = blob_client.download_blob().readall()
            snapshot_data = json.loads(blob_data)
            
            # Extract annotations
            annotations = snapshot_data.get('annotations', [])
            df = pd.DataFrame(annotations)
            
            # Save to file
            os.makedirs(os.path.dirname(restore_path), exist_ok=True)
            df.to_csv(restore_path, index=False)
            
            logger.info(f"Restored snapshot {blob_name} to {restore_path}")
            return True
            
        except AzureError as e:
            logger.error(f"Failed to restore snapshot: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error restoring snapshot: {e}")
            return False
    
    def create_milestone_backup(
        self,
        annotations_df: pd.DataFrame,
        milestone_name: str,
        description: str = ""
    ) -> Optional[str]:
        """
        Create a named milestone backup.
        
        Args:
            annotations_df: Annotation data
            milestone_name: Name for this milestone
            description: Description of the milestone
            
        Returns:
            Blob name if successful
        """
        metadata = {
            'milestone_name': milestone_name,
            'description': description,
            'created_by': 'user'
        }
        
        return self.create_snapshot(
            annotations_df,
            metadata=metadata,
            snapshot_type='milestone'
        )
    
    def get_backup_statistics(self) -> Dict:
        """
        Get statistics about backups in Azure storage.
        
        Returns:
            Dictionary with backup statistics
        """
        if not self.is_enabled():
            return {
                'enabled': False,
                'total_backups': 0,
                'total_size_mb': 0
            }
        
        try:
            backups = self.list_backups(limit=1000)
            total_size = sum(b['size'] for b in backups)
            
            # Count by type
            by_type = {}
            for backup in backups:
                backup_type = backup['name'].split('/')[0]
                by_type[backup_type] = by_type.get(backup_type, 0) + 1
            
            return {
                'enabled': True,
                'total_backups': len(backups),
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'by_type': by_type,
                'latest_backup': backups[0] if backups else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get backup statistics: {e}")
            return {
                'enabled': True,
                'error': str(e)
            }
    
    # =========================================================================
    # DISASTER RECOVERY METHODS
    # =========================================================================
    
    def create_full_backup(self, description: str = "") -> Dict:
        """
        Create a complete disaster recovery backup of all critical files.
        
        This backs up:
        - All data files (training_data.csv, bootstrap data, annotations)
        - Configuration files (discipline_codes.json)
        - Training history and state
        - Model checkpoints (if any)
        
        Args:
            description: Optional description for this backup
            
        Returns:
            Dictionary with backup results
        """
        if not self.is_enabled():
            return {'success': False, 'error': 'Azure backup not enabled'}
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_folder = f"disaster_recovery/{timestamp}"
        
        results = {
            'success': True,
            'timestamp': timestamp,
            'backup_folder': backup_folder,
            'files_backed_up': [],
            'files_failed': [],
            'total_size_bytes': 0
        }
        
        # Backup critical data files
        for file_path in CRITICAL_DATA_FILES:
            if os.path.exists(file_path):
                blob_name = f"{backup_folder}/{file_path}"
                if self._upload_file(file_path, blob_name):
                    size = os.path.getsize(file_path)
                    results['files_backed_up'].append({
                        'local_path': file_path,
                        'blob_name': blob_name,
                        'size': size
                    })
                    results['total_size_bytes'] += size
                else:
                    results['files_failed'].append(file_path)
        
        # Backup config files
        for file_path in CRITICAL_CONFIG_FILES:
            if os.path.exists(file_path):
                blob_name = f"{backup_folder}/{file_path}"
                if self._upload_file(file_path, blob_name):
                    size = os.path.getsize(file_path)
                    results['files_backed_up'].append({
                        'local_path': file_path,
                        'blob_name': blob_name,
                        'size': size
                    })
                    results['total_size_bytes'] += size
                else:
                    results['files_failed'].append(file_path)
        
        # Backup model files
        for model_dir in MODEL_DIRECTORIES:
            if os.path.exists(model_dir):
                for root, dirs, files in os.walk(model_dir):
                    for file in files:
                        if file.endswith(('.bin', '.pt', '.pth', '.json', '.txt', '.safetensors')):
                            file_path = os.path.join(root, file)
                            blob_name = f"{backup_folder}/{file_path}"
                            if self._upload_file(file_path, blob_name):
                                size = os.path.getsize(file_path)
                                results['files_backed_up'].append({
                                    'local_path': file_path,
                                    'blob_name': blob_name,
                                    'size': size
                                })
                                results['total_size_bytes'] += size
                            else:
                                results['files_failed'].append(file_path)
        
        # Create manifest file
        manifest = {
            'backup_type': 'disaster_recovery',
            'timestamp': datetime.now().isoformat(),
            'description': description,
            'files': results['files_backed_up'],
            'total_files': len(results['files_backed_up']),
            'total_size_bytes': results['total_size_bytes']
        }
        
        manifest_blob = f"{backup_folder}/manifest.json"
        self._upload_json(manifest, manifest_blob)
        
        results['manifest_blob'] = manifest_blob
        results['success'] = len(results['files_failed']) == 0
        
        logger.info(f"Full backup created: {len(results['files_backed_up'])} files, "
                   f"{results['total_size_bytes'] / 1024:.1f} KB")
        
        return results
    
    def _upload_file(self, local_path: str, blob_name: str) -> bool:
        """Upload a file to blob storage."""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            with open(local_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
            return True
        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            return False
    
    def _upload_json(self, data: dict, blob_name: str) -> bool:
        """Upload JSON data to blob storage."""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            json_data = json.dumps(data, indent=2, default=str)
            blob_client.upload_blob(json_data, overwrite=True)
            return True
        except Exception as e:
            logger.error(f"Failed to upload JSON to {blob_name}: {e}")
            return False
    
    def list_disaster_recovery_backups(self) -> List[Dict]:
        """
        List all disaster recovery backups.
        
        Returns:
            List of backup manifests
        """
        if not self.is_enabled():
            return []
        
        try:
            backups = []
            blob_list = self.container_client.list_blobs(
                name_starts_with="disaster_recovery/"
            )
            
            # Find manifest files
            manifest_blobs = [b for b in blob_list if b.name.endswith('manifest.json')]
            
            for blob in manifest_blobs:
                try:
                    # Download and parse manifest
                    blob_client = self.blob_service_client.get_blob_client(
                        container=self.container_name,
                        blob=blob.name
                    )
                    manifest_data = blob_client.download_blob().readall()
                    manifest = json.loads(manifest_data)
                    manifest['blob_name'] = blob.name
                    manifest['folder'] = '/'.join(blob.name.split('/')[:-1])
                    backups.append(manifest)
                except Exception as e:
                    logger.error(f"Failed to parse manifest {blob.name}: {e}")
            
            # Sort by timestamp (newest first)
            backups.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return backups
            
        except Exception as e:
            logger.error(f"Failed to list DR backups: {e}")
            return []
    
    def restore_full_backup(self, backup_folder: str) -> Dict:
        """
        Restore all files from a disaster recovery backup.
        
        Args:
            backup_folder: The backup folder path (e.g., "disaster_recovery/20260105_120000")
            
        Returns:
            Dictionary with restore results
        """
        if not self.is_enabled():
            return {'success': False, 'error': 'Azure backup not enabled'}
        
        results = {
            'success': True,
            'files_restored': [],
            'files_failed': [],
            'backup_folder': backup_folder
        }
        
        try:
            # List all blobs in the backup folder
            blob_list = list(self.container_client.list_blobs(
                name_starts_with=f"{backup_folder}/"
            ))
            
            for blob in blob_list:
                # Skip manifest
                if blob.name.endswith('manifest.json'):
                    continue
                
                # Calculate local path by removing backup folder prefix
                local_path = blob.name.replace(f"{backup_folder}/", "")
                
                try:
                    # Download blob
                    blob_client = self.blob_service_client.get_blob_client(
                        container=self.container_name,
                        blob=blob.name
                    )
                    blob_data = blob_client.download_blob().readall()
                    
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
                    
                    # Write file
                    with open(local_path, 'wb') as f:
                        f.write(blob_data)
                    
                    results['files_restored'].append({
                        'blob_name': blob.name,
                        'local_path': local_path,
                        'size': blob.size
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to restore {blob.name}: {e}")
                    results['files_failed'].append({
                        'blob_name': blob.name,
                        'error': str(e)
                    })
            
            results['success'] = len(results['files_failed']) == 0
            logger.info(f"Restored {len(results['files_restored'])} files from {backup_folder}")
            
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        return results
    
    def get_local_files_status(self) -> Dict:
        """
        Get status of all critical local files.
        
        Returns:
            Dictionary showing which files exist and their sizes
        """
        status = {
            'data_files': [],
            'config_files': [],
            'model_files': [],
            'total_size_bytes': 0
        }
        
        # Check data files
        for file_path in CRITICAL_DATA_FILES:
            file_info = {
                'path': file_path,
                'exists': os.path.exists(file_path),
                'size': 0
            }
            if file_info['exists']:
                file_info['size'] = os.path.getsize(file_path)
                file_info['modified'] = datetime.fromtimestamp(
                    os.path.getmtime(file_path)
                ).isoformat()
                status['total_size_bytes'] += file_info['size']
            status['data_files'].append(file_info)
        
        # Check config files
        for file_path in CRITICAL_CONFIG_FILES:
            file_info = {
                'path': file_path,
                'exists': os.path.exists(file_path),
                'size': 0
            }
            if file_info['exists']:
                file_info['size'] = os.path.getsize(file_path)
                status['total_size_bytes'] += file_info['size']
            status['config_files'].append(file_info)
        
        # Check model files
        for model_dir in MODEL_DIRECTORIES:
            if os.path.exists(model_dir):
                for root, dirs, files in os.walk(model_dir):
                    for file in files:
                        if file.endswith(('.bin', '.pt', '.pth', '.json', '.safetensors')):
                            file_path = os.path.join(root, file)
                            size = os.path.getsize(file_path)
                            status['model_files'].append({
                                'path': file_path,
                                'exists': True,
                                'size': size
                            })
                            status['total_size_bytes'] += size
        
        return status


class AutoBackupManager:
    """
    Manager for automatic backup scheduling.
    
    Triggers backups based on annotation events and schedules.
    """
    
    def __init__(self, backup_service: AzureBlobBackupService):
        """
        Initialize auto backup manager.
        
        Args:
            backup_service: Azure backup service instance
        """
        self.backup_service = backup_service
        self.annotations_since_backup = 0
        self.backup_threshold = int(os.getenv('AUTO_BACKUP_THRESHOLD', '10'))
    
    def on_annotation_saved(self, annotations_df: pd.DataFrame) -> None:
        """
        Called when an annotation is saved.
        
        Args:
            annotations_df: Current annotation data
        """
        self.annotations_since_backup += 1
        
        # Trigger backup if threshold reached
        if self.annotations_since_backup >= self.backup_threshold:
            self.backup_service.create_snapshot(
                annotations_df,
                metadata={'trigger': 'auto', 'count': self.annotations_since_backup},
                snapshot_type='auto'
            )
            self.annotations_since_backup = 0
    
    def create_session_backup(self, annotations_df: pd.DataFrame) -> Optional[str]:
        """
        Create a backup at the end of an annotation session.
        
        Args:
            annotations_df: Annotation data
            
        Returns:
            Blob name if successful
        """
        return self.backup_service.create_snapshot(
            annotations_df,
            metadata={'trigger': 'session_end'},
            snapshot_type='session'
        )
