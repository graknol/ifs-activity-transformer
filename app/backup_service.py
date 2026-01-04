"""
Azure Blob Storage backup service for annotation data.

This module provides automatic and manual backup of valuable training annotations
to Azure Blob Storage, ensuring data is never lost.
"""
import os
import json
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import AzureError
import logging

logger = logging.getLogger(__name__)


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
