"""
Database connection and data retrieval utilities for Oracle DB.
Following repository pattern and dependency injection principles.
"""
import os
import oracledb
import pandas as pd
from typing import Optional, Dict
from app.config import DatabaseConfig


class OracleDBConnection:
    """
    Handles Oracle database connections and data retrieval.
    
    This class follows the repository pattern, separating data access
    from business logic. Uses dependency injection for configuration.
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize database connection with configuration.
        
        Args:
            config: Database configuration. If None, loads from environment.
        """
        self.config = config or DatabaseConfig.from_env()
        self.connection = None
    
    def connect(self) -> bool:
        """
        Establish connection to Oracle database.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            dsn = f"{self.config.host}:{self.config.port}/{self.config.service}"
            self.connection = oracledb.connect(
                user=self.config.user,
                password=self.config.password,
                dsn=dsn
            )
            return True
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return False
    
    def disconnect(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def __enter__(self):
        """Context manager entry: establish connection."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: close connection."""
        self.disconnect()
    
    def fetch_training_data(self, query: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch training data from Oracle database.
        
        Args:
            query: Custom SQL query. If None, uses default query.
        
        Returns:
            DataFrame with columns: old_activity, new_activity, description
        """
        if not self.connection:
            if not self.connect():
                return pd.DataFrame()
        
        if query is None:
            # Default query - modify based on your actual table schema
            query = """
                SELECT 
                    old_activity_id,
                    old_activity_name,
                    new_activity_path,
                    activity_description
                FROM activity_mapping
                WHERE new_activity_path IS NOT NULL
            """
        
        try:
            df = pd.read_sql(query, self.connection)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def execute_custom_query(self, query: str) -> pd.DataFrame:
        """
        Execute a custom SQL query and return results as DataFrame.
        
        WARNING: This method executes arbitrary SQL queries. Only use with
        trusted input. In production, implement query validation or use
        parameterized queries with allowed patterns.
        
        Args:
            query: SQL query to execute
        
        Returns:
            DataFrame with query results
        """
        if not self.connection:
            if not self.connect():
                return pd.DataFrame()
        
        try:
            df = pd.read_sql(query, self.connection)
            return df
        except Exception as e:
            print(f"Error executing query: {e}")
            return pd.DataFrame()
    
    def get_activity_statistics(self) -> dict:
        """
        Get statistics about activities in the database.
        
        Returns:
            Dictionary with statistics
        """
        if not self.connection:
            if not self.connect():
                return {}
        
        try:
            cursor = self.connection.cursor()
            
            # Count total activities
            cursor.execute("SELECT COUNT(*) FROM activity_mapping")
            total_count = cursor.fetchone()[0]
            
            # Count mapped activities
            cursor.execute("""
                SELECT COUNT(*) FROM activity_mapping 
                WHERE new_activity_path IS NOT NULL
            """)
            mapped_count = cursor.fetchone()[0]
            
            # Count unique new activities
            cursor.execute("""
                SELECT COUNT(DISTINCT new_activity_path) 
                FROM activity_mapping 
                WHERE new_activity_path IS NOT NULL
            """)
            unique_new_activities = cursor.fetchone()[0]
            
            cursor.close()
            
            return {
                'total_activities': total_count,
                'mapped_activities': mapped_count,
                'unmapped_activities': total_count - mapped_count,
                'unique_new_activities': unique_new_activities
            }
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}
