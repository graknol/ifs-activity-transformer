"""
Database connection and data retrieval utilities for Oracle DB.
Following repository pattern and dependency injection principles.

This module connects to IFSCTRN and queries IFS10PRD via database link
to fetch activity data for training the classification model.
"""
import os
import warnings
import oracledb
import pandas as pd
from typing import Optional, Dict, List
from app.config import DatabaseConfig, DisciplineConfig

# Suppress pandas SQLAlchemy warning - oracledb works fine with read_sql
warnings.filterwarnings('ignore', message='.*pandas only supports SQLAlchemy.*')


class OracleDBConnection:
    """
    Handles Oracle database connections and data retrieval.
    
    This class follows the repository pattern, separating data access
    from business logic. Uses dependency injection for configuration.
    
    Connection setup:
    - Connects to IFSCTRN database
    - Queries IFS10PRD via @IFS10PRD database link
    - READ-ONLY operations only (no UPDATE/INSERT/DELETE)
    """
    
    def __init__(
        self, 
        config: Optional[DatabaseConfig] = None,
        discipline_config: Optional[DisciplineConfig] = None
    ):
        """
        Initialize database connection with configuration.
        
        Args:
            config: Database configuration. If None, loads from environment.
            discipline_config: Discipline configuration with SQL queries.
        """
        self.config = config or DatabaseConfig.from_env()
        self.discipline_config = discipline_config or DisciplineConfig.load()
        self.connection = None
        
        # Load SQL queries from discipline config
        self._sql_queries = self.discipline_config._data.get('sql_queries', {})
    
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
    
    def get_available_queries(self) -> List[str]:
        """
        Get list of available query names from configuration.
        
        Returns:
            List of query names (e.g., 'training_data_full', 'discipline_distribution')
        """
        excluded = {'description', 'connection_info'}
        return [k for k in self._sql_queries.keys() if k not in excluded]
    
    def get_query_info(self, query_name: str) -> Dict[str, str]:
        """
        Get information about a specific query.
        
        Args:
            query_name: Name of the query
            
        Returns:
            Dict with 'description' and 'query' keys
        """
        return self._sql_queries.get(query_name, {})

    def get_training_query(
        self, 
        query_type: str = 'full',
        order_projects_only: bool = False,
        with_hierarchy: bool = False
    ) -> str:
        """
        Get the SQL query for fetching training data.
        
        Args:
            query_type: 'full', 'sample', or custom query name
            order_projects_only: If True, filter to 1.ORD, 2.ORD, 3.ORD projects
            with_hierarchy: If True, include parent sub-project hierarchy
            
        Returns:
            SQL query string
        """
        # Determine which query to use
        if order_projects_only:
            query_name = 'training_data_order_projects'
        elif with_hierarchy:
            query_name = 'training_data_with_hierarchy'
        elif query_type == 'sample':
            query_name = 'training_data_sample'
        elif query_type == 'full':
            query_name = 'training_data_full'
        else:
            query_name = query_type
        
        query_config = self._sql_queries.get(query_name, {})
        return query_config.get('query', self._get_default_training_query(query_type == 'sample'))
    
    def _get_default_training_query(self, sample: bool = False) -> str:
        """
        Get default training query if not configured.
        
        Args:
            sample: If True, limit to 10000 rows
            
        Returns:
            Default SQL query string
        """
        base_query = """
            SELECT 
                a.activity_seq,
                a.project_id,
                a.sub_project_id,
                a.activity_no,
                a.short_name,
                a.description AS activity_description,
                a.rowstate AS activity_status,
                a.early_start,
                a.early_finish,
                a.hours_planned AS plan_hrs,
                a.estimated_progress AS progress,
                SUBSTR(a.sub_project_id, 1, 2) AS discipline_code,
                SUBSTR(a.sub_project_id, 1, 1) AS phase_prefix,
                sp.description AS sub_project_description,
                sp.parent_sub_project_id,
                p.name AS project_name,
                p.description AS project_description,
                p.customer_id,
                p.category1_id AS project_category
            FROM IFSAPP.activity_tab a
            JOIN IFSAPP.sub_project_tab sp 
                ON a.project_id = sp.project_id 
                AND a.sub_project_id = sp.sub_project_id
            JOIN IFSAPP.project_tab p 
                ON a.project_id = p.project_id
            WHERE a.rowstate IN ('Closed', 'Completed')
                AND a.description IS NOT NULL
                AND LENGTH(TRIM(a.description)) > 5
        """
        
        if sample:
            base_query += "\n                AND ROWNUM <= 10000"
        
        return base_query

    def fetch_training_data(
        self, 
        query: Optional[str] = None,
        query_type: str = 'full',
        order_projects_only: bool = False,
        with_hierarchy: bool = False
    ) -> pd.DataFrame:
        """
        Fetch training data from Oracle database via IFS10PRD db link.
        
        Args:
            query: Custom SQL query. If provided, other options are ignored.
            query_type: 'full' (~528K rows), 'sample' (10K rows), or custom query name
            order_projects_only: Filter to 1.ORD, 2.ORD, 3.ORD projects only
            with_hierarchy: Include parent sub-project hierarchy context
        
        Returns:
            DataFrame with columns:
            - activity_seq: Unique activity identifier
            - project_id: Project identifier
            - sub_project_id: Sub-project ID (first 2 chars = discipline code)
            - activity_no: Activity number
            - short_name: Short activity name/code
            - activity_description: Full text description (PRIMARY ML input)
            - activity_status: Status (Closed/Completed)
            - early_start: Planned start date
            - early_finish: Planned finish date  
            - plan_hrs: Planned hours (HOURS_PLANNED)
            - progress: Completion percentage (ESTIMATED_PROGRESS)
            - discipline_code: Extracted 2-letter discipline code (LABEL SOURCE)
            - phase_prefix: First letter of discipline code (phase indicator)
            - sub_project_description: Sub-project description (context)
            - parent_sub_project_id: Parent sub-project for hierarchy
            - project_name: Project name (context)
            - project_description: Project description (context)
            - customer_id: Customer identifier
            - project_category_id: Project category (1.ORD, 2.ORD, etc.)
        """
        if not self.connection:
            if not self.connect():
                return pd.DataFrame()
        
        if query is None:
            query = self.get_training_query(
                query_type=query_type,
                order_projects_only=order_projects_only,
                with_hierarchy=with_hierarchy
            )
        
        try:
            df = pd.read_sql(query, self.connection)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def fetch_training_data_sample(self, limit: int = 10000) -> pd.DataFrame:
        """
        Fetch a sample of training data for quick testing.
        
        Args:
            limit: Maximum number of rows to fetch
            
        Returns:
            DataFrame with sample of activity data
        """
        return self.fetch_training_data(query_type='sample')
    
    def fetch_order_projects_data(self) -> pd.DataFrame:
        """
        Fetch training data filtered to order projects only.
        
        Order projects (1.ORD, 2.ORD, 3.ORD) are customer EPCI projects,
        which are the most relevant for activity classification.
        
        Returns:
            DataFrame with order project activities only
        """
        return self.fetch_training_data(order_projects_only=True)
    
    def fetch_data_with_hierarchy(self) -> pd.DataFrame:
        """
        Fetch training data with full sub-project hierarchy context.
        
        Includes parent sub-project descriptions for richer context.
        
        Returns:
            DataFrame with hierarchy information
        """
        return self.fetch_training_data(with_hierarchy=True)

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
    
    def get_activity_count_by_status(self) -> pd.DataFrame:
        """
        Get count of activities grouped by status.
        
        Returns:
            DataFrame with rowstate and count columns
        """
        query_config = self._sql_queries.get('activity_count', {})
        query = query_config.get('query', """
            SELECT rowstate, COUNT(*) as count 
            FROM IFSAPP.activity_tab 
            GROUP BY rowstate 
            ORDER BY count DESC
        """)
        return self.execute_custom_query(query)
    
    def get_discipline_distribution(self) -> pd.DataFrame:
        """
        Get distribution of discipline codes in completed activities.
        
        Returns:
            DataFrame with discipline_code and count columns
        """
        query_config = self._sql_queries.get('discipline_distribution', {})
        query = query_config.get('query', """
            SELECT SUBSTR(sub_project_id, 1, 2) AS discipline_code, COUNT(*) AS count 
            FROM IFSAPP.activity_tab 
            WHERE rowstate IN ('Closed', 'Completed') 
                AND description IS NOT NULL 
            GROUP BY SUBSTR(sub_project_id, 1, 2) 
            ORDER BY count DESC
        """)
        return self.execute_custom_query(query)

    def get_activity_statistics(self) -> dict:
        """
        Get statistics about activities in the database.
        
        Returns:
            Dictionary with statistics including:
            - total_activities: Total count of all activities
            - completed_activities: Count of Closed + Completed
            - activities_with_description: Count with valid descriptions
            - discipline_code_coverage: Dict of code -> count
        """
        if not self.connection:
            if not self.connect():
                return {}
        
        try:
            cursor = self.connection.cursor()
            
            # Count total activities
            cursor.execute("SELECT COUNT(*) FROM IFSAPP.activity_tab")
            total_count = cursor.fetchone()[0]
            
            # Count completed activities (Closed + Completed)
            cursor.execute("""
                SELECT COUNT(*) FROM IFSAPP.activity_tab 
                WHERE rowstate IN ('Closed', 'Completed')
            """)
            completed_count = cursor.fetchone()[0]
            
            # Count activities with valid descriptions
            cursor.execute("""
                SELECT COUNT(*) FROM IFSAPP.activity_tab 
                WHERE rowstate IN ('Closed', 'Completed')
                    AND description IS NOT NULL 
                    AND LENGTH(TRIM(description)) > 5
            """)
            with_description_count = cursor.fetchone()[0]
            
            cursor.close()
            
            return {
                'total_activities': total_count,
                'completed_activities': completed_count,
                'activities_with_description': with_description_count,
                'training_ready': with_description_count,
                'coverage_pct': round(with_description_count / total_count * 100, 2) if total_count > 0 else 0
            }
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}
    
    def validate_discipline_codes(self) -> Dict[str, any]:
        """
        Validate that discipline codes in the database match configuration.
        
        Returns:
            Dictionary with validation results:
            - total_codes_in_db: Number of unique codes found
            - mapped_codes: Codes that exist in discipline_codes.json
            - unmapped_codes: Codes found in DB but not in config
            - unused_codes: Codes in config but not found in DB
        """
        # Get codes from database
        df = self.get_discipline_distribution()
        if df.empty:
            return {'error': 'Could not fetch discipline distribution'}
        
        db_codes = set(df['discipline_code'].dropna().unique())
        
        # Get codes from config
        config_codes = set()
        for phase_data in self.discipline_config.discipline_codes.values():
            config_codes.update(phase_data.get('codes', {}).keys())
        
        return {
            'total_codes_in_db': len(db_codes),
            'total_codes_in_config': len(config_codes),
            'mapped_codes': sorted(db_codes & config_codes),
            'unmapped_codes': sorted(db_codes - config_codes),
            'unused_codes': sorted(config_codes - db_codes),
            'mapping_coverage_pct': round(len(db_codes & config_codes) / len(db_codes) * 100, 2) if db_codes else 0
        }
