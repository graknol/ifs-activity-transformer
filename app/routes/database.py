"""Database-related API routes."""
from flask import request
from app.utils import ResponseBuilder, DataManager, DataFrameHelper, FileValidator


def get_service(service_type: str):
    """Helper to get service from container."""
    from flask import current_app
    return current_app.container.resolve(service_type)


def register_database_routes(app):
    """Register database routes."""
    
    @app.route('/api/database/connect', methods=['POST'])
    def connect_database():
        """Connect to Oracle database."""
        try:
            db_connection = get_service('db_connection')
            success = db_connection.connect()
            
            if success:
                stats = db_connection.get_activity_statistics()
                
                # Map statistics to UI-expected keys
                ui_stats = {
                    'total_activities': stats.get('total_activities', 0),
                    'mapped_activities': stats.get('activities_with_description', 0),
                    'unmapped_activities': stats.get('total_activities', 0) - stats.get('activities_with_description', 0),
                    'unique_new_activities': stats.get('training_ready', 0),
                    'coverage_pct': stats.get('coverage_pct', 0)
                }
                
                return ResponseBuilder.success(
                    'Connected to database successfully',
                    {'statistics': ui_stats}
                )
            else:
                return ResponseBuilder.error('Failed to connect to database', 500)
                
        except Exception as e:
            return ResponseBuilder.from_exception(e, "Database connection")
    
    @app.route('/api/database/fetch-data', methods=['POST'])
    def fetch_data():
        """Fetch training data from database."""
        try:
            db_connection = get_service('db_connection')
            
            data = request.get_json() or {}
            custom_query = data.get('query', None) if data.get('query') else None
            query_type = data.get('query_type', 'sample')  # Default to sample for UI
            use_cache = data.get('use_cache', False)  # Option to load from cache
            
            # If use_cache is True and cache exists, load from cache
            if use_cache and DataManager.has_cached_data():
                df = DataManager.load_training_data()
                if df is not None:
                    cache_info = DataManager.get_cache_info()
                    return ResponseBuilder.success(
                        f'Loaded {len(df)} records from cache (saved {cache_info.get("modified", "unknown")})',
                        DataFrameHelper.get_preview(df)
                    )
            
            # Auto-reconnect if connection was lost
            if not db_connection.connection:
                print("Connection not established, attempting to connect...")
                if not db_connection.connect():
                    return ResponseBuilder.error('Not connected to database. Please connect first.', 400)
            
            print(f"Fetching data with query_type={query_type}, custom_query={bool(custom_query)}")
            
            # Fetch data
            if custom_query:
                df = db_connection.fetch_training_data(query=custom_query)
            else:
                df = db_connection.fetch_training_data(query_type=query_type)
            
            if df.empty:
                # Check if it's a connection issue
                try:
                    # Test simple query
                    test_df = db_connection.execute_custom_query(
                        "SELECT COUNT(*) as cnt FROM IFSAPP.activity_tab WHERE ROWNUM = 1"
                    )
                    if test_df.empty:
                        return ResponseBuilder.error(
                            'Database connection issue - cannot query IFSAPP schema', 
                            500
                        )
                except Exception as test_error:
                    return ResponseBuilder.error(
                        f'Database query failed: {str(test_error)}', 
                        500
                    )
                
                return ResponseBuilder.error(
                    'No data retrieved - query returned empty result set', 
                    404
                )
            
            # Save data
            DataManager.save_training_data(df)
            
            return ResponseBuilder.success(
                f'Retrieved {len(df)} records',
                DataFrameHelper.get_preview(df)
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return ResponseBuilder.from_exception(e, "Data fetch")
    
    @app.route('/api/upload-csv', methods=['POST'])
    def upload_csv():
        """Upload CSV file with training data."""
        try:
            file = request.files.get('file')
            
            # Validate file
            is_valid, error_msg = FileValidator.validate_csv_upload(file)
            if not is_valid:
                return ResponseBuilder.error(error_msg, 400)
            
            # Save and validate
            file.save(DataManager.TRAINING_DATA_PATH)
            df = DataManager.load_training_data()
            
            if df is None:
                return ResponseBuilder.error('Failed to read uploaded file', 500)
            
            return ResponseBuilder.success(
                f'Uploaded {len(df)} records',
                DataFrameHelper.get_preview(df)
            )
            
        except Exception as e:
            return ResponseBuilder.from_exception(e, "File upload")

    @app.route('/api/database/cache-info', methods=['GET'])
    def get_cache_info():
        """Get information about cached training data."""
        try:
            cache_info = DataManager.get_cache_info()
            
            if cache_info is None:
                return ResponseBuilder.success(
                    'No cached data found',
                    {'cache': {'exists': False}}
                )
            
            return ResponseBuilder.success(
                f'Cache found: {cache_info.get("row_count", 0)} records, {cache_info.get("size_mb", 0)} MB',
                {'cache': cache_info}
            )
            
        except Exception as e:
            return ResponseBuilder.from_exception(e, "Cache info")

    @app.route('/api/database/load-cache', methods=['POST'])
    def load_cached_data():
        """Load training data from cache."""
        try:
            if not DataManager.has_cached_data():
                return ResponseBuilder.error('No cached data found. Fetch data from database first.', 404)
            
            df = DataManager.load_training_data()
            if df is None:
                return ResponseBuilder.error('Failed to load cached data', 500)
            
            cache_info = DataManager.get_cache_info()
            
            return ResponseBuilder.success(
                f'Loaded {len(df)} records from cache',
                {
                    **DataFrameHelper.get_preview(df),
                    'cache': cache_info
                }
            )
            
        except Exception as e:
            return ResponseBuilder.from_exception(e, "Load cache")

    @app.route('/api/database/clear-cache', methods=['POST'])
    def clear_cache():
        """Clear cached training data."""
        try:
            if not DataManager.has_cached_data():
                return ResponseBuilder.success('No cache to clear')
            
            if DataManager.clear_cache():
                return ResponseBuilder.success('Cache cleared successfully')
            else:
                return ResponseBuilder.error('Failed to clear cache', 500)
            
        except Exception as e:
            return ResponseBuilder.from_exception(e, "Clear cache")
