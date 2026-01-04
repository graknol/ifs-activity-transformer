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
                return ResponseBuilder.success(
                    'Connected to database successfully',
                    {'statistics': stats}
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
            
            if not db_connection.connection:
                return ResponseBuilder.error('Not connected to database', 400)
            
            data = request.get_json()
            custom_query = data.get('query', None)
            
            df = db_connection.fetch_training_data(custom_query)
            
            if df.empty:
                return ResponseBuilder.error('No data retrieved from database', 404)
            
            # Save data
            DataManager.save_training_data(df)
            
            return ResponseBuilder.success(
                f'Retrieved {len(df)} records',
                DataFrameHelper.get_preview(df)
            )
            
        except Exception as e:
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
