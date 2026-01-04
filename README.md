# IFS Activity Transformer

A fine tuning application for classification of legacy IFS activities to new project structure in IFS Cloud. This web application uses RoBERTa BigBird transformer model to learn from historical activity mappings and predict appropriate activity paths for new activities.

## Features

- 🗄️ **Oracle Database Integration**: Connect directly to your Oracle database to fetch activity mapping data
- 📁 **CSV File Upload**: Alternative data loading via CSV file upload
- 🤖 **RoBERTa BigBird Model**: State-of-the-art transformer model for sequence classification
- 📊 **Training Dashboard**: Real-time training progress monitoring with metrics
- 🎯 **Prediction Interface**: Single and batch prediction capabilities
- 📈 **Model Evaluation**: Performance metrics and confidence scores
- 🏗️ **Modern Architecture**: Application factory pattern with dependency injection
- 📋 **Best Practices**: Type hints, configuration management, and DRY principles

## Architecture

This project follows Python best practices and design patterns similar to C#'s IServiceCollection:

- **Application Factory Pattern**: Creates app instances with proper configuration
- **Dependency Injection**: Service container for managing dependencies
- **Configuration Management**: Type-safe configuration with dataclasses
- **Separation of Concerns**: Clear separation between routes, services, and data access
- **Testability**: Easy to test with mock services

See [.github/instructions.md](.github/instructions.md) for detailed documentation on patterns and practices.

## Installation

### Prerequisites

- Python 3.8 or higher
- Oracle Database (optional, if using database connection)
- 4GB+ RAM (8GB+ recommended for training)
- GPU recommended for faster training (optional)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/graknol/ifs-activity-transformer.git
cd ifs-activity-transformer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your database credentials and configuration
```

## Configuration

Edit the `.env` file with your settings:

```env
# Oracle Database Configuration
ORACLE_USER=your_username
ORACLE_PASSWORD=your_password
ORACLE_HOST=your_host
ORACLE_PORT=1521
ORACLE_SERVICE=your_service_name

# Flask Configuration
FLASK_SECRET_KEY=your_secret_key_here
FLASK_ENV=development

# Model Configuration
MODEL_NAME=google/bigbird-roberta-base
MAX_LENGTH=512
BATCH_SIZE=8
LEARNING_RATE=2e-5
NUM_EPOCHS=3
SAVE_DIR=./models/saved
```

## Usage

### Starting the Application

Run the web application:
```bash
python run.py
```

Or using the module directly:
```bash
python -m app.app
```

The application will start on `http://localhost:5000`

### Workflow

1. **Load Data**
   - Navigate to the "Data" page
   - Connect to Oracle database OR upload a CSV file
   - Preview your data to ensure it's loaded correctly

2. **Train Model**
   - Go to the "Train" page
   - Configure training parameters:
     - Text column: Column containing activity descriptions
     - Label column: Column containing target activity paths
     - Validation split: Portion of data for validation (0.1-0.5)
   - Click "Start Training" and monitor progress

3. **Make Predictions**
   - Visit the "Predict" page
   - Enter single or multiple activity descriptions
   - Review predictions with confidence scores
   - Download results as CSV

## Data Format

### CSV File Format

Your CSV file should contain at least these columns:
- `activity_description`: Text description of the activity
- `new_activity_path`: Target activity path (e.g., "MainProject->SubProject->Activity")

Example:
```csv
activity_description,new_activity_path
"Installation of equipment at site A","Infrastructure->Installation->Equipment"
"Software development for module B","Software->Development->Module B"
```

### Database Schema

The default query expects a table with:
- `old_activity_id`: Original activity identifier
- `old_activity_name`: Original activity name
- `new_activity_path`: New hierarchical activity path
- `activity_description`: Activity description text

Modify the query in `app/database.py` to match your schema.

## Model Information

The application uses **RoBERTa BigBird** (`google/bigbird-roberta-base`), which:
- Handles long sequences (up to 4096 tokens)
- Uses sparse attention for efficiency
- Provides state-of-the-art performance on text classification
- Pre-trained on large corpus, fine-tuned on your data

### Training Tips

- More training data (500+ examples) generally improves performance
- Ensure balanced representation of all activity types
- Training time depends on data size and hardware (15-60 minutes typical)
- GPU acceleration significantly speeds up training
- Models are automatically saved after training

### Prediction Confidence

- **>90%**: Very high confidence - likely correct
- **70-90%**: Good confidence - review recommended
- **50-70%**: Moderate confidence - manual verification needed
- **<50%**: Low confidence - manual mapping recommended

## Project Structure

```
ifs-activity-transformer/
├── .github/
│   └── instructions.md     # Python best practices & patterns
├── app/
│   ├── app.py              # Flask application factory
│   ├── config.py           # Configuration classes
│   ├── database.py         # Oracle DB connection utilities
│   ├── model.py            # Model training and inference
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css   # Application styles
│   │   └── js/
│   │       ├── database.js # Database page functionality
│   │       ├── train.js    # Training page functionality
│   │       └── predict.js  # Prediction page functionality
│   └── templates/
│       ├── index.html      # Home page
│       ├── database.html   # Data management page
│       ├── train.html      # Training page
│       └── predict.html    # Prediction page
├── services/
│   ├── __init__.py
│   └── container.py        # Dependency injection container
├── data/                   # Training data storage
├── models/                 # Saved models directory
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── run.py                 # Application entry point
└── README.md              # This file
```

## Code Architecture

The application follows Python best practices with patterns similar to C#:

### Dependency Injection

```python
# Service container registration
from services.container import ServiceContainer

container = ServiceContainer()
container.register_singleton('db', OracleDBConnection(config))
container.register_transient('classifier', lambda: ActivityClassifier(config))

# Service resolution
db = container.resolve('db')
```

### Application Factory

```python
# Create app with configuration
from app.app import create_app

app = create_app()  # Uses Config.from_env()

# Or with custom config
custom_config = Config()
app = create_app(custom_config)
```

### Type-Safe Configuration

```python
# Configuration with dataclasses
from app.config import Config, DatabaseConfig, ModelConfig

config = Config.from_env()
db_config = config.database  # Type-safe access
model_config = config.model
```

See [.github/instructions.md](.github/instructions.md) for comprehensive documentation on:
- Dependency injection patterns
- Configuration management
- Service layer pattern
- Repository pattern
- Testing strategies
- DRY principles

## Development
│   │   ├── css/
│   │   │   └── style.css   # Application styles
│   │   └── js/
│   │       ├── database.js # Database page functionality
│   │       ├── train.js    # Training page functionality
│   │       └── predict.js  # Prediction page functionality
│   └── templates/
│       ├── index.html      # Home page
│       ├── database.html   # Data management page
│       ├── train.html      # Training page
│       └── predict.html    # Prediction page
├── data/                   # Training data storage
├── models/                 # Saved models directory
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── run.py                 # Application entry point
└── README.md              # This file
```

## Development

### Running in Development Mode

```bash
export FLASK_ENV=development
python run.py
```

### Running in Production

Use a production WSGI server like Gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app.app:app
```

**Important Production Notes:**
- Use a single worker (`-w 1`) for training to avoid concurrency issues with global state
- For multi-worker deployments, implement proper state management (Redis, database, or job queue)
- Configure authentication and authorization for the web interface
- Validate and sanitize all user inputs, especially custom SQL queries
- Use HTTPS in production environments
- Set strong `FLASK_SECRET_KEY` in production

## Troubleshooting

### Database Connection Issues
- Verify Oracle client is installed
- Check credentials in `.env` file
- Ensure network connectivity to database server

### Training Issues
- Check available memory (8GB+ recommended)
- Reduce `BATCH_SIZE` if out of memory
- Ensure training data has valid labels

### Model Loading Issues
- Verify model was saved successfully after training
- Check `SAVE_DIR` path in `.env`
- Ensure sufficient disk space

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on GitHub.
