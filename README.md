# IFS Activity Transformer

A fine tuning application for classification of legacy IFS activities to new project structure in IFS Cloud. This web application uses advanced active learning with multi-label classification, combining RoBERTa BigBird transformer with tabular features for optimal activity categorization.

## Features

- 🗄️ **Oracle Database Integration**: Connect directly to your Oracle database to fetch activity mapping data
- 📁 **CSV File Upload**: Alternative data loading via CSV file upload
- ✏️ **Active Learning Annotation**: Interactive UI with uncertainty sampling - prioritizes samples the model is least confident about
- 🏷️ **Multi-Label Classification**: Categorize activities with multiple labels (procurement, jobcard, milestone, expense, etc.)
- 🔬 **Hybrid Model Architecture**: Combines RoBERTa text encoding with tabular numerical features
- ☁️ **Azure Blob Backup**: Automatic and manual snapshots of annotations to Azure Storage - never lose valuable training data
- 📊 **Training Dashboard**: Real-time training progress monitoring with metrics
- 🎯 **Prediction Interface**: Single and batch prediction capabilities with confidence scores
- 📈 **Model Evaluation**: Performance metrics and confidence scores
- 🏗️ **Modern Architecture**: Application factory pattern with dependency injection
- 📋 **Best Practices**: Type hints, configuration management, and DRY principles

## Data Protection with Azure Backup

The application includes comprehensive backup capabilities to protect valuable annotation data:

### Automatic Backups
- **Auto-snapshots**: Automatically creates backups every N annotations (configurable)
- **Session backups**: Saves data at the end of annotation sessions
- **Zero data loss**: All manual tagging and user answers are preserved

### Manual Backups
- **On-demand backups**: Create backups anytime with a single click
- **Milestone backups**: Create named snapshots at important points (e.g., "After 100 annotations")
- **Backup manager UI**: View, manage, and restore from previous backups

### Restoration
- **Easy recovery**: Restore annotations from any previous backup
- **Backup history**: View all backups with timestamps and sizes
- **Version control**: Keep multiple versions of your annotation data

### Configuration
Add to your `.env` file:
```bash
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=your_account;AccountKey=your_key;EndpointSuffix=core.windows.net
AZURE_CONTAINER_NAME=ifs-annotations
AUTO_BACKUP_THRESHOLD=10
```

## Active Learning Workflow

The application implements a sophisticated **two-phase active learning approach**:

### Phase 1: Multi-Label Embedding
1. **Data Loading**: Import activities from Oracle DB or CSV with custom SQL queries
2. **Uncertainty Sampling**: Model identifies activities it's least confident about
3. **Intelligent Annotation**: Users label prioritized samples one at a time
4. **Sanity Checks**: Occasionally shows confident predictions for validation
5. **Category Assignment**: Multi-label classification (procurement, jobcard, milestone, etc.)

### Phase 2: WBS Decoding (Future)
- Use category embeddings to predict hierarchical WBS paths
- Hierarchical classifier or seq2seq model for structure prediction

### Why This Approach Works
- **Uncertainty Sampling**: Maximizes model improvement per labeled sample
- **Multi-Label**: Captures the multi-faceted nature of activities
- **Hybrid Features**: Leverages both text descriptions and numerical data (# jobcards, procurement lines, etc.)
- **Human-in-the-Loop**: Domain expertise guides model learning
- **Interpretable**: Two-phase approach allows iterative refinement

## Architecture

This project follows Python best practices and design patterns similar to C#'s IServiceCollection:

### Clean Architecture Principles
- **Application Factory Pattern**: Creates app instances with proper configuration (116 lines vs 678 lines before refactoring)
- **Dependency Injection**: Service container for managing dependencies
- **Configuration Management**: Type-safe configuration with dataclasses
- **Modular Routes**: Organized by feature area (pages, database, training, prediction, annotation, backup)
- **DRY Utilities**: Centralized common operations (DataManager, ResponseBuilder, FileValidator)
- **Separation of Concerns**: Clear separation between routes, services, and data access
- **Testability**: Easy to test with mock services

### Project Structure
```
app/
├── __init__.py
├── app.py                 # Application factory (116 lines - refactored!)
├── config.py              # Type-safe configuration
├── database.py            # Oracle DB repository
├── model.py               # RoBERTa classifier service
├── active_learning.py     # Uncertainty sampling service
├── multilabel_model.py    # Hybrid multi-label classifier
├── backup_service.py      # Azure Blob backup service
├── utils.py               # DRY utilities (NEW - 258 lines)
├── routes/                # Modular routes (NEW)
│   ├── __init__.py       # Route registration
│   ├── pages.py          # HTML page routes
│   ├── database.py       # Database API routes
│   ├── training.py       # Training API routes
│   ├── prediction.py     # Prediction API routes
│   ├── annotation.py     # Annotation API routes
│   └── backup.py         # Backup API routes
├── static/               # CSS, JavaScript
└── templates/            # HTML templates

services/
├── __init__.py
└── container.py          # DI container implementation
```

### DRY Improvements
- **83% reduction** in main app.py (678 → 116 lines)
- **Eliminated duplication**: Common patterns extracted to utilities
- **Centralized data management**: DataManager class handles all file operations
- **Unified response building**: ResponseBuilder ensures consistent API responses
- **Reusable validation**: FileValidator for common file checks
- **Modular routes**: Each feature area has its own route module

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
