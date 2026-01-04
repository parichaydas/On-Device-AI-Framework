# Backend

Flask-based REST API backend for the On-Device AI Framework.

## Features

- RESTful API endpoints for AI model queries
- CORS support for frontend integration
- Health check and model information endpoints
- Document indexing for vector search
- Error handling and logging
- Configuration management
- Integration with Framework module

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```

## Running the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

- `GET /health` - Health check
- `GET /api/info` - API information
- `GET /api/models` - Available models
- `POST /api/query` - Process AI queries (`{ "query": "text" }`)
- `POST /api/index` - Index documents (`{ "docs": { "id1": "text", ... } }`)

## Development

For development with auto-reload:
```bash
FLASK_ENV=development FLASK_DEBUG=True python app.py
```

## Project Structure

```
backend/
├── app.py           # Main Flask application
├── config.py        # Configuration management
├── utils.py         # Utility functions
├── requirements.txt # Python dependencies
├── .env.example    # Environment variables template
└── README.md       # This file
```
