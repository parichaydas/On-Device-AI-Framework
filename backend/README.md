Backend

Flask-based REST API backend for the On-Device AI Framework.

## Features

- RESTful API endpoints for AI model queries
- CORS support for frontend integration
- Health check and model information endpoints
- Error handling and logging
- Configuration management
- Document indexing for vector search
- Framework integration

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
- `POST /api/query` - Process AI queries (request: { "query": "text" })
- `POST /api/index` - Index documents (request: { "docs": { "id1": "text", ... } })

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

## Configuration

Create a `.env` file with:

```
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_PORT=5000
SECRET_KEY=your-secret-key-here
FRONTEND_URL=http://localhost:3000
LOG_LEVEL=INFO
```

## Testing

### Health Check
```bash
curl http://localhost:5000/health
```

### Test Query
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test query"}'
```

### Get Models
```bash
curl http://localhost:5000/api/models
```

## Deployment

### Production Build
```bash
pip install -r requirements.txt
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker
```bash
docker build -t ai-framework-backend .
docker run -p 5000:5000 -e FLASK_ENV=production ai-framework-backend
```

## Troubleshooting

### Port Already in Use
```bash
FLASK_PORT=5001 python app.py
```

### CORS Errors
- Verify backend is running
- Check FRONTEND_URL in .env
- Verify Flask-CORS is installed

### Import Errors
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`
- Check Python version (3.8+)

## Dependencies

- Flask 2.3.0
- Flask-CORS 4.0.0
- Python-dotenv 1.0.0
- Gunicorn 21.2.0

---

Version: 1.0.0
Status: Active Development
Last Updated: January 2026
