Design and Implementation of an On-Device AI Framework

This is a comprehensive full-stack framework for building and deploying on-device AI applications with a modern web interface.

## Project Overview

This is a complete AI framework that combines:
- Backend API: Flask-based REST API for AI model inference and document indexing
- Frontend UI: React-based web interface for user interaction
- Framework Core: AI model integration with embedding and indexing capabilities

## Quick Start

### Backend Setup
```
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

### Frontend Setup
```
cd frontend
npm install
npm start
```

## Contents

- Framework/: core docs, examples, source, and tools for building the on-device AI framework.
- backend/: Flask REST API for queries and document indexing
- frontend/: React web interface for user interaction

## API Endpoints

- GET /health — health check
- GET /api/info — API information
- GET /api/models — available models
- POST /api/query — JSON payload { "query": "text" }
- POST /api/index — JSON payload { "docs": { "id1": "text", ... } }

## Documentation

- Backend README: Detailed backend documentation
- Frontend README: Frontend setup and features
- Framework Documentation: Framework integration guide

## Development

For development with auto-reload:
```
cd backend
FLASK_ENV=development FLASK_DEBUG=True python app.py
```

```
cd frontend
npm start
```

## License

This project is open source and available under the MIT License.

---

Version: 1.0.0
Status: Active Development
Last Updated: January 2026