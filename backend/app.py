import os
import sys
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Try to load framework
framework = None
try:
    SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Framework', 'src'))
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    from app import FrameworkApp
    framework = FrameworkApp()
    logger.info("Framework loaded successfully")
except Exception as e:
    logger.warning(f"Framework not available: {str(e)}")

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'On-Device AI Framework Backend'
    }), 200


@app.route('/api/query', methods=['POST'])
def handle_query():
    """Handle AI queries from the frontend"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query parameter'}), 400
        
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        logger.info(f"Processing query: {query[:100]}...")
        
        # Process the query
        if framework:
            response = framework.query(query, k=5)
            response_text = str(response)
        else:
            response_text = process_query(query)
        
        return jsonify({
            'query': query,
            'response': response_text,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@app.route('/api/info', methods=['GET'])
def get_info():
    """Get API information"""
    return jsonify({
        'name': 'On-Device AI Framework',
        'version': '1.0.0',
        'description': 'Backend API for on-device AI inference',
        'endpoints': {
            'health': '/health',
            'info': '/api/info',
            'query': '/api/query',
            'models': '/api/models',
            'index': '/api/index'
        }
    }), 200


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    return jsonify({
        'models': [
            {
                'id': 'default',
                'name': 'Default Model',
                'type': 'language',
                'status': 'ready'
            }
        ],
        'total': 1
    }), 200


@app.route('/api/index', methods=['POST'])
def index_documents():
    """Index documents for vector search"""
    try:
        data = request.get_json()
        
        if not data or 'docs' not in data:
            return jsonify({'error': 'Missing docs parameter'}), 400
        
        docs = data.get('docs', {})
        
        if framework:
            framework.index_documents(docs)
            count = len(docs)
        else:
            count = len(docs)
        
        logger.info(f"Indexed {count} documents")
        
        return jsonify({
            'status': 'success',
            'indexed': count,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error indexing documents: {str(e)}")
        return jsonify({
            'error': 'Indexing failed',
            'details': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'path': request.path
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error'
    }), 500


def process_query(query):
    """
    Process the AI query
    This is a placeholder - integrate with your actual AI model here
    """
    try:
        response = f"Processed query: {query}\n\nThis is a placeholder response. Integrate your AI model here."
        return response
    except Exception as e:
        logger.error(f"Error in process_query: {str(e)}")
        return f"Error processing query: {str(e)}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
