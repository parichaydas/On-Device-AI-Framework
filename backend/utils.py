import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def log_request(method, path, status_code):
    """Log API requests"""
    logger.info(f"{method} {path} - {status_code}")

def format_error_response(error_code, message, details=None):
    """Format error responses"""
    response = {
        'error': {
            'code': error_code,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
    }
    if details:
        response['error']['details'] = details
    return response

def format_success_response(data, message='Success'):
    """Format success responses"""
    return {
        'status': 'success',
        'message': message,
        'data': data,
        'timestamp': datetime.now().isoformat()
    }
