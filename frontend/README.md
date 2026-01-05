Frontend

React-based web interface for the On-Device AI Framework.

## Features

- Query Interface: Clean textarea for submitting AI queries
- Real-time Responses: Instant feedback from backend API
- Responsive Design: Works on desktop, tablet, and mobile
- Loading States: Visual indicators during processing
- Error Handling: User-friendly error messages
- Modern UI: Purple gradient design with smooth animations

## Installation

### Prerequisites
- Node.js 14 or higher
- npm or yarn

### Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. (Optional) Create `.env` file for configuration:
```bash
cp .env.example .env
```

## Running the Application

### Development Mode
```bash
npm start
```
Opens http://localhost:3000 in your browser

### Production Build
```bash
npm run build
```
Creates optimized production build in build/ directory

### Testing
```bash
npm test
```

## Project Structure

```
frontend/
├── src/
│   ├── App.js          # Main React component with query logic
│   ├── App.css         # Component styling
│   ├── index.js        # React entry point
│   ├── index.css       # Global styles
│
├── public/
│   └── index.html      # HTML template
│
├── package.json        # Dependencies and scripts
├── .env.example        # Environment variables template
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

## Configuration

Create a `.env` file with:

```
REACT_APP_API_URL=http://localhost:5000
REACT_APP_API_TIMEOUT=30000
```

## Components

### App Component (App.js)
Main component handling:
- Query input form
- API communication
- Response display
- Loading state management
- Error handling

## API Integration

The frontend communicates with the backend API:

Endpoint: POST http://localhost:5000/api/query

Request:
```javascript
{
  "query": "user's question"
}
```

Response:
```javascript
{
  "query": "...",
  "response": "...",
  "timestamp": "2026-01-05T...",
  "status": "success"
}
```

## Styling

Color Scheme:
- Primary Gradient: Purple (#667eea to #764ba2)
- Background: Light gray (#f5f5f5)
- Text: Dark gray (#333, #666)

Responsive Breakpoints:
- Mobile: < 600px
- Tablet: 600px - 1024px
- Desktop: > 1024px

## Development Workflow

1. Start development server: npm start
2. Make changes to src files
3. Hot reload automatically applies changes
4. Test in browser at http://localhost:3000
5. Check console for errors

## Deployment

### Build for Production
```bash
npm run build
```

### Deploy Static Files
```bash
# Deploy to hosting service (Vercel, Netlify, GitHub Pages, etc.)
```

## Troubleshooting

### Port Already in Use
```bash
PORT=3001 npm start
```

### API Connection Issues
- Verify backend is running on http://localhost:5000
- Check CORS settings
- Review browser console for errors

### Build Errors
```bash
rm -rf node_modules
npm install
npm start
```

## Dependencies

### Production
- react: UI library
- react-dom: React DOM rendering

### Development
- react-scripts: Build and test scripts

## Future Enhancements

- Dark mode support
- Query history
- Multiple conversation threads
- Export results
- Advanced filtering
- Model selection dropdown

---

Version: 1.0.0
Status: Active Development
Last Updated: January 2026
