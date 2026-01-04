import React, { useState } from 'react';
import './App.css';

function App() {
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!message.trim()) return;

    setLoading(true);
    try {
      const result = await fetch('http://localhost:5000/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: message }),
      });

      const data = await result.json();
      setResponse(data.response || 'No response received');
    } catch (error) {
      setResponse(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>On-Device AI Framework</h1>
        <p>Query the AI Model</p>
      </header>

      <main className="container">
        <form onSubmit={handleSubmit} className="query-form">
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Enter your query here..."
            disabled={loading}
            rows="4"
          />
          <button type="submit" disabled={loading}>
            {loading ? 'Processing...' : 'Submit Query'}
          </button>
        </form>

        {response && (
          <div className="response-box">
            <h2>Response:</h2>
            <p>{response}</p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
