// Prediction JavaScript

let batchResults = [];

async function predictSingle() {
    const text = document.getElementById('single-text').value.trim();
    
    if (!text) {
        alert('Please enter an activity description');
        return;
    }
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                texts: [text]
            })
        });
        
        const data = await response.json();
        
        if (data.success && data.predictions.length > 0) {
            displaySingleResult(data.predictions[0]);
        } else {
            alert(`Error: ${data.message}`);
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

function displaySingleResult(prediction) {
    const resultDiv = document.getElementById('single-result');
    const predictedPath = document.getElementById('predicted-path');
    const confidence = document.getElementById('confidence');
    const confidenceFill = document.getElementById('confidence-fill');
    
    resultDiv.style.display = 'block';
    predictedPath.textContent = prediction.predicted_activity;
    
    const confidencePercent = (prediction.confidence * 100).toFixed(2);
    confidence.textContent = confidencePercent + '%';
    confidenceFill.style.width = confidencePercent + '%';
    
    // Change color based on confidence
    if (prediction.confidence > 0.9) {
        confidenceFill.style.backgroundColor = '#28a745'; // Green
    } else if (prediction.confidence > 0.7) {
        confidenceFill.style.backgroundColor = '#ffc107'; // Yellow
    } else {
        confidenceFill.style.backgroundColor = '#dc3545'; // Red
    }
}

async function predictBatch() {
    const text = document.getElementById('batch-text').value.trim();
    
    if (!text) {
        alert('Please enter activity descriptions');
        return;
    }
    
    // Split by lines and filter empty lines
    const texts = text.split('\n').filter(line => line.trim() !== '');
    
    if (texts.length === 0) {
        alert('No valid activity descriptions found');
        return;
    }
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                texts: texts
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            batchResults = data.predictions;
            displayBatchResults(data.predictions);
        } else {
            alert(`Error: ${data.message}`);
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

function displayBatchResults(predictions) {
    const resultsDiv = document.getElementById('batch-results');
    const resultsBody = document.getElementById('results-body');
    
    resultsDiv.style.display = 'block';
    
    let html = '';
    predictions.forEach((pred, index) => {
        const confidencePercent = (pred.confidence * 100).toFixed(2);
        let confidenceClass = '';
        
        if (pred.confidence > 0.9) {
            confidenceClass = 'confidence-high';
        } else if (pred.confidence > 0.7) {
            confidenceClass = 'confidence-medium';
        } else {
            confidenceClass = 'confidence-low';
        }
        
        html += `
            <tr>
                <td>${index + 1}</td>
                <td>${pred.text}</td>
                <td>${pred.predicted_activity}</td>
                <td class="${confidenceClass}">${confidencePercent}%</td>
            </tr>
        `;
    });
    
    resultsBody.innerHTML = html;
}

function downloadResults() {
    if (batchResults.length === 0) {
        alert('No results to download');
        return;
    }
    
    // Create CSV content
    let csv = 'Activity Description,Predicted Path,Confidence\n';
    
    batchResults.forEach(pred => {
        const text = `"${pred.text.replace(/"/g, '""')}"`;
        const path = `"${pred.predicted_activity.replace(/"/g, '""')}"`;
        const confidence = pred.confidence.toFixed(4);
        csv += `${text},${path},${confidence}\n`;
    });
    
    // Create download link
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'predictions.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}
