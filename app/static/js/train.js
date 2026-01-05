// Training management JavaScript

let statusCheckInterval = null;

async function startTraining() {
    const textColumn = document.getElementById('text-column').value;
    const labelColumn = document.getElementById('label-column').value;
    const testSize = parseFloat(document.getElementById('test-size').value);
    const trainBtn = document.getElementById('train-btn');
    const progressDiv = document.getElementById('training-progress');
    
    // Validation
    if (!textColumn || !labelColumn) {
        alert('Please provide both text and label column names');
        return;
    }
    
    if (testSize < 0.1 || testSize > 0.5) {
        alert('Validation split must be between 0.1 and 0.5');
        return;
    }
    
    trainBtn.disabled = true;
    trainBtn.textContent = 'Training...';
    progressDiv.style.display = 'block';
    
    try {
        const response = await fetch('/api/train/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text_column: textColumn,
                label_column: labelColumn,
                test_size: testSize
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Start polling for status
            startStatusPolling();
        } else {
            alert(`Error: ${data.message}`);
            trainBtn.disabled = false;
            trainBtn.textContent = 'Start Training';
            progressDiv.style.display = 'none';
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
        trainBtn.disabled = false;
        trainBtn.textContent = 'Start Training';
        progressDiv.style.display = 'none';
    }
}

function startStatusPolling() {
    statusCheckInterval = setInterval(checkTrainingStatus, 2000);
}

function stopStatusPolling() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
    }
}

async function checkTrainingStatus() {
    try {
        const response = await fetch('/api/train/status');
        const data = await response.json();
        
        updateProgressUI(data);
        
        if (!data.is_training && data.progress === 100) {
            stopStatusPolling();
            displayMetrics(data.metrics);
            
            const trainBtn = document.getElementById('train-btn');
            trainBtn.disabled = false;
            trainBtn.textContent = 'Start Training';
        }
    } catch (error) {
        console.error('Error checking status:', error);
    }
}

function updateProgressUI(status) {
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    const statusMessage = document.getElementById('status-message');
    
    progressFill.style.width = `${status.progress}%`;
    progressText.textContent = `${status.progress}%`;
    statusMessage.textContent = status.message;
    
    // Update status message class
    if (status.progress === 100) {
        statusMessage.className = 'status-message success';
    } else if (status.message.includes('Error')) {
        statusMessage.className = 'status-message error';
    } else {
        statusMessage.className = 'status-message info';
    }
}

function displayMetrics(metrics) {
    const metricsDiv = document.getElementById('training-metrics');
    
    if (!metrics || !metrics.eval_metrics) {
        return;
    }
    
    metricsDiv.style.display = 'block';
    
    // Display training loss
    const trainLoss = document.getElementById('train-loss');
    if (metrics.train_metrics && metrics.train_metrics.train_loss) {
        trainLoss.textContent = metrics.train_metrics.train_loss.toFixed(4);
    }
    
    // Display validation accuracy
    const valAccuracy = document.getElementById('val-accuracy');
    if (metrics.eval_metrics.eval_accuracy) {
        valAccuracy.textContent = (metrics.eval_metrics.eval_accuracy * 100).toFixed(2) + '%';
    }
    
    // Display validation loss
    const valLoss = document.getElementById('val-loss');
    if (metrics.eval_metrics.eval_loss) {
        valLoss.textContent = metrics.eval_metrics.eval_loss.toFixed(4);
    }
}
