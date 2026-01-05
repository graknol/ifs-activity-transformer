// Database management JavaScript

function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    event.target.classList.add('active');
}

async function connectDatabase() {
    const btn = document.getElementById('connect-btn');
    const statusDiv = document.getElementById('db-status');
    
    btn.disabled = true;
    btn.textContent = 'Connecting...';
    statusDiv.textContent = 'Connecting to database...';
    statusDiv.className = 'status-message info';
    
    try {
        const response = await fetch('/api/database/connect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            statusDiv.textContent = data.message;
            statusDiv.className = 'status-message success';
            
            // Show statistics
            if (data.statistics) {
                document.getElementById('db-stats').style.display = 'grid';
                document.getElementById('total-activities').textContent = data.statistics.total_activities || 0;
                document.getElementById('mapped-activities').textContent = data.statistics.mapped_activities || 0;
                document.getElementById('unmapped-activities').textContent = data.statistics.unmapped_activities || 0;
                document.getElementById('unique-activities').textContent = data.statistics.unique_new_activities || 0;
            }
            
            // Show fetch section
            document.getElementById('fetch-section').style.display = 'block';
            btn.textContent = 'Connected';
        } else {
            statusDiv.textContent = data.message;
            statusDiv.className = 'status-message error';
            btn.disabled = false;
            btn.textContent = 'Connect to Database';
        }
    } catch (error) {
        statusDiv.textContent = `Error: ${error.message}`;
        statusDiv.className = 'status-message error';
        btn.disabled = false;
        btn.textContent = 'Connect to Database';
    }
}

async function fetchData() {
    const customQuery = document.getElementById('custom-query').value.trim();
    const queryType = document.getElementById('query-type').value;
    const statusDiv = document.getElementById('db-status');
    const fetchBtn = document.getElementById('fetch-btn');
    const fetchProgress = document.getElementById('fetch-progress');
    
    // Show loading state
    fetchBtn.disabled = true;
    fetchProgress.style.display = 'inline-block';
    statusDiv.textContent = 'Fetching data... This may take a moment for large datasets.';
    statusDiv.className = 'status-message info';
    
    try {
        const response = await fetch('/api/database/fetch-data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: customQuery || null,
                query_type: queryType
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            statusDiv.textContent = data.message;
            statusDiv.className = 'status-message success';
            displayDataPreview(data);
        } else {
            statusDiv.textContent = data.message;
            statusDiv.className = 'status-message error';
        }
    } catch (error) {
        statusDiv.textContent = `Error: ${error.message}`;
        statusDiv.className = 'status-message error';
    } finally {
        fetchBtn.disabled = false;
        fetchProgress.style.display = 'none';
    }
}

function handleFileSelect() {
    const fileInput = document.getElementById('csv-file');
    const fileName = document.getElementById('file-name');
    const uploadBtn = document.getElementById('upload-btn');
    
    if (fileInput.files.length > 0) {
        fileName.textContent = `Selected: ${fileInput.files[0].name}`;
        uploadBtn.style.display = 'inline-block';
    }
}

async function uploadCSV() {
    const fileInput = document.getElementById('csv-file');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a file first');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    const uploadBtn = document.getElementById('upload-btn');
    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Uploading...';
    
    try {
        const response = await fetch('/api/upload-csv', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert(data.message);
            displayDataPreview(data);
        } else {
            alert(`Error: ${data.message}`);
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Upload File';
    }
}

function displayDataPreview(data) {
    const previewDiv = document.getElementById('data-preview');
    const rowCount = document.getElementById('row-count');
    const columnList = document.getElementById('column-list');
    const tableHead = document.getElementById('table-head');
    const tableBody = document.getElementById('table-body');
    
    previewDiv.style.display = 'block';
    rowCount.textContent = data.row_count;
    columnList.textContent = data.columns.join(', ');
    
    // Create table header
    let headerHTML = '<tr>';
    data.columns.forEach(col => {
        headerHTML += `<th>${col}</th>`;
    });
    headerHTML += '</tr>';
    tableHead.innerHTML = headerHTML;
    
    // Create table body
    let bodyHTML = '';
    data.preview.forEach(row => {
        bodyHTML += '<tr>';
        data.columns.forEach(col => {
            bodyHTML += `<td>${row[col] !== null ? row[col] : ''}</td>`;
        });
        bodyHTML += '</tr>';
    });
    tableBody.innerHTML = bodyHTML;
}

// Check for cached data on page load
async function checkCacheStatus() {
    try {
        const response = await fetch('/api/database/cache-info');
        const data = await response.json();
        
        const cacheStatusDiv = document.getElementById('cache-status');
        if (!cacheStatusDiv) return;
        
        if (data.cache && data.cache.exists) {
            cacheStatusDiv.innerHTML = `
                <div class="cache-info">
                    <strong>📦 Cached Data Available:</strong> 
                    ${data.cache.row_count.toLocaleString()} records (${data.cache.size_mb} MB)
                    <br><small>Last updated: ${new Date(data.cache.modified).toLocaleString()}</small>
                    <div class="cache-actions" style="margin-top: 8px;">
                        <button onclick="loadFromCache()" class="btn btn-success btn-sm">Load from Cache</button>
                        <button onclick="clearCache()" class="btn btn-warning btn-sm">Clear Cache</button>
                    </div>
                </div>
            `;
            cacheStatusDiv.style.display = 'block';
        } else {
            cacheStatusDiv.innerHTML = '<div class="cache-info muted">No cached data. Fetch from database to create cache.</div>';
            cacheStatusDiv.style.display = 'block';
        }
    } catch (error) {
        console.error('Error checking cache:', error);
    }
}

async function loadFromCache() {
    const statusDiv = document.getElementById('db-status');
    statusDiv.textContent = 'Loading data from cache...';
    statusDiv.className = 'status-message info';
    
    try {
        const response = await fetch('/api/database/load-cache', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        if (data.success) {
            statusDiv.textContent = data.message;
            statusDiv.className = 'status-message success';
            displayDataPreview(data);
        } else {
            statusDiv.textContent = data.message;
            statusDiv.className = 'status-message error';
        }
    } catch (error) {
        statusDiv.textContent = `Error: ${error.message}`;
        statusDiv.className = 'status-message error';
    }
}

async function clearCache() {
    if (!confirm('Are you sure you want to clear the cached data?')) {
        return;
    }
    
    try {
        const response = await fetch('/api/database/clear-cache', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert('Cache cleared successfully');
            checkCacheStatus();
            
            // Hide data preview
            const previewDiv = document.getElementById('data-preview');
            if (previewDiv) {
                previewDiv.style.display = 'none';
            }
        } else {
            alert(`Error: ${data.message}`);
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

// Initialize cache check on page load
document.addEventListener('DOMContentLoaded', checkCacheStatus);
