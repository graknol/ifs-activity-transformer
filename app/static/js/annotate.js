// Annotation interface JavaScript

let currentActivity = null;
let currentIndex = 0;
let activities = [];
let annotations = {};
let categories = [
    'procurement',
    'jobcard',
    'milestone',
    'expense',
    'engineering',
    'construction',
    'planning',
    'quality'
];

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadCategories();
    checkDataStatusAndLoad();
    checkBackupStatus();
});

async function checkDataStatusAndLoad() {
    const loadingMessage = document.getElementById('loading-message');
    
    try {
        // First check if data is loaded
        const statusResponse = await fetch('/api/annotate/data-status');
        const statusData = await statusResponse.json();
        
        if (!statusData.data_loaded) {
            // No data loaded - show helpful message
            loadingMessage.innerHTML = `
                <div class="alert alert-warning" style="padding: 20px; background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; margin: 20px 0;">
                    <h3 style="margin-top: 0; color: #856404;">⚠️ No Training Data Loaded</h3>
                    <p>Before you can annotate activities, you need to load training data.</p>
                    <p><strong>Steps:</strong></p>
                    <ol>
                        <li>Go to the <a href="/database" style="color: #0056b3; font-weight: bold;">Data page</a></li>
                        <li>Connect to the Oracle database or upload a CSV file</li>
                        <li>Fetch training data (or load from cache if available)</li>
                        <li>Return here to start annotating</li>
                    </ol>
                    <a href="/database" class="btn btn-primary" style="margin-top: 10px;">Go to Data Page →</a>
                </div>
            `;
            
            // Update progress to show 0 activities
            updateProgress(statusData.statistics || { total_samples: 0, labeled_samples: 0 });
            return;
        }
        
        // Data is loaded, proceed to load activities
        loadingMessage.textContent = `Loading activities from ${statusData.cache?.row_count?.toLocaleString() || 'cached'} records...`;
        await loadActivities();
        
    } catch (error) {
        console.error('Error checking data status:', error);
        loadingMessage.innerHTML = `<p class="error">Error checking data status: ${error.message}</p>`;
    }
}

// Backup management functions
async function checkBackupStatus() {
    try {
        const response = await fetch('/api/backup/status');
        const data = await response.json();
        
        if (data.success && data.backup_enabled) {
            const statusDiv = document.getElementById('backup-status');
            const statusText = document.getElementById('backup-status-text');
            const lastBackup = document.getElementById('last-backup-time');
            
            statusDiv.style.display = 'block';
            statusText.textContent = 'Enabled';
            
            if (data.statistics && data.statistics.latest_backup) {
                const backupDate = new Date(data.statistics.latest_backup.created);
                lastBackup.textContent = backupDate.toLocaleString();
            }
        }
    } catch (error) {
        console.error('Error checking backup status:', error);
    }
}

async function createManualBackup() {
    try {
        const response = await fetch('/api/backup/create', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                type: 'manual'
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert('Backup created successfully!');
            checkBackupStatus();
        } else {
            alert('Backup failed: ' + data.message);
        }
    } catch (error) {
        console.error('Error creating backup:', error);
        alert('Failed to create backup');
    }
}

async function createMilestoneBackup() {
    const name = document.getElementById('milestone-name').value;
    const description = document.getElementById('milestone-desc').value;
    
    if (!name) {
        alert('Please enter a milestone name');
        return;
    }
    
    try {
        const response = await fetch('/api/backup/create', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                type: 'milestone',
                milestone_name: name,
                description: description
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert('Milestone backup created!');
            document.getElementById('milestone-name').value = '';
            document.getElementById('milestone-desc').value = '';
            loadBackupList();
        } else {
            alert('Backup failed: ' + data.message);
        }
    } catch (error) {
        console.error('Error creating milestone:', error);
        alert('Failed to create milestone backup');
    }
}

async function showBackupManager() {
    document.getElementById('backup-modal').style.display = 'block';
    await loadBackupList();
}

function closeBackupManager() {
    document.getElementById('backup-modal').style.display = 'none';
}

async function loadBackupList() {
    try {
        const response = await fetch('/api/backup/list?limit=50');
        const data = await response.json();
        
        const listDiv = document.getElementById('backup-list');
        
        if (data.success && data.backups.length > 0) {
            listDiv.innerHTML = '<table style="width: 100%; border-collapse: collapse;">' +
                '<tr style="border-bottom: 2px solid #ddd;"><th>Name</th><th>Date</th><th>Size</th><th>Action</th></tr>' +
                data.backups.map(backup => `
                    <tr style="border-bottom: 1px solid #eee;">
                        <td style="padding: 0.5rem;">${backup.name}</td>
                        <td style="padding: 0.5rem;">${new Date(backup.created).toLocaleString()}</td>
                        <td style="padding: 0.5rem;">${(backup.size / 1024).toFixed(2)} KB</td>
                        <td style="padding: 0.5rem;">
                            <button onclick="restoreBackup('${backup.name}')" class="btn btn-sm">Restore</button>
                        </td>
                    </tr>
                `).join('') +
                '</table>';
        } else {
            listDiv.innerHTML = '<p>No backups available</p>';
        }
    } catch (error) {
        console.error('Error loading backups:', error);
        document.getElementById('backup-list').innerHTML = '<p>Error loading backups</p>';
    }
}

async function restoreBackup(blobName) {
    if (!confirm('Are you sure you want to restore this backup? This will create a new file with restored data.')) {
        return;
    }
    
    try {
        const response = await fetch('/api/backup/restore', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                blob_name: blobName
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert('Backup restored successfully! Check the data folder for the restored file.');
        } else {
            alert('Restore failed: ' + data.message);
        }
    } catch (error) {
        console.error('Error restoring backup:', error);
        alert('Failed to restore backup');
    }
}

function loadCategories() {
    const grid = document.getElementById('categories-grid');
    grid.innerHTML = '';
    
    categories.forEach(category => {
        const div = document.createElement('div');
        div.className = 'category-checkbox';
        div.id = `category-${category}`;
        div.onclick = () => toggleCategory(category);
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `checkbox-${category}`;
        checkbox.value = category;
        
        const label = document.createElement('span');
        label.className = 'category-label';
        label.textContent = category.charAt(0).toUpperCase() + category.slice(1);
        
        div.appendChild(checkbox);
        div.appendChild(label);
        grid.appendChild(div);
    });
}

function toggleCategory(category) {
    const div = document.getElementById(`category-${category}`);
    const checkbox = document.getElementById(`checkbox-${category}`);
    
    checkbox.checked = !checkbox.checked;
    
    if (checkbox.checked) {
        div.classList.add('selected');
    } else {
        div.classList.remove('selected');
    }
}

async function loadActivities() {
    const loadingMessage = document.getElementById('loading-message');
    
    try {
        const response = await fetch('/api/annotate/get-batch', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                batch_size: 50,
                sampling_strategy: 'least_confidence'
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            activities = data.activities;
            updateProgress(data.statistics);
            
            if (activities.length > 0) {
                currentIndex = 0;
                displayActivity(activities[currentIndex]);
                loadingMessage.style.display = 'none';
                document.getElementById('activity-card').style.display = 'block';
            } else {
                loadingMessage.innerHTML = 
                    '<p>No activities available for annotation. All activities may already be labeled.</p>';
            }
        } else {
            // Show error with hint if available
            let errorHtml = `<p class="error">Error loading activities: ${data.message}</p>`;
            if (data.details && data.details.hint) {
                errorHtml += `<p><em>${data.details.hint}</em></p>`;
                errorHtml += `<a href="/database" class="btn btn-primary">Go to Data Page</a>`;
            }
            loadingMessage.innerHTML = errorHtml;
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to load activities');
    }
}

function displayActivity(activity) {
    currentActivity = activity;
    
    // Update header
    document.getElementById('activity-name').textContent = activity.name || 'Unnamed Activity';
    document.getElementById('activity-id').textContent = activity.id;
    
    // Update description
    document.getElementById('activity-text').textContent = activity.description;
    
    // Update confidence badge
    const uncertaintyScore = activity.uncertainty_score || 0;
    const badge = document.getElementById('confidence-badge');
    badge.textContent = `Uncertainty: ${(uncertaintyScore * 100).toFixed(1)}%`;
    
    if (uncertaintyScore > 0.6) {
        badge.className = 'confidence-badge confidence-low';
    } else if (uncertaintyScore > 0.3) {
        badge.className = 'confidence-badge confidence-medium';
    } else {
        badge.className = 'confidence-badge confidence-high';
    }
    
    // Update features
    displayFeatures(activity.features || {});
    
    // Update model predictions
    displayPredictions(activity.predicted_categories || [], activity.max_confidence || 0);
    
    // Load existing annotations if any
    loadExistingAnnotations(activity.id);
}

function displayFeatures(features) {
    const grid = document.getElementById('features-grid');
    grid.innerHTML = '';
    
    const featureNames = {
        'hierarchy_level': 'Hierarchy Level',
        'num_jobcards': '# Job Cards',
        'num_procurement_lines': '# Procurement Lines',
        'total_expenses': 'Total Expenses',
        'num_time_registrations': '# Time Registrations'
    };
    
    for (const [key, value] of Object.entries(features)) {
        const div = document.createElement('div');
        div.className = 'feature-item';
        
        const label = document.createElement('div');
        label.className = 'feature-label';
        label.textContent = featureNames[key] || key;
        
        const valueDiv = document.createElement('div');
        valueDiv.className = 'feature-value';
        valueDiv.textContent = value;
        
        div.appendChild(label);
        div.appendChild(valueDiv);
        grid.appendChild(div);
    }
}

function displayPredictions(predictedCategories, confidence) {
    const container = document.getElementById('prediction-tags');
    container.innerHTML = '';
    
    if (predictedCategories.length === 0) {
        container.innerHTML = '<p><em>No predictions available</em></p>';
    } else {
        predictedCategories.forEach(category => {
            const tag = document.createElement('span');
            tag.className = 'prediction-tag';
            tag.textContent = category.charAt(0).toUpperCase() + category.slice(1);
            container.appendChild(tag);
        });
    }
    
    document.getElementById('model-confidence').textContent = (confidence * 100).toFixed(1);
}

function loadExistingAnnotations(activityId) {
    // Clear all checkboxes first
    categories.forEach(category => {
        const checkbox = document.getElementById(`checkbox-${category}`);
        const div = document.getElementById(`category-${category}`);
        checkbox.checked = false;
        div.classList.remove('selected');
    });
    
    // Load saved annotations
    if (annotations[activityId]) {
        annotations[activityId].forEach(category => {
            const checkbox = document.getElementById(`checkbox-${category}`);
            const div = document.getElementById(`category-${category}`);
            if (checkbox) {
                checkbox.checked = true;
                div.classList.add('selected');
            }
        });
    }
}

function getSelectedCategories() {
    return categories.filter(category => {
        return document.getElementById(`checkbox-${category}`).checked;
    });
}

async function saveAndNext() {
    await saveCurrentAnnotation();
    
    if (currentIndex < activities.length - 1) {
        currentIndex++;
        displayActivity(activities[currentIndex]);
    } else {
        showCompletedMessage();
    }
}

async function saveAndPrevious() {
    await saveCurrentAnnotation();
    
    if (currentIndex > 0) {
        currentIndex--;
        displayActivity(activities[currentIndex]);
    }
}

async function saveCurrentAnnotation() {
    const selectedCategories = getSelectedCategories();
    const activityId = currentActivity.id;
    
    // Save locally
    annotations[activityId] = selectedCategories;
    
    // Save to server
    try {
        const response = await fetch('/api/annotate/save', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                activity_id: activityId,
                categories: selectedCategories,
                uncertainty_score: currentActivity.uncertainty_score
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            updateProgress(data.statistics);
        }
    } catch (error) {
        console.error('Error saving annotation:', error);
        alert('Failed to save annotation');
    }
}

function skipActivity() {
    if (currentIndex < activities.length - 1) {
        currentIndex++;
        displayActivity(activities[currentIndex]);
    } else {
        showCompletedMessage();
    }
}

function updateProgress(statistics) {
    if (statistics) {
        document.getElementById('labeled-count').textContent = statistics.labeled_samples || 0;
        document.getElementById('total-count').textContent = statistics.total_samples || 0;
        document.getElementById('completion-rate').textContent = 
            (statistics.completion_rate || 0).toFixed(1);
        
        const progressBar = document.getElementById('progress-bar-fill');
        progressBar.style.width = (statistics.completion_rate || 0) + '%';
    }
}

function showCompletedMessage() {
    document.getElementById('activity-card').style.display = 'none';
    document.getElementById('completed-message').style.display = 'block';
}

function loadNextBatch() {
    document.getElementById('completed-message').style.display = 'none';
    document.getElementById('loading-message').style.display = 'block';
    currentIndex = 0;
    activities = [];
    loadActivities();
}

function toggleSqlEditor() {
    const section = document.getElementById('sql-editor-section');
    const toggle = document.getElementById('sql-toggle');
    
    if (section.style.display === 'none') {
        section.style.display = 'block';
        toggle.textContent = '▲';
    } else {
        section.style.display = 'none';
        toggle.textContent = '▼';
    }
}

async function updateQuery() {
    const query = document.getElementById('sql-query').value;
    
    try {
        const response = await fetch('/api/annotate/update-query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert('Query updated successfully. Reloading activities...');
            loadActivities();
        } else {
            alert('Error updating query: ' + data.message);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to update query');
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Number keys 1-8 for quick category selection
    if (e.key >= '1' && e.key <= '8') {
        const index = parseInt(e.key) - 1;
        if (index < categories.length) {
            toggleCategory(categories[index]);
        }
    }
    
    // Arrow keys for navigation
    if (e.key === 'ArrowRight' && e.ctrlKey) {
        e.preventDefault();
        saveAndNext();
    }
    
    if (e.key === 'ArrowLeft' && e.ctrlKey) {
        e.preventDefault();
        saveAndPrevious();
    }
    
    // Space to skip
    if (e.key === ' ' && e.ctrlKey) {
        e.preventDefault();
        skipActivity();
    }
});
