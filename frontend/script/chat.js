document.addEventListener('DOMContentLoaded', function() {
    const chatContainer = document.getElementById('chatContainer');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const typingIndicator = document.getElementById('typingIndicator');

    // No PIN prompt; use a default key
    if (!localStorage.getItem('userPin')) {
        localStorage.setItem('userPin', 'default');
    }
    // Load chat history on page load
    loadChatHistory();

    // Auto-resize textarea
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    // Load chat history from localStorage
    function getHistoryKey() {
        const pin = localStorage.getItem('userPin') || 'default';
        return `chatHistory_${pin}`;
    }

    function loadChatHistory() {
        const savedHistory = localStorage.getItem(getHistoryKey());
        if (savedHistory) {
            const messages = JSON.parse(savedHistory);
            // Clear the default AI message
            chatContainer.innerHTML = '';
            // Load all saved messages
            messages.forEach(msg => {
                addMessageToDOM(msg.text, msg.type, false);
            });
        }
    }

    // Save chat history to localStorage
    function saveChatHistory() {
        const messages = [];
        const messageElements = chatContainer.querySelectorAll('.message');
        messageElements.forEach(element => {
            const text = element.querySelector('p').textContent;
            const type = element.classList.contains('user-message') ? 'user' : 'ai';
            messages.push({ text, type, timestamp: new Date().toISOString() });
        });
        localStorage.setItem(getHistoryKey(), JSON.stringify(messages));
    }

// Clear chat history
function clearChatHistory() {
    // Create a custom confirmation with buttons
    const confirmDiv = document.createElement('div');
    confirmDiv.innerHTML = `
        <div style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
                   background: linear-gradient(135deg, #2c3e50, #34495e); color: white;
                   padding: 30px; border-radius: 15px; box-shadow: 0 20px 40px rgba(0,0,0,0.5);
                   z-index: 10001; text-align: center; max-width: 400px;">
            <h3 style="margin-bottom: 20px;">Clear Chat History</h3>
            <p style="margin-bottom: 25px; color: rgba(255,255,255,0.8);">Are you sure you want to clear all chat history? This action cannot be undone.</p>
            <div style="display: flex; gap: 15px; justify-content: center;">
                <button onclick="confirmClearHistory()" style="background: #e74c3c; color: white; border: none;
                             padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: 600;">
                    Yes, Clear All
                </button>
                <button onclick="cancelClearHistory()" style="background: #3b82f6; color: white; border: none;
                             padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: 600;">
                    Cancel
                </button>
            </div>
        </div>
    `;
    document.body.appendChild(confirmDiv);

    window.confirmClearHistory = function() {
        localStorage.removeItem(getHistoryKey());
        chatContainer.innerHTML = '';
        addMessageToDOM("Hello! I'm your personal AI assistant. How can I help you today?", 'ai', false);
        // Close the history modal after clearing
        closeHistoryModal();
        document.body.removeChild(confirmDiv);
        showNotification('Chat history cleared successfully!', 'success');
    };

    window.cancelClearHistory = function() {
        document.body.removeChild(confirmDiv);
    };
}

    // Send message function
    function sendMessage() {
        const message = userInput.value.trim();
        if (message) {
            // Add user message
            addMessage(message, 'user');
            userInput.value = '';
            userInput.style.height = 'auto';

            // Show typing indicator
            typingIndicator.style.display = 'block';

            // Simulate AI response after a delay
            // Call Python backend using pywebview
window.pywebview.api.get_ai_response(message)
.then(response => {
typingIndicator.style.display = 'none';
if (response.status === "ok") {
    addMessage(response.answer, 'ai');
} else {
    addMessage("⚠️ Error: " + response.message, 'ai');
}
// Save history after AI response
saveChatHistory();
})
.catch(err => {
typingIndicator.style.display = 'none';
addMessage("⚠️ Failed to reach backend: " + err, 'ai');
// Save history even if there's an error
saveChatHistory();
});

        }
    }

    // Add message to chat (with save option)
    function addMessage(text, type, saveToHistory = true) {
        addMessageToDOM(text, type, saveToHistory);
    }

    // Add message to DOM
    function addMessageToDOM(text, type, saveToHistory = true) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        messageDiv.innerHTML = `
            <p>${text}</p>
            ${type === 'ai' ? `
            <div class="feedback-buttons">
                <button class="feedback-button" title="Like">👍</button>
                <button class="feedback-button" title="Dislike">👎</button>
            </div>
            ` : ''}
        `;
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;

        // Save to history if requested
        if (saveToHistory) {
            saveChatHistory();
        }
    }

    // Event listeners
    sendButton.addEventListener('click', sendMessage);

    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Add navigation functionality
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', function() {
            navItems.forEach(i => i.classList.remove('active'));
            this.classList.add('active');
        });
    });

    // Add clear history button functionality
    window.clearChatHistory = clearChatHistory;

    // Add logout functionality
    window.logout = logout;

    // Add history modal functionality
    window.showChatHistory = showChatHistory;
    window.closeHistoryModal = closeHistoryModal;

    // Add settings functionality
    window.showSettings = showSettings;
    window.closeSettingsModal = closeSettingsModal;
    window.updateFontSize = updateFontSize;
    window.saveSettings = saveSettings;
    window.resetSettings = resetSettings;

    // Add help center functionality
    window.showHelpCenter = showHelpCenter;
    window.closeHelpModal = closeHelpModal;
    window.submitFeedback = submitFeedback;

    // Add notification functionality
    window.showNotification = showNotification;

    // Load saved settings on page load
    loadSettings();
});

// Show chat history in modal
function showChatHistory() {
    const modal = document.getElementById('historyModal');
    const historyContent = document.getElementById('historyContent');

    const savedHistory = localStorage.getItem('chatHistory');
    if (savedHistory) {
        const messages = JSON.parse(savedHistory);
        let historyHTML = '';

        messages.forEach(msg => {
            const timestamp = new Date(msg.timestamp).toLocaleString();
            historyHTML += `
                <div class="history-message ${msg.type}">
                    <div>${msg.text}</div>
                    <div class="history-timestamp">${timestamp}</div>
                </div>
            `;
        });

        historyContent.innerHTML = historyHTML;
    } else {
        historyContent.innerHTML = '<div class="no-history">No chat history found.</div>';
    }

    modal.style.display = 'flex';
}

// Close history modal
function closeHistoryModal() {
    document.getElementById('historyModal').style.display = 'none';
}

// Logout function
function logout() {
    // Create a custom confirmation with buttons
    const confirmDiv = document.createElement('div');
    confirmDiv.innerHTML = `
        <div style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
                   background: linear-gradient(135deg, #2c3e50, #34495e); color: white;
                   padding: 30px; border-radius: 15px; box-shadow: 0 20px 40px rgba(0,0,0,0.5);
                   z-index: 10001; text-align: center; max-width: 400px;">
            <h3 style="margin-bottom: 20px;">Logout</h3>
            <p style="margin-bottom: 25px; color: rgba(255,255,255,0.8);">Are you sure you want to logout?</p>
            <div style="display: flex; gap: 15px; justify-content: center;">
                <button onclick="confirmLogout()" style="background: #e74c3c; color: white; border: none;
                             padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: 600;">
                    Yes, Logout
                </button>
                <button onclick="cancelLogout()" style="background: #3b82f6; color: white; border: none;
                             padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: 600;">
                    Cancel
                </button>
            </div>
        </div>
    `;
    document.body.appendChild(confirmDiv);

    window.confirmLogout = function() {
        // Clear any session data if needed
        // Redirect to upload page
        document.body.removeChild(confirmDiv);
        window.location.href = 'admin.html';
    };

    window.cancelLogout = function() {
        document.body.removeChild(confirmDiv);
    };
}

// Close modal when clicking outside
window.onclick = function(event) {
    const historyModal = document.getElementById('historyModal');
    const settingsModal = document.getElementById('settingsModal');
    const helpModal = document.getElementById('helpModal');
    if (event.target === historyModal) {
        closeHistoryModal();
    }
    if (event.target === settingsModal) {
        closeSettingsModal();
    }
    if (event.target === helpModal) {
        closeHelpModal();
    }
}

// Settings Modal Functions
function showSettings() {
    document.getElementById('settingsModal').style.display = 'flex';
    loadSettings(); // Load current settings into the modal
}

function closeSettingsModal() {
    document.getElementById('settingsModal').style.display = 'none';
}

// Font size update function
function updateFontSize(value) {
    document.getElementById('fontSizeValue').textContent = value;
    document.body.style.fontSize = value + 'px';
}

// Load settings from localStorage
function loadSettings() {
    const savedTheme = localStorage.getItem('chatTheme') || 'dark';
    const savedFontSize = localStorage.getItem('chatFontSize') || '16';

    // Apply theme
    document.body.className = 'theme-' + savedTheme;

    // Apply font size
    document.body.style.fontSize = savedFontSize + 'px';
    document.getElementById('fontSizeSlider').value = savedFontSize;
    document.getElementById('fontSizeValue').textContent = savedFontSize;

    // Set radio button
    document.querySelector(`input[name="theme"][value="${savedTheme}"]`).checked = true;
}

// Save settings to localStorage
function saveSettings() {
    const selectedTheme = document.querySelector('input[name="theme"]:checked').value;
    const fontSize = document.getElementById('fontSizeSlider').value;

    // Save to localStorage
    localStorage.setItem('chatTheme', selectedTheme);
    localStorage.setItem('chatFontSize', fontSize);

    // Apply settings immediately
    document.body.className = 'theme-' + selectedTheme;
    document.body.style.fontSize = fontSize + 'px';

    // Show success message
    showNotification('Settings saved successfully!', 'success');
    closeSettingsModal();
}

// Reset settings to default
function resetSettings() {
    showNotification('Are you sure you want to reset all settings to default?', 'warning', 5000);

    // Create a custom confirmation with buttons
    const confirmDiv = document.createElement('div');
    confirmDiv.innerHTML = `
        <div style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
                   background: linear-gradient(135deg, #2c3e50, #34495e); color: white;
                   padding: 30px; border-radius: 15px; box-shadow: 0 20px 40px rgba(0,0,0,0.5);
                   z-index: 10001; text-align: center; max-width: 400px;">
            <h3 style="margin-bottom: 20px;">Reset Settings</h3>
            <p style="margin-bottom: 25px; color: rgba(255,255,255,0.8);">Are you sure you want to reset all settings to default?</p>
            <div style="display: flex; gap: 15px; justify-content: center;">
                <button onclick="confirmReset()" style="background: #e74c3c; color: white; border: none;
                             padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: 600;">
                    Yes, Reset
                </button>
                <button onclick="cancelReset()" style="background: #3b82f6; color: white; border: none;
                             padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: 600;">
                    Cancel
                </button>
            </div>
        </div>
    `;
    document.body.appendChild(confirmDiv);

    window.confirmReset = function() {
        localStorage.removeItem('chatTheme');
        localStorage.removeItem('chatFontSize');

        // Reset to default values
        document.body.className = '';
        document.body.style.fontSize = '16px';
        document.getElementById('fontSizeSlider').value = '16';
        document.getElementById('fontSizeValue').textContent = '16';
        document.querySelector('input[name="theme"][value="dark"]').checked = true;

        showNotification('Settings reset to default!', 'success');
        document.body.removeChild(confirmDiv);
        closeSettingsModal();
    };

    window.cancelReset = function() {
        document.body.removeChild(confirmDiv);
    };
}

// Help Center Functions
function showHelpCenter() {
    document.getElementById('helpModal').style.display = 'flex';
}

function closeHelpModal() {
    document.getElementById('helpModal').style.display = 'none';
}

// Notification system
function showNotification(message, type = 'info', duration = 3000) {
    const notification = document.getElementById('notification');
    const icon = notification.querySelector('.notification-icon');
    const messageEl = notification.querySelector('.notification-message');

    // Set message
    messageEl.textContent = message;

    // Set icon and type
    notification.className = `notification ${type}`;
    switch(type) {
        case 'success':
            icon.className = 'notification-icon fas fa-check-circle';
            break;
        case 'error':
            icon.className = 'notification-icon fas fa-exclamation-circle';
            break;
        case 'warning':
            icon.className = 'notification-icon fas fa-exclamation-triangle';
            break;
        default:
            icon.className = 'notification-icon fas fa-info-circle';
    }

    // Show notification
    notification.style.display = 'block';
    setTimeout(() => notification.classList.add('show'), 100);

    // Auto hide
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.style.display = 'none', 300);
    }, duration);
}

// Submit feedback function
function submitFeedback() {
    const feedbackText = document.getElementById('feedbackText').value.trim();

    if (!feedbackText) {
        showNotification('Please enter your feedback before submitting.', 'warning');
        return;
    }

    // Here you would typically send the feedback to your backend
    // For now, we'll just show a success message and save to localStorage
    const feedback = {
        text: feedbackText,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent
    };

    // Save feedback to localStorage (in a real app, this would go to a server)
    const existingFeedback = JSON.parse(localStorage.getItem('userFeedback') || '[]');
    existingFeedback.push(feedback);
    localStorage.setItem('userFeedback', JSON.stringify(existingFeedback)); document.addEventListener('DOMContentLoaded', function() {
            const chatContainer = document.getElementById('chatContainer');
            const userInput = document.getElementById('userInput');
            const sendButton = document.getElementById('sendButton');
            const typingIndicator = document.getElementById('typingIndicator');

            // Load chat history on page load
            loadChatHistory();

            // Auto-resize textarea
            userInput.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = (this.scrollHeight) + 'px';
            });

            // Load chat history from localStorage
            function loadChatHistory() {
                const savedHistory = localStorage.getItem('chatHistory');
                if (savedHistory) {
                    const messages = JSON.parse(savedHistory);
                    // Clear the default AI message
                    chatContainer.innerHTML = '';
                    // Load all saved messages
                    messages.forEach(msg => {
                        addMessageToDOM(msg.text, msg.type, false);
                    });
                }
            }

            // Save chat history to localStorage
            function saveChatHistory() {
                const messages = [];
                const messageElements = chatContainer.querySelectorAll('.message');
                messageElements.forEach(element => {
                    const text = element.querySelector('p').textContent;
                    const type = element.classList.contains('user-message') ? 'user' : 'ai';
                    messages.push({ text, type, timestamp: new Date().toISOString() });
                });
                localStorage.setItem('chatHistory', JSON.stringify(messages));
            }

        // Clear chat history
        function clearChatHistory() {
            // Create a custom confirmation with buttons
            const confirmDiv = document.createElement('div');
            confirmDiv.innerHTML = `
                <div style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
                           background: linear-gradient(135deg, #2c3e50, #34495e); color: white;
                           padding: 30px; border-radius: 15px; box-shadow: 0 20px 40px rgba(0,0,0,0.5);
                           z-index: 10001; text-align: center; max-width: 400px;">
                    <h3 style="margin-bottom: 20px;">Clear Chat History</h3>
                    <p style="margin-bottom: 25px; color: rgba(255,255,255,0.8);">Are you sure you want to clear all chat history? This action cannot be undone.</p>
                    <div style="display: flex; gap: 15px; justify-content: center;">
                        <button onclick="confirmClearHistory()" style="background: #e74c3c; color: white; border: none;
                                     padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: 600;">
                            Yes, Clear All
                        </button>
                        <button onclick="cancelClearHistory()" style="background: #3b82f6; color: white; border: none;
                                     padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: 600;">
                            Cancel
                        </button>
                    </div>
                </div>
            `;
            document.body.appendChild(confirmDiv);

            window.confirmClearHistory = function() {
                localStorage.removeItem('chatHistory');
                chatContainer.innerHTML = '';
                addMessageToDOM("Hello! I'm your personal AI assistant. How can I help you today?", 'ai', false);
                // Close the history modal after clearing
                closeHistoryModal();
                document.body.removeChild(confirmDiv);
                showNotification('Chat history cleared successfully!', 'success');
            };

            window.cancelClearHistory = function() {
                document.body.removeChild(confirmDiv);
            };
        }

            // Send message function
            function sendMessage() {
                const message = userInput.value.trim();
                if (message) {
                    // Add user message
                    addMessage(message, 'user');
                    userInput.value = '';
                    userInput.style.height = 'auto';

                    // Show typing indicator
                    typingIndicator.style.display = 'block';

                    // Simulate AI response after a delay
                    // Call Python backend using pywebview
window.pywebview.api.get_ai_response(message)
.then(response => {
        typingIndicator.style.display = 'none';
        if (response.status === "ok") {
            addMessage(response.answer, 'ai');
        } else {
            addMessage("⚠️ Error: " + response.message, 'ai');
        }
        // Save history after AI response
        saveChatHistory();
    })
    .catch(err => {
        typingIndicator.style.display = 'none';
        addMessage("⚠️ Failed to reach backend: " + err, 'ai');
        // Save history even if there's an error
        saveChatHistory();
    });

                }
            }

            // Add message to chat (with save option)
            function addMessage(text, type, saveToHistory = true) {
                addMessageToDOM(text, type, saveToHistory);
            }

            // Add message to DOM
            function addMessageToDOM(text, type, saveToHistory = true) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${type}-message`;
                messageDiv.innerHTML = `
                    <p>${text}</p>
                    ${type === 'ai' ? `
                    <div class="feedback-buttons">
                        <button class="feedback-button" title="Like">👍</button>
                        <button class="feedback-button" title="Dislike">👎</button>
                    </div>
                    ` : ''}
                `;
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;

                // Save to history if requested
                if (saveToHistory) {
                    saveChatHistory();
                }
            }

            // Event listeners
            sendButton.addEventListener('click', sendMessage);

            userInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });

            // Add navigation functionality
            const navItems = document.querySelectorAll('.nav-item');
            navItems.forEach(item => {
                item.addEventListener('click', function() {
                    navItems.forEach(i => i.classList.remove('active'));
                    this.classList.add('active');
                });
            });

            // Add clear history button functionality
            window.clearChatHistory = clearChatHistory;

            // Add logout functionality
            window.logout = logout;

            // Add history modal functionality
            window.showChatHistory = showChatHistory;
            window.closeHistoryModal = closeHistoryModal;

            // Add settings functionality
            window.showSettings = showSettings;
            window.closeSettingsModal = closeSettingsModal;
            window.updateFontSize = updateFontSize;
            window.saveSettings = saveSettings;
            window.resetSettings = resetSettings;

            // Add help center functionality
            window.showHelpCenter = showHelpCenter;
            window.closeHelpModal = closeHelpModal;
            window.submitFeedback = submitFeedback;

            // Add notification functionality
            window.showNotification = showNotification;

            // Load saved settings on page load
            loadSettings();
        });

        // Show chat history in modal
        function showChatHistory() {
            const modal = document.getElementById('historyModal');
            const historyContent = document.getElementById('historyContent');

            const savedHistory = localStorage.getItem('chatHistory');
            if (savedHistory) {
                const messages = JSON.parse(savedHistory);
                let historyHTML = '';

                messages.forEach(msg => {
                    const timestamp = new Date(msg.timestamp).toLocaleString();
                    historyHTML += `
                        <div class="history-message ${msg.type}">
                            <div>${msg.text}</div>
                            <div class="history-timestamp">${timestamp}</div>
                        </div>
                    `;
                });

                historyContent.innerHTML = historyHTML;
            } else {
                historyContent.innerHTML = '<div class="no-history">No chat history found.</div>';
            }

            modal.style.display = 'flex';
        }

        // Close history modal
        function closeHistoryModal() {
            document.getElementById('historyModal').style.display = 'none';
        }

        // Logout function
        function logout() {
            // Create a custom confirmation with buttons
            const confirmDiv = document.createElement('div');
            confirmDiv.innerHTML = `
                <div style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
                           background: linear-gradient(135deg, #2c3e50, #34495e); color: white;
                           padding: 30px; border-radius: 15px; box-shadow: 0 20px 40px rgba(0,0,0,0.5);
                           z-index: 10001; text-align: center; max-width: 400px;">
                    <h3 style="margin-bottom: 20px;">Logout</h3>
                    <p style="margin-bottom: 25px; color: rgba(255,255,255,0.8);">Are you sure you want to logout?</p>
                    <div style="display: flex; gap: 15px; justify-content: center;">
                        <button onclick="confirmLogout()" style="background: #e74c3c; color: white; border: none;
                                     padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: 600;">
                            Yes, Logout
                        </button>
                        <button onclick="cancelLogout()" style="background: #3b82f6; color: white; border: none;
                                     padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: 600;">
                            Cancel
                        </button>
                    </div>
                </div>
            `;
            document.body.appendChild(confirmDiv);

            window.confirmLogout = function() {
                // Clear any session data if needed
                // Redirect to upload page
                document.body.removeChild(confirmDiv);
                window.location.href = 'admin.html';
            };

            window.cancelLogout = function() {
                document.body.removeChild(confirmDiv);
            };
        }

        // Close modal when clicking outside
        window.onclick = function(event) {
            const historyModal = document.getElementById('historyModal');
            const settingsModal = document.getElementById('settingsModal');
            const helpModal = document.getElementById('helpModal');
            if (event.target === historyModal) {
                closeHistoryModal();
            }
            if (event.target === settingsModal) {
                closeSettingsModal();
            }
            if (event.target === helpModal) {
                closeHelpModal();
            }
        }

        // Settings Modal Functions
        function showSettings() {
            document.getElementById('settingsModal').style.display = 'flex';
            loadSettings(); // Load current settings into the modal
        }

        function closeSettingsModal() {
            document.getElementById('settingsModal').style.display = 'none';
        }

        // Font size update function
        function updateFontSize(value) {
            document.getElementById('fontSizeValue').textContent = value;
            document.body.style.fontSize = value + 'px';
        }

        // Load settings from localStorage
        function loadSettings() {
            const savedTheme = localStorage.getItem('chatTheme') || 'dark';
            const savedFontSize = localStorage.getItem('chatFontSize') || '16';

            // Apply theme
            document.body.className = 'theme-' + savedTheme;

            // Apply font size
            document.body.style.fontSize = savedFontSize + 'px';
            document.getElementById('fontSizeSlider').value = savedFontSize;
            document.getElementById('fontSizeValue').textContent = savedFontSize;

            // Set radio button
            document.querySelector(`input[name="theme"][value="${savedTheme}"]`).checked = true;
        }

        // Save settings to localStorage
        function saveSettings() {
            const selectedTheme = document.querySelector('input[name="theme"]:checked').value;
            const fontSize = document.getElementById('fontSizeSlider').value;

            // Save to localStorage
            localStorage.setItem('chatTheme', selectedTheme);
            localStorage.setItem('chatFontSize', fontSize);

            // Apply settings immediately
            document.body.className = 'theme-' + selectedTheme;
            document.body.style.fontSize = fontSize + 'px';

            // Show success message
            showNotification('Settings saved successfully!', 'success');
            closeSettingsModal();
        }

        // Reset settings to default
        function resetSettings() {
            showNotification('Are you sure you want to reset all settings to default?', 'warning', 5000);

            // Create a custom confirmation with buttons
            const confirmDiv = document.createElement('div');
            confirmDiv.innerHTML = `
                <div style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
                           background: linear-gradient(135deg, #2c3e50, #34495e); color: white;
                           padding: 30px; border-radius: 15px; box-shadow: 0 20px 40px rgba(0,0,0,0.5);
                           z-index: 10001; text-align: center; max-width: 400px;">
                    <h3 style="margin-bottom: 20px;">Reset Settings</h3>
                    <p style="margin-bottom: 25px; color: rgba(255,255,255,0.8);">Are you sure you want to reset all settings to default?</p>
                    <div style="display: flex; gap: 15px; justify-content: center;">
                        <button onclick="confirmReset()" style="background: #e74c3c; color: white; border: none;
                                     padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: 600;">
                            Yes, Reset
                        </button>
                        <button onclick="cancelReset()" style="background: #3b82f6; color: white; border: none;
                                     padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: 600;">
                            Cancel
                        </button>
                    </div>
                </div>
            `;
            document.body.appendChild(confirmDiv);

            window.confirmReset = function() {
                localStorage.removeItem('chatTheme');
                localStorage.removeItem('chatFontSize');

                // Reset to default values
                document.body.className = '';
                document.body.style.fontSize = '16px';
                document.getElementById('fontSizeSlider').value = '16';
                document.getElementById('fontSizeValue').textContent = '16';
                document.querySelector('input[name="theme"][value="dark"]').checked = true;

                showNotification('Settings reset to default!', 'success');
                document.body.removeChild(confirmDiv);
                closeSettingsModal();
            };

            window.cancelReset = function() {
                document.body.removeChild(confirmDiv);
            };
        }

        // Help Center Functions
        function showHelpCenter() {
            document.getElementById('helpModal').style.display = 'flex';
        }

        function closeHelpModal() {
            document.getElementById('helpModal').style.display = 'none';
        }

        // Notification system
        function showNotification(message, type = 'info', duration = 3000) {
            const notification = document.getElementById('notification');
            const icon = notification.querySelector('.notification-icon');
            const messageEl = notification.querySelector('.notification-message');

            // Set message
            messageEl.textContent = message;

            // Set icon and type
            notification.className = `notification ${type}`;
            switch(type) {
                case 'success':
                    icon.className = 'notification-icon fas fa-check-circle';
                    break;
                case 'error':
                    icon.className = 'notification-icon fas fa-exclamation-circle';
                    break;
                case 'warning':
                    icon.className = 'notification-icon fas fa-exclamation-triangle';
                    break;
                default:
                    icon.className = 'notification-icon fas fa-info-circle';
            }

            // Show notification
            notification.style.display = 'block';
            setTimeout(() => notification.classList.add('show'), 100);

            // Auto hide
            setTimeout(() => {
                notification.classList.remove('show');
                setTimeout(() => notification.style.display = 'none', 300);
            }, duration);
        }

        // Submit feedback function
        function submitFeedback() {
            const feedbackText = document.getElementById('feedbackText').value.trim();

            if (!feedbackText) {
                showNotification('Please enter your feedback before submitting.', 'warning');
                return;
            }

            // Here you would typically send the feedback to your backend
            // For now, we'll just show a success message and save to localStorage
            const feedback = {
                text: feedbackText,
                timestamp: new Date().toISOString(),
                userAgent: navigator.userAgent
            };

            // Save feedback to localStorage (in a real app, this would go to a server)
            const existingFeedback = JSON.parse(localStorage.getItem('userFeedback') || '[]');
            existingFeedback.push(feedback);
            localStorage.setItem('userFeedback', JSON.stringify(existingFeedback));

            // Show success message
            showNotification('Thank you for your feedback! We appreciate your input.', 'success');

            // Clear the form
            document.getElementById('feedbackText').value = '';

            // Close the modal
            closeHelpModal();
        }

    // Show success message
    showNotification('Thank you for your feedback! We appreciate your input.', 'success');

    // Clear the form
    document.getElementById('feedbackText').value = '';

    // Close the modal
    closeHelpModal();
}
