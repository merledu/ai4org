document.addEventListener('DOMContentLoaded', function() {
  const chatContainer = document.getElementById('chatContainer');
  const userInput = document.getElementById('userInput');
  const sendButton = document.getElementById('sendButton');
  const typingIndicator = document.getElementById('typingIndicator');

  // Auto-resize textarea
  userInput.addEventListener('input', function() {
      this.style.height = 'auto';
      this.style.height = (this.scrollHeight) + 'px';
  });

  // Function to add a message to the chat
  function addMessage(text, isUser) {
      const messageDiv = document.createElement('div');
      messageDiv.className = `message ${isUser ? 'user-message' : 'ai-message'}`;

      const messageText = document.createElement('p');
      messageText.textContent = text;
      messageDiv.appendChild(messageText);

      if (!isUser) {
          const feedbackDiv = document.createElement('div');
          feedbackDiv.className = 'feedback-buttons';

          const thumbsUp = document.createElement('button');
          thumbsUp.className = 'feedback-button';
          thumbsUp.title = 'Like';
          thumbsUp.textContent = '👍';
          thumbsUp.addEventListener('click', function() {
              thumbsUp.classList.toggle('active');
              thumbsDown.classList.remove('active');
              console.log('User liked the response');
          });

          const thumbsDown = document.createElement('button');
          thumbsDown.className = 'feedback-button';
          thumbsDown.title = 'Dislike';
          thumbsDown.textContent = '👎';
          thumbsDown.addEventListener('click', function() {
              thumbsDown.classList.toggle('active');
              thumbsUp.classList.remove('active');
              console.log('User disliked the response');
          });

          feedbackDiv.appendChild(thumbsUp);
          feedbackDiv.appendChild(thumbsDown);
          messageDiv.appendChild(feedbackDiv);
      }

      chatContainer.appendChild(messageDiv);
      chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  // Function to simulate AI response
  function getAIResponse(userMessage) {
      const responses = [
          "I understand you're asking about " + userMessage + ". Here's what I can tell you...",
          "That's an interesting question about " + userMessage + ". Let me think...",
          "I can help with " + userMessage + ". Here's some information...",
          "Regarding " + userMessage + ", here's what I found...",
          userMessage + " is a great topic. Here are my thoughts..."
      ];

      return responses[Math.floor(Math.random() * responses.length)];
  }

  // Handle send button click
  sendButton.addEventListener('click', function() {
      const message = userInput.value.trim();
      if (message) {
          addMessage(message, true);
          userInput.value = '';
          userInput.style.height = 'auto';

          // Show typing indicator
          typingIndicator.style.display = 'block';
          chatContainer.scrollTop = chatContainer.scrollHeight;

          // Simulate AI thinking and responding
          setTimeout(function() {
              typingIndicator.style.display = 'none';
              const aiResponse = getAIResponse(message);
              addMessage(aiResponse, false);
          }, 1500 + Math.random() * 2000);
      }
  });

  // Handle Enter key press (Shift+Enter for new line)
  userInput.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          sendButton.click();
      }
  });

  // Focus the input field when the page loads
  userInput.focus();
});
