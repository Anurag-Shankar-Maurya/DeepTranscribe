document.addEventListener('DOMContentLoaded', () => {
  // Create floating circular button
  const floatButton = document.createElement('button');
  floatButton.id = 'chatbot-float-button';
  floatButton.textContent = 'Ⓜ️';
  floatButton.title = 'Open Chatbot';
  floatButton.style.position = 'fixed';
  floatButton.style.bottom = '20px';
  floatButton.style.right = '20px';
  floatButton.style.width = '50px';
  floatButton.style.height = '50px';
  floatButton.style.borderRadius = '50%';
  floatButton.style.border = 'none';
  floatButton.style.backgroundColor = '#4f46e5'; // Indigo-600
  floatButton.style.color = 'white';
  floatButton.style.fontSize = '24px';
  floatButton.style.cursor = 'pointer';
  floatButton.style.zIndex = '10000';
  floatButton.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';

  // Create chatbot container (hidden initially)
  const chatbotContainer = document.createElement('div');
  chatbotContainer.id = 'chatbot-container';
  chatbotContainer.style.position = 'fixed';
  chatbotContainer.style.bottom = '80px';
  chatbotContainer.style.right = '20px';
  chatbotContainer.style.width = '350px';
  chatbotContainer.style.height = '500px'; // fixed height
  chatbotContainer.style.backgroundColor = 'white';
  chatbotContainer.style.border = '1px solid #ccc';
  chatbotContainer.style.borderRadius = '10px';
  chatbotContainer.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';
  chatbotContainer.style.display = 'flex';
  chatbotContainer.style.flexDirection = 'column';
  chatbotContainer.style.fontFamily = 'Arial, sans-serif';
  chatbotContainer.style.zIndex = '10000';
  chatbotContainer.style.overflow = 'hidden';
  chatbotContainer.style.transition = 'transform 0.3s ease-in-out';
  chatbotContainer.style.display = 'none'; // hidden initially

  // Create header with close button
  const header = document.createElement('div');
  header.style.backgroundColor = '#4f46e5'; // Indigo-600
  header.style.color = 'white';
  header.style.padding = '10px';
  header.style.display = 'flex';
  header.style.justifyContent = 'space-between';
  header.style.alignItems = 'center';

  const title = document.createElement('div');
  title.textContent = 'Chatbot';
  title.style.fontWeight = 'bold';
  title.style.fontSize = '16px';

  const closeButton = document.createElement('button');
  closeButton.textContent = '✕';
  closeButton.title = 'Close Chatbot';
  closeButton.style.background = 'none';
  closeButton.style.border = 'none';
  closeButton.style.color = 'white';
  closeButton.style.fontSize = '20px';
  closeButton.style.cursor = 'pointer';

  header.appendChild(title);
  header.appendChild(closeButton);
  chatbotContainer.appendChild(header);

  // Create messages container
  const messagesContainer = document.createElement('div');
  messagesContainer.id = 'chatbot-messages';
  messagesContainer.style.flex = '1';
  messagesContainer.style.padding = '10px';
  messagesContainer.style.overflowY = 'auto';
  messagesContainer.style.backgroundColor = '#f9fafb'; // Gray-50
  messagesContainer.style.display = 'flex';
  messagesContainer.style.flexDirection = 'column';
  messagesContainer.style.gap = '10px';
  chatbotContainer.appendChild(messagesContainer);

  // Create input container
  const inputContainer = document.createElement('div');
  inputContainer.style.display = 'flex';
  inputContainer.style.borderTop = '1px solid #ccc';

  const inputField = document.createElement('input');
  inputField.type = 'text';
  inputField.placeholder = 'Type your message...';
  inputField.style.flex = '1';
  inputField.style.border = 'none';
  inputField.style.padding = '10px';
  inputField.style.fontSize = '14px';
  inputField.style.outline = 'none';

  const sendButton = document.createElement('button');
  sendButton.textContent = 'Send';
  sendButton.style.backgroundColor = '#4f46e5'; // Indigo-600
  sendButton.style.color = 'white';
  sendButton.style.border = 'none';
  sendButton.style.padding = '10px 15px';
  sendButton.style.cursor = 'pointer';
  sendButton.style.fontSize = '14px';

  inputContainer.appendChild(inputField);
  inputContainer.appendChild(sendButton);
  chatbotContainer.appendChild(inputContainer);

  document.body.appendChild(floatButton);
  document.body.appendChild(chatbotContainer);

  // Show chatbot container and hide float button
  function openChatbot() {
    floatButton.style.display = 'none';
    chatbotContainer.style.display = 'flex';
    inputField.focus();
    addMessage("Hello! How can I assist you today?", 'bot');
  }

  // Hide chatbot container and show float button
  function closeChatbot() {
    chatbotContainer.style.display = 'none';
    floatButton.style.display = 'block';
    clearMessages();
  }

  // Clear messages container
  function clearMessages() {
    messagesContainer.innerHTML = '';
  }

  // Basic Markdown parser
  function parseMarkdown(text) {
    // Bold: **text**
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Italic: *text*
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // Code inline: `code`
    text = text.replace(/`(.*?)`/g, '<code>$1</code>');
    // Code block: ```code```
    text = text.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    // Links: [text](url)
    text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    // Headers: # ## ###
    text = text.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    text = text.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    text = text.replace(/^# (.*$)/gim, '<h1>$1</h1>');
    // Lists: - item or * item
    text = text.replace(/^\- (.*$)/gim, '<li>$1</li>');
    text = text.replace(/^\* (.*$)/gim, '<li>$1</li>');
    // Wrap consecutive <li> in <ul>
    text = text.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');
    return text;
  }

  // Add message to chat window with alignment and bubble style
  function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.style.marginBottom = '10px';
    messageDiv.style.padding = '8px 12px';
    messageDiv.style.borderRadius = '15px';
    messageDiv.style.maxWidth = '100%';
    messageDiv.style.fontSize = '14px';
    messageDiv.style.wordWrap = 'break-word';
    messageDiv.style.whiteSpace = 'pre-wrap';
    messageDiv.style.display = 'inline-block';

    if (sender === 'user') {
      messageDiv.style.backgroundColor = '#4f46e5'; // Indigo-600
      messageDiv.style.color = 'white';
      messageDiv.style.alignSelf = 'flex-end';
      messageDiv.style.textAlign = 'right';
      messageDiv.style.borderBottomRightRadius = '0';
    } else {
      messageDiv.style.backgroundColor = '#e0e7ff'; // Indigo-100
      messageDiv.style.color = '#1e293b'; // Gray-800
      messageDiv.style.alignSelf = 'flex-start';
      messageDiv.style.textAlign = 'left';
      messageDiv.style.borderBottomLeftRadius = '0';
    }

    messageDiv.innerHTML = parseMarkdown(text);
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  // Send message to backend
  async function sendMessage() {
    const message = inputField.value.trim();
    if (!message) return;

    addMessage(message, 'user');
    showTypingIndicator();
    inputField.value = '';
    inputField.disabled = true;
    sendButton.disabled = true;

    try {
      const response = await fetch('/api/chatbot/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCookie('csrftoken'),
        },
        body: JSON.stringify({ message }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      hideTypingIndicator();
      if (data.error) {
        addMessage('Error: ' + data.error, 'bot');
      } else {
        addMessage(data.response, 'bot');
      }
    } catch (error) {
      hideTypingIndicator();
      addMessage('Error: ' + error.message, 'bot');
    } finally {
      inputField.disabled = false;
      sendButton.disabled = false;
      inputField.focus();
    }
  }

  // Event listeners
  floatButton.addEventListener('click', openChatbot);
  closeButton.addEventListener('click', closeChatbot);
  sendButton.addEventListener('click', sendMessage);
  inputField.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      sendMessage();
    }
  });

  // Helper function to get CSRF token cookie
  function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
      const cookies = document.cookie.split(';');
      for (let i = 0; i < cookies.length; i++) {
        const cookie = cookies[i].trim();
        // Does this cookie string begin with the name we want?
        if (cookie.substring(0, name.length + 1) === (name + '=')) {
          cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
          break;
        }
      }
    }
    return cookieValue;
  }

  // Show typing indicator
  function showTypingIndicator() {
    console.log('showTypingIndicator called');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.id = 'typingIndicator';
    typingDiv.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    `;
    messagesContainer.appendChild(typingDiv);
    scrollToBottom();
  }

  // Hide typing indicator
  function hideTypingIndicator() {
    console.log('hideTypingIndicator called');
    const typingDiv = document.getElementById('typingIndicator');
    if (typingDiv) {
      typingDiv.remove();
      scrollToBottom();
    }
  }

  // Scroll to bottom helper
  function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }
});
