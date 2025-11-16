document.addEventListener('DOMContentLoaded', () => {
  // Create floating circular button
  const floatButton = document.createElement('button');
  floatButton.id = 'chatbot-float-button';
  floatButton.textContent = '💬';
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

  // Create header with close button and memory selector
  const header = document.createElement('div');
  header.style.backgroundColor = '#4f46e5'; // Indigo-600
  header.style.color = 'white';
  header.style.padding = '10px';
  header.style.display = 'flex';
  header.style.flexDirection = 'column';

  const headerRow = document.createElement('div');
  headerRow.style.display = 'flex';
  headerRow.style.justifyContent = 'space-between';
  headerRow.style.alignItems = 'center';
  headerRow.style.marginBottom = '8px';

  const title = document.createElement('div');
  title.textContent = 'AI Assistant (RAG)';
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

  headerRow.appendChild(title);
  headerRow.appendChild(closeButton);

  // Memory type selector
  const memoryRow = document.createElement('div');
  memoryRow.style.display = 'flex';
  memoryRow.style.alignItems = 'center';
  memoryRow.style.gap = '8px';

  const memoryLabel = document.createElement('label');
  memoryLabel.textContent = 'Memory:';
  memoryLabel.style.fontSize = '12px';
  memoryLabel.style.whiteSpace = 'nowrap';

  const memorySelect = document.createElement('select');
  memorySelect.id = 'memory-type-select';
  memorySelect.style.flex = '1';
  memorySelect.style.padding = '4px 8px';
  memorySelect.style.borderRadius = '4px';
  memorySelect.style.border = 'none';
  memorySelect.style.fontSize = '12px';
  memorySelect.style.backgroundColor = 'white';
  memorySelect.style.color = '#4f46e5';
  memorySelect.style.cursor = 'pointer';

  const memoryOptions = [
    { value: 'summary_buffer', label: 'Summary Buffer (Recommended)' },
    { value: 'buffer', label: 'Recent Messages Only' },
    { value: 'summary', label: 'Summary Only' },
    { value: 'graph', label: 'Knowledge Graph' }
  ];

  memoryOptions.forEach(opt => {
    const option = document.createElement('option');
    option.value = opt.value;
    option.textContent = opt.label;
    if (opt.value === 'summary_buffer') {
      option.selected = true;
    }
    memorySelect.appendChild(option);
  });

  memoryRow.appendChild(memoryLabel);
  memoryRow.appendChild(memorySelect);

  header.appendChild(headerRow);
  header.appendChild(memoryRow);
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

  // Add message to chat window with alignment and bubble style
  function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.style.marginBottom = '10px';
    messageDiv.style.padding = '8px 12px';
    messageDiv.style.borderRadius = '15px';
    messageDiv.style.maxWidth = '80%';
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

    messageDiv.textContent = text;
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  // Send message to backend
  async function sendMessage() {
    const message = inputField.value.trim();
    if (!message) return;

    const memoryType = document.getElementById('memory-type-select').value;

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
        body: JSON.stringify({ 
          message: message,
          memory_type: memoryType 
        }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      // Delay hiding typing indicator to test visibility
      setTimeout(() => {
        hideTypingIndicator();
        if (data.error) {
          addMessage('Error: ' + data.error, 'bot');
        } else {
          addMessage(data.response, 'bot');
        }
      }, 1000);
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
