document.addEventListener('DOMContentLoaded', () => {
  // Create floating circular button
  const floatButton = document.createElement('button');
  floatButton.id = 'chatbot-float-button';
  floatButton.innerHTML = '💬';
  floatButton.title = 'Open AI Assistant';
  floatButton.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 60px;
    height: 60px;
    border-radius: 50%;
    border: none;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    font-size: 24px;
    cursor: pointer;
    z-index: 10000;
    box-shadow: 0 4px 20px rgba(79, 70, 229, 0.3);
    transition: all 0.3s ease;
  `;

  // Add hover effect
  floatButton.addEventListener('mouseenter', () => {
    floatButton.style.transform = 'scale(1.1)';
    floatButton.style.boxShadow = '0 6px 25px rgba(79, 70, 229, 0.4)';
  });

  floatButton.addEventListener('mouseleave', () => {
    floatButton.style.transform = 'scale(1)';
    floatButton.style.boxShadow = '0 4px 20px rgba(79, 70, 229, 0.3)';
  });

  // Create chatbot container (hidden initially)
  const chatbotContainer = document.createElement('div');
  chatbotContainer.id = 'chatbot-container';
  chatbotContainer.style.cssText = `
    position: fixed;
    bottom: 90px;
    right: 20px;
    width: 380px;
    height: 550px;
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    display: none;
    flex-direction: column;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    z-index: 10000;
    overflow: hidden;
    transition: all 0.3s ease;
  `;

  // Create header with close button
  const header = document.createElement('div');
  header.style.cssText = `
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    padding: 16px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-radius: 16px 16px 0 0;
  `;

  const title = document.createElement('div');
  title.innerHTML = `
    <div style="font-weight: 600; font-size: 16px;">AI Assistant</div>
    <div style="font-size: 12px; opacity: 0.9;">Ask about your transcripts</div>
  `;

  const closeButton = document.createElement('button');
  closeButton.innerHTML = '✕';
  closeButton.title = 'Close Assistant';
  closeButton.style.cssText = `
    background: none;
    border: none;
    color: white;
    font-size: 18px;
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
    transition: background-color 0.2s;
  `;

  closeButton.addEventListener('mouseenter', () => {
    closeButton.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
  });

  closeButton.addEventListener('mouseleave', () => {
    closeButton.style.backgroundColor = 'transparent';
  });

  header.appendChild(title);
  header.appendChild(closeButton);
  chatbotContainer.appendChild(header);

  // Create messages container
  const messagesContainer = document.createElement('div');
  messagesContainer.id = 'chatbot-messages';
  messagesContainer.style.cssText = `
    flex: 1;
    padding: 16px;
    overflow-y: auto;
    background: #f9fafb;
    display: flex;
    flex-direction: column;
    gap: 12px;
  `;
  chatbotContainer.appendChild(messagesContainer);

  // Create input container
  const inputContainer = document.createElement('div');
  inputContainer.style.cssText = `
    display: flex;
    border-top: 1px solid #e5e7eb;
    background: white;
    padding: 16px;
    gap: 8px;
    border-radius: 0 0 16px 16px;
  `;

  const inputField = document.createElement('textarea');
  inputField.placeholder = 'Ask about your transcripts...';
  inputField.style.cssText = `
    flex: 1;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 14px;
    outline: none;
    resize: none;
    min-height: 20px;
    max-height: 80px;
    font-family: inherit;
    transition: border-color 0.2s;
  `;

  inputField.addEventListener('focus', () => {
    inputField.style.borderColor = '#4f46e5';
  });

  inputField.addEventListener('blur', () => {
    inputField.style.borderColor = '#d1d5db';
  });

  // Auto-resize textarea
  inputField.addEventListener('input', () => {
    inputField.style.height = 'auto';
    inputField.style.height = Math.min(inputField.scrollHeight, 80) + 'px';
  });

  const sendButton = document.createElement('button');
  sendButton.innerHTML = '➤';
  sendButton.style.cssText = `
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 12px;
    cursor: pointer;
    font-size: 16px;
    transition: all 0.2s;
    min-width: 44px;
    height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
  `;

  sendButton.addEventListener('mouseenter', () => {
    sendButton.style.transform = 'scale(1.05)';
  });

  sendButton.addEventListener('mouseleave', () => {
    sendButton.style.transform = 'scale(1)';
  });

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
    
    // Add welcome message if no messages exist
    if (messagesContainer.children.length === 0) {
      addMessage("👋 Hello! I'm your AI assistant. I can help you with:\n\n• Searching your transcripts\n• Summarizing conversations\n• Analyzing speakers\n• Finding specific information\n\nWhat would you like to know?", 'bot');
    }
  }

  // Hide chatbot container and show float button
  function closeChatbot() {
    chatbotContainer.style.display = 'none';
    floatButton.style.display = 'block';
  }

  // Add message to chat window with improved styling
  function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.style.cssText = `
      display: flex;
      ${sender === 'user' ? 'justify-content: flex-end;' : 'justify-content: flex-start;'}
      margin-bottom: 8px;
    `;

    const messageBubble = document.createElement('div');
    messageBubble.style.cssText = `
      max-width: 85%;
      padding: 12px 16px;
      border-radius: 16px;
      word-wrap: break-word;
      white-space: pre-wrap;
      font-size: 14px;
      line-height: 1.4;
      ${sender === 'user' 
        ? `
          background: linear-gradient(135deg, #4f46e5, #7c3aed);
          color: white;
          border-bottom-right-radius: 4px;
          margin-left: 20px;
        ` 
        : `
          background: white;
          color: #374151;
          border: 1px solid #e5e7eb;
          border-bottom-left-radius: 4px;
          margin-right: 20px;
          box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        `
      }
    `;

    messageBubble.textContent = text;
    messageDiv.appendChild(messageBubble);
    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
  }

  // Show typing indicator with improved animation
  function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typingIndicator';
    typingDiv.style.cssText = `
      display: flex;
      justify-content: flex-start;
      margin-bottom: 8px;
    `;

    const typingBubble = document.createElement('div');
    typingBubble.style.cssText = `
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 16px;
      border-bottom-left-radius: 4px;
      padding: 12px 16px;
      margin-right: 20px;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    `;

    const dotsContainer = document.createElement('div');
    dotsContainer.style.cssText = `
      display: flex;
      gap: 4px;
      align-items: center;
    `;

    for (let i = 0; i < 3; i++) {
      const dot = document.createElement('div');
      dot.style.cssText = `
        width: 6px;
        height: 6px;
        background-color: #9ca3af;
        border-radius: 50%;
        animation: typingBounce 1.4s infinite ease-in-out both;
        animation-delay: ${i * 0.2}s;
      `;
      dotsContainer.appendChild(dot);
    }

    // Add CSS animation
    if (!document.getElementById('typing-animation-style')) {
      const style = document.createElement('style');
      style.id = 'typing-animation-style';
      style.textContent = `
        @keyframes typingBounce {
          0%, 80%, 100% {
            transform: scale(0.8);
            opacity: 0.5;
          }
          40% {
            transform: scale(1.2);
            opacity: 1;
          }
        }
      `;
      document.head.appendChild(style);
    }

    typingBubble.appendChild(dotsContainer);
    typingDiv.appendChild(typingBubble);
    messagesContainer.appendChild(typingDiv);
    scrollToBottom();
  }

  // Hide typing indicator
  function hideTypingIndicator() {
    const typingDiv = document.getElementById('typingIndicator');
    if (typingDiv) {
      typingDiv.remove();
    }
  }

  // Send message to backend with improved error handling
  async function sendMessage() {
    const message = inputField.value.trim();
    if (!message) return;

    // Validate message length
    if (message.length > 2000) {
      addMessage('Message too long. Please limit to 2000 characters.', 'bot');
      return;
    }

    addMessage(message, 'user');
    showTypingIndicator();
    inputField.value = '';
    inputField.style.height = 'auto';
    inputField.disabled = true;
    sendButton.disabled = true;
    sendButton.style.opacity = '0.6';

    try {
      const response = await fetch('/api/chatbot/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCookie('csrftoken'),
        },
        body: JSON.stringify({ message }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `HTTP ${response.status}`);
      }

      hideTypingIndicator();
      
      if (data.error) {
        addMessage('❌ ' + data.error, 'bot');
      } else {
        addMessage(data.response, 'bot');
        
        // Show user stats if available
        if (data.user_stats) {
          const stats = data.user_stats;
          if (stats.total_transcripts > 0) {
            console.log('User stats:', stats);
          }
        }
      }
    } catch (error) {
      hideTypingIndicator();
      console.error('Chatbot error:', error);
      addMessage('❌ Sorry, I encountered an error. Please try again in a moment.', 'bot');
    } finally {
      inputField.disabled = false;
      sendButton.disabled = false;
      sendButton.style.opacity = '1';
      inputField.focus();
    }
  }

  // Event listeners
  floatButton.addEventListener('click', openChatbot);
  closeButton.addEventListener('click', closeChatbot);
  sendButton.addEventListener('click', sendMessage);
  
  inputField.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
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
        if (cookie.substring(0, name.length + 1) === (name + '=')) {
          cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
          break;
        }
      }
    }
    return cookieValue;
  }

  // Scroll to bottom helper
  function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  // Close chatbot when clicking outside
  document.addEventListener('click', (e) => {
    if (!chatbotContainer.contains(e.target) && !floatButton.contains(e.target)) {
      if (chatbotContainer.style.display === 'flex') {
        // Don't auto-close, let user manually close
      }
    }
  });
});