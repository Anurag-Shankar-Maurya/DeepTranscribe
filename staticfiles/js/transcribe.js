// Audio recording and transcription handling

class TranscriptionManager {
  constructor() {
    this.socket = null;
    this.mediaRecorder = null;
    this.audioContext = null;
    this.isRecording = false;
    this.transcriptId = null;
    this.speakerColors = [
  '#3898f3', // primary blue
  '#f59e0b', // amber
  '#10b981', // green
  '#ef4444', // red
  '#8b5cf6', // purple
  '#ec4899', // pink
  '#14b8a6', // teal
  '#f97316', // orange
  '#6366f1', // indigo
  '#22d3ee', // cyan
  '#eab308', // yellow
  '#4ade80', // light green
  '#f43f5e', // rose
  '#a855f7', // violet
  '#0ea5e9', // sky blue
  '#34d399', // emerald
  '#facc15', // gold
  '#fb7185', // light red
  '#7c3aed', // deep purple
  '#06b6d4', // turquoise
];
  }

  async initialize() {
    // Set up WebSocket connection
    this.setupWebSocket();

    // Set up audio context
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }

  setupWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/transcribe/`;

    this.socket = new WebSocket(wsUrl);

    this.socket.onopen = () => {
      console.log('WebSocket connection established');
      this.updateStatus('WebSocket connected');
    };

    this.socket.onclose = (event) => {
      console.log('WebSocket connection closed', event);
      this.updateStatus('WebSocket disconnected');
      this.stopRecording();
    };

    this.socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.updateStatus('WebSocket error');
    };

    this.socket.onmessage = (event) => {
      this.handleWebSocketMessage(event);
    };
  }

  handleWebSocketMessage(event) {
    try {
      const data = JSON.parse(event.data);

      if (data.error) {
        this.showError(data.error);
        return;
      }

      if (data.status === 'transcription_started') {
        this.transcriptId = data.transcript_id;
        this.updateStatus('Transcription started');
      } else if (data.status === 'transcription_stopped') {
        this.updateStatus('Transcription stopped');
      } else if (data.type === 'transcript_segment') {
        this.displayTranscriptSegment(data);
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  }

  async startRecording() {
    if (this.isRecording) return;

    try {
      // Get title from input
      const title = document.getElementById('transcript-title').value || 'Untitled Transcript';

      // Get settings
      const settings = {
        model: document.getElementById('model').value,
        language: document.getElementById('language').value,
        diarize: document.getElementById('diarize').checked,
        punctuate: document.getElementById('punctuate').checked,
        numerals: document.getElementById('numerals').checked,
        smart_format: document.getElementById('smart_format').checked,
      };

      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Start transcription on the server
      this.socket.send(JSON.stringify({
        command: 'start_transcription',
        title: title,
        settings: settings
      }));

      // Set up audio processing
      const audioSource = this.audioContext.createMediaStreamSource(stream);
      const processor = this.audioContext.createScriptProcessor(4096, 1, 1);

      audioSource.connect(processor);
      processor.connect(this.audioContext.destination);

      processor.onaudioprocess = (e) => {
        if (!this.isRecording) return;

        // Get audio data
        const inputData = e.inputBuffer.getChannelData(0);

        // Convert to 16-bit PCM
        const pcmData = this.floatTo16BitPCM(inputData);

        // Send audio data to server
        this.socket.send(pcmData);
      };

      // Set up media recorder for visualizations
      this.mediaRecorder = new MediaRecorder(stream);
      this.mediaRecorder.start();

      // Update UI state
      this.isRecording = true;
      this.updateButtonState();
      this.updateStatus('Recording...');
      this.startVisualization(stream);

      // Clear previous transcript
      document.getElementById('transcript-container').innerHTML = '';

    } catch (error) {
      console.error('Error starting recording:', error);
      this.showError('Could not access microphone');
    }
  }

  stopRecording() {
    if (!this.isRecording) return;

    // Stop server transcription
    this.socket.send(JSON.stringify({
      command: 'stop_transcription'
    }));

    // Stop media recorder if exists
    if (this.mediaRecorder) {
      this.mediaRecorder.stop();
      this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
      this.mediaRecorder = null;
    }

    // Update UI state
    this.isRecording = false;
    this.updateButtonState();
    this.updateStatus('Stopped');
    this.stopVisualization();
  }

  updateButtonState() {
    const startBtn = document.getElementById('start-recording');
    const stopBtn = document.getElementById('stop-recording');

    if (this.isRecording) {
      startBtn.disabled = true;
      stopBtn.disabled = false;
    } else {
      startBtn.disabled = false;
      stopBtn.disabled = true;
    }
  }

  updateStatus(message) {
    document.getElementById('status').textContent = message;
  }

  showError(message) {
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');

    setTimeout(() => {
      errorDiv.classList.add('hidden');
    }, 5000);
  }

  displayTranscriptSegment(data) {
    const container = document.getElementById('transcript-container');

    // Check if there's already a segment for this speaker
    const speakerId = data.speaker !== null ? data.speaker : 'unknown';
    const speakerSelector = `[data-speaker="${speakerId}"]`;
    let speakerDiv = container.querySelector(speakerSelector);

    if (!speakerDiv) {
      // Create new speaker div
      speakerDiv = document.createElement('div');
      speakerDiv.className = 'mb-4 p-4 rounded-lg';
      speakerDiv.dataset.speaker = speakerId;

      // Assign color based on speaker ID
      const colorIndex = data.speaker !== null ? (data.speaker % this.speakerColors.length) : 0;
      const speakerColor = this.speakerColors[colorIndex];
      speakerDiv.style.backgroundColor = `${speakerColor}15`; // 15% opacity
      speakerDiv.style.borderLeft = `4px solid ${speakerColor}`;

      // Add speaker label
      const speakerLabel = document.createElement('div');
      speakerLabel.className = 'font-semibold text-sm mb-1';
      speakerLabel.textContent = data.speaker !== null ? `Speaker ${data.speaker}` : 'Unidentified Speaker';
      speakerLabel.style.color = speakerColor;
      speakerDiv.appendChild(speakerLabel);

      // Add text container
      const textContainer = document.createElement('div');
      textContainer.className = 'text-content';
      speakerDiv.appendChild(textContainer);

      container.appendChild(speakerDiv);
    }

    // Update or add text
    const textContent = speakerDiv.querySelector('.text-content');
    textContent.textContent = data.text;

    // Scroll to bottom
    container.scrollTop = container.scrollHeight;
  }

  floatTo16BitPCM(input) {
    const output = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
      const s = Math.max(-1, Math.min(1, input[i]));
      output[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return output.buffer;
  }

  startVisualization(stream) {
    const canvas = document.getElementById('audio-visualizer');
    if (!canvas) return;

    const canvasCtx = canvas.getContext('2d');
    const analyser = this.audioContext.createAnalyser();
    const source = this.audioContext.createMediaStreamSource(stream);

    source.connect(analyser);
    analyser.fftSize = 256;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

    const draw = () => {
      if (!this.isRecording) return;

      requestAnimationFrame(draw);

      analyser.getByteFrequencyData(dataArray);

      canvasCtx.fillStyle = 'rgb(20, 20, 20)';
      canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

      const barWidth = (canvas.width / bufferLength) * 2.5;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = dataArray[i] / 2;

        // Use gradient colors based on frequency
        const hue = (i / bufferLength) * 220 + 180; // blue to purple range
        canvasCtx.fillStyle = `hsl(${hue}, 80%, 50%)`;

        canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);

        x += barWidth + 1;
      }
    };

    draw();
  }

  stopVisualization() {
    const canvas = document.getElementById('audio-visualizer');
    if (!canvas) return;

    const canvasCtx = canvas.getContext('2d');
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
  }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  const transcriptionManager = new TranscriptionManager();

  // Initialize transcription manager
  transcriptionManager.initialize().then(() => {
    console.log('Transcription manager initialized');
  }).catch(error => {
    console.error('Error initializing transcription manager:', error);
  });

  // Set up event listeners
  document.getElementById('start-recording').addEventListener('click', () => {
    transcriptionManager.startRecording();
  });

  document.getElementById('stop-recording').addEventListener('click', () => {
    transcriptionManager.stopRecording();
  });
  
  // Fetch available models and languages
  fetch('/api/settings/')
    .then(response => response.json())
    .then(data => {
      const modelSelect = document.getElementById('model');
      const languageSelect = document.getElementById('language');
      
      // Populate models
      data.deepgram.models.forEach(model => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = model;
        if (model === 'nova-3') option.selected = true;
        modelSelect.appendChild(option);
      });
      
      // Populate languages
      data.deepgram.languages.forEach(language => {
        const option = document.createElement('option');
        option.value = language.code;
        option.textContent = language.name;
        if (language.code === 'multi') option.selected = true;
        languageSelect.appendChild(option);
      });
    })
    .catch(error => {
      console.error('Error fetching settings:', error);
    });
});