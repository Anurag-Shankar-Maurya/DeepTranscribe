# DeepTranscribe

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Django Version](https://img.shields.io/badge/django-5.2-green.svg)](https://www.djangoproject.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

DeepTranscribe is a comprehensive Django-based web application that leverages AI-powered transcription and intelligent chatbot services. Built with modern web technologies, it integrates seamlessly with Google's Gemma3 and Deepgram APIs to deliver advanced features including real-time transcription, transcript analytics, and context-aware conversational AI.

## 🚀 Features

### Core Functionality
- **Real-time Transcription**: Advanced speech-to-text with speaker diarization and automatic punctuation
- **AI-Powered Chatbot**: Context-aware conversational assistant using transcript data and chat history
- **Transcript Management**: Complete CRUD operations with support for multiple export formats
- **Analytics & Insights**: Sentiment analysis, entity extraction, speaker activity tracking, and timeline generation

### Technical Features
- **User Authentication**: Secure registration, login, and profile management system
- **REST API**: Comprehensive API endpoints with token and session authentication
- **Real-time Communication**: WebSocket support via Django Channels and Daphne for live features
- **Multi-format Export**: Export transcripts as JSON, plain text, or PDF
- **CORS Support**: Ready for frontend integration and cross-origin requests

## 📋 Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Contributing](#contributing)
- [License](#license)

## 🛠 Installation

### Prerequisites

- Python 3.10 or higher
- PostgreSQL database (recommended) or SQLite for development
- Git

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Anurag-Shankar-Maurya/DeepTranscribe.git
   cd DeepTranscribe
   ```

2. **Create Virtual Environment**
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**

   Create a `.env` file in the project root:
   ```env
   SECRET_KEY=your_django_secret_key_here
   DEBUG=True
   ALLOWED_HOSTS=localhost,127.0.0.1
   DATABASE_URL=postgresql://user:password@localhost:5432/deeptranscribe
   DEEPGRAM_API_KEY=your_deepgram_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

5. **Database Setup**
   ```bash
   python manage.py migrate
   ```

6. **Create Superuser** (Optional)
   ```bash
   python manage.py createsuperuser
   ```

7. **Collect Static Files**
   ```bash
   python manage.py collectstatic --noinput
   ```

## ⚙ Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SECRET_KEY` | Django secret key for security | Yes |
| `DEBUG` | Enable/disable debug mode | Yes |
| `ALLOWED_HOSTS` | Comma-separated list of allowed hosts | Yes |
| `DATABASE_URL` | Database connection URL | Yes |
| `DEEPGRAM_API_KEY` | Deepgram API key for transcription | Yes |
| `GEMINI_API_KEY` | Gemini API key for AI features | Yes |

### API Keys Setup

1. **Deepgram API**: Sign up at [Deepgram](https://console.deepgram.com/) and obtain your API key
2. **Gemma3 API**: Obtain your API key from [Google AI Studio](https://aistudio.google.com/) or your Gemma3 provider

## 🚀 Usage

### Development Server

Choose one of the following methods to run the application:

**Option 1: Django Development Server**
```bash
python manage.py runserver
```

**Option 2: ASGI Server (Recommended for real-time features)**
```bash
daphne -p 8000 transcriber.asgi:application
```

Access the application at `http://localhost:8000`

### Production Deployment

The application is configured for deployment on Vercel. Use the provided `vercel.json` configuration for seamless deployment.

## 📚 API Documentation

DeepTranscribe provides a comprehensive REST API for integration. Key endpoints include:

- `POST /api/transcribe/` - Upload and transcribe audio files
- `GET /api/transcripts/` - List user transcripts
- `POST /api/chat/` - Interact with AI chatbot
- `GET /api/analytics/{transcript_id}/` - Get transcript analytics

For detailed API documentation, visit `/api/docs/` when the server is running.

## 🏗 Project Structure

```
DeepTranscribe/
├── api/                    # API application
│   ├── models.py          # Database models for chat and embeddings
│   ├── views.py           # API endpoints
│   ├── serializers.py     # Data serialization
│   └── chatbot_service.py # AI chatbot logic
├── core/                   # Core application
│   ├── models.py          # Transcript and user models
│   ├── views.py           # Main views and transcription logic
│   └── consumers.py       # WebSocket consumers
├── users/                  # User management
│   ├── views.py           # Authentication views
│   └── forms.py           # User forms
├── transcriber/            # Django project settings
│   ├── settings.py        # Main configuration
│   ├── urls.py            # URL routing
│   └── asgi.py            # ASGI configuration
├── static/                 # Static assets
├── templates/              # HTML templates
└── requirements.txt        # Python dependencies
```

## 🛠 Technology Stack

### Backend
- **Django 5.2** - Web framework
- **Django REST Framework** - API development
- **Django Channels** - WebSocket support
- **Daphne** - ASGI server

### AI & ML
- **Google Gemma3** - Language model integration
- **Deepgram API** - Speech-to-text transcription
- **FAISS** - Vector similarity search (for embeddings)

### Database & Storage
- **PostgreSQL** - Primary database
- **SQLite** - Development database
- **Redis** - Caching (optional)

### Frontend
- **HTML5/CSS3** - Base templates
- **JavaScript** - Client-side interactions
- **Bootstrap** - UI framework

### Deployment
- **Vercel** - Cloud platform
- **Gunicorn** - WSGI server

## 🤝 Contributing

We welcome contributions to DeepTranscribe! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Write tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Django](https://www.djangoproject.com/) - The web framework
- [Google Gemma3](https://ai.google.dev/gemma) - AI language models
- [Deepgram](https://deepgram.com/) - Speech recognition
- [Django REST Framework](https://www.django-rest-framework.org/) - API toolkit

## 📞 Support

For questions, issues, or contributions, please:

- Open an issue on [GitHub](https://github.com/Anurag-Shankar-Maurya/DeepTranscribe/issues)
- Contact the maintainers

---

**Made with ❤️ by [Anurag Shankar Maurya](https://github.com/Anurag-Shankar-Maurya)**
