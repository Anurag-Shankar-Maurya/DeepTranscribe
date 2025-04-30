# DeepTranscribe

## Project Overview
DeepTranscribe is a Django-based web application that provides advanced transcription and chatbot services powered by AI. It integrates with OpenAI and Deepgram APIs to offer features such as transcript management, intelligent chat assistance, sentiment analysis, entity extraction, timeline generation, and transcript summarization. The application supports real-time transcription, user authentication, and export of transcripts in multiple formats (JSON, TXT, PDF).


## Features
- User registration, login, and profile management
- Real-time transcription with speaker diarization and punctuation
- Transcript management: list, view details, edit, delete
- Export transcripts as JSON, plain text, or PDF
- AI-powered chatbot assistant with context-aware responses based on transcripts and chat history
- Transcript summarization and analytics (sentiment, speaker activity, entity extraction, timeline)
- REST API endpoints with token and session authentication
- WebSocket support for real-time features using Django Channels and Daphne
- CORS support for frontend integration

## Installation

### Prerequisites
- Python 3.10+
- PostgreSQL database
- Virtual environment tool (optional but recommended)

### Setup Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Anurag-Shankar-Maurya/DeepTranscribe.git
   cd DeepTranscribe
   ```

2. Create a virtual environment:
   - On Unix or MacOS:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - On Windows CMD:
     ```cmd
     python -m venv venv
     venv\Scripts\activate.bat
     ```
   - On Windows PowerShell:
     ```powershell
     python -m venv venv
     venv\Scripts\Activate.ps1
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   Create a `.env` file in the project root with the following variables:
   ```
   SECRET_KEY=your_django_secret_key
   DEBUG=True
   ALLOWED_HOSTS=localhost,127.0.0.1
   DATABASE_URL=sqlite:///db.sqlite3
   DEEPGRAM_API_KEY=your_deepgram_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

5. Apply database migrations:
   ```bash
   python manage.py migrate
   ```

6. Create a superuser (optional, for admin access):
   ```bash
   python manage.py createsuperuser
   ```

7. Collect static files:
   ```bash
   python manage.py collectstatic
   ```

## Running the Application

### Development Server

You can run the development server using either Daphne (ASGI) or the standard Django runserver command.

- Using Django's built-in development server:
```bash
python manage.py runserver
```

Then open your browser at `http://localhost:8000`.

### Usage
- Register a new user or log in.
- Access the transcription page to upload or record audio for transcription.
- Manage your transcripts: view, edit, delete, and export.
- Use the AI chatbot assistant to query transcript data and get intelligent responses.
- View your user profile and manage your Deepgram API key information.
daphne -p 8000 transcriber.asgi:application

## Project Structure
- `api/`: Contains chatbot service, API views, serializers, models, and utilities.
- `core/`: Core app with transcript models, views, and PDF export functionality.
- `users/`: User authentication, registration, and profile management.
- `transcriber/`: Django project settings, URLs, ASGI and WSGI configurations.
- `static/`: Static assets including CSS, JS, and fonts.
- `templates/`: HTML templates for core and user interfaces.

## Environment Variables
- `SECRET_KEY`: Django secret key.
- `DEBUG`: Enable/disable debug mode.
- `ALLOWED_HOSTS`: Allowed hosts for Django.
- `DATABASE_URL`: Database connection URL.
- `DEEPGRAM_API_KEY`: API key for Deepgram transcription service.
- `OPENAI_API_KEY`: API key for OpenAI services.

## Dependencies
See `requirements.txt` for a full list of Python packages used, including:
- Django 5.2
- Django REST Framework
- Django Channels and Daphne
- Deepgram SDK
- OpenAI Python client
- ReportLab for PDF generation
- Psycopg2 for PostgreSQL
