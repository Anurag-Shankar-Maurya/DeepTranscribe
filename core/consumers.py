import json
import asyncio
import logging
import os
import threading
import ssl
import websocket
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.conf import settings as django_settings
from .models import Transcript, TranscriptSegment, TranscriptSettings

logger = logging.getLogger(__name__)


class TranscribeConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for real-time transcription using Deepgram WebSocket API."""

    # Audio settings
    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_SIZE = 4096
    AUDIO_ENCODING = "linear16"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws = None
        self.ws_thread = None
        self.ws_ready = False  # Flag to track WebSocket readiness
        self.transcript_id = None
        self.speaker_mapping = {}
        self.speaker_counter = 0
        self.current_speaker = None
        self.streaming_active = False
        self.loop = None  # Will hold reference to the consumer's event loop

    async def connect(self):
        self.user = self.scope["user"]
        logger.info(f"WebSocket connect attempt - User: {self.user}, Authenticated: {self.user.is_authenticated}")
        if not self.user.is_authenticated:
            logger.warning(f"Rejecting WebSocket connection - user not authenticated")
            await self.close()
            return

        # Capture the event loop when connection is established
        self.loop = asyncio.get_running_loop()
        await self.accept()
        logger.info(f"✓ WebSocket connection accepted for user {self.user.username}")

    async def disconnect(self, close_code):
        logger.info(f"WebSocket disconnect - Code: {close_code}, User: {getattr(self, 'user', 'unknown')}")
        self.streaming_active = False
        if self.ws:
            self.ws.close()
            self.ws = None

        if self.transcript_id:
            await self.update_transcript_status(self.transcript_id, True)
        logger.info(f"✓ WebSocket disconnected")

    async def receive(self, text_data=None, bytes_data=None):
        if text_data:
            data = json.loads(text_data)
            command = data.get('command')

            if command == 'start_transcription':
                logger.info(f"Received start_transcription command")
                title = data.get('title', 'Untitled Transcript')
                settings_data = data.get('settings', {})
                await self.start_transcription(title, settings_data)

            elif command == 'stop_transcription':
                logger.info(f"Received stop_transcription command")
                await self.stop_transcription()

        elif bytes_data:
            # Log state on first audio chunk
            if not hasattr(self, '_audio_chunk_count'):
                self._audio_chunk_count = 0
                logger.info(f"First audio chunk received - ws={self.ws is not None}, ws_ready={self.ws_ready}, streaming_active={self.streaming_active}")
            
            # Forward audio data to Deepgram
            if self.ws and self.ws_ready and self.streaming_active:
                try:
                    self.ws.send(bytes_data, opcode=websocket.ABNF.OPCODE_BINARY)
                    self._audio_chunk_count += 1
                    if self._audio_chunk_count == 1:
                        logger.info(f"✓ First audio chunk forwarded to Deepgram ({len(bytes_data)} bytes)")
                    elif self._audio_chunk_count % 50 == 0:
                        logger.info(f"→ Sent {self._audio_chunk_count} audio chunks to Deepgram ({len(bytes_data)} bytes/chunk)")
                except Exception as e:
                    logger.error(f"Error sending audio data to Deepgram: {e}", exc_info=True)
            else:
                if not hasattr(self, '_dropped_warning_shown'):
                    logger.warning(f"Audio dropped: ws={self.ws is not None}, ws_ready={self.ws_ready}, streaming_active={self.streaming_active}")
                    self._dropped_warning_shown = True

    async def start_transcription(self, title, settings_data):
        # Reset state for new transcription
        self._audio_chunk_count = 0
        if hasattr(self, '_dropped_warning_shown'):
            del self._dropped_warning_shown
            
        self.transcript_id = await self.create_transcript(title, settings_data)
        transcript_settings = await self.get_transcript_settings(self.transcript_id)
        
        # Get audio settings from client or use defaults
        sample_rate = settings_data.get('sample_rate', 48000)
        channels = settings_data.get('channels', 1)
        encoding = settings_data.get('encoding', 'linear16')

        # Build Deepgram WebSocket URL with parameters
        features = {
            "diarize": str(transcript_settings.diarize).lower(),
            "punctuate": str(transcript_settings.punctuate).lower(),
            "numerals": str(transcript_settings.numerals).lower(),
            "smart_format": str(transcript_settings.smart_format).lower(),
            "interim_results": "false",
            "endpointing": "100",
            "vad_events": "true"
        }
        base_url = f"wss://api.deepgram.com/v1/listen?model={transcript_settings.model}&language={transcript_settings.language}"
        base_url += f"&encoding={encoding}&sample_rate={sample_rate}&channels={channels}"
        feature_params = "&".join([f"{key}={value}" for key, value in features.items()])
        deepgram_url = f"{base_url}&{feature_params}"

        # Validate Deepgram API key
        deepgram_api_key = os.getenv('DEEPGRAM_API_KEY')
        if not deepgram_api_key:
            logger.error("DEEPGRAM_API_KEY is not set")
            await self.send(text_data=json.dumps({
                'error': 'Server configuration error: Deepgram API key missing'
            }))
            await self.stop_transcription()
            return

        logger.info(f"Starting transcription with title: {title}")
        logger.info(f"Settings: model={transcript_settings.model}, language={transcript_settings.language}")
        logger.info(f"Deepgram URL: {deepgram_url}")
        self.streaming_active = True

        def on_open(ws):
            logger.info("✓ Deepgram WebSocket connection OPENED")
            self.ws_ready = True
            # Audio will be sent from client via WebSocket

        def on_message(ws, message):
            try:
                data = json.loads(message)
                logger.info(f"✓ Received message from Deepgram: type={data.get('type', 'unknown')}, is_final={data.get('is_final', False)}")
                logger.debug(f"Message structure: {list(data.keys())}")
                if 'channel' in data and 'alternatives' in data['channel']:
                    alternatives = data['channel']['alternatives']
                    if alternatives:
                        logger.info(f"  → Alternatives count: {len(alternatives)}")
                        asyncio.run_coroutine_threadsafe(
                            self.process_transcript(data),
                            self.loop
                        )
                    else:
                        logger.debug("No alternatives in channel message")
                else:
                    logger.debug(f"Message structure unexpected: {list(data.keys())}")
            except Exception as e:
                logger.error(f"Error processing Deepgram message: {e}", exc_info=True)

        def on_error(ws, error):
            logger.error(f"✗ Deepgram WebSocket error: {error}", exc_info=True)
            if hasattr(error, '__traceback__'):
                import traceback
                logger.error("".join(traceback.format_exception(type(error), error, error.__traceback__)))
            self.ws_ready = False
            try:
                asyncio.run_coroutine_threadsafe(
                    self.send(text_data=json.dumps({
                        'error': f"Deepgram error: {str(error)}"
                    })),
                    self.loop
                )
            except Exception as e:
                logger.error(f"Failed to send error message: {e}")
            try:
                asyncio.run_coroutine_threadsafe(
                    self.stop_transcription(),
                    self.loop
                )
            except Exception as e:
                logger.error(f"Failed to stop transcription: {e}")

        def on_close(ws, close_status_code, close_msg):
            logger.info(f"Deepgram WebSocket closed: {close_status_code} - {close_msg}")
            self.streaming_active = False
            self.ws = None
            self.ws_ready = False
            asyncio.run_coroutine_threadsafe(
                self.send(text_data=json.dumps({
                    'status': 'transcription_stopped',
                    'code': close_status_code,
                    'reason': close_msg
                })),
                self.loop
            )

        logger.debug(f"Initializing Deepgram WebSocket with URL: {deepgram_url}")
        logger.debug(f"Using Deepgram API key (first 10 chars): {deepgram_api_key[:10]}...")
        
        try:
            self.ws = websocket.WebSocketApp(
                deepgram_url,
                header=[f"Authorization: Token {deepgram_api_key}"],
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
        except Exception as e:
            logger.error(f"Failed to create WebSocketApp: {e}", exc_info=True)
            await self.send(text_data=json.dumps({
                'error': f'Failed to initialize WebSocket: {str(e)}'
            }))
            return

        self.ws_ready = False
        try:
            self.ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            self.ws_thread.start()
            logger.info("WebSocket thread started")
        except Exception as e:
            logger.error(f"Failed to start WebSocket thread: {e}", exc_info=True)
            await self.send(text_data=json.dumps({
                'error': f'Failed to start WebSocket: {str(e)}'
            }))
            return

        # Wait for WebSocket to be ready (timeout after 15 seconds)
        timeout = 15
        start_time = asyncio.get_event_loop().time()
        check_count = 0
        while not self.ws_ready:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                logger.error(f"Timeout waiting for Deepgram WebSocket connection after {elapsed:.1f}s")
                await self.send(text_data=json.dumps({
                    'error': 'Failed to connect to Deepgram WebSocket - timeout'
                }))
                await self.stop_transcription()
                return
            check_count += 1
            if check_count % 10 == 0:
                logger.debug(f"Waiting for WebSocket connection... ({elapsed:.1f}s elapsed)")
            await asyncio.sleep(0.1)

        await self.send(text_data=json.dumps({
            'status': 'transcription_started',
            'transcript_id': self.transcript_id
        }))

    async def stop_transcription(self):
        if self.streaming_active:
            self.streaming_active = False
            if self.ws:
                self.ws.close()
                self.ws = None
                self.ws_ready = False

            if self.transcript_id:
                await self.update_transcript_status(self.transcript_id, True)
            await self.send(text_data=json.dumps({
                'status': 'transcription_stopped',
                'transcript_id': self.transcript_id
            }))

    async def process_transcript(self, transcript):
        try:
            logger.debug(f"Processing transcript with keys: {list(transcript.keys())}")
            result = transcript.get('channel', {}).get('alternatives', [{}])[0]
            text = result.get('transcript', '').strip()
            is_final = transcript.get('is_final', False)
            
            # Log the first result in full to see what Deepgram is sending
            if not hasattr(self, '_logged_first_result'):
                logger.info(f"FIRST RESULT FROM DEEPGRAM: {result}")
                self._logged_first_result = True
            
            logger.info(f"Transcript segment: text='{text}' (len={len(text)}), is_final={is_final}")
            
            if not text:
                logger.debug(f"Empty transcript, skipping. Confidence: {result.get('confidence', 'N/A')}")
                return

            words = result.get('words', [])
            speaker_id = self._get_primary_speaker(words)

            if speaker_id is not None and speaker_id not in self.speaker_mapping:
                self.speaker_mapping[speaker_id] = self.speaker_counter
                self.speaker_counter += 1

            mapped_speaker = self.speaker_mapping.get(speaker_id, None)

            start_time = words[0].get('start') if words else 0
            end_time = words[-1].get('end') if words else 0
            confidence = result.get('confidence', 0)

            segment_id = await self.create_transcript_segment(
                self.transcript_id, text, mapped_speaker, start_time, end_time, confidence
            )

            logger.info(f"✓ Sending transcript segment to client: '{text[:30]}...'")
            await self.send(text_data=json.dumps({
                'type': 'transcript_segment',
                'segment_id': segment_id,
                'text': text,
                'speaker': mapped_speaker,
                'start_time': start_time,
                'end_time': end_time,
                'confidence': confidence
            }))
        except Exception as e:
            logger.error(f"Error processing transcript: {e}", exc_info=True)
            try:
                await self.send(text_data=json.dumps({
                    'error': f"Error processing transcript: {str(e)}"
                }))
            except Exception as send_error:
                logger.error(f"Failed to send error message: {send_error}")

    def _get_primary_speaker(self, words):
        speaker_counts = {}
        for word in words:
            spk = word.get('speaker')
            if spk is not None:
                speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
        if speaker_counts:
            return max(speaker_counts, key=speaker_counts.get)
        return None

    @database_sync_to_async
    def create_transcript(self, title, settings_data):
        transcript = Transcript.objects.create(
            title=title,
            user=self.user,
            is_complete=False
        )
        TranscriptSettings.objects.create(
            transcript=transcript,
            model=settings_data.get('model', 'nova-3'),
            language=settings_data.get('language', 'en-US'),
            diarize=settings_data.get('diarize', True),
            punctuate=settings_data.get('punctuate', True),
            numerals=settings_data.get('numerals', True),
            smart_format=settings_data.get('smart_format', True)
        )
        return transcript.id

    @database_sync_to_async
    def create_transcript_segment(self, transcript_id, text, speaker, start_time, end_time, confidence):
        segment = TranscriptSegment.objects.create(
            transcript_id=transcript_id,
            text=text,
            speaker=speaker,
            start_time=start_time,
            end_time=end_time,
            confidence=confidence
        )
        return segment.id

    @database_sync_to_async
    def update_transcript_status(self, transcript_id, is_complete):
        transcript = Transcript.objects.get(id=transcript_id)
        transcript.is_complete = is_complete
        segments = transcript.segments.all()
        if segments.exists():
            last_segment = segments.order_by('-end_time').first()
            transcript.duration = last_segment.end_time
        transcript.save()

    @database_sync_to_async
    def get_transcript_settings(self, transcript_id):
        transcript = Transcript.objects.get(id=transcript_id)
        return transcript.settings