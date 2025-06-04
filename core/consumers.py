import json
import asyncio
import logging
import os
import threading
import websocket
import pyaudio
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.conf import settings
from .models import Transcript, TranscriptSegment, TranscriptSettings
from api.chatbot_service import ChatbotService

logger = logging.getLogger(__name__)

class TranscribeConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for real-time transcription using Deepgram WebSocket API."""

    # Audio settings
    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_SIZE = 4096
    FORMAT = pyaudio.paInt16
    AUDIO_ENCODING = "linear16"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ws = None
        self.ws_thread = None
        self.ws_ready = False
        self.transcript_id = None
        self.speaker_mapping = {}
        self.speaker_counter = 0
        self.current_speaker = None
        self.streaming_active = False
        self.loop = None
        self.audio_thread = None
        self.pyaudio_instance = None
        self.audio_stream = None
        self.chatbot_service = None

    async def connect(self):
        self.user = self.scope["user"]
        if not self.user.is_authenticated:
            await self.close()
            return

        self.loop = asyncio.get_running_loop()
        self.chatbot_service = ChatbotService(self.user)
        await self.accept()
        logger.info(f"WebSocket connection established for user {self.user.username}")

    async def disconnect(self, close_code):
        self.streaming_active = False
        if self.ws:
            self.ws.close()
            self.ws = None

        self.stop_audio_stream()

        if self.transcript_id:
            await self.update_transcript_status(self.transcript_id, True)
        logger.info(f"WebSocket connection closed for user {self.user.username}")

    async def receive(self, text_data=None, bytes_data=None):
        if text_data:
            data = json.loads(text_data)
            command = data.get('command')

            if command == 'start_transcription':
                title = data.get('title', 'Untitled Transcript')
                settings_data = data.get('settings', {})
                await self.start_transcription(title, settings_data)

            elif command == 'stop_transcription':
                await self.stop_transcription()

    async def start_transcription(self, title, settings_data):
        self.transcript_id = await self.create_transcript(title, settings_data)
        transcript_settings = await self.get_transcript_settings(self.transcript_id)

        features = {
            "diarize": str(transcript_settings.diarize).lower(),
            "punctuate": str(transcript_settings.punctuate).lower(),
            "numerals": str(transcript_settings.numerals).lower(),
            "smart_format": str(transcript_settings.smart_format).lower(),
            "interim_results": "false",
            "endpointing": "100",
            "paragraphs": "true",
            "vad_events": "true",
            "filler_words": "true",
            "sentiment": "true",
        }
        base_url = f"wss://api.deepgram.com/v1/listen?model={transcript_settings.model}&language={transcript_settings.language}"
        base_url += f"&encoding={self.AUDIO_ENCODING}&sample_rate={self.SAMPLE_RATE}&channels={self.CHANNELS}"
        feature_params = "&".join([f"{key}={value}" for key, value in features.items()])
        deepgram_url = f"{base_url}&{feature_params}"

        deepgram_api_key = os.getenv('DEEPGRAM_API_KEY')
        if not deepgram_api_key:
            logger.error("DEEPGRAM_API_KEY is not set")
            await self.send(text_data=json.dumps({
                'error': 'Server configuration error: Deepgram API key missing'
            }))
            await self.stop_transcription()
            return

        self.streaming_active = True

        def on_open(ws):
            logger.info("Deepgram WebSocket connection opened")
            self.ws_ready = True
            self.start_audio_stream(ws)

        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 'channel' in data and 'alternatives' in data['channel']:
                    asyncio.run_coroutine_threadsafe(
                        self.process_transcript(data),
                        self.loop
                    )
            except Exception as e:
                logger.error(f"Error processing Deepgram message: {e}")

        def on_error(ws, error):
            logger.error(f"Deepgram WebSocket error: {error}")
            self.ws_ready = False
            asyncio.run_coroutine_threadsafe(
                self.send(text_data=json.dumps({
                    'error': f"Deepgram error: {error}"
                })),
                self.loop
            )
            asyncio.run_coroutine_threadsafe(
                self.stop_transcription(),
                self.loop
            )

        def on_close(ws, close_status_code, close_msg):
            logger.info(f"Deepgram WebSocket closed: {close_status_code} - {close_msg}")
            self.streaming_active = False
            self.ws = None
            self.ws_ready = False
            self.stop_audio_stream()
            asyncio.run_coroutine_threadsafe(
                self.send(text_data=json.dumps({
                    'status': 'transcription_stopped',
                    'code': close_status_code,
                    'reason': close_msg
                })),
                self.loop
            )

        logger.debug(f"Initializing Deepgram WebSocket with URL: {deepgram_url}")
        self.ws = websocket.WebSocketApp(
            deepgram_url,
            header=[f"Authorization: Token {deepgram_api_key}"],
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        self.ws_ready = False
        self.ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.ws_thread.start()

        timeout = 5
        start_time = asyncio.get_event_loop().time()
        while not self.ws_ready:
            if asyncio.get_event_loop().time() - start_time > timeout:
                logger.error("Timeout waiting for Deepgram WebSocket connection")
                await self.send(text_data=json.dumps({
                    'error': 'Failed to connect to Deepgram WebSocket'
                }))
                await self.stop_transcription()
                return
            await asyncio.sleep(0.1)

        await self.send(text_data=json.dumps({
            'status': 'transcription_started',
            'transcript_id': self.transcript_id
        }))

    def start_audio_stream(self, ws):
        """Start the audio streaming in a separate thread."""
        if self.audio_thread and self.audio_thread.is_alive():
            return

        self.audio_thread = threading.Thread(
            target=self.stream_audio,
            args=(ws,),
            daemon=True
        )
        self.audio_thread.start()
        logger.info("Audio streaming thread started")

    def stream_audio(self, ws):
        """Captures and streams audio to Deepgram."""
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            self.audio_stream = self.pyaudio_instance.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE // 2
            )

            logger.info("Microphone streaming started...")

            while self.streaming_active and ws and ws.sock and ws.sock.connected:
                try:
                    data = self.audio_stream.read(self.CHUNK_SIZE // 2, exception_on_overflow=False)
                    if self.streaming_active and ws and ws.sock and ws.sock.connected:
                        ws.send(data, opcode=websocket.ABNF.OPCODE_BINARY)
                except IOError as e:
                    if e.errno == pyaudio.paInputOverflowed:
                        continue
                    logger.error(f"IO Error in audio streaming: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error in audio streaming: {e}")
                    break

        except Exception as e:
            logger.error(f"Audio streaming error: {e}")
        finally:
            self.stop_audio_stream()

    def stop_audio_stream(self):
        """Stop and clean up the audio stream."""
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}")

        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
                self.pyaudio_instance = None
            except Exception as e:
                logger.error(f"Error terminating PyAudio: {e}")

    async def stop_transcription(self):
        if self.streaming_active:
            self.streaming_active = False
            if self.ws:
                self.ws.close()
                self.ws = None
                self.ws_ready = False

            self.stop_audio_stream()

            if self.transcript_id:
                await self.update_transcript_status(self.transcript_id, True)
            await self.send(text_data=json.dumps({
                'status': 'transcription_stopped',
                'transcript_id': self.transcript_id
            }))

    async def process_transcript(self, transcript):
        try:
            result = transcript.get('channel', {}).get('alternatives', [{}])[0]
            text = result.get('transcript', '').strip()
            if not text:
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

            # Store in Pinecone
            await self.loop.run_in_executor(
                None,
                self.chatbot_service.store_transcript_segment,
                str(segment_id),
                str(self.transcript_id),
                text,
                str(mapped_speaker) if mapped_speaker is not None else "Unknown",
                start_time,
                end_time
            )

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
            logger.error(f"Error processing transcript: {e}")
            await self.send(text_data=json.dumps({
                'error': f"Error processing transcript: {str(e)}"
            }))

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