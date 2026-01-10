import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'transcriber.settings')
from django.core.asgi import get_asgi_application
# Initialize Django ASGI app first so apps are loaded before importing local modules
django_asgi_app = get_asgi_application()

from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import core.routing

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AuthMiddlewareStack(
        URLRouter(
            core.routing.websocket_urlpatterns
        )
    ),
})