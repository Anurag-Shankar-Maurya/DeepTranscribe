from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm
from django.conf import settings
from .forms import UserRegistrationForm, UserUpdateForm


def register(request):
    """Register a new user."""
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}! You can now log in.')
            return redirect('users:login')
    else:
        form = UserRegistrationForm()
    return render(request, 'users/register.html', {'form': form})


@login_required
def profile(request):
    """Update user profile."""
    if request.method == 'POST':
        form = UserUpdateForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, 'Your profile has been updated!')
            return redirect('users:profile')
    else:
        form = UserUpdateForm(instance=request.user)
    
    return render(request, 'users/profile.html', {'form': form})


@login_required
def api_key(request):
    """Display Deepgram API key information."""
    # For security, don't show the actual API key
    api_key_prefix = settings.DEEPGRAM_API_KEY[:4] + '...' if settings.DEEPGRAM_API_KEY else None
    return render(request, 'users/api_key.html', {'api_key_prefix': api_key_prefix})