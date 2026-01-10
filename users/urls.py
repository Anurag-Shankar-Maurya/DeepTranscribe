from django.urls import path, reverse_lazy
from django.contrib.auth import views as auth_views
from . import views

app_name = 'users'

urlpatterns = [
    path('register/', views.register, name='register'),
    path('login/', auth_views.LoginView.as_view(template_name='users/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),

    # Password change
    path(
        'password_change/',
        auth_views.PasswordChangeView.as_view(
            template_name='users/change_password.html',
            success_url=reverse_lazy('users:password_change_done'),
        ),
        name='change_password',
    ),
    path(
        'password_change/done/',
        auth_views.PasswordChangeDoneView.as_view(
            template_name='users/change_password_done.html'
        ),
        name='password_change_done',
    ),

    path('profile/', views.profile, name='profile'),
]