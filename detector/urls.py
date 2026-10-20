from django.urls import path

from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('sms/', views.sms_detection, name='sms'),
    path('email/', views.email_detection, name='email'),
    path('url/', views.url_detection, name='url'),
    path('file-scan/', views.file_scan_view, name='file_scan'),
    path('history/', views.history_view, name='history'),
    path('history/delete/<int:id>/', views.delete_history, name='delete_history'),
    path('history/clear/', views.clear_history, name='clear_history'),
    path('predict/', views.predict_json, name='predict'),
    path('dashboard/sms/', views.authenticated_sms_detection, name='sms_detection'),
    path('dashboard/email/', views.authenticated_email_detection, name='email_detection'),
    path('dashboard/url/', views.authenticated_url_detection, name='url_detection'),
]
