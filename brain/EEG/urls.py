from django.urls import path
from . import views

urlpatterns = [
    # Auth
    path('',          views.landing,       name='landing'),
    path('register/', views.register_view, name='register'),
    path('login/',    views.login_view,    name='login'),
    path('logout/',   views.logout_view,   name='logout'),

    # Main pages
    path('dashboard/',           views.dashboard,  name='dashboard'),
    path('upload/',              views.upload,     name='upload'),
    path('results/<uuid:session_id>/', views.results, name='results'),
    path('journal/',             views.journal,    name='journal'),
    path('chat/<uuid:session_id>/',    views.chat,    name='chat'),
    
   
    path('about/',               views.about,      name='about'),

    # AJAX / API
    path('api/chat/<uuid:session_id>/',    views.api_chat,           name='api_chat'),
    path('api/journal/stats/',             views.api_journal_stats,  name='api_journal_stats'),
    path('api/session/<uuid:session_id>/delete/', views.api_delete_session, name='api_delete_session'),
]
