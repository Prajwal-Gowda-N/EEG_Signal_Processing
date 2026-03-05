from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from .models import JournalEntry


class EEGUploadForm(forms.Form):
    eeg_file = forms.FileField(
        label='Upload EEG File',
        help_text='Accepted formats: .csv, .txt, .npy',
        widget=forms.FileInput(attrs={
            'accept': '.csv,.txt,.npy',
            'class': 'file-input',
            'id': 'eeg-file-input',
        })
    )
    notes = forms.CharField(
        required=False,
        label='Session Notes (optional)',
        widget=forms.Textarea(attrs={
            'rows': 2,
            'placeholder': 'e.g. Recorded after exercise, feeling stressed...',
            'class': 'form-textarea',
        })
    )


class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model  = User
        fields = ['username', 'email', 'password1', 'password2']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-input'


class LoginForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-input'


class JournalNoteForm(forms.ModelForm):
    class Meta:
        model  = JournalEntry
        fields = ['user_note']
        widgets = {
            'user_note': forms.Textarea(attrs={
                'rows': 3,
                'placeholder': 'Add a personal note about how you felt...',
                'class': 'form-textarea',
            })
        }


class ChatForm(forms.Form):
    message = forms.CharField(
        widget=forms.TextInput(attrs={
            'placeholder': 'Ask a follow-up question...',
            'class': 'chat-input',
            'autocomplete': 'off',
        }),
        max_length=500,
    )
