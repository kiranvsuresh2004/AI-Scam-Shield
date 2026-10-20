from django.conf import settings
from django.db import models


class DetectionHistory(models.Model):
    DETECTION_CHOICES = [
        ('sms', 'SMS'),
        ('email', 'Email'),
        ('url', 'URL'),
        ('file', 'File'),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='detections')
    detection_type = models.CharField(max_length=20, choices=DETECTION_CHOICES)
    input_text = models.TextField()
    file_name = models.CharField(max_length=255, blank=True, default='')
    file_extension = models.CharField(max_length=20, blank=True, default='')
    prediction = models.CharField(max_length=30)
    risk_score = models.IntegerField()
    risk_level = models.CharField(max_length=20, default='LOW')
    risk_signals = models.TextField(default='[]')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.user.username} - {self.detection_type} - {self.prediction}'
