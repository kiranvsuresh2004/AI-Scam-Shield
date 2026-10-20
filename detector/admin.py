from django.contrib import admin

from .models import DetectionHistory


@admin.register(DetectionHistory)
class DetectionHistoryAdmin(admin.ModelAdmin):
    list_display = ('user', 'detection_type', 'prediction', 'risk_score', 'created_at')
    list_filter = ('detection_type', 'prediction')
    search_fields = ('user__username', 'input_text', 'prediction')
    ordering = ('-created_at',)
