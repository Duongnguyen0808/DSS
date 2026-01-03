"""
Django Admin Configuration
"""
from django.contrib import admin
from .models import LoanPrediction


@admin.register(LoanPrediction)
class LoanPredictionAdmin(admin.ModelAdmin):
    """Admin interface cho LoanPrediction"""
    
    list_display = [
        'id', 'person_age', 'person_income', 'loan_amnt',
        'decision', 'probability_default', 'final_score', 'created_at'
    ]
    
    list_filter = ['decision', 'risk_level', 'created_at']
    
    search_fields = ['id', 'person_age', 'person_income']
    
    readonly_fields = [
        'ahp_score', 'probability_default', 'final_score',
        'decision', 'recommended_amount', 'created_at'
    ]
    
    fieldsets = (
        ('Thông tin khách hàng', {
            'fields': ('person_age', 'person_income', 'person_home_ownership', 'person_emp_length')
        }),
        ('Thông tin khoản vay', {
            'fields': ('loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate', 'loan_percent_income')
        }),
        ('Thông tin tín dụng', {
            'fields': ('cb_person_default_on_file', 'cb_person_cred_hist_length')
        }),
        ('Kết quả AHP', {
            'fields': ('ahp_score', 'ahp_weights', 'ahp_criteria_scores', 'consistency_ratio')
        }),
        ('Kết quả AI', {
            'fields': ('probability_default', 'model_accuracy')
        }),
        ('Quyết định DSS', {
            'fields': ('final_score', 'decision', 'risk_level', 'recommended_amount', 'max_loan_amount', 'explanation')
        }),
        ('Metadata', {
            'fields': ('created_at',)
        }),
    )
    
    date_hierarchy = 'created_at'
    
    def has_add_permission(self, request):
        """Không cho phép thêm mới từ admin"""
        return False
