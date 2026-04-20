"""
Django Models - Lưu trữ lịch sử dự đoán
"""
from django.db import models
from django.utils import timezone


class LoanPrediction(models.Model):
    """Model lưu trữ lịch sử dự đoán cho vay"""
    
    # Thông tin khách hàng
    person_name = models.CharField(max_length=100, verbose_name="Họ và tên", default="Khách hàng", null=True, blank=True)
    person_age = models.IntegerField(verbose_name="Tuổi")
    person_income = models.IntegerField(verbose_name="Thu nhập")
    person_home_ownership = models.CharField(max_length=20, verbose_name="Nhà ở")
    person_emp_length = models.FloatField(verbose_name="Số năm làm việc")
    
    # Thông tin khoản vay
    loan_intent = models.CharField(max_length=50, verbose_name="Mục đích vay")
    loan_grade = models.CharField(max_length=10, verbose_name="Xếp hạng")
    loan_amnt = models.IntegerField(verbose_name="Số tiền vay yêu cầu")
    loan_int_rate = models.FloatField(verbose_name="Lãi suất")
    loan_percent_income = models.FloatField(verbose_name="Tỷ lệ vay/Thu nhập")
    
    # Thông tin tín dụng
    cb_person_default_on_file = models.CharField(max_length=5, verbose_name="Từng vỡ nợ")
    cb_person_cred_hist_length = models.IntegerField(verbose_name="Lịch sử tín dụng (năm)")
    
    # Kết quả AHP
    ahp_score = models.FloatField(verbose_name="Điểm AHP")
    ahp_weights = models.JSONField(verbose_name="Trọng số AHP", null=True)
    ahp_criteria_scores = models.JSONField(verbose_name="Điểm từng tiêu chí", null=True)
    consistency_ratio = models.FloatField(verbose_name="CR", null=True)
    
    # Kết quả AI
    probability_default = models.FloatField(verbose_name="Xác suất nợ xấu")
    model_accuracy = models.FloatField(verbose_name="Độ chính xác model", null=True)
    
    # Kết quả DSS
    final_score = models.FloatField(verbose_name="Điểm cuối cùng")
    decision = models.CharField(max_length=50, verbose_name="Quyết định")
    risk_level = models.CharField(max_length=50, verbose_name="Mức độ rủi ro")
    recommended_amount = models.IntegerField(verbose_name="Số tiền đề xuất")
    max_loan_amount = models.IntegerField(verbose_name="Hạn mức tối đa")
    explanation = models.TextField(verbose_name="Giải thích")
    
    # Metadata
    created_at = models.DateTimeField(default=timezone.now, verbose_name="Thời gian dự đoán")
    
    class Meta:
        db_table = 'loan_predictions'
        verbose_name = 'Dự đoán cho vay'
        verbose_name_plural = 'Lịch sử dự đoán cho vay'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Dự đoán #{self.id} - {self.decision} - {self.created_at.strftime('%d/%m/%Y %H:%M')}"


class CreditRiskDatasetRow(models.Model):
    """Model lưu raw dataset (credit_risk_dataset.csv)"""

    person_age = models.IntegerField(verbose_name="Tuổi")
    person_income = models.IntegerField(verbose_name="Thu nhập")
    person_home_ownership = models.CharField(max_length=20, verbose_name="Nhà ở")
    person_emp_length = models.FloatField(null=True, blank=True, verbose_name="Số năm làm việc")

    loan_intent = models.CharField(max_length=50, verbose_name="Mục đích vay")
    loan_grade = models.CharField(max_length=10, verbose_name="Xếp hạng")
    loan_amnt = models.IntegerField(verbose_name="Số tiền vay")
    loan_int_rate = models.FloatField(null=True, blank=True, verbose_name="Lãi suất")
    loan_percent_income = models.FloatField(null=True, blank=True, verbose_name="Tỷ lệ vay/Thu nhập")

    cb_person_default_on_file = models.CharField(max_length=5, verbose_name="Từng vỡ nợ")
    cb_person_cred_hist_length = models.IntegerField(verbose_name="Lịch sử tín dụng (năm)")

    loan_status = models.IntegerField(verbose_name="Trạng thái khoản vay")  # 0/1

    created_at = models.DateTimeField(default=timezone.now, verbose_name="Thời gian import")

    class Meta:
        db_table = 'credit_risk_dataset'
        verbose_name = 'Dòng dữ liệu Credit Risk'
        verbose_name_plural = 'Dataset Credit Risk'
        indexes = [
            models.Index(fields=['loan_status']),
            models.Index(fields=['loan_grade']),
            models.Index(fields=['loan_intent']),
        ]

    def __str__(self):
        return f"DatasetRow #{self.id} - status={self.loan_status}"
