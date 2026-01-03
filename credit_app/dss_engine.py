"""
DSS Engine - Decision Support System
Kết hợp AHP và AI để ra quyết định cho vay
"""

class DSSEngine:
    """
    DSS Engine kết hợp AHP và Machine Learning
    
    Logic quyết định:
    - Tính Final_Score = AHP_Score × (1 - Probability_of_Default)
    - Probability < 0.20 → Duyệt
    - 0.20 ≤ Probability < 0.35 → Duyệt có điều kiện  
    - Probability ≥ 0.35 → Không duyệt
    """
    
    def __init__(self):
        self.thresholds = {
            'approve': 0.20,      # P < 0.20
            'conditional': 0.35    # 0.20 <= P < 0.35
        }
    
    def make_decision(self, ahp_score, probability_default, loan_amount, income):
        """
        Ra quyết định cho vay
        
        Parameters:
            ahp_score (float): Điểm AHP (0-100)
            probability_default (float): Xác suất nợ xấu (0-1)
            loan_amount (int): Số tiền vay yêu cầu
            income (int): Thu nhập
        
        Returns:
            dict: {
                'decision': str,
                'decision_class': str,
                'risk_level': str,
                'final_score': float,
                'recommended_amount': int,
                'max_loan_amount': int,
                'explanation': str
            }
        """
        # Tính Final Score
        final_score = ahp_score * (1 - probability_default)
        
        # Tính hạn mức vay tối đa dựa trên thu nhập và rủi ro
        max_loan_amount = self._calculate_max_loan(income, probability_default, ahp_score)
        
        # Ra quyết định
        if probability_default < self.thresholds['approve']:
            decision = "DUYỆT"
            decision_class = "success"
            risk_level = "Rủi ro thấp"
            
            # Duyệt toàn bộ nếu trong hạn mức
            if loan_amount <= max_loan_amount:
                recommended_amount = loan_amount
                explanation = f"Khách hàng có hồ sơ tốt. Duyệt toàn bộ {loan_amount:,}đ yêu cầu."
            else:
                recommended_amount = max_loan_amount
                explanation = f"Khách hàng tốt nhưng số tiền vay vượt hạn mức an toàn. Đề xuất duyệt {max_loan_amount:,}đ."
                
        elif probability_default < self.thresholds['conditional']:
            decision = "DUYỆT CÓ ĐIỀU KIỆN"
            decision_class = "warning"
            risk_level = "Rủi ro trung bình"
            
            # Giảm 30% hạn mức cho trường hợp có điều kiện
            safe_amount = int(max_loan_amount * 0.7)
            
            if loan_amount <= safe_amount:
                recommended_amount = loan_amount
                explanation = f"Duyệt {loan_amount:,}đ với điều kiện: Thế chấp tài sản hoặc người bảo lãnh."
            else:
                recommended_amount = safe_amount
                explanation = f"Đề xuất duyệt {safe_amount:,}đ (giảm từ {loan_amount:,}đ) với điều kiện: Thế chấp tài sản và kiểm tra định kỳ."
                
        else:
            decision = "TỪ CHỐI"
            decision_class = "danger"
            risk_level = "Rủi ro cao"
            recommended_amount = 0
            explanation = f"Xác suất vỡ nợ cao ({probability_default*100:.1f}%). Không đủ điều kiện cho vay tại thời điểm này. Đề xuất: Cải thiện hồ sơ tín dụng và tái nộp sau 6 tháng."
        
        return {
            'decision': decision,
            'decision_class': decision_class,
            'risk_level': risk_level,
            'final_score': round(final_score, 2),
            'recommended_amount': recommended_amount,
            'max_loan_amount': max_loan_amount,
            'explanation': explanation,
            'probability_percent': round(probability_default * 100, 2)
        }
    
    def _calculate_max_loan(self, income, probability, ahp_score):
        """
        Tính hạn mức vay tối đa
        
        Công thức: Max_Loan = Income × DTI_Ratio × Risk_Adjustment
        - DTI_Ratio: Debt-to-Income ratio tối đa (30-50%)
        - Risk_Adjustment: Điều chỉnh theo rủi ro và AHP score
        """
        # DTI ratio cơ bản (40% thu nhập)
        base_dti = 0.40
        
        # Điều chỉnh theo xác suất nợ xấu
        if probability < 0.15:
            risk_factor = 1.25  # Tăng 25% cho khách hàng rất tốt
        elif probability < 0.25:
            risk_factor = 1.0
        elif probability < 0.35:
            risk_factor = 0.75  # Giảm 25%
        else:
            risk_factor = 0.5   # Giảm 50% cho rủi ro cao
        
        # Điều chỉnh theo AHP score
        if ahp_score >= 80:
            ahp_factor = 1.2
        elif ahp_score >= 60:
            ahp_factor = 1.0
        elif ahp_score >= 40:
            ahp_factor = 0.8
        else:
            ahp_factor = 0.6
        
        # Tính hạn mức
        max_loan = income * base_dti * risk_factor * ahp_factor
        
        # Làm tròn đến 1 triệu
        max_loan = round(max_loan / 1000000) * 1000000
        
        # Giới hạn tối thiểu và tối đa
        max_loan = max(5000000, min(max_loan, income * 5))  # Min 5M, Max 5x thu nhập
        
        return int(max_loan)


# Singleton instance
_dss_engine = None

def get_dss_engine():
    """Lấy hoặc tạo DSS engine instance"""
    global _dss_engine
    if _dss_engine is None:
        _dss_engine = DSSEngine()
    return _dss_engine
