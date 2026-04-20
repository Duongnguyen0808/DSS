"""
DSS Engine - Decision Support System
Kết hợp AHP và AI để ra quyết định cho vay
"""

import math

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
        """
        # Tính Final Score
        final_score = ahp_score * (1 - probability_default)
        probability_percent = round(probability_default * 100, 2)
        
        # Tính hạn mức vay tối đa dựa trên thu nhập và rủi ro
        max_loan_amount = self._calculate_max_loan(income, probability_default, ahp_score)

        # Một số chỉ số giải thích để khách hàng hiểu vì sao được/không được vay.
        loan_to_income_ratio = (loan_amount / income) if income > 0 else 0
        loan_to_income_percent = loan_to_income_ratio * 100
        current_profile_ratio = (max_loan_amount / income) if income > 0 else 0
        required_income_current_profile = (
            math.ceil(loan_amount / current_profile_ratio)
            if current_profile_ratio > 0 else loan_amount
        )
        # Kịch bản tốt nhất theo policy hiện tại: risk_multiple=4.0 và ahp_factor=1.1 => 4.4x thu nhập.
        best_case_ratio = 4.4
        required_income_best_case = math.ceil(loan_amount / best_case_ratio)
        
        def format_vnd(amount):
            return f"{amount:,}"
            
        loan_amount_fmt = format_vnd(loan_amount)
        max_loan_amount_fmt = format_vnd(max_loan_amount)
        required_income_current_profile_fmt = format_vnd(required_income_current_profile)
        required_income_best_case_fmt = format_vnd(required_income_best_case)
        
        # Ra quyết định
        if probability_default < self.thresholds['approve']:
            decision = "DUYỆT"
            decision_class = "success"
            risk_level = "Rủi ro thấp"
            
            # Duyệt toàn bộ nếu trong hạn mức
            if loan_amount <= max_loan_amount:
                recommended_amount = loan_amount
                explanation = (
                    f"✅ Đề xuất: Phê duyệt toàn bộ khoản vay\n\n"
                    f"Dựa trên phân tích chuyên sâu của hệ thống:\n"
                    f"• Điểm tín nhiệm (AHP): {ahp_score:.2f}/100\n"
                    f"• Xác suất rủi ro (AI): {probability_percent}%\n"
                    f"• Mức độ rủi ro: Thấp.\n\n"
                    f"Hồ sơ của khách hàng rất tốt. Hệ thống tính toán hạn mức tối đa an toàn có thể cấp là {max_loan_amount_fmt} VNĐ.\n"
                    f"Khoản vay yêu cầu ({loan_amount_fmt} VNĐ) nằm trong giới hạn an toàn. Có thể tiến hành giải ngân ngay."
                )
            else:
                recommended_amount = max_loan_amount
                explanation = (
                    f"⚠️ Đề xuất: Phê duyệt một phần khoản vay\n\n"
                    f"Dựa trên phân tích chuyên sâu của hệ thống:\n"
                    f"• Điểm tín nhiệm (AHP): {ahp_score:.2f}/100\n"
                    f"• Xác suất rủi ro (AI): {probability_percent}%\n"
                    f"• Mức độ rủi ro: Thấp.\n\n"
                    f"Mặc dù hồ sơ tốt, nhưng số tiền khách hàng yêu cầu vay ({loan_amount_fmt} VNĐ) vượt mức an toàn dựa trên năng lực tài chính hiện tại.\n"
                    f"Để đảm bảo an toàn tín dụng, đề xuất giảm mức giải ngân xuống tối đa {max_loan_amount_fmt} VNĐ."
                )
                
        elif probability_default < self.thresholds['conditional']:
            decision = "DUYỆT CÓ ĐIỀU KIỆN"
            decision_class = "warning"
            risk_level = "Rủi ro trung bình"
            
            # Giảm 30% hạn mức cho trường hợp có điều kiện
            safe_amount = int(max_loan_amount * 0.7)
            safe_amount_fmt = format_vnd(safe_amount)
            
            if loan_amount <= safe_amount:
                recommended_amount = loan_amount
                explanation = (
                    f"⚠️ Đề xuất: Phê duyệt kèm điều kiện bổ sung\n\n"
                    f"Dựa trên phân tích chuyên sâu của hệ thống:\n"
                    f"• Điểm tín nhiệm (AHP): {ahp_score:.2f}/100\n"
                    f"• Xác suất rủi ro (AI): {probability_percent}%\n"
                    f"• Mức độ rủi ro: Trung bình.\n\n"
                    f"Hồ sơ có một số điểm lưu ý, tiềm ẩn mức độ rủi ro nhất định. Khoản vay yêu cầu là {loan_amount_fmt} VNĐ.\n"
                    f"Có thể duyệt cấp tín dụng với số tiền này nhưng BẮT BUỘC phải yêu cầu thế chấp tài sản có giá trị tương đương hoặc có người bảo lãnh uy tín."
                )
            else:
                recommended_amount = safe_amount
                explanation = (
                    f"⚠️ Đề xuất: Cắt giảm hạn mức & Yêu cầu điều kiện\n\n"
                    f"Dựa trên phân tích chuyên sâu của hệ thống:\n"
                    f"• Điểm tín nhiệm (AHP): {ahp_score:.2f}/100\n"
                    f"• Xác suất rủi ro (AI): {probability_percent}%\n"
                    f"• Mức độ rủi ro: Trung bình.\n\n"
                    f"Số tiền yêu cầu ({loan_amount_fmt} VNĐ) là quá cao so với mức độ rủi ro hiện tại.\n"
                    f"Phương án 1 (nếu giữ nguyên hồ sơ hiện tại): chỉ nên vay tối đa {safe_amount_fmt} VNĐ.\n"
                    f"Phương án 2 (muốn vay gần mức yêu cầu): cần giảm xác suất rủi ro xuống dưới 35%, nâng điểm hồ sơ (AHP) lên từ 60+ và chứng minh dòng tiền ổn định hơn."
                )
                
        else:
            decision = "TỪ CHỐI"
            decision_class = "danger"
            risk_level = "Rủi ro cao"
            # Vẫn đưa ra hạn mức gợi ý để người dùng biết mốc có thể cân nhắc vay lại.
            provisional_safe_amount = round((income * 0.3) / 1000000) * 1000000
            provisional_safe_amount = max(5000000, provisional_safe_amount)
            recommended_amount = int(min(max_loan_amount, provisional_safe_amount))
            recommended_amount_fmt = format_vnd(recommended_amount)
            explanation = (
                f"❌ Đề xuất: Từ chối hồ sơ\n\n"
                f"Vì sao chưa thể duyệt khoản {loan_amount_fmt} VNĐ:\n"
                f"• Xác suất rủi ro AI: {probability_percent}% (cao hơn ngưỡng 35%).\n"
                f"• Tỷ lệ khoản vay/thu nhập hiện tại: {loan_to_income_ratio:.2f} ({loan_to_income_percent:.1f}%).\n"
                f"• Hồ sơ hiện tại chỉ phù hợp hạn mức tối đa khoảng {max_loan_amount_fmt} VNĐ.\n\n"
                f"Để có thể vay gần mức {loan_amount_fmt} VNĐ, khách hàng cần cải thiện:\n"
                f"1) Giảm rủi ro xuống dưới 35% (ưu tiên dưới 25%).\n"
                f"2) Nâng chất lượng hồ sơ (AHP) lên ít nhất 60 điểm.\n"
                f"3) Chứng minh thu nhập cao hơn: tối thiểu khoảng {required_income_current_profile_fmt} VNĐ/năm nếu các yếu tố rủi ro giữ như hiện tại; hoặc khoảng {required_income_best_case_fmt} VNĐ/năm trong kịch bản hồ sơ tốt.\n"
                f"4) Bổ sung tài sản bảo đảm/đồng trả nợ để giảm áp lực rủi ro.\n\n"
                f"Nếu chưa cải thiện ngay, hệ thống khuyến nghị chỉ nên vay mức an toàn khoảng {recommended_amount_fmt} VNĐ ở lần nộp lại gần nhất."
            )
        
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

        Chính sách mới dùng bội số thu nhập năm để trực quan hơn:
        Max_Loan = Income_Annual × Risk_Multiple × AHP_Adjustment

        - Risk_Multiple: theo mức rủi ro tổng quát
        - AHP_Adjustment: tinh chỉnh theo chất lượng hồ sơ
        """
        # Bội số thu nhập năm theo xác suất rủi ro.
        if probability < 0.15:
            risk_multiple = 4.0
        elif probability < 0.25:
            risk_multiple = 3.0
        elif probability < 0.35:
            risk_multiple = 2.0
        else:
            risk_multiple = 1.2

        # Điều chỉnh nhẹ theo AHP score.
        if ahp_score >= 80:
            ahp_factor = 1.1
        elif ahp_score >= 60:
            ahp_factor = 1.0
        elif ahp_score >= 40:
            ahp_factor = 0.9
        else:
            ahp_factor = 0.8

        # Tính hạn mức theo thu nhập năm.
        max_loan = income * risk_multiple * ahp_factor

        # Làm tròn đến 1 triệu
        max_loan = round(max_loan / 1000000) * 1000000

        # Giới hạn tối thiểu và tối đa
        max_loan = max(5000000, min(max_loan, income * 5))  # Min 5M, Max 5x thu nhập năm

        return int(max_loan)


# Singleton instance
_dss_engine = None

def get_dss_engine():
    """Lấy hoặc tạo DSS engine instance"""
    global _dss_engine
    if _dss_engine is None:
        _dss_engine = DSSEngine()
    return _dss_engine
