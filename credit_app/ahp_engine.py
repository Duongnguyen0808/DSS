"""
AHP Engine - Analytic Hierarchy Process
Xác định trọng số các tiêu chí và tính điểm AHP cho khách hàng
"""

import numpy as np

class AHPEngine:
    """
    AHP Engine để tính trọng số tiêu chí và điểm AHP
    
    Tiêu chí đánh giá:
    1. Thu nhập (Income)
    2. Nghề nghiệp/Thâm niên (Employment)
    3. Lịch sử tín dụng (Credit History)
    4. Dư nợ/Thu nhập (Debt Ratio)
    5. Tài sản đảm bảo (Collateral)
    """
    
    def __init__(self):
        """Khởi tạo ma trận so sánh cặp và tính trọng số"""
        # Ma trận so sánh cặp (Pairwise Comparison Matrix)
        # Thang đo Saaty: 1=Equal, 3=Moderate, 5=Strong, 7=Very strong, 9=Extreme
        # Hàng i, cột j: tiêu chí i quan trọng hơn tiêu chí j bao nhiêu lần
        
        self.criteria = [
            'Thu nhập',          # Income
            'Nghề nghiệp',       # Employment  
            'Lịch sử tín dụng',  # Credit History
            'Dư nợ/Thu nhập',    # Debt Ratio
            'Tài sản đảm bảo'    # Collateral
        ]
        
        # Ma trận so sánh cặp 5x5
        # Thu nhập > Lịch sử TD > Nghề nghiệp > Dư nợ > Tài sản
        self.pairwise_matrix = np.array([
            [1,   2,   1/2, 3,   5],   # Thu nhập
            [1/2, 1,   1/3, 2,   4],   # Nghề nghiệp
            [2,   3,   1,   3,   5],   # Lịch sử tín dụng
            [1/3, 1/2, 1/3, 1,   3],   # Dư nợ/Thu nhập
            [1/5, 1/4, 1/5, 1/3, 1]    # Tài sản đảm bảo
        ])
        
        # Tính trọng số và kiểm tra tính nhất quán
        self.weights, self.cr = self._calculate_weights()
        
    def _calculate_weights(self):
        """
        Tính trọng số các tiêu chí bằng phương pháp Eigenvector
        
        Returns:
            weights (np.array): Trọng số chuẩn hóa
            cr (float): Consistency Ratio (CR < 0.1 là chấp nhận được)
        """
        n = len(self.pairwise_matrix)
        
        # Bước 1: Tính tổng mỗi cột
        column_sums = self.pairwise_matrix.sum(axis=0)
        
        # Bước 2: Chuẩn hóa ma trận (chia mỗi phần tử cho tổng cột)
        normalized_matrix = self.pairwise_matrix / column_sums
        
        # Bước 3: Tính trung bình mỗi hàng = trọng số
        weights = normalized_matrix.mean(axis=1)
        
        # Bước 4: Kiểm tra tính nhất quán (Consistency Check)
        # Tính λ_max (eigenvalue lớn nhất)
        weighted_sum = self.pairwise_matrix @ weights
        lambda_max = (weighted_sum / weights).mean()
        
        # Tính Consistency Index (CI)
        ci = (lambda_max - n) / (n - 1)
        
        # Random Index (RI) cho n=5
        ri = 1.12
        
        # Tính Consistency Ratio (CR)
        cr = ci / ri
        
        return weights, cr
    
    def calculate_score(self, customer_data):
        """
        Tính điểm AHP cho một khách hàng
        
        Parameters:
            customer_data (dict): {
                'income': int,
                'employment_length': float,
                'credit_history_length': int,
                'loan_percent_income': float,
                'home_ownership': str
            }
        
        Returns:
            dict: {
                'ahp_score': float (0-100),
                'criteria_scores': dict,
                'weights': dict,
                'consistency_ratio': float
            }
        """
        # Tính điểm từng tiêu chí (0-100)
        income_score = self._score_income(customer_data['income'])
        employment_score = self._score_employment(customer_data['employment_length'])
        credit_score = self._score_credit_history(customer_data['credit_history_length'])
        debt_score = self._score_debt_ratio(customer_data['loan_percent_income'])
        collateral_score = self._score_collateral(customer_data['home_ownership'])
        
        # Vector điểm các tiêu chí
        scores = np.array([
            income_score,
            employment_score,
            credit_score,
            debt_score,
            collateral_score
        ])
        
        # Tính điểm tổng hợp AHP = tổng (trọng số × điểm)
        ahp_score = np.dot(self.weights, scores)
        
        return {
            'ahp_score': round(float(ahp_score), 2),
            'criteria_scores': {
                'Thu nhập': round(float(income_score), 2),
                'Nghề nghiệp': round(float(employment_score), 2),
                'Lịch sử tín dụng': round(float(credit_score), 2),
                'Dư nợ/Thu nhập': round(float(debt_score), 2),
                'Tài sản đảm bảo': round(float(collateral_score), 2)
            },
            'weights': {
                self.criteria[i]: round(float(self.weights[i]), 4)
                for i in range(len(self.criteria))
            },
            'consistency_ratio': round(float(self.cr), 4)
        }
    
    def _score_income(self, income):
        """Điểm thu nhập (0-100)"""
        if income >= 200000:
            return 100
        elif income >= 150000:
            return 90
        elif income >= 100000:
            return 75
        elif income >= 60000:
            return 60
        elif income >= 40000:
            return 45
        elif income >= 30000:
            return 30
        else:
            return 15
    
    def _score_employment(self, years):
        """Điểm thâm niên nghề nghiệp (0-100)"""
        if years >= 10:
            return 100
        elif years >= 7:
            return 85
        elif years >= 5:
            return 70
        elif years >= 3:
            return 55
        elif years >= 2:
            return 40
        elif years >= 1:
            return 25
        else:
            return 10
    
    def _score_credit_history(self, years):
        """Điểm lịch sử tín dụng (0-100)"""
        if years >= 15:
            return 100
        elif years >= 10:
            return 85
        elif years >= 7:
            return 70
        elif years >= 5:
            return 55
        elif years >= 3:
            return 40
        elif years >= 1:
            return 25
        else:
            return 10
    
    def _score_debt_ratio(self, ratio):
        """Điểm tỷ lệ dư nợ/thu nhập (0-100) - càng thấp càng tốt"""
        if ratio <= 0.1:
            return 100
        elif ratio <= 0.2:
            return 85
        elif ratio <= 0.3:
            return 70
        elif ratio <= 0.4:
            return 50
        elif ratio <= 0.5:
            return 30
        elif ratio <= 0.6:
            return 15
        else:
            return 5
    
    def _score_collateral(self, home_ownership):
        """Điểm tài sản đảm bảo (0-100)"""
        scores = {
            'MORTGAGE': 100,  # Có thế chấp nhà
            'OWN': 80,        # Sở hữu nhà
            'RENT': 40,       # Thuê nhà
            'OTHER': 30       # Khác
        }
        return scores.get(home_ownership.upper(), 50)
    
    def get_weights_info(self):
        """Trả về thông tin về trọng số và tính nhất quán"""
        return {
            'criteria': self.criteria,
            'weights': {
                self.criteria[i]: round(float(self.weights[i]), 4)
                for i in range(len(self.criteria))
            },
            'consistency_ratio': round(float(self.cr), 4),
            'is_consistent': self.cr < 0.1,
            'pairwise_matrix': self.pairwise_matrix.tolist()
        }


# Singleton instance
_ahp_engine = None

def get_ahp_engine():
    """Lấy hoặc tạo AHP engine instance"""
    global _ahp_engine
    if _ahp_engine is None:
        _ahp_engine = AHPEngine()
    return _ahp_engine
