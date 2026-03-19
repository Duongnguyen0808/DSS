import numpy as np


class AHPEngine:
    """
    AHP Engine - Cấu trúc 3 tầng:
      Tầng 1: Mục tiêu (Đánh giá rủi ro tín dụng)
      Tầng 2: Tiêu chí (5 tiêu chí) - ma trận 5×5
      Tầng 3: Phương án (Duyệt / Duyệt có điều kiện / Từ chối) - 5 ma trận 3×3
    """

    ALTERNATIVES = ['Duyệt', 'Duyệt có điều kiện', 'Từ chối']

    def __init__(self, custom_matrix=None):
        self.criteria = [
            'Thu nhập',
            'Nghề nghiệp',
            'Lịch sử tín dụng',
            'Dư nợ/Thu nhập',
            'Tài sản đảm bảo',
        ]
        self.is_custom = custom_matrix is not None

        if custom_matrix is not None:
            self.pairwise_matrix = np.array(custom_matrix)
        else:
            self.pairwise_matrix = np.array([
                [1,    2,    1/2,  3,    5  ],   # Thu nhập
                [1/2,  1,    1/3,  2,    4  ],   # Nghề nghiệp
                [2,    3,    1,    3,    5  ],   # Lịch sử tín dụng
                [1/3,  1/2,  1/3,  1,    3  ],   # Dư nợ/Thu nhập
                [1/5,  1/4,  1/5,  1/3,  1  ],   # Tài sản đảm bảo
            ])

        self.weights, self.cr = self._calculate_weights()

    # ══════════════════════════════════════════════
    # TẦNG 2 — Trọng số tiêu chí (ma trận 5×5)
    # ══════════════════════════════════════════════
    def _calculate_weights(self):
        n = len(self.pairwise_matrix)
        column_sums = self.pairwise_matrix.sum(axis=0)
        normalized_matrix = self.pairwise_matrix / column_sums
        weights = normalized_matrix.mean(axis=1)

        weighted_sum = self.pairwise_matrix @ weights
        consistency_vector = weighted_sum / weights
        lambda_max = consistency_vector.mean()
        ci = (lambda_max - n) / (n - 1)
        ri = 1.12
        cr = ci / ri

        self.column_sums = column_sums
        self.normalized_matrix = normalized_matrix
        self.weighted_sum = weighted_sum
        self.consistency_vector = consistency_vector
        self.lambda_max = lambda_max
        self.ci = ci
        self.ri = ri
        return weights, cr

    # ══════════════════════════════════════════════
    # TẦNG 3 — Ma trận so sánh phương án (3×3)
    # ══════════════════════════════════════════════
    def _get_alternative_matrix(self, criterion_score):
        """
        Xây dựng ma trận 3×3 so sánh cặp phương án dựa trên điểm tiêu chí (0-100).
        Hàng/Cột: [Duyệt (D), Duyệt có điều kiện (DC), Từ chối (TC)]
        """
        s = criterion_score

        if s >= 75:
            # Rất tốt → D >> DC > TC
            return np.array([
                [1,    4,    8   ],
                [1/4,  1,    3   ],
                [1/8,  1/3,  1   ],
            ])
        elif s >= 60:
            # Tốt → D > DC > TC
            return np.array([
                [1,    2,    5   ],
                [1/2,  1,    3   ],
                [1/5,  1/3,  1   ],
            ])
        elif s >= 45:
            # Trung bình → DC ≈ D > TC
            return np.array([
                [1,    1/2,  3   ],
                [2,    1,    4   ],
                [1/3,  1/4,  1   ],
            ])
        elif s >= 30:
            # Yếu → TC > DC > D
            return np.array([
                [1,    1/3,  1/5 ],
                [3,    1,    1/3 ],
                [5,    3,    1   ],
            ])
        else:
            # Rất yếu → TC >> DC > D
            return np.array([
                [1,    1/5,  1/8 ],
                [5,    1,    1/3 ],
                [8,    3,    1   ],
            ])

    def _priority_vector(self, matrix):
        """Tính priority vector và trả về cả normalized matrix + col_sums."""
        col_sums = matrix.sum(axis=0)
        normalized = matrix / col_sums
        priority = normalized.mean(axis=1)
        return priority, col_sums, normalized

    def _consistency_check_3x3(self, matrix, priority):
        """Kiểm tra CR cho ma trận 3×3 (RI = 0.58)."""
        n = 3
        weighted_sum = matrix @ priority
        consistency_vector = weighted_sum / priority
        lambda_max = consistency_vector.mean()
        ci = (lambda_max - n) / (n - 1)
        ri = 0.58
        cr = ci / ri
        return round(float(lambda_max), 4), round(float(ci), 4), round(float(cr), 4), consistency_vector

    def calculate_alternative_matrices(self, criteria_scores_dict):
        """
        Tính AHP Tầng 3: so sánh 3 phương án theo từng tiêu chí.

        Parameters:
            criteria_scores_dict: {'Thu nhập': 75, 'Nghề nghiệp': 55, ...}

        Returns:
            dict với ma trận, priority vector, điểm tổng hợp, xếp hạng.
        """
        alts = self.ALTERNATIVES
        alt_matrices, priority_vectors, consistency_data = [], [], []
        normalized_matrices, col_sums_list = [], []

        for criterion in self.criteria:
            score = criteria_scores_dict.get(criterion, 50)
            matrix = self._get_alternative_matrix(score)
            priority, col_sums, normalized = self._priority_vector(matrix)
            lmax, ci, cr, consistency_vector = self._consistency_check_3x3(matrix, priority)        

            alt_matrices.append(matrix.tolist())
            priority_vectors.append(priority.tolist())
            normalized_matrices.append([[round(v, 4) for v in row] for row in normalized.tolist()])
            col_sums_list.append([round(float(v), 4) for v in col_sums.tolist()])
            consistency_data.append({
                'criterion': criterion,
                'score': score,
                'lambda_max': lmax,
                'ci': ci,
                'cr': cr,
                'consistency_vector': [round(float(v), 4) for v in consistency_vector.tolist()],
                'is_consistent': cr < 0.1,
            })

        # Tổng hợp: Final_Score[j] = Σ_k (weight_k × priority_k[j])
        priority_matrix = np.array(priority_vectors)   # (5, 3)
        final_scores = self.weights @ priority_matrix   # (3,)

        ranking = sorted(
            [{'alternative': alts[j], 'score': round(float(final_scores[j]), 4)}
             for j in range(3)],
            key=lambda x: x['score'],
            reverse=True,
        )

        return {
            'alternatives': alts,
            'criteria': self.criteria,
            'alternative_matrices': alt_matrices,
            'normalized_matrices': normalized_matrices,
            'col_sums_list': col_sums_list,
            'priority_vectors': priority_vectors,
            'consistency_data': consistency_data,
            'final_scores': {alts[j]: round(float(final_scores[j]), 4) for j in range(3)},
            'final_scores_list': [round(float(s), 4) for s in final_scores],
            'ranking': ranking,
            'best_alternative': ranking[0]['alternative'],
            'weights': self.weights.tolist(),
        }

    # ══════════════════════════════════════════════
    # Tính điểm AHP cho khách hàng
    # ══════════════════════════════════════════════
    def calculate_score(self, customer_data):
        income_s     = self._score_income(customer_data['income'])
        emp_s        = self._score_employment(customer_data['employment_length'])
        credit_s     = self._score_credit_history(customer_data['credit_history_length'])
        debt_s       = self._score_debt_ratio(customer_data['loan_percent_income'])
        collateral_s = self._score_collateral(customer_data['home_ownership'])

        scores = np.array([income_s, emp_s, credit_s, debt_s, collateral_s])
        ahp_score = float(np.dot(self.weights, scores))

        criteria_scores = {
            'Thu nhập':         round(float(income_s), 2),
            'Nghề nghiệp':      round(float(emp_s), 2),
            'Lịch sử tín dụng': round(float(credit_s), 2),
            'Dư nợ/Thu nhập':   round(float(debt_s), 2),
            'Tài sản đảm bảo':  round(float(collateral_s), 2),
        }

        alt_result = self.calculate_alternative_matrices(criteria_scores)

        return {
            'ahp_score': round(ahp_score, 2),
            'criteria_scores': criteria_scores,
            'weights': {
                self.criteria[i]: round(float(self.weights[i]), 4)
                for i in range(len(self.criteria))
            },
            'consistency_ratio': round(float(self.cr), 4),
            'alternative_result': alt_result,
            'ahp_details': {
                'pairwise_matrix':   self.pairwise_matrix.tolist(),
                'normalized_matrix': self.normalized_matrix.tolist(),
                'column_sums':       self.column_sums.tolist(),
                'weighted_sum':      self.weighted_sum.tolist(),
                'consistency_vector': self.consistency_vector.tolist(),
                'lambda_max': round(float(self.lambda_max), 4),
                'ci':         round(float(self.ci), 4),
                'ri':         float(self.ri),
                'criteria':   self.criteria,
                'weights_array': self.weights.tolist(),
            },
        }

    # ── Hàm chấm điểm từng tiêu chí ──
    def _score_income(self, income):
        if income >= 200000: return 100
        elif income >= 150000: return 90
        elif income >= 100000: return 75
        elif income >= 60000: return 60
        elif income >= 40000: return 45
        elif income >= 30000: return 30
        else: return 15

    def _score_employment(self, years):
        if years >= 10: return 100
        elif years >= 7: return 85
        elif years >= 5: return 70
        elif years >= 3: return 55
        elif years >= 2: return 40
        elif years >= 1: return 25
        else: return 10

    def _score_credit_history(self, years):
        if years >= 15: return 100
        elif years >= 10: return 85
        elif years >= 7: return 70
        elif years >= 5: return 55
        elif years >= 3: return 40
        elif years >= 1: return 25
        else: return 10

    def _score_debt_ratio(self, ratio):
        if ratio <= 0.1: return 100
        elif ratio <= 0.2: return 85
        elif ratio <= 0.3: return 70
        elif ratio <= 0.4: return 50
        elif ratio <= 0.5: return 30
        elif ratio <= 0.6: return 15
        else: return 5

    def _score_collateral(self, home_ownership):
        return {
            'MORTGAGE': 100, 'OWN': 80, 'RENT': 40, 'OTHER': 30
        }.get(home_ownership.upper(), 50)

    def get_weights_info(self):
        return {
            'criteria': self.criteria,
            'weights': {
                self.criteria[i]: round(float(self.weights[i]), 4)
                for i in range(len(self.criteria))
            },
            'consistency_ratio': round(float(self.cr), 4),
            'is_consistent': self.cr < 0.1,
            'pairwise_matrix': self.pairwise_matrix.tolist(),
        }


# Singleton instance
_ahp_engine = None


def get_ahp_engine(custom_matrix=None):
    global _ahp_engine
    if custom_matrix is not None:
        return AHPEngine(custom_matrix=custom_matrix)
    if _ahp_engine is None:
        _ahp_engine = AHPEngine()
    return _ahp_engine
