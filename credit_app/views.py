"""
Credit App Views - Tích hợp AHP, AI và DSS
"""
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages
from django.db import models
import pandas as pd
import numpy as np
import pickle

from .models import LoanPrediction
from .ahp_engine import get_ahp_engine
from .dss_engine import get_dss_engine


# Global cache cho model
_model_cache = None


def get_or_load_model():
    """Load Random Forest model đã train sẵn + scaler"""
    global _model_cache
    
    if _model_cache is not None:
        return _model_cache
    
    try:
        with open('credit_risk_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        with open('model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # ⭐ Load scaler (QUAN TRỌNG)
        try:
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            print("✓ Scaler loaded!")
        except FileNotFoundError:
            scaler = None
            print("⚠️  No scaler found - predictions may be inaccurate!")
        
        _model_cache = {
            'model': model,
            'encoders': encoders,
            'feature_names': feature_names,
            'metadata': metadata,
            'scaler': scaler
        }
        
        # Print accuracy if available
        if isinstance(metadata, dict) and 'accuracy' in metadata:
            print(f"✓ Model loaded! Accuracy: {metadata['accuracy']:.2%}")
            if 'roc_auc' in metadata:
                print(f"✓ ROC-AUC: {metadata['roc_auc']:.4f}")
        else:
            print(f"✓ Model loaded successfully!")
        return _model_cache
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None


def index(request):
    """Trang chủ - Form nhập dữ liệu"""
    return render(request, 'index.html')


def predict(request):
    """Xử lý dự đoán và lưu kết quả"""
    if request.method != 'POST':
        return redirect('credit_app:index')
    
    try:
        # 1. Lấy dữ liệu từ form
        print(f"DEBUG - POST data: {request.POST}")
        
        person_age = int(request.POST.get('person_age'))
        
        # Thu nhập - lấy từ hidden field
        person_income_str = request.POST.get('person_income', '').strip()
        if not person_income_str:
            raise ValueError("Thu nhập không được để trống")
        person_income = int(person_income_str)
        
        person_emp_length = float(request.POST.get('person_emp_length'))
        person_home_ownership = request.POST.get('person_home_ownership')
        loan_intent = request.POST.get('loan_intent')
        loan_grade = request.POST.get('loan_grade')
        
        # Số tiền vay - lấy từ hidden field
        loan_amnt_str = request.POST.get('loan_amnt', '').strip()
        if not loan_amnt_str:
            raise ValueError("Số tiền vay không được để trống")
        loan_amnt = int(loan_amnt_str)
        
        loan_int_rate = float(request.POST.get('loan_int_rate'))
        cb_person_default_on_file = request.POST.get('cb_person_default_on_file')
        cb_person_cred_hist_length = int(request.POST.get('cb_person_cred_hist_length'))
        
        loan_percent_income = loan_amnt / person_income if person_income > 0 else 0
        
        # 2. Load ML model
        model_data = get_or_load_model()
        if model_data is None:
            messages.error(request, 'Không thể load model AI')
            return redirect('credit_app:index')
        
        # 3. Tính điểm AHP
        ahp_engine = get_ahp_engine()
        ahp_result = ahp_engine.calculate_score({
            'income': person_income,
            'employment_length': person_emp_length,
            'credit_history_length': cb_person_cred_hist_length,
            'loan_percent_income': loan_percent_income,
            'home_ownership': person_home_ownership
        })
        
        # 4. Dự đoán bằng AI - PHẢI GIỐNG CHÍNH XÁC train_model.py
        
        # ⭐ Step 1: Lưu original values TRƯỚC KHI transform
        loan_amnt_original = loan_amnt
        person_income_original = person_income
        
        # ⭐ Step 2: Apply transformations (GIỐNG train_model.py)
        loan_amnt_transformed = np.sqrt(max(0, loan_amnt))
        person_income_transformed = 1 / np.log(max(1, person_income) + 1)
        loan_percent_income_transformed = min(loan_percent_income, 1.0)
        
        # ⭐ Step 3: Create engineered features (GIỐNG train_model.py)
        # Age group
        if person_age <= 25:
            age_group = 'Young'
        elif person_age <= 35:
            age_group = 'Adult'
        elif person_age <= 50:
            age_group = 'Middle'
        else:
            age_group = 'Senior'
        
        # Income level (dựa trên original income)
        if person_income_original <= 30000:
            income_level = 'Low'
        elif person_income_original <= 60000:
            income_level = 'Medium'
        elif person_income_original <= 100000:
            income_level = 'High'
        else:
            income_level = 'VeryHigh'
        
        # Employment stability
        if person_emp_length <= 2:
            employment_stability = 'New'
        elif person_emp_length <= 5:
            employment_stability = 'Stable'
        elif person_emp_length <= 10:
            employment_stability = 'Experienced'
        else:
            employment_stability = 'Veteran'
        
        # Debt burden
        if loan_percent_income <= 0.2:
            debt_burden = 'Low'
        elif loan_percent_income <= 0.4:
            debt_burden = 'Medium'
        elif loan_percent_income <= 0.6:
            debt_burden = 'High'
        else:
            debt_burden = 'VeryHigh'
        
        # Interaction features
        income_to_loan = person_income_transformed / (loan_amnt_transformed + 1)
        credit_per_age = cb_person_cred_hist_length / (person_age + 1)
        
        # ⭐ Step 4: Tạo input data với ĐẦY ĐỦ 20 features
        input_data = {
            'person_age': person_age,
            'person_income': person_income_transformed,
            'person_home_ownership': person_home_ownership,
            'person_emp_length': person_emp_length,
            'loan_intent': loan_intent,
            'loan_grade': loan_grade,
            'loan_amnt': loan_amnt_transformed,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_percent_income_transformed,
            'cb_person_default_on_file': cb_person_default_on_file,
            'cb_person_cred_hist_length': cb_person_cred_hist_length,
            'loan_amnt_original': loan_amnt_original,
            'person_income_original': person_income_original,
            'age_group': age_group,
            'income_level': income_level,
            'employment_stability': employment_stability,
            'debt_burden': debt_burden,
            'income_to_loan': income_to_loan,
            'credit_per_age': credit_per_age
        }
        
        # Encode categorical variables
        for col, encoder in model_data['encoders'].items():
            if col in input_data:
                try:
                    input_data[col] = encoder.transform([str(input_data[col])])[0]
                except:
                    input_data[col] = 0
        
        # Create DataFrame
        df = pd.DataFrame([input_data])[model_data['feature_names']]
        
        # ⭐ Apply StandardScaler (nếu có)
        if model_data['scaler'] is not None:
            df_scaled = model_data['scaler'].transform(df)
            df = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)
        
        # Predict
        probability_default = float(model_data['model'].predict_proba(df)[0][1])
        
        # 5. Ra quyết định DSS
        dss_engine = get_dss_engine()
        dss_result = dss_engine.make_decision(
            ahp_score=ahp_result['ahp_score'],
            probability_default=probability_default,
            loan_amount=loan_amnt,
            income=person_income
        )
        
        # 6. Lưu vào database
        prediction = LoanPrediction.objects.create(
            # Thông tin khách hàng
            person_age=person_age,
            person_income=person_income,
            person_home_ownership=person_home_ownership,
            person_emp_length=person_emp_length,
            
            # Thông tin khoản vay
            loan_intent=loan_intent,
            loan_grade=loan_grade,
            loan_amnt=loan_amnt,
            loan_int_rate=loan_int_rate,
            loan_percent_income=loan_percent_income,
            
            # Thông tin tín dụng
            cb_person_default_on_file=cb_person_default_on_file,
            cb_person_cred_hist_length=cb_person_cred_hist_length,
            
            # Kết quả AHP
            ahp_score=ahp_result['ahp_score'],
            ahp_weights=ahp_result['weights'],
            ahp_criteria_scores=ahp_result['criteria_scores'],
            consistency_ratio=ahp_result['consistency_ratio'],
            
            # Kết quả AI
            probability_default=probability_default,
            model_accuracy=model_data['metadata']['accuracy'],
            
            # Kết quả DSS
            final_score=dss_result['final_score'],
            decision=dss_result['decision'],
            risk_level=dss_result['risk_level'],
            recommended_amount=dss_result['recommended_amount'],
            max_loan_amount=dss_result['max_loan_amount'],
            explanation=dss_result['explanation']
        )
        
        # 7. Hiển thị kết quả
        context = {
            'prediction_id': prediction.id,
            
            # Input data
            'input_data': {
                'Tuổi': person_age,
                'Thu nhập': f"{person_income:,}đ",
                'Số năm làm việc': person_emp_length,
                'Nhà ở': person_home_ownership,
                'Mục đích vay': loan_intent,
                'Xếp hạng': loan_grade,
                'Số tiền vay': f"{loan_amnt:,}đ",
                'Lãi suất': f"{loan_int_rate}%",
                'Từng vỡ nợ': cb_person_default_on_file,
                'Lịch sử TD': f"{cb_person_cred_hist_length} năm",
            },
            
            # AHP results
            'ahp_score': ahp_result['ahp_score'],
            'ahp_weights': ahp_result['weights'],
            'criteria_scores': ahp_result['criteria_scores'],
            'consistency_ratio': ahp_result['consistency_ratio'],
            'is_consistent': ahp_result['consistency_ratio'] < 0.1,
            
            # AI results
            'probability': probability_default * 100,
            'model_accuracy': model_data['metadata'].get('accuracy', 0.93) * 100 if isinstance(model_data['metadata'], dict) else 93.0,
            
            # DSS results
            'final_score': dss_result['final_score'],
            'decision': dss_result['decision'],
            'decision_class': dss_result['decision_class'],
            'risk_level': dss_result['risk_level'],
            'recommended_amount': f"{dss_result['recommended_amount']:,}đ",
            'max_loan_amount': f"{dss_result['max_loan_amount']:,}đ",
            'explanation': dss_result['explanation'],
        }
        
        # Render kết quả trong trang index.html thay vì result.html
        return render(request, 'index.html', context)
        
    except Exception as e:
        import traceback
        print(f"ERROR in predict(): {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        messages.error(request, f'Lỗi: {str(e)}')
        return redirect('credit_app:index')


def history(request):
    """Xem lịch sử dự đoán"""
    predictions = LoanPrediction.objects.all().order_by('-created_at')[:50]  # 50 gần nhất
    
    # Tính toán thống kê
    total_predictions = LoanPrediction.objects.count()
    approved_count = LoanPrediction.objects.filter(decision="DUYỆT").count()
    conditional_count = LoanPrediction.objects.filter(decision="DUYỆT CÓ ĐIỀU KIỆN").count()
    rejected_count = LoanPrediction.objects.filter(decision="TỪ CHỐI").count()
    
    # Tính xác suất trung bình
    avg_probability = LoanPrediction.objects.aggregate(
        avg_prob=models.Avg('probability_default')
    )['avg_prob'] or 0
    
    context = {
        'predictions': predictions,
        'total_predictions': total_predictions,
        'approved_count': approved_count,
        'conditional_count': conditional_count,
        'rejected_count': rejected_count,
        'avg_probability': avg_probability,
    }
    
    return render(request, 'history.html', context)


def about(request):
    """Trang giới thiệu về hệ thống DSS"""
    ahp_engine = get_ahp_engine()
    weights_info = ahp_engine.get_weights_info()
    
    model_data = get_or_load_model()
    
    context = {
        'ahp_info': weights_info,
        'model_info': model_data['metadata'] if model_data else None,
    }
    
    return render(request, 'about.html', context)
