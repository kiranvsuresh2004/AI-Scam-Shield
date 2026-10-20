import json
import logging
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse, urlsplit

import joblib
import pandas as pd
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.models import Count, Q
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.urls import reverse
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST

from url_feature_extractor import URL_FEATURE_COLUMNS, extract_url_features

from .file_processing import extract_text_from_uploaded_file, validate_file_upload
from .models import DetectionHistory

MODEL_DIR = Path(settings.BASE_DIR) / 'model'
logger = logging.getLogger(__name__)

SPAM_KEYWORDS = [
    'free', 'winner', 'claim', 'click', 'urgent', 'limited', 'prize',
    'verify', 'bonus', 'reward', 'cash', 'account', 'password', 'update',
    'immediate', 'secure', 'alert', 'lottery', 'gift', 'locked', 'suspended',
    'bank', 'invoice', 'payment', 'suspicious', 'refund', 'complaint'
]

SAFE_KEYWORDS = [
    'hello', 'meeting', 'schedule', 'project', 'invoice', 'receipt',
    'thanks', 'team', 'status', 'document', 'report', 'approved',
    'appointment', 'confirmation', 'follow-up', 'order', 'customer'
]

MODEL_FILES = {
    'sms_model': 'sms_svm_model.pkl',
    'sms_vectorizer': 'sms_vectorizer.pkl',
    'email_model': 'email_svm_model.pkl',
    'email_vectorizer': 'email_vectorizer.pkl',
    'url_model': 'random_forest_url_model.pkl',
}
GUEST_TRIAL_SESSION_KEY = 'guest_detection_trials'
GUEST_TRIAL_TYPES = {'sms', 'email', 'url'}


class ModelUnavailableError(RuntimeError):
    pass


@lru_cache(maxsize=None)
def load_model_artifact(model_name):
    file_name = MODEL_FILES.get(model_name)
    if file_name is None:
        raise ValueError('Invalid model artifact name.')

    model_path = MODEL_DIR / file_name
    if not model_path.is_file():
        raise ModelUnavailableError(
            'A required detection model is missing. Restore the model files or retrain the model.'
        )
    try:
        return joblib.load(str(model_path))
    except Exception as exc:
        logger.exception('Unable to load detection model artifact %s.', model_path.name)
        raise ModelUnavailableError(
            'A required detection model could not be loaded. Check the model files or retrain the model.'
        ) from exc


def guest_trial_used(request, detection_type):
    if request.user.is_authenticated or detection_type not in GUEST_TRIAL_TYPES:
        return False
    used_trials = request.session.get(GUEST_TRIAL_SESSION_KEY, {})
    return bool(used_trials.get(detection_type))


def record_guest_trial(request, detection_type):
    if request.user.is_authenticated or detection_type not in GUEST_TRIAL_TYPES:
        return
    used_trials = request.session.get(GUEST_TRIAL_SESSION_KEY, {})
    used_trials[detection_type] = True
    request.session[GUEST_TRIAL_SESSION_KEY] = used_trials


def normalize_prediction(raw_value):
    value = str(raw_value).strip().lower()
    if value in {'1', 'spam', 'scam', 'phishing', 'fraud', 'yes', 'true'}:
        return 1
    if value in {'0', 'ham', 'safe', 'legit', 'genuine', 'no', 'false'}:
        return 0
    try:
        return int(float(raw_value))
    except (TypeError, ValueError):
        return 1 if value else 0


def clean_email_text(text):
    text = str(text or '')
    text = text.lower()
    text = ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in text)
    text = ' '.join(text.split())
    return text


def model_confidence(model, features, prediction=None):
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(features)[0]
        classes = list(getattr(model, 'classes_', []))
        if prediction is not None and prediction in classes:
            index = classes.index(prediction)
            return round(float(probs[index]) * 100, 2)
        return round(float(max(probs)) * 100, 2)
    if hasattr(model, 'decision_function'):
        decision = model.decision_function(features)[0]
        confidence = 1 / (1 + math.exp(-abs(float(decision))))
        return round(confidence * 100, 2)
    return 85.0


def model_scam_probability(model, features, prediction):
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)[0]
        classes = list(getattr(model, 'classes_', []))
        if 1 in classes:
            return round(float(probabilities[classes.index(1)]) * 100, 2)
        return 100.0 if prediction == 1 else 0.0

    if hasattr(model, 'decision_function'):
        decision = float(model.decision_function(features)[0])
        classes = list(getattr(model, 'classes_', []))
        if len(classes) == 2:
            margin = max(-60.0, min(60.0, decision))
            probability_of_second_class = 1 / (1 + math.exp(-margin))
            second_class_is_scam = normalize_prediction(classes[1]) == 1
            probability = probability_of_second_class if second_class_is_scam else 1 - probability_of_second_class
            return round(probability * 100, 2)
    return 100.0 if prediction == 1 else 0.0


def find_keyword_matches(text, keywords):
    lowered = str(text or '').lower()
    matches = []
    for keyword in keywords:
        if keyword.lower() in lowered:
            matches.append(keyword)
    return matches


def get_risk_status(is_scam, confidence_pct):
    if not is_scam:
        return 'NO RISK', 0
    if confidence_pct >= 75:
        return 'CRITICAL', int(round(confidence_pct))
    if confidence_pct >= 50:
        return 'HIGH', int(round(confidence_pct))
    if confidence_pct >= 25:
        return 'MEDIUM', int(round(confidence_pct))
    return 'LOW', int(round(confidence_pct))


def risk_level_for_score(risk_score):
    if risk_score >= 75:
        return 'CRITICAL'
    if risk_score >= 50:
        return 'HIGH'
    if risk_score >= 25:
        return 'MEDIUM'
    if risk_score > 0:
        return 'LOW'
    return 'NO RISK'


def analyze_message_context(text):
    lowered = str(text or '').lower()
    signals = []
    contextual_score = 0

    sentence_text = re.sub(r'\brs\.(?=\s*\d)', 'rs', lowered)
    sentences = re.split(r'(?<=[.!?])\s+|[\r\n]+', sentence_text)
    credential_terms = (
        r'(?:password|passcode|otp|one[- ]time (?:password|code)|pin|security code|'
        r'cvv|cvc|card details|banking details|login credentials)'
    )
    request_verbs = (
        r'(?:enter(?:ing)?|type|provid(?:e|ing)|shar(?:e|ing)|send(?:ing)?|'
        r'submit(?:ting)?|confirm|reply with|email us|text us|give us|supply|'
        r'forward|disclose|tell us|update)'
    )
    credential_request = False
    for sentence in sentences:
        is_refusal = re.search(
            r'\b(?:never|not|don\'t|do not|won\'t|will not|nobody will)\b'
            r'.{0,60}\b(?:ask|request|share|send|enter|provide|give|submit|disclose)\b',
            sentence,
        )
        asks_for_credentials = re.search(rf'\b{request_verbs}\b.{{0,55}}\b{credential_terms}\b|\b{credential_terms}\b.{{0,45}}\b{request_verbs}\b', sentence)
        if asks_for_credentials and not is_refusal:
            credential_request = True
            break

    urgency = bool(re.search(
        r'\b(?:urgent|immediately|right away|action required|final warning|'
        r'within \d+ hours?|expires? soon|before (?:it|this) expires?)\b',
        lowered,
    ))
    account_threat = bool(re.search(
        r'\b(?:account|card|access|service|profile)\b.{0,55}\b'
        r'(?:(?:will be|may be|has been|is|was)\s+)?'
        r'(?:suspend\w*|block\w*|lock\w*|clos\w*|disabl\w*|terminat\w*|'
        r'compromis\w*|restrict\w*)\b|'
        r'\b(?:suspended|compromised|locked|blocked|disabled)\b',
        lowered,
    ))
    verification_request = bool(re.search(
        r'\b(?:verify|confirm|validate|update|restore|unlock|secure|reactivate)\b'
        r'.{0,55}\b(?:your\s+)?(?:account|bank(?:ing)? details|identity|card|'
        r'payment|profile|login|access)\b|'
        r'\bclick\b.{0,30}\b(?:here|link)\b.{0,40}\b(?:verify|confirm|update)\b',
        lowered,
    ))
    link_action = bool(re.search(
        r'\b(?:click|follow|open|visit|use)\b.{0,35}\b(?:link|url|here|below|page)\b',
        lowered,
    ))
    urls = re.findall(r'(?:https?://|www\.)[^\s<>()\[\]{}"\']+', text, flags=re.IGNORECASE)
    suspicious_url = False
    for url in urls:
        candidate = url.rstrip('.,!?;:')
        hostname = (urlparse(candidate if '://' in candidate else f'http://{candidate}').hostname or '').lower()
        labels = hostname.split('.')
        suspicious_url = suspicious_url or (
            '@' in candidate
            or bool(re.fullmatch(r'\d{1,3}(?:\.\d{1,3}){3}', hostname))
            or 'xn--' in hostname
            or any(shortener in hostname for shortener in ('bit.ly', 'tinyurl', 't.co', 'ow.ly', 'rb.gy'))
            or (labels and labels[-1] in {'tk', 'ml', 'ga', 'cf', 'gq', 'top', 'xyz', 'click', 'zip'})
            or bool(re.search(r'(?:bank|paypal|microsoft|apple)[-_](?:secure|verify|login|account)', hostname))
        )

    investment_claim = False
    prize_fee_request = False
    job_fee_request = False
    money_request = r'\b(?:pay|send|transfer|deposit|fee|charge|upfront|registration|processing|onboarding)\b'
    for sentence in sentences:
        investment = bool(re.search(r'\b(?:invest(?:ment)?|capital)\b', sentence))
        return_claim = bool(re.search(
            r'\b(?:earn|make|profit|return|yield|grow|double|triple|multiply|payout)\b',
            sentence,
        ))
        money_values = [
            float(value.replace(',', ''))
            for value in re.findall(
                r'(?:\brs\.?|\binr\b|₹|\$|\busd\b)\s*([\d,]+(?:\.\d+)?)',
                sentence,
            )
        ]
        high_return = len(money_values) >= 2 and max(money_values) >= 2 * min(money_values)
        guaranteed_return = bool(re.search(
            r'\b(?:guaranteed?|assured|risk[- ]free|no risk)\b',
            sentence,
        )) and not bool(re.search(r'\b(?:not|never|no)\s+(?:be\s+)?guaranteed?\b', sentence))
        rapid_return = bool(re.search(
            r'\b(?:within|in)\s+\d+\s+(?:minutes?|hours?|days?)\b',
            sentence,
        ))
        if investment and return_claim and (high_return or guaranteed_return or rapid_return):
            investment_claim = True

        prize_offer = bool(re.search(
            r'\b(?:won|winner|lottery|prize|jackpot|reward|payout|winnings|award|draw)\b',
            sentence,
        ))
        fee_to_claim = bool(re.search(
            r'\b(?:pay|send|transfer|deposit|fee|tax|charge|processing charge|release fee)\b'
            r'.{0,55}\b(?:claim|release|collect|receive|prize|winnings|payout)\b|'
            r'\b(?:claim|release|collect|receive|prize|winnings|payout|award)\b'
            r'.{0,55}\b(?:fee|tax|charge|deposit|payment)\b',
            sentence,
        ))
        if prize_offer and fee_to_claim:
            prize_fee_request = True

    job_offer = bool(re.search(
        r'\b(?:job|work from home|from home|remote role|remote work|position|'
        r'employment|hiring|interview|onboarding|work opportunity)\b',
        lowered,
    ))
    fee_for_job = bool(re.search(
        rf'{money_request}.{{0,100}}\b(?:start|begin|work|job|position|interview|'
        r'training|earn|earnings|opportunity|role)\b|'
        r'\b(?:start|begin|work|job|position|interview|training|earn|earnings|'
        rf'opportunity|role)\b.{{0,100}}{money_request}',
        lowered,
    ))
    if job_offer and fee_for_job:
        job_fee_request = True

    if credential_request:
        signals.append('request for passwords or sensitive credentials')
        contextual_score += 45
    if urgency:
        signals.append('urgent or time-pressured language')
        contextual_score += 12
    if account_threat:
        signals.append('account suspension or compromise threat')
        contextual_score += 25
    if verification_request:
        signals.append('request to verify or update account details')
        contextual_score += 18
    if suspicious_url:
        signals.append('suspicious URL or domain')
        contextual_score += 35
    if urls and (credential_request or verification_request or urgency):
        signals.append('link paired with a sensitive action')
        contextual_score += 15
    elif link_action and (credential_request or verification_request or account_threat):
        signals.append('link instruction paired with an account or credential request')
        contextual_score += 12
    if urgency and credential_request:
        contextual_score += 20
    if account_threat and (verification_request or credential_request):
        contextual_score += 20
    if suspicious_url and (credential_request or verification_request or link_action):
        contextual_score += 20
    if investment_claim:
        signals.append('unrealistic or guaranteed investment-return promise')
        contextual_score += 65
    if job_fee_request:
        signals.append('advance payment requested for a job or earnings offer')
        contextual_score += 65
    if prize_fee_request:
        signals.append('fee requested to claim a prize or lottery payout')
        contextual_score += 65

    contextual_score = min(contextual_score, 100)
    return signals, contextual_score


def assess_email_risk(text, phishing_probability):
    signals, contextual_score = analyze_message_context(text)
    risk_score = int(round(0.2 * phishing_probability + 0.8 * contextual_score))
    return signals, risk_score, risk_level_for_score(risk_score)


def assess_sms_risk(text, spam_probability):
    signals, contextual_score = analyze_message_context(text)
    risk_score = int(round(0.2 * spam_probability + 0.8 * contextual_score))
    return signals, risk_score, contextual_score


def prediction_name_for(detection_type, is_scam):
    if detection_type == 'sms':
        return 'Spam' if is_scam else 'Genuine'
    return 'Phishing' if is_scam else 'Genuine'


def infer_uploaded_content_type(extension, extracted_text):
    if extension == '.eml':
        return 'email', 'Email'

    lines = [line.strip() for line in extracted_text.splitlines() if line.strip()]
    header_names = {
        match.lower()
        for match in re.findall(r'(?im)^(from|to|subject|date):\s*\S.*$', extracted_text)
    }
    if 'subject' in header_names and len(header_names.intersection({'from', 'to'})) > 0:
        return 'email', 'Email'
    if re.search(r'(?i)\b(?:dear|hello)\s+.{1,60},', extracted_text) and re.search(
        r'(?i)\b(?:regards|sincerely|best wishes|thank you)\b', extracted_text
    ):
        return 'email', 'Email'

    if extension == '.txt':
        return 'sms', 'SMS-like text (plain-text fallback)'
    if len(extracted_text) <= 500 and len(lines) <= 5:
        return 'sms', 'SMS-like text'
    return 'email', 'Email (document fallback)'


def analyze_uploaded_file(uploaded_file, user=None):
    file_name, extension = validate_file_upload(uploaded_file)
    extracted = extract_text_from_uploaded_file(uploaded_file, file_name)
    warnings = list(extracted.get('warnings', []))
    ocr_blocker = next(
        (
            warning for warning in warnings
            if 'install tesseract' in warning.lower() or 'ocr is unavailable' in warning.lower()
        ),
        None,
    )
    if ocr_blocker:
        raise ValueError(ocr_blocker)

    if extension == '.eml':
        body = extracted.get('body', '').strip()
        if not body:
            raise ValueError('The email contains no readable message body; headers alone are not enough to scan it.')
        joined_text = '\n'.join(filter(None, [extracted.get('subject', '').strip(), body]))
    else:
        joined_text = extracted.get('text', '').strip()

    if not joined_text:
        actionable_warning = next(
            (
                warning for warning in warnings
                if any(term in warning.lower() for term in ('ocr', 'pymupdf', 'pillow'))
            ),
            None,
        )
        if actionable_warning:
            raise ValueError(actionable_warning)
        raise ValueError('No readable text could be extracted from this file. Upload a clearer file and try again.')

    detection_type, content_type = infer_uploaded_content_type(extension, joined_text)
    if content_type.endswith('fallback)'):
        warnings.append(
            'The message type could not be identified with certainty; the displayed fallback analysis type was used.'
        )

    result = run_detection_for_type(detection_type, joined_text)
    urls = list(dict.fromkeys(extracted.get('urls', [])))
    suspicious_urls = []
    url_scores = []
    if urls:
        for url in urls[:6]:
            try:
                url_result = run_detection_for_type('url', url, None)
            except ValueError:
                continue
            if url_result.get('is_scam'):
                url_scores.append(int(url_result.get('risk_score', 0)))
                suspicious_urls.append(url)
        if url_scores:
            warnings.append(f'{len(url_scores)} suspicious URL(s) were detected in the uploaded file.')

    risk_score = result['risk_score']
    signals = list(result['detected_indicators'])
    maximum_url_score = max(url_scores, default=0)
    if urls:
        risk_score = min(100, int(round((risk_score * 0.75) + (maximum_url_score * 0.25))))
    is_scam = result['is_scam'] or maximum_url_score >= 50
    if is_scam:
        risk_score = max(25, risk_score)
    risk_level = risk_level_for_score(risk_score)
    if suspicious_urls:
        signals.append('suspicious URLs detected in the extracted content')

    if warnings:
        warnings = list(dict.fromkeys(warnings))
        signals.extend(warning for warning in warnings if warning)

    if is_scam and not signals:
        signals.append(f'file text resembles known {detection_type} scam patterns in the trained model')
    prediction = int(is_scam)
    prediction_label = 'SCAM' if is_scam else 'SAFE'
    prediction_name = prediction_name_for(detection_type, is_scam)
    result.update({
        'result': prediction_name,
        'prediction': prediction,
        'prediction_name': prediction_name,
        'prediction_label': prediction_label,
        'is_scam': is_scam,
        'risk_score': risk_score,
        'risk_level': risk_level,
        'entered_text': joined_text,
        'entered_url': None,
        'file_name': file_name,
        'file_extension': extension,
        'content_type': content_type,
        'extraction_status': (
            'OCR text extracted'
            if extracted.get('ocr_used')
            else 'Text extracted'
        ),
        'ocr_status': (
            'Succeeded'
            if extracted.get('ocr_used')
            else 'Unavailable' if any('ocr' in warning.lower() for warning in warnings)
            else 'Not needed'
        ),
        'sender': extracted.get('sender'),
        'subject': extracted.get('subject'),
        'message': 'This file contains suspicious content and may be fraudulent.' if is_scam else 'This file appears safe based on the uploaded content and extracted text.',
        'detected_indicators': signals[:12],
        'suspicious_urls': suspicious_urls,
        'warnings': warnings,
        'risk_elements': {'suspicious': signals[:8], 'safe': []},
        'disclaimer': 'Risk score combines the selected text classifier, contextual indicators, and URL analysis when available; it is not a confidence percentage.',
    })
    result.pop('confidence', None)
    result.pop('probability', None)

    if user is not None and user.is_authenticated:
        record = DetectionHistory.objects.create(
            user=user,
            detection_type='file',
            input_text=(extracted.get('subject') or joined_text[:180]),
            file_name=file_name,
            file_extension=extension,
            prediction=prediction_name,
            risk_score=risk_score,
            risk_level=risk_level,
            risk_signals=json.dumps(result['detected_indicators']),
        )
        result['detected_at'] = timezone.localtime(record.created_at).strftime('%d %b %Y, %I:%M %p')

    return result


def run_detection_for_type(detection_type, text, user=None):
    detection_type = str(detection_type or '').lower().strip()
    text = str(text or '').strip()
    if not detection_type or not text:
        raise ValueError('Please enter some text or a URL.')

    prediction_label = 'SAFE'
    confidence_pct = 0.0
    risk_level = 'LOW'
    detected_indicators = []
    risk_score = 0
    prediction_name = 'Genuine'
    is_scam = False
    probability = 0

    if detection_type == 'sms':
        sms_model = load_model_artifact('sms_model')
        sms_vectorizer = load_model_artifact('sms_vectorizer')
        features = sms_vectorizer.transform([text])
        raw_prediction = sms_model.predict(features)[0]
        model_prediction = normalize_prediction(raw_prediction)
        confidence_pct = model_confidence(sms_model, features, raw_prediction)
        scam_probability = model_scam_probability(sms_model, features, model_prediction)
        detected_indicators, risk_score, contextual_score = assess_sms_risk(text, scam_probability)
        is_scam = model_prediction == 1 or contextual_score >= 40
        if is_scam:
            risk_score = max(25, risk_score)
        risk_level = risk_level_for_score(risk_score)
        if is_scam and model_prediction != 1:
            confidence_pct = max(confidence_pct, contextual_score, scam_probability)
        if is_scam and not detected_indicators:
            detected_indicators.append('message resembles known spam patterns in the trained model')
        prediction = int(is_scam)
        prediction_label = 'SCAM' if is_scam else 'SAFE'
        prediction_name = prediction_name_for('sms', is_scam)
        probability = int(round(max(scam_probability, contextual_score))) if is_scam else 0

    elif detection_type == 'email':
        cleaned_email = clean_email_text(text)
        email_model = load_model_artifact('email_model')
        email_vectorizer = load_model_artifact('email_vectorizer')
        features = email_vectorizer.transform([cleaned_email])
        raw_prediction = email_model.predict(features)[0]
        prediction = normalize_prediction(raw_prediction)
        confidence_pct = model_confidence(email_model, features, prediction)
        phishing_probability = model_scam_probability(email_model, features, prediction)
        detected_indicators, risk_score, risk_level = assess_email_risk(text, phishing_probability)
        _, contextual_score = analyze_message_context(text)
        is_scam = prediction == 1 or contextual_score >= 40
        if is_scam:
            risk_score = max(25, risk_score)
        risk_level = risk_level_for_score(risk_score)
        if is_scam and prediction != 1:
            confidence_pct = max(confidence_pct, contextual_score, phishing_probability)
        if is_scam and not detected_indicators:
            detected_indicators.append('email text resembles known phishing patterns in the trained model')
        prediction = int(is_scam)
        prediction_name = prediction_name_for('email', is_scam)
        prediction_label = 'SCAM' if is_scam else 'SAFE'
        probability = int(round(max(phishing_probability, contextual_score))) if is_scam else 0

    elif detection_type == 'url':
        parsed_url = urlparse(text)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError('Please enter a valid full URL, for example https://example.com')
        url_model = load_model_artifact('url_model')
        features = extract_url_features(text)
        feature_df = pd.DataFrame([features], columns=URL_FEATURE_COLUMNS)
        url_probabilities = url_model.predict_proba(feature_df)[0]
        raw_prediction = url_model.predict(feature_df)[0]
        prediction = normalize_prediction(raw_prediction)
        confidence = float(max(url_probabilities))
        confidence_pct = round(confidence * 100, 2)
        is_scam = prediction == 0
        prediction_name = prediction_name_for('url', is_scam)
        prediction_label = 'SCAM' if is_scam else 'SAFE'
        detected_indicators = [feature_name for feature_name, value in zip(URL_FEATURE_COLUMNS, features) if value == -1]
        risk_percent = round((1 - confidence) * 100, 2) if prediction == 1 else round(confidence * 100, 2)
        if risk_percent >= 75:
            risk_level = 'CRITICAL'
        elif risk_percent >= 50:
            risk_level = 'HIGH'
        elif risk_percent >= 25:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        risk_score = int(round(risk_percent))
        probability = int(round(confidence_pct)) if is_scam else 0

    else:
        raise ValueError('Invalid detection type.')

    safe_keywords = find_keyword_matches(text, SAFE_KEYWORDS)
    suspected_keywords = (
        detected_indicators
        if detection_type in {'sms', 'email'}
        else find_keyword_matches(text, SPAM_KEYWORDS)
    )

    if user is not None and user.is_authenticated:
        DetectionHistory.objects.create(
            user=user,
            detection_type=detection_type,
            input_text=text,
            prediction=prediction_name,
            risk_score=risk_score,
            risk_level=risk_level,
            risk_signals=json.dumps(detected_indicators),
        )

    result = {
        'result': prediction_name,
        'prediction': prediction,
        'prediction_name': prediction_name,
        'prediction_label': prediction_label,
        'is_scam': is_scam,
        'risk_score': risk_score,
        'probability': probability,
        'confidence': confidence_pct,
        'risk_level': risk_level,
        'entered_text': text,
        'entered_url': text if detection_type == 'url' else None,
        'message': 'This content appears suspicious and may be fraudulent.' if is_scam else 'This content appears safe based on the trained AI model.',
        'detected_indicators': detected_indicators,
        'risk_elements': {
            'suspicious': suspected_keywords[:8],
            'safe': safe_keywords[:8],
        },
        'disclaimer': 'AI Scam Shield provides an ML-based assessment and cannot guarantee that a URL is safe.',
    }

    return result


def home_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request, 'detector/index.html')


def register_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip().lower()
        password = request.POST.get('password', '')
        confirm_password = request.POST.get('confirm_password', '')

        if not username or len(username) < 3 or len(username) > 30:
            messages.error(request, 'Username must be 3-30 characters using letters, numbers, or underscores.')
        elif not re.fullmatch(r'[A-Za-z0-9_]+', username):
            messages.error(request, 'Username can only contain letters, numbers, and underscores.')
        elif User.objects.filter(username__iexact=username).exists():
            messages.error(request, 'That username is already registered.')
        elif not email or '@' not in email:
            messages.error(request, 'Enter a valid email address.')
        elif User.objects.filter(email__iexact=email).exists():
            messages.error(request, 'That email address is already registered.')
        elif len(password) < 8 or len(password) > 128:
            messages.error(request, 'Password must be between 8 and 128 characters.')
        elif password != confirm_password:
            messages.error(request, 'Passwords do not match.')
        else:
            User.objects.create_user(username=username, email=email, password=password)
            return redirect('login')

    return render(request, 'detector/register.html')


def login_view(request):
    next_url = request.POST.get('next') or request.GET.get('next')
    if not url_has_allowed_host_and_scheme(
        next_url,
        allowed_hosts={request.get_host()},
        require_https=request.is_secure(),
    ) or urlsplit(next_url).path == reverse('login'):
        next_url = ''

    if request.user.is_authenticated:
        return redirect(next_url or 'dashboard')

    if request.method == 'POST':
        identity = request.POST.get('identity', '').strip()
        password = request.POST.get('password', '')

        user = User.objects.filter(username__iexact=identity).first() or User.objects.filter(email__iexact=identity).first()
        authenticated_user = authenticate(request, username=user.username, password=password) if user else None
        if authenticated_user is not None:
            auth_login(request, authenticated_user)
            return redirect(next_url or 'dashboard')

        if user is not None and user.check_password(password):
            auth_login(request, user)
            return redirect(next_url or 'dashboard')

        messages.error(request, 'Invalid username/email or password.')

    return render(request, 'detector/login.html', {'next': next_url})


@require_POST
@login_required
def logout_view(request):
    auth_logout(request)
    return redirect('login')


@login_required
def dashboard_view(request):
    stats = DetectionHistory.objects.filter(user=request.user).aggregate(
        total_checks=Count('id'),
        sms_checks=Count('id', filter=Q(detection_type='sms')),
        email_checks=Count('id', filter=Q(detection_type='email')),
        url_checks=Count('id', filter=Q(detection_type='url')),
        file_checks=Count('id', filter=Q(detection_type='file')),
        spam_checks=Count('id', filter=Q(prediction__in=['Spam', 'Phishing'])),
        genuine_checks=Count('id', filter=Q(prediction='Genuine')),
    )
    return render(request, 'detector/dashboard.html', {'stats': stats, 'username': request.user.username})


def _render_detection_page(request, template_name, detection_type, text=None):
    result = None
    if request.method == 'POST':
        text_value = request.POST.get('text', '') if text is None else text
        if text_value:
            if guest_trial_used(request, detection_type):
                return redirect('register')
            try:
                result = run_detection_for_type(detection_type, text_value, request.user if request.user.is_authenticated else None)
                record_guest_trial(request, detection_type)
            except ModelUnavailableError as exc:
                messages.error(request, str(exc))
            except ValueError as exc:
                messages.error(request, str(exc))
            except Exception:
                logger.exception('Detection failed for type %s.', detection_type)
                messages.error(request, 'Detection could not be completed due to an internal error. Please try again.')
    return render(request, template_name, {'result': result, 'text': (text or ''), 'detection_type': detection_type})


@login_required
def file_scan_view(request):
    result = None
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')
        if uploaded_file is None:
            messages.error(request, 'Please choose a file to upload.')
        else:
            try:
                result = analyze_uploaded_file(uploaded_file, request.user)
            except ModelUnavailableError as exc:
                messages.error(request, str(exc))
            except ValueError as exc:
                messages.error(request, str(exc))
            except Exception:
                logger.exception('Uploaded-file detection failed.')
                messages.error(request, 'File detection could not be completed due to an internal error. Please try again.')
    return render(request, 'detector/file_upload.html', {'result': result})


def sms_detection(request):
    return _render_detection_page(request, 'detector/sms.html', 'sms')


def email_detection(request):
    return _render_detection_page(request, 'detector/email.html', 'email')


def url_detection(request):
    return _render_detection_page(request, 'detector/url.html', 'url')


@login_required
def authenticated_sms_detection(request):
    return _render_detection_page(request, 'smsd.html', 'sms')


@login_required
def authenticated_email_detection(request):
    return _render_detection_page(request, 'emaild.html', 'email')


@login_required
def authenticated_url_detection(request):
    return _render_detection_page(request, 'urld.html', 'url')


@login_required
def history_view(request):
    history_records = []
    for record in DetectionHistory.objects.filter(user=request.user):
        try:
            risk_signals = json.loads(record.risk_signals or '[]')
        except json.JSONDecodeError:
            risk_signals = []
        record_input = record.file_name or record.input_text
        if record.detection_type == 'file':
            record_input = record.file_name or record.input_text
        history_records.append({
            'id': record.id,
            'detection_type': record.detection_type,
            'input_text': record_input,
            'prediction': record.prediction,
            'risk_score': record.risk_score,
            'risk_signals': risk_signals,
            'created_at': record.created_at,
            'formatted_date': timezone.localtime(record.created_at).strftime('%d %b %Y, %I:%M %p'),
        })
    return render(request, 'detector/history.html', {'history_records': history_records})


@require_POST
@login_required
def delete_history(request, id):
    record = DetectionHistory.objects.filter(id=id, user=request.user).first()
    if record is None:
        messages.error(request, 'That record could not be found.')
        return redirect('history')
    record.delete()
    return redirect('history')


@require_POST
@login_required
def clear_history(request):
    DetectionHistory.objects.filter(user=request.user).delete()
    return redirect('history')


def predict_json(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method.'}, status=405)

    try:
        payload = json.loads(request.body.decode('utf-8')) if request.body else {}
    except json.JSONDecodeError:
        payload = request.POST.dict()

    detection_type = str(payload.get('type') or payload.get('detection_type') or '').lower().strip()
    text = str(payload.get('text') or payload.get('input_text') or '').strip()

    if guest_trial_used(request, detection_type):
        return JsonResponse(
            {
                'error': 'Your free trial for this detection type has been used. Register to continue.',
                'register_url': reverse('register'),
            },
            status=403,
        )

    try:
        result = run_detection_for_type(detection_type, text, request.user if request.user.is_authenticated else None)
    except ModelUnavailableError as exc:
        return JsonResponse({'error': str(exc)}, status=503)
    except ValueError as exc:
        return JsonResponse({'error': str(exc)}, status=400)
    except Exception:
        logger.exception('JSON detection failed for type %s.', detection_type)
        return JsonResponse(
            {'error': 'Detection could not be completed due to an internal error. Please try again.'},
            status=500,
        )

    record_guest_trial(request, detection_type)
    return JsonResponse(result)


home = home_view
register = register_view
login = login_view
logout = logout_view
dashboard = dashboard_view
sms = sms_detection
email = email_detection
url = url_detection
history = history_view
delete_history_record = delete_history
clear_history_view = clear_history
predict = predict_json
file_scan = file_scan_view
