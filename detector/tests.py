import io
import json
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase
from django.urls import reverse
from django.utils import timezone

from .models import DetectionHistory
from .views import (
    ModelUnavailableError,
    analyze_message_context,
    run_detection_for_type,
)


SPAM_EXAMPLES = [
    'Invest Rs. 1,000 today and earn Rs. 10,000 within 24 hours. Guaranteed profit!',
    'Congratulations! You have won a lottery. Pay a processing fee to claim your prize.',
    'Earn Rs. 3,000 daily from home. Pay a registration fee to start working.',
    'Your account will be blocked immediately. Click this suspicious link and enter your password.',
    'Send us your OTP and PIN to unlock your bank account.',
]

GENUINE_EXAMPLES = [
    'Your salary has been credited to your bank account successfully.',
    'Your monthly bank statement is available in the official banking application.',
    'Our project meeting is scheduled for tomorrow at 10 AM.',
    'Never share your OTP or password with anyone.',
    'Your invoice has been received. We will process the payment next week.',
]


class MessageDetectionTests(TestCase):
    def assert_message_classification(self, detection_type, text, is_scam):
        result = run_detection_for_type(detection_type, text)
        self.assertEqual(result['is_scam'], is_scam, text)
        self.assertEqual(result['prediction'], int(is_scam), text)
        self.assertEqual(result['prediction_label'], 'SCAM' if is_scam else 'SAFE', text)
        self.assertEqual(result['prediction_name'], 'Spam' if detection_type == 'sms' and is_scam else (
            'Phishing' if detection_type == 'email' and is_scam else 'Genuine'
        ), text)
        if is_scam:
            self.assertGreaterEqual(result['risk_score'], 25, text)
            self.assertTrue(result['detected_indicators'], text)
        else:
            self.assertLess(result['risk_score'], 25, text)

    def test_required_spam_examples_for_sms(self):
        for text in SPAM_EXAMPLES:
            with self.subTest(text=text):
                self.assert_message_classification('sms', text, True)

    def test_required_genuine_examples_for_sms(self):
        for text in GENUINE_EXAMPLES:
            with self.subTest(text=text):
                self.assert_message_classification('sms', text, False)

    def test_required_spam_examples_for_email(self):
        for text in SPAM_EXAMPLES:
            with self.subTest(text=text):
                self.assert_message_classification('email', text, True)

    def test_required_genuine_examples_for_email(self):
        for text in GENUINE_EXAMPLES:
            with self.subTest(text=text):
                self.assert_message_classification('email', text, False)

    def test_scam_context_generalizes_to_unfamiliar_wording(self):
        messages = [
            'Place capital with us and we guarantee it will multiply tenfold by tomorrow.',
            'Your ticket was selected in a draw. Transfer a release fee to collect your winnings.',
            'The remote position pays Rs. 4,000 daily. Send a refundable onboarding deposit before your interview.',
        ]
        for detection_type in ('sms', 'email'):
            for text in messages:
                with self.subTest(detection_type=detection_type, text=text):
                    self.assert_message_classification(detection_type, text, True)

    def test_single_financial_keywords_do_not_create_contextual_risk(self):
        messages = [
            'The bank is open today.',
            'Your account settings are unchanged.',
            'The payment is listed in the ledger.',
            'We received an offer from the vendor.',
            'Money is discussed in the report.',
            'The investment committee meets next week.',
        ]
        for text in messages:
            with self.subTest(text=text):
                signals, contextual_score = analyze_message_context(text)
                self.assertEqual(signals, [])
                self.assertEqual(contextual_score, 0)

    def test_sms_and_email_scans_keep_authenticated_history(self):
        user = User.objects.create_user(username='scanner', password='test-password-123')
        client = Client()
        self.assertTrue(client.login(username='scanner', password='test-password-123'))

        response = client.post('/sms/', {'text': SPAM_EXAMPLES[0]})

        self.assertEqual(response.status_code, 200)
        record = DetectionHistory.objects.get(user=user, detection_type='sms')
        self.assertEqual(record.prediction, 'Spam')
        self.assertGreater(record.risk_score, 0)

    def test_history_displays_saved_timestamp_in_indian_time(self):
        user = User.objects.create_user(username='timecheck', password='test-password-123')
        record = DetectionHistory.objects.create(
            user=user,
            detection_type='sms',
            input_text='A test message',
            prediction='Genuine',
            risk_score=0,
        )
        client = Client()
        self.assertTrue(client.login(username='timecheck', password='test-password-123'))

        response = client.get('/history/')

        expected_timestamp = timezone.localtime(record.created_at).strftime('%d %b %Y, %I:%M %p')
        self.assertContains(response, expected_timestamp)
        self.assertEqual(timezone.get_default_timezone_name(), 'Asia/Kolkata')

    def test_plain_text_scam_upload_uses_sms_model_and_saves_file_history(self):
        user = User.objects.create_user(username='filecheck', password='test-password-123')
        client = Client()
        self.assertTrue(client.login(username='filecheck', password='test-password-123'))
        uploaded_file = SimpleUploadedFile(
            'message.txt',
            b'Send us your OTP and PIN to unlock your bank account.',
            content_type='text/plain',
        )

        response = client.post('/file-scan/', {'file': uploaded_file})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'SMS-like text (plain-text fallback)')
        self.assertContains(response, 'Spam')
        self.assertNotIn('confidence', response.context['result'])
        self.assertNotIn('probability', response.context['result'])
        record = DetectionHistory.objects.get(user=user, detection_type='file')
        self.assertEqual(record.prediction, 'Spam')
        self.assertEqual(record.file_name, 'message.txt')
        self.assertTrue(record.created_at)

    def test_email_upload_uses_email_model_and_needs_a_message_body(self):
        user = User.objects.create_user(username='emailcheck', password='test-password-123')
        client = Client()
        self.assertTrue(client.login(username='emailcheck', password='test-password-123'))
        email_file = SimpleUploadedFile(
            'notice.eml',
            (
                b'From: security@example.invalid\r\n'
                b'To: customer@example.invalid\r\n'
                b'Subject: Account verification required\r\n\r\n'
                b'Your account will be blocked immediately. Send us your OTP and password.'
            ),
            content_type='message/rfc822',
        )

        response = client.post('/file-scan/', {'file': email_file})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Detected Content Type')
        self.assertContains(response, 'Phishing')
        record = DetectionHistory.objects.get(user=user, detection_type='file')
        self.assertEqual(record.prediction, 'Phishing')

        legitimate_email = SimpleUploadedFile(
            'statement.eml',
            (
                b'From: statements@example.invalid\r\n'
                b'To: customer@example.invalid\r\n'
                b'Subject: Monthly statement ready\r\n\r\n'
                b'Your monthly bank statement is available in the official banking application. '
                b'Open the app to review it. We will never ask you to share your password or OTP.'
            ),
            content_type='message/rfc822',
        )
        response = client.post('/file-scan/', {'file': legitimate_email})
        self.assertContains(response, 'Genuine')
        self.assertEqual(
            DetectionHistory.objects.get(
                user=user,
                detection_type='file',
                file_name='statement.eml',
            ).prediction,
            'Genuine',
        )

        header_only = SimpleUploadedFile(
            'headers.eml',
            b'From: security@example.invalid\r\nSubject: Account alert\r\n\r\n',
            content_type='message/rfc822',
        )
        response = client.post('/file-scan/', {'file': header_only})
        self.assertContains(response, 'headers alone are not enough to scan it')
        self.assertEqual(DetectionHistory.objects.filter(user=user, detection_type='file').count(), 2)

    def test_text_pdf_upload_extracts_content_before_prediction(self):
        import pymupdf

        user = User.objects.create_user(username='pdfcheck', password='test-password-123')
        client = Client()
        self.assertTrue(client.login(username='pdfcheck', password='test-password-123'))
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text(
            (72, 72),
            'Your account will be blocked immediately. Send us your OTP and password.',
        )
        pdf_file = SimpleUploadedFile(
            'notice.pdf',
            document.tobytes(),
            content_type='application/pdf',
        )
        document.close()

        response = client.post('/file-scan/', {'file': pdf_file})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Text extracted')
        self.assertContains(response, 'Spam')
        self.assertTrue(DetectionHistory.objects.filter(
            user=user,
            detection_type='file',
            file_name='notice.pdf',
        ).exists())

    def test_image_and_scanned_pdf_without_ocr_fail_without_saving_a_prediction(self):
        import pymupdf
        from PIL import Image, ImageDraw

        user = User.objects.create_user(username='ocrcheck', password='test-password-123')
        client = Client()
        self.assertTrue(client.login(username='ocrcheck', password='test-password-123'))
        image = Image.new('RGB', (600, 120), 'white')
        ImageDraw.Draw(image).text((10, 10), 'Your account is blocked. Send your OTP now.', fill='black')
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        png_content = image_bytes.getvalue()

        document = pymupdf.open()
        page = document.new_page()
        page.insert_image(pymupdf.Rect(30, 30, 570, 140), stream=png_content)
        scanned_pdf = document.tobytes()
        document.close()

        with patch('detector.file_processing.pytesseract', None):
            image_response = client.post(
                '/file-scan/',
                {'file': SimpleUploadedFile('message.png', png_content, content_type='image/png')},
            )
            pdf_response = client.post(
                '/file-scan/',
                {'file': SimpleUploadedFile('scan.pdf', scanned_pdf, content_type='application/pdf')},
            )

        self.assertContains(image_response, 'Tesseract')
        self.assertContains(pdf_response, 'Tesseract')
        self.assertFalse(DetectionHistory.objects.filter(user=user, detection_type='file').exists())

    def test_empty_and_corrupt_file_uploads_do_not_create_predictions(self):
        user = User.objects.create_user(username='badfile', password='test-password-123')
        client = Client()
        self.assertTrue(client.login(username='badfile', password='test-password-123'))

        empty_file = SimpleUploadedFile('empty.txt', b'', content_type='text/plain')
        response = client.post('/file-scan/', {'file': empty_file})
        self.assertContains(response, 'empty')

        corrupt_pdf = SimpleUploadedFile('broken.pdf', b'not a PDF', content_type='application/pdf')
        response = client.post('/file-scan/', {'file': corrupt_pdf})
        self.assertContains(response, 'corrupted')
        self.assertFalse(DetectionHistory.objects.filter(user=user, detection_type='file').exists())

    def test_guest_gets_one_web_trial_per_detection_type(self):
        client = Client()
        guest_scans = [
            ('/sms/', SPAM_EXAMPLES[0]),
            ('/email/', GENUINE_EXAMPLES[1]),
            ('/url/', 'https://example.com'),
        ]

        for path, text in guest_scans:
            with self.subTest(path=path):
                response = client.post(path, {'text': text})
                self.assertEqual(response.status_code, 200)

        for path, text in guest_scans:
            with self.subTest(path=path):
                response = client.post(path, {'text': text})
                self.assertRedirects(response, '/register/', fetch_redirect_response=False)

    def test_guest_json_trial_requires_registration_after_one_scan(self):
        client = Client()
        payload = {'type': 'sms', 'text': SPAM_EXAMPLES[0]}

        first_response = client.post(
            '/predict/',
            data=json.dumps(payload),
            content_type='application/json',
        )
        second_response = client.post(
            '/predict/',
            data=json.dumps(payload),
            content_type='application/json',
        )

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 403)
        self.assertEqual(second_response.json()['register_url'], '/register/')

    def test_public_home_and_protected_routes_resolve_without_duplicate_paths(self):
        client = Client()
        home_response = client.get('/')
        self.assertEqual(home_response.status_code, 200)
        self.assertTemplateUsed(home_response, 'detector/index.html')

        protected_routes = [
            ('dashboard', '/dashboard/'),
            ('sms_detection', '/dashboard/sms/'),
            ('email_detection', '/dashboard/email/'),
            ('url_detection', '/dashboard/url/'),
        ]
        for route_name, expected_path in protected_routes:
            with self.subTest(route_name=route_name):
                self.assertEqual(reverse(route_name), expected_path)
                response = client.get(expected_path)
                self.assertEqual(response.status_code, 302)
                self.assertEqual(response.url, f'/login/?next={expected_path}')

    def test_login_navigation_detection_history_and_logout_flow(self):
        user = User.objects.create_user(username='routecheck', password='test-password-123')
        client = Client()

        protected_response = client.get('/dashboard/sms/')
        self.assertEqual(protected_response.status_code, 302)
        login_response = client.get(protected_response.url)
        self.assertContains(login_response, 'name="next"')

        login_response = client.post(
            '/login/',
            {
                'identity': 'routecheck',
                'password': 'test-password-123',
                'next': '/dashboard/sms/',
            },
        )
        self.assertRedirects(login_response, '/dashboard/sms/', fetch_redirect_response=False)
        self.assertRedirects(client.get('/'), '/dashboard/', fetch_redirect_response=False)

        detection_pages = [
            ('/dashboard/sms/', SPAM_EXAMPLES[0], 'smsd.html', 'Spam', 'sms'),
            ('/dashboard/email/', SPAM_EXAMPLES[3], 'emaild.html', 'Phishing', 'email'),
            ('/dashboard/url/', 'https://example.com', 'urld.html', None, 'url'),
        ]
        for path, text, template_name, expected_prediction, detection_type in detection_pages:
            with self.subTest(path=path):
                page_response = client.get(path)
                self.assertEqual(page_response.status_code, 200)
                self.assertTemplateUsed(page_response, template_name)

                prediction_response = client.post(path, {'text': text})
                self.assertEqual(prediction_response.status_code, 200)
                self.assertTemplateUsed(prediction_response, template_name)
                record = DetectionHistory.objects.get(user=user, detection_type=detection_type)
                if expected_prediction:
                    self.assertEqual(record.prediction, expected_prediction)

        dashboard_response = client.get('/dashboard/')
        self.assertEqual(dashboard_response.status_code, 200)
        self.assertEqual(dashboard_response.context['stats']['total_checks'], 3)
        self.assertEqual(
            DetectionHistory.objects.filter(user=user).count(),
            3,
        )

        logout_response = client.post('/logout/')
        self.assertRedirects(logout_response, '/login/', fetch_redirect_response=False)
        self.assertRedirects(
            client.get('/dashboard/email/'),
            '/login/?next=/dashboard/email/',
            fetch_redirect_response=False,
        )

    def test_url_detection_contract_remains_available(self):
        result = run_detection_for_type('url', 'https://example.com')
        self.assertIn(result['prediction_label'], {'SCAM', 'SAFE'})
        self.assertEqual(result['is_scam'], result['prediction_label'] == 'SCAM')
        self.assertGreaterEqual(result['risk_score'], 0)
        self.assertLessEqual(result['risk_score'], 100)

    def test_missing_model_returns_service_error_without_stack_trace(self):
        client = Client()
        with patch(
            'detector.views.load_model_artifact',
            side_effect=ModelUnavailableError('A required detection model is missing.'),
        ):
            response = client.post(
                '/predict/',
                data=json.dumps({'type': 'sms', 'text': SPAM_EXAMPLES[0]}),
                content_type='application/json',
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {'error': 'A required detection model is missing.'})

    def test_prediction_error_does_not_expose_internal_details(self):
        client = Client()
        with patch(
            'detector.views.load_model_artifact',
            side_effect=RuntimeError('private model path'),
        ), self.assertLogs('detector.views', level='ERROR'):
            response = client.post(
                '/predict/',
                data=json.dumps({'type': 'sms', 'text': SPAM_EXAMPLES[0]}),
                content_type='application/json',
            )

        self.assertEqual(response.status_code, 500)
        self.assertNotIn('private model path', response.content.decode())

    def test_empty_and_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            run_detection_for_type('sms', '  ')
        with self.assertRaises(ValueError):
            run_detection_for_type('url', 'not-a-full-url')
