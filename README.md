# AI Scam Shield

A Django-based scam detection project for SMS, email, and URL analysis.

## Stack
- Frontend: HTML, CSS, JavaScript
- Backend: Python Django
- Database: SQLite
- ML models: joblib-loaded SVM, Naive Bayes, logistic regression, and random forest models

## Project structure
- `manage.py`
- `scamshield/` – Django project settings and routing
- `detector/` – app for auth, detection, history, templates, and admin
- `model/` – trained ML model files
- `dataset/` – datasets used for model training and testing
- `training/` – training scripts
- `static/` – CSS and JS assets
- `url_feature_extractor.py` – URL feature extraction logic

## Run locally
```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe manage.py migrate
.\.venv\Scripts\python.exe manage.py runserver
```

## File scans and timestamps

Detection history uses Django's timezone-aware timestamps and displays them in
Indian Standard Time (`Asia/Kolkata`). Existing UTC timestamps are retained and
converted for display; no database migration is needed.

File scans accept `.txt`, `.eml`, `.pdf`, `.png`, `.jpg`, `.jpeg`, and `.webp`
files up to 10 MB. The Python packages for PDF and image handling are listed in
`requirements.txt`. Scanned PDFs and images additionally require the native
Tesseract OCR engine. On Windows, the app detects the standard
`C:\Program Files\Tesseract-OCR\tesseract.exe` installation path. If Tesseract
is installed elsewhere, set the `TESSERACT_CMD` environment variable to its
executable path. Restart the Django server after installing or configuring
Tesseract. OCR-dependent uploads return a clear error when the OCR engine is
unavailable rather than producing a prediction from missing text.

## Detection tests

Run the focused SMS, email, URL, history, and missing-model checks from the project root:

```powershell
.\.venv\Scripts\python.exe manage.py test detector.tests
```

Unauthenticated visitors can run one SMS scan, one email scan, and one URL scan per browser session. Further scans of a type redirect to registration (or return HTTP 403 with the registration URL from the JSON endpoint). Signing in bypasses the guest trial limit. Because the trial is session-based, clearing browser cookies or using another browser starts a new guest session.

SMS and email combine their existing TF-IDF classifiers with contextual signals. Strong combinations—such as unrealistic promised returns, fees tied to a prize or job, credential requests, and account threats—can override a safe model prediction. A single financial keyword does not trigger the contextual override. The risk score and indicators include the detected signals.

## Retrain the models

Run from the project root so the scripts use the project datasets and write model artifacts into `model\`:

```powershell
.\.venv\Scripts\python.exe training\sms_training.py
.\.venv\Scripts\python.exe training\email_training.py
```

Each trainer checks the 0=Genuine/Safe and 1=Spam/Scam label mapping, removes duplicate message text before the stratified split, and reports per-class precision, recall, F1, and a confusion matrix. Review the measured metrics before replacing or distributing trained artifacts; they are not a guarantee of performance on unseen messages.

## Verify in the web interface

Start Django from the project root:

```powershell
.\.venv\Scripts\python.exe manage.py runserver
```

Open `http://127.0.0.1:8000/sms/` or `http://127.0.0.1:8000/email/`, submit a message, and check the prediction, risk score, and explanation signals. Stop the development server with `Ctrl+C`.
