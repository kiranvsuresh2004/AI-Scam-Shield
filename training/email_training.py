import re
from pathlib import Path

import pandas as pd
import joblib

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_DIR / "dataset" / "phishing_email.csv"
MODEL_DIR = PROJECT_DIR / "model"


# ==========================
# Helpers
# ==========================

def clean_email_text(text):
    text = str(text or "")
    text = text.lower()
    text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


BANKING_CONTEXT_EXAMPLES = [
    ("Your monthly statement is ready in the official banking app. Sign in through the app to review it.", 0),
    ("The latest statement for your bank account is available in our official mobile banking application.", 0),
    ("Monthly account statements are published in the official banking application for customers to review.", 0),
    ("Your account statement is ready. Open the bank's official app to view the document.", 0),
    ("We have made your monthly statement available through the bank's usual online banking service.", 0),
    ("Please use the official bank application to review the statement posted to your account.", 0),
    ("A new statement has been added to your account. You can view it after signing in to the banking app.", 0),
    ("Your latest account statement can now be viewed in the mobile application you normally use for banking.", 0),
    ("A debit card purchase of Rs. 1,240 was approved. Contact support using the number on your card if this was not you.", 0),
    ("Your card payment was completed successfully. The receipt is available in your recent account activity.", 0),
    ("A payment using your debit card has posted to your account. No further action is required.", 0),
    ("Your recent card transaction was approved. Review the details in the official mobile banking app.", 0),
    ("The purchase on your credit card has been processed. This is a confirmation for your records.", 0),
    ("Thank you for using your card. The completed transaction appears in your normal banking history.", 0),
    ("Your monthly card payment was received. A confirmation is available through online banking.", 0),
    ("Your transfer has completed successfully. The transaction receipt is available in online banking.", 0),
    ("Your current account balance has been updated after today's scheduled deposit.", 0),
    ("A deposit has been credited to your account and is included in the balance shown in the bank app.", 0),
    ("Your scheduled transfer is complete. You can review the transaction in your account history.", 0),
    ("A sign-in to digital banking was completed from a recognized device. Review recent activity in the official app.", 0),
    ("This is a notification that your online banking login was successful from a device you have used before.", 0),
    ("We received your card payment. Your receipt is available in the account activity section of our mobile app.", 0),
    ("Your new debit card has been dispatched. Delivery updates are available through the bank's official website.", 0),
    ("Scheduled maintenance for online banking will take place this weekend. No action is needed.", 0),
    ("Your monthly interest credit has been posted to your savings account.", 0),
    ("Your direct deposit was received and is reflected in your account balance.", 0),
    ("Online banking enrollment is confirmed. We will never request your password or one-time code by email.", 0),
    ("A recurring card payment was processed successfully. Manage scheduled payments in the official banking application.", 0),
    ("Your bank's customer service hours will change next month. Service information is available in the mobile app.", 0),
    ("We have updated the terms for our savings account. Read the notice after signing in to the official website.", 0),
    ("Your replacement card request has been received. Track its status using the bank's regular application.", 0),
    ("Urgent: your account will be suspended unless you verify your details using this link and provide your password.", 1),
    ("Your bank account is at risk of closure. Confirm your identity and password on this page immediately.", 1),
    ("Immediate action is required to keep your account active. Click the link and submit your banking password.", 1),
    ("We will disable your online banking today unless you verify your account using the link in this message.", 1),
    ("Your statement access has been suspended. Enter your PIN and one-time code at the link to restore it.", 1),
    ("A security review is required now. Reply with your password and OTP to protect your bank account.", 1),
    ("Your debit card has been blocked. Click the link and enter your PIN to restore access immediately.", 1),
    ("Your card will be permanently locked unless you confirm its number and security code on our verification page.", 1),
    ("A payment is pending. Sign in through this link and provide your card PIN to release the funds.", 1),
    ("We detected unusual activity. Send your one-time code and online banking password to prevent account closure.", 1),
    ("Your account has been compromised. Verify your login credentials immediately using the link below.", 1),
    ("Confirm your identity and banking details now or your account access will be suspended.", 1),
    ("Final warning: confirm your banking details today or your account access will be disabled.", 1),
    ("Claim your refund by signing in at this shortened link and entering your card number and security code.", 1),
    ("You are owed a card refund. Click here and enter your online banking password to claim it today.", 1),
    ("Your account is compromised. Follow this link now to reset your password and confirm your PIN.", 1),
    ("A payment is on hold. Reply with your OTP and password so we can release the transaction.", 1),
    ("Verify your card immediately at http://bank-secure-login.top and avoid permanent suspension.", 1),
    ("Your bank profile will be disabled. Open http://secure-account-check.xyz and enter your password.", 1),
    ("Your online banking will be locked today. Open the link below and submit your login credentials.", 1),
    ("An unauthorized transfer was reported. Confirm your account number and password through this page.", 1),
    ("Act now to reverse the card charge by entering your PIN and one-time password on our verification page.", 1),
    ("Your bank profile needs urgent validation. Click here and provide your password to keep services active.", 1),
    ("A transfer from your account is blocked. Verify your password and one-time code using this link immediately.", 1),
    ("Your card payment failed. Submit your card number, expiry date, and PIN here to process it again.", 1),
    ("We will close your account unless you confirm your banking login and security code today.", 1),
    ("Urgent security notice: reply with your OTP so our team can stop the suspicious transaction.", 1),
    ("Your account balance is frozen. Click this link and enter your password to restore access now.", 1),
    ("Confirm your debit card details and PIN to receive the pending refund before it expires.", 1),
    ("Your mobile banking access is suspended. Update your credentials at this page to unlock it.", 1),
    ("A recent login was flagged. Provide your password and PIN immediately to prevent account closure.", 1),
]


# ==========================
# Load Email Dataset
# ==========================

data = pd.read_csv(DATASET_PATH)

if "text_combined" not in data.columns or "label" not in data.columns:
    raise ValueError("Email dataset must contain 'text_combined' and 'label' columns.")

print("Dataset Loaded Successfully")
print(data.head())


# ==========================
# Select Features and Labels
# ==========================

data = data[["text_combined", "label"]].copy()
data["text_combined"] = data["text_combined"].map(clean_email_text)
data["label"] = pd.to_numeric(data["label"], errors="raise").astype(int)
if set(data["label"].unique()) != {0, 1}:
    raise ValueError("Email labels must use 0 for SAFE and 1 for SCAM.")
data = data[data["text_combined"].ne("")]

conflicting = data.groupby("text_combined")["label"].nunique().gt(1)
if conflicting.any():
    raise ValueError("Duplicate email text has conflicting labels; resolve those rows before training.")
duplicate_count = int(data.duplicated(subset=["text_combined"]).sum())
data = data.drop_duplicates(subset=["text_combined"]).reset_index(drop=True)
print(f"Email rows after removing {duplicate_count} duplicate texts: {len(data)}")
print(f"Label counts (0=SAFE, 1=SCAM): {data['label'].value_counts().sort_index().to_dict()}")

X = data["text_combined"]
y = data["label"]


# ==========================
# TF-IDF Conversion
# ==========================

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 3),
    min_df=2,
    max_features=30000,
    sublinear_tf=True
)


# ==========================
# Stratified Train/Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Keep curated banking examples in training only, so test metrics remain independent.
augmentation_texts = [text for text, _ in BANKING_CONTEXT_EXAMPLES for _ in range(40)]
augmentation_labels = [label for _, label in BANKING_CONTEXT_EXAMPLES for _ in range(40)]
X_train = pd.concat([X_train.reset_index(drop=True), pd.Series(augmentation_texts)], ignore_index=True)
y_train = pd.concat([y_train.reset_index(drop=True), pd.Series(augmentation_labels)], ignore_index=True)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF Completed")
print("Training shape:", X_train_tfidf.shape)
print("Test shape:", X_test_tfidf.shape)


# ==========================
# Calibrated Linear SVM
# ==========================

base_model = LinearSVC(
    C=1.0,
    class_weight="balanced"
)

svm_model = CalibratedClassifierCV(
    estimator=base_model,
    method="sigmoid",
    cv=3
)

svm_model.fit(X_train_tfidf, y_train)

svm_pred = svm_model.predict(X_test_tfidf)

print("\nClassification Report:")
print(classification_report(
    y_test,
    svm_pred,
    labels=[0, 1],
    target_names=["SAFE", "SCAM"],
    digits=4
))
print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_pred, labels=[0, 1]))

MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(
    svm_model,
    MODEL_DIR / "email_svm_model.pkl"
)

joblib.dump(
    vectorizer,
    MODEL_DIR / "email_vectorizer.pkl"
)


# ==========================
# Test Custom Emails
# ==========================

examples = [
    (
        "Legitimate banking email",
        "Your monthly bank statement for September 2026 is now available. You can securely view your statement by logging in through the official banking app or by visiting your bank's official website. For your security, we will never ask you to share your password, PIN, OTP, or CV through email. If you did not request or expect this statement, please contact our customer support through the official contact details listed on our website."
    ),
    (
        "Phishing email",
        "Congratulations! You have won a free cash prize. Click here now to verify your account and claim your payout before it expires."
    )
]

for label, email in examples:
    cleaned_email = clean_email_text(email)
    email_vector = vectorizer.transform([cleaned_email])
    prediction = svm_model.predict(email_vector)[0]
    probabilities = svm_model.predict_proba(email_vector)[0]
    print(f"\n{label}: predicted={prediction}, probabilities={probabilities}")
    if prediction == 1:
        print("Prediction: ⚠️ SCAM")
    else:
        print("Prediction: ✅ SAFE")