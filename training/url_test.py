import joblib
import pandas as pd

from url_feature_extractor import extract_features


# ==============================
# LOAD MODEL
# ==============================

model = joblib.load("model/random_forest_url_model.pkl")


# ==============================
# USER INPUT
# ==============================

url = input("Enter website URL: ")


# ==============================
# EXTRACT FEATURES
# ==============================

features = extract_features(url)

feature_names = [
    "having_IP_Address",
    "URL_Length",
    "Shortining_Service",
    "having_At_Symbol",
    "double_slash_redirecting",
    "Prefix_Suffix",
    "having_Sub_Domain",
    "SSLfinal_State",
    "Domain_registeration_length",
    "Favicon",
    "port",
    "HTTPS_token",
    "Request_URL",
    "URL_of_Anchor",
    "Links_in_tags",
    "SFH",
    "Submitting_to_email",
    "Abnormal_URL",
    "Redirect",
    "on_mouseover",
    "RightClick",
    "popUpWidnow",
    "Iframe",
    "age_of_domain",
    "DNSRecord",
    "web_traffic",
    "Page_Rank",
    "Google_Index",
    "Links_pointing_to_page",
    "Statistical_report"
]

print("\nExpected feature count:", len(feature_names))
print("Actual feature count:", len(features))
print("Feature order:", feature_names)
print("\nExtracted Features:")
print(features)


# ==============================
# CONVERT TO DATAFRAME
# ==============================

X = pd.DataFrame(
    [features],
    columns=feature_names
)


# ==============================
# PREDICTION
# ==============================

prediction = model.predict(X)[0]

probability = model.predict_proba(X)[0]

print("\n========== RESULT ==========")
print("URL:", url)
print("Prediction value:", prediction)
print("Probability:", probability)

if prediction == 0:
    print("🚨 PHISHING WEBSITE")
else:
    print("✅ LEGITIMATE WEBSITE")