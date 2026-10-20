import re
import socket
from urllib.parse import urlparse


URL_FEATURE_COLUMNS = [
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


SHORTENING_SERVICES = [
    "bit.ly", "goo.gl", "tinyurl", "t.co", "ow.ly", "is.gd",
    "tiny.cc", "buff.ly", "bit.do", "adf.ly", "rb.gy"
]

SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "bid", "loan", "top", "club"
}


def _safe_int(value):
    return int(value) if value is not None else 0


def _score(valid, suspicious):
    if suspicious:
        return -1
    if valid:
        return 1
    return 0


def extract_url_features(url):
    raw_url = str(url or "").strip()
    parsed = urlparse(raw_url)
    scheme = parsed.scheme.lower()
    domain = parsed.netloc.lower()
    hostname = parsed.hostname.lower() if parsed.hostname else ""
    path = parsed.path.lower()

    if not raw_url:
        return [0] * 30

    has_ip = bool(re.search(r"\d+\.\d+\.\d+\.\d+", raw_url))
    url_length = len(raw_url)
    url_length_score = 1 if url_length < 54 else 0 if url_length <= 75 else -1

    shortened = any(service in raw_url.lower() for service in SHORTENING_SERVICES)
    has_at_symbol = "@" in raw_url
    double_slash_redirect = raw_url.lower().find("//") > 7
    prefix_suffix = "-" in hostname
    dots = hostname.count(".")
    has_sub_domain = dots > 2
    ssl_final_state = 1 if scheme == "https" else -1

    suspicious_tld = hostname.split(".")[-1] in SUSPICIOUS_TLDS if hostname else False
    domain_registration_length = -1 if suspicious_tld else 1

    favicon = 1
    port_value = parsed.port
    port = 1 if port_value is None else -1

    https_token = -1 if "https" in hostname else 1

    request_url = 1 if not hostname else 0
    url_of_anchor = 1 if not hostname else 0
    links_in_tags = 1 if not hostname else 0
    sfh = 1 if not path or "about:blank" in raw_url.lower() else -1
    submitting_to_email = -1 if "mailto:" in raw_url.lower() else 1
    abnormal_url = 1 if hostname and hostname in raw_url.lower() else -1
    redirect = 1 if raw_url.lower().count("//") > 1 else -1

    on_mouseover = 1
    right_click = 1
    pop_up_window = 1
    iframe = 1

    age_of_domain = 1 if hostname and len(hostname) > 10 else -1

    dns_record = 1
    try:
        socket.gethostbyname(hostname)
    except socket.gaierror:
        dns_record = -1

    web_traffic = 1 if hostname and ("google" in hostname or "microsoft" in hostname or "apple" in hostname) else -1
    page_rank = 1 if hostname and len(hostname) > 6 else -1
    google_index = 1 if hostname and not suspicious_tld else -1
    links_pointing_to_page = 1 if hostname and scheme == "https" else -1
    statistical_report = -1 if (
        has_ip or shortened or has_at_symbol or double_slash_redirect or prefix_suffix or has_sub_domain
    ) else 1

    features = [
        _safe_int(1 if not has_ip else -1),
        _safe_int(url_length_score),
        _safe_int(-1 if shortened else 1),
        _safe_int(-1 if has_at_symbol else 1),
        _safe_int(-1 if double_slash_redirect else 1),
        _safe_int(-1 if prefix_suffix else 1),
        _safe_int(-1 if has_sub_domain else 1),
        _safe_int(ssl_final_state),
        _safe_int(domain_registration_length),
        _safe_int(favicon),
        _safe_int(port),
        _safe_int(https_token),
        _safe_int(request_url),
        _safe_int(url_of_anchor),
        _safe_int(links_in_tags),
        _safe_int(sfh),
        _safe_int(submitting_to_email),
        _safe_int(abnormal_url),
        _safe_int(redirect),
        _safe_int(on_mouseover),
        _safe_int(right_click),
        _safe_int(pop_up_window),
        _safe_int(iframe),
        _safe_int(age_of_domain),
        _safe_int(dns_record),
        _safe_int(web_traffic),
        _safe_int(page_rank),
        _safe_int(google_index),
        _safe_int(links_pointing_to_page),
        _safe_int(statistical_report),
    ]

    if len(features) != len(URL_FEATURE_COLUMNS):
        raise ValueError(f"URL feature extraction produced {len(features)} values; expected {len(URL_FEATURE_COLUMNS)}.")

    return features


def extract_features(url):
    return extract_url_features(url)