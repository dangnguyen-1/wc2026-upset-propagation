"""Country → FIFA confederation map for Phase 1+ HFA β bonus.

Used by MatchContext construction in validate_phase1.py and predict_match runs.
Keys match Kaggle dataset's `home_team` / `away_team` strings exactly (e.g.,
"South Korea", "Côte d'Ivoire" → "Ivory Coast" per Kaggle convention,
"Czech Republic" → "Czech Republic" not "Czechia").

Confederations (6 FIFA + UNKNOWN sentinel):
    UEFA       — 55 European associations
    CONMEBOL   — 10 South American
    CONCACAF   — North/Central America + Caribbean (incl. associates)
    AFC        — Asia + Australia (since 2006)
    CAF        — Africa
    OFC        — Oceania (sans Australia post-2006)
    UNKNOWN    — non-FIFA entities (CONIFA, regional/sub-national teams)

KNOWN APPROXIMATION (uses CURRENT 2026 confederation for all historical matches):
    Australia OFC → AFC (transitioned 2006)
    Israel    AFC → UEFA (transitioned 1992)
    Kazakhstan AFC → UEFA (transitioned 2002)
Safe for Phase 1 validation (WC2018+) and WC2026 — no pre-transition matches affected.

UNKNOWN behavior: hfa_log_goals treats 'UNKNOWN' confed as never matching
host_confederation, so β=0 — safe fallback for sub-national CONIFA teams that
don't compete in international tournaments anyway.
"""

from __future__ import annotations

UEFA = "UEFA"
CONMEBOL = "CONMEBOL"
CONCACAF = "CONCACAF"
AFC = "AFC"
CAF = "CAF"
OFC = "OFC"
UNKNOWN = "UNKNOWN"


COUNTRY_TO_CONFEDERATION: dict[str, str] = {
    # ── UEFA (55) ──────────────────────────────────────────────────────────
    "Albania": UEFA, "Andorra": UEFA, "Armenia": UEFA, "Austria": UEFA,
    "Azerbaijan": UEFA, "Belarus": UEFA, "Belgium": UEFA,
    "Bosnia and Herzegovina": UEFA, "Bulgaria": UEFA, "Croatia": UEFA,
    "Cyprus": UEFA, "Czech Republic": UEFA, "Czechia": UEFA,
    "Denmark": UEFA, "England": UEFA, "Estonia": UEFA,
    "Faroe Islands": UEFA, "Finland": UEFA, "France": UEFA,
    "Georgia": UEFA, "Germany": UEFA, "Gibraltar": UEFA, "Greece": UEFA,
    "Hungary": UEFA, "Iceland": UEFA, "Israel": UEFA, "Italy": UEFA,
    "Kazakhstan": UEFA, "Kosovo": UEFA, "Latvia": UEFA,
    "Liechtenstein": UEFA, "Lithuania": UEFA, "Luxembourg": UEFA,
    "Malta": UEFA, "Moldova": UEFA, "Montenegro": UEFA,
    "Netherlands": UEFA, "North Macedonia": UEFA, "Northern Ireland": UEFA,
    "Norway": UEFA, "Poland": UEFA, "Portugal": UEFA,
    "Republic of Ireland": UEFA, "Ireland": UEFA, "Romania": UEFA,
    "Russia": UEFA, "San Marino": UEFA, "Scotland": UEFA,
    "Serbia": UEFA, "Slovakia": UEFA, "Slovenia": UEFA, "Spain": UEFA,
    "Sweden": UEFA, "Switzerland": UEFA, "Turkey": UEFA, "Türkiye": UEFA,
    "Ukraine": UEFA, "Wales": UEFA,

    # ── CONMEBOL (10) ──────────────────────────────────────────────────────
    "Argentina": CONMEBOL, "Bolivia": CONMEBOL, "Brazil": CONMEBOL,
    "Chile": CONMEBOL, "Colombia": CONMEBOL, "Ecuador": CONMEBOL,
    "Paraguay": CONMEBOL, "Peru": CONMEBOL, "Uruguay": CONMEBOL,
    "Venezuela": CONMEBOL,

    # ── CONCACAF (35 FIFA members + 6 associates) ─────────────────────────
    "Anguilla": CONCACAF, "Antigua and Barbuda": CONCACAF, "Aruba": CONCACAF,
    "Bahamas": CONCACAF, "Barbados": CONCACAF, "Belize": CONCACAF,
    "Bermuda": CONCACAF, "Bonaire": CONCACAF,
    "British Virgin Islands": CONCACAF, "Canada": CONCACAF,
    "Cayman Islands": CONCACAF, "Costa Rica": CONCACAF, "Cuba": CONCACAF,
    "Curaçao": CONCACAF, "Curacao": CONCACAF,
    "Dominica": CONCACAF, "Dominican Republic": CONCACAF,
    "El Salvador": CONCACAF, "French Guiana": CONCACAF, "Grenada": CONCACAF,
    "Guadeloupe": CONCACAF, "Guatemala": CONCACAF, "Guyana": CONCACAF,
    "Haiti": CONCACAF, "Honduras": CONCACAF, "Jamaica": CONCACAF,
    "Martinique": CONCACAF, "Mexico": CONCACAF, "Montserrat": CONCACAF,
    "Nicaragua": CONCACAF, "Panama": CONCACAF, "Puerto Rico": CONCACAF,
    "Saint Barthélemy": CONCACAF,
    "Saint Kitts and Nevis": CONCACAF, "Saint Lucia": CONCACAF,
    "Saint Martin": CONCACAF, "Saint Pierre and Miquelon": CONCACAF,
    "Saint Vincent and the Grenadines": CONCACAF,
    "Sint Maarten": CONCACAF, "Suriname": CONCACAF,
    "Trinidad and Tobago": CONCACAF,
    "Turks and Caicos Islands": CONCACAF,
    "United States": CONCACAF, "USA": CONCACAF,
    "United States Virgin Islands": CONCACAF,
    "Greenland": CONCACAF,  # admitted to CONCACAF 2024

    # ── AFC (47) ───────────────────────────────────────────────────────────
    "Afghanistan": AFC, "Australia": AFC, "Bahrain": AFC, "Bangladesh": AFC,
    "Bhutan": AFC, "Brunei": AFC, "Cambodia": AFC,
    "China": AFC, "China PR": AFC, "Chinese Taipei": AFC, "Taiwan": AFC,
    "Guam": AFC, "Hong Kong": AFC, "India": AFC, "Indonesia": AFC,
    "Iran": AFC, "Iraq": AFC, "Japan": AFC, "Jordan": AFC,
    "Kuwait": AFC, "Kyrgyzstan": AFC, "Laos": AFC, "Lebanon": AFC,
    "Macau": AFC, "Malaysia": AFC, "Maldives": AFC, "Mongolia": AFC,
    "Myanmar": AFC, "Nepal": AFC, "North Korea": AFC,
    "Korea Republic": AFC, "South Korea": AFC,
    "Northern Mariana Islands": AFC,
    "Oman": AFC, "Pakistan": AFC, "Palestine": AFC, "Philippines": AFC,
    "Qatar": AFC, "Saudi Arabia": AFC, "Singapore": AFC, "Sri Lanka": AFC,
    "Syria": AFC, "Tajikistan": AFC, "Thailand": AFC,
    "Timor-Leste": AFC, "East Timor": AFC,
    "Turkmenistan": AFC, "United Arab Emirates": AFC,
    "Uzbekistan": AFC, "Vietnam": AFC, "Yemen": AFC,

    # ── CAF (54 FIFA members + 2 associates) ──────────────────────────────
    "Algeria": CAF, "Angola": CAF, "Benin": CAF, "Botswana": CAF,
    "Burkina Faso": CAF, "Burundi": CAF, "Cameroon": CAF,
    "Cape Verde": CAF, "Cabo Verde": CAF,
    "Central African Republic": CAF, "Chad": CAF, "Comoros": CAF,
    "Congo": CAF, "DR Congo": CAF, "Democratic Republic of the Congo": CAF,
    "Djibouti": CAF, "Egypt": CAF, "Equatorial Guinea": CAF,
    "Eritrea": CAF, "Eswatini": CAF, "Swaziland": CAF,
    "Ethiopia": CAF, "Gabon": CAF, "Gambia": CAF, "Ghana": CAF,
    "Guinea": CAF, "Guinea-Bissau": CAF,
    "Ivory Coast": CAF, "Côte d'Ivoire": CAF,
    "Kenya": CAF, "Lesotho": CAF, "Liberia": CAF, "Libya": CAF,
    "Madagascar": CAF, "Malawi": CAF, "Mali": CAF, "Mauritania": CAF,
    "Mauritius": CAF, "Mayotte": CAF, "Morocco": CAF, "Mozambique": CAF,
    "Namibia": CAF, "Niger": CAF, "Nigeria": CAF, "Réunion": CAF,
    "Reunion": CAF, "Rwanda": CAF, "Saint Helena": CAF,
    "São Tomé and Príncipe": CAF, "Senegal": CAF, "Seychelles": CAF,
    "Sierra Leone": CAF, "Somalia": CAF, "South Africa": CAF,
    "South Sudan": CAF, "Sudan": CAF, "Tanzania": CAF, "Togo": CAF,
    "Tunisia": CAF, "Uganda": CAF, "Zambia": CAF, "Zanzibar": CAF,
    "Zimbabwe": CAF,

    # ── OFC (11 FIFA members + associates) ────────────────────────────────
    "American Samoa": OFC, "Cook Islands": OFC, "Fiji": OFC,
    "Kiribati": OFC, "New Caledonia": OFC, "New Zealand": OFC,
    "Papua New Guinea": OFC, "Samoa": OFC, "Solomon Islands": OFC,
    "Tahiti": OFC, "Tonga": OFC, "Tuvalu": OFC, "Vanuatu": OFC,
    "Marshall Islands": OFC, "Micronesia": OFC, "Palau": OFC,
}


def get_confederation(country: str) -> str:
    """Return current (2026) confederation for `country`, or 'UNKNOWN' if unmapped.

    Non-FIFA entities present in dataset (CONIFA members, sub-national teams,
    historical/territorial sides like Catalonia, Basque Country, Tibet, Sápmi,
    Yorkshire, etc.) → 'UNKNOWN'. predict_match's β=0 fallback handles this safely.
    """
    return COUNTRY_TO_CONFEDERATION.get(country, UNKNOWN)
