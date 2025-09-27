# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import os
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# -------------------- warnings --------------------
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------- file paths --------------------
MODEL_PATH = " my_model_compressed.pkl"
CSV_PATH = "Skin_Disease_Dataset_Extended.csv"
IMAGE_FOLDER = "image"

# -------------------- load model & data --------------------
model = joblib.load(MODEL_PATH)
df = pd.read_csv(CSV_PATH)
SYMPTOMS = [c for c in df.columns if c != "prognosis"]
DISEASES = sorted(df["prognosis"].unique())

# -------------------- UI language --------------------
language = st.radio("🌐 Select Language / மொழியைத் தேர்ந்தெடுக்கவும்:", ["English", "தமிழ்"])

# -------------------- UI text --------------------
UI_TEXT = {
    "English": {
        "title": "🩺 Skin Disease Prediction App",
        "select_symptoms": "Select your symptoms",
        "choose_symptoms": "Choose from the list:",
        "predict": "Predict",
        "warning": "⚠️ Please select at least one symptom.",
        "predictions": "🔮 Top 3 Predictions",
        "confirm_question": "✅ Do any of these images match your condition?",
        "confirm_button": "Confirm",
        "you_selected": "You selected",
        "precautions": "🛡️ Recommended Precautions",
        "medicines": "💊 Common Medicines / Treatments",
        "no_precautions": "ℹ️ No precautions available. Please consult a doctor.",
        "no_meds": "ℹ️ No medicine information available. Consult a doctor.",
        "final_info": "📌 If this matches your condition, follow the precautions and medicines only after consulting a doctor."
    },
    "தமிழ்": {
        "title": "🩺 தோல் நோய் கணிப்பு பயன்பாடு",
        "select_symptoms": "உங்களுடைய அறிகுறிகளைத் தேர்ந்தெடுக்கவும்",
        "choose_symptoms": "பட்டியலிலிருந்து தேர்ந்தெடுக்கவும்:",
        "predict": "கணிக்கவும்",
        "warning": "⚠️ குறைந்தது ஒரு அறிகுறியைத் தேர்ந்தெடுக்கவும்.",
        "predictions": "🔮 முன்னிலை 3 கணிப்புகள்",
        "confirm_question": "✅ இவற்றில் உங்கள் நிலையை தொடர்புத்தான் என்று தோன்றுமா?",
        "confirm_button": "உறுதிப்படுத்தவும்",
        "you_selected": "நீங்கள் தேர்ந்தெடுத்தது",
        "precautions": "🛡️ பரிந்துரைக்கப்பட்ட முன்னெச்சரிக்கை",
        "medicines": "💊 பொது மருந்துகள் / சிகிச்சைகள்",
        "no_precautions": "ℹ️ முன்னெச்சரிக்கை இல்லை. தயவுசெய்து மருத்துவரை அணுகவும்.",
        "no_meds": "ℹ️ மருந்து தகவல் கிடைக்கவில்லை. மருத்துவரை அணுகவும்.",
        "final_info": "📌 இது உங்கள் நிலைக்கு பொருந்தினால் மருத்துவரை அணுகி முன்னெச்சரிக்கை மற்றும் மருந்துகளை பின்பற்றவும்."
    }
}
INSTRUCTIONS = {
    "English": """
📋 **Instructions to use the Skin Disease Prediction App:**
1. Select your symptoms from the list below.
2. Click on **Predict** to get top 3 possible skin diseases.
3. Look at the images and confirm which disease matches your condition.
4. Once confirmed, you can view recommended **Precautions** and **Medicines**.
5. Always consult a doctor before following any precautions or medicines.
""",
    "தமிழ்": """
📋 **பயன்படுத்தும் வழிமுறைகள்:**
1. கீழுள்ள பட்டியலில் இருந்து உங்கள் அறிகுறிகளை தேர்ந்தெடுக்கவும்.
2. **கணிக்கவும்** பொத்தானை அழுத்தி முன்னிலை 3 தோல் நோய்களைப் பார்க்கவும்.
3. படங்களைப் பார்த்து உங்கள் நிலைக்கு பொருந்திய நோயை உறுதிப்படுத்தவும்.
4. உறுதிப்படுத்திய பின்னர் பரிந்துரைக்கப்பட்ட **முன்னெச்சரிக்கை** மற்றும் **மருந்துகள்** பார்க்கலாம்.
5. எந்த முன்னெச்சரிக்கையும் அல்லது மருந்தையும் பயன்படுத்துவதற்கு முன் மருத்துவரை அணுகவும்.
"""
}

st.title(UI_TEXT[language]["title"])
with st.expander("📋 Instructions / வழிமுறைகள்"):
    st.markdown(INSTRUCTIONS[language])
st.header(UI_TEXT[language]["select_symptoms"])
# -------------------- symptom translations --------------------
SYMPTOM_TRANSLATIONS = {
    "itching": "குடைச்சல்",
    "skin_rash": "தோல் சிரங்கு",
    "nodal_skin_eruptions": "நோடல் தோல் வெடிப்புகள்",
    "blister": "புண்",
    "pus_filled_pimples": "சீழ் நிரம்பிய முகப்பருக்கள்",
    "scurring": "புண்களின் மேல் புழுவிழிப்பு",
    "scarring": "காயமடைந்த அடையாளம்",
    "red_spots_over_body": "உடலில் சிவப்பு புள்ளிகள்",
    "high_fever": "உயர் ஜலநோய்",
    "fatigue": "சோர்வு",
    "headache": "தலைவலி",
    "loss_of_appetite": "உணவில்லாமை",
    "swelling_on_skin": "தோலில் வீக்கம்",
    "hives": "உர்டிகாரியா",
    "weeping_skin": "பீதியான தோல்",
    "thickened_skin": "துடுப்பான தோல்",
    "skin_ulcers": "தோல் புண்",
    "burning_sensation": "எரிச்சல் உணர்வு",
    "dry_skin": "உலர் தோல்",
    "facial_redness": "முகம் சிவப்பு",
    "pigmentation_spots": "தோல் நிறமாற்றப் புள்ளிகள்",
    "oily_skin": "கொழுப்பான தோல்",
    "cold_sores": "குளிர் புண்கள்",
    "flaky_skin": "உதிரும் தோல்",
    "swollen_face": "முக வீக்கம்",
    "dry_patches": "உலர் பகுதிகள்",
    "peeling": "தோல் உரிதல்",
    "sores": "காயங்கள்",
    "redness": "சிவப்பு",
    "blisters": "புண்கள்",
    "malaise": "மனஅழுத்தம்",
    "facial_warts": "முக வேர்ட்ஸ்"
}

# -------------------- PRECAUTIONS --------------------
PRECAUTIONS = {
    "perioral dermatitis": {
        "English": [
            "Avoid steroid creams unless prescribed",
            "Wash face with gentle cleanser only",
            "Stop using heavy cosmetics",
            "Consult dermatologist for proper treatment"
        ],
        "தமிழ்": [
            "மருத்துவர் பரிந்துரைக்கும் வரை ஸ்டீராய்டு கிரீம்களை தவிர்க்கவும்",
            "முகத்தை மென்மையான சுத்திகரிப்பால் மட்டும் கழுவவும்",
            "கனமான மேக்கப்பை நிறுத்தவும்",
            "தோல் மருத்துவரை அணுகவும்"
        ]
    },
    "contact dermatitis": {
        "English": [
            "Avoid contact with the allergen/irritant",
            "Wear gloves while handling chemicals",
            "Apply soothing creams or ointments",
            "Consult doctor if rash spreads"
        ],
        "தமிழ்": [
            "ஒவ்வாமி அல்லது எரிச்சல் தரும் பொருட்களுடன் தொடர்பு இல்லாமல் இருங்கள்",
            "ரசாயனங்களை கையாளும் போது கையுறைகள் அணியவும்",
            "தணிவு கிரீம்களை பயன்படுத்தவும்",
            "சிரங்கு பரவினால் மருத்துவரை அணுகவும்"
        ]
    },
    "rosacea": {
        "English": [
            "Avoid alcohol, spicy foods, and hot drinks",
            "Use sunscreen daily",
            "Avoid harsh skin products",
            "Consult dermatologist for long-term care"
        ],
        "தமிழ்": [
            "மதுபானம், கார உணவுகள் மற்றும் சூடான பானங்களை தவிர்க்கவும்",
            "தினமும் சன்ஸ்கிரீன் பயன்படுத்தவும்",
            "கடுமையான தோல் தயாரிப்புகளை தவிர்க்கவும்",
            "நீண்டகால சிகிச்சைக்காக தோல் மருத்துவரை அணுகவும்"
        ]
    },
    "psoriasis": {
        "English": [
            "Moisturize skin regularly",
            "Avoid stress, alcohol, and smoking",
            "Take prescribed medicines",
            "Expose skin to safe amounts of sunlight"
        ],
        "தமிழ்": [
            "தோலை அடிக்கடி ஈரப்பதப்படுத்தவும்",
            "மனஅழுத்தம், மதுபானம் மற்றும் புகைப்பிடித்தலை தவிர்க்கவும்",
            "மருத்துவர் பரிந்துரைத்த மருந்துகளை எடுத்துக்கொள்ளவும்",
            "பாதுகாப்பான அளவு சூரிய வெளிச்சத்திற்கு தோலை வெளிப்படுத்தவும்"
        ]
    },
    "acne": {
        "English": [
            "Wash face twice daily with mild cleanser",
            "Avoid oily or greasy cosmetics",
            "Do not squeeze or pick pimples",
            "Maintain a healthy diet and hydrate well"
        ],
        "தமிழ்": [
            "மென்மையான சுத்திகரிப்பால் முகத்தை நாளில் இரண்டு முறை கழுவவும்",
            "எண்ணெய்/கொழுப்பு நிறைந்த அழகு சாதனங்களை தவிர்க்கவும்",
            "முகப்பருக்களை அழுத்தவோ எடுக்கவோ வேண்டாம்",
            "ஆரோக்கியமான உணவுமுறையை பின்பற்றவும் மற்றும் நிறைய தண்ணீர் குடிக்கவும்"
        ]
    },
    "allergy": {
        "English": [
            "Avoid known allergens",
            "Take antihistamines if required",
            "Wear protective clothing outdoors",
            "Consult a doctor for allergy testing"
        ],
        "தமிழ்": [
            "அறிந்த ஒவ்வாமிகளை தவிர்க்கவும்",
            "தேவையானால் ஆன்டிஹிஸ்டமின்களை எடுத்துக்கொள்ளவும்",
            "வெளியில் பாதுகாப்பு உடைகள் அணியவும்",
            "ஒவ்வாமை பரிசோதனைக்காக மருத்துவரை அணுகவும்"
        ]
    },
    "impetigo": {
        "English": [
            "Wash skin gently with soap and water",
            "Avoid scratching or touching sores",
            "Take prescribed antibiotics",
            "Maintain proper hygiene"
        ],
        "தமிழ்": [
            "தோலை சோப்பும் நீராலும் மெதுவாக கழுவவும்",
            "காயங்களை கீறவோ தொடவோ வேண்டாம்",
            "மருத்துவர் பரிந்துரைத்த ஆன்டிபயாடிக்களை எடுத்துக்கொள்ளவும்",
            "சரியான சுகாதாரத்தை பேணவும்"
        ]
    },
    "ringworm": {
        "English": [
            "Keep skin clean and dry",
            "Do not share clothes or towels",
            "Use antifungal cream as prescribed",
            "Wash bedding and clothes regularly"
        ],
        "தமிழ்": [
            "தோலை சுத்தமாகவும் உலர்ந்தாகவும் வைத்திருங்கள்",
            "ஆடைகள் மற்றும் துணிகளை பகிராதீர்கள்",
            "மருத்துவர் பரிந்துரைத்த பூஞ்சை கிரீம் பயன்படுத்தவும்",
            "படுக்கை மற்றும் ஆடைகளை அடிக்கடி கழுவவும்"
        ]
    },
    "chicken pox": {
        "English": [
            "Avoid scratching blisters",
            "Take antihistamines for itching",
            "Stay isolated until blisters dry",
            "Drink plenty of fluids and rest"
        ],
        "தமிழ்": [
            "புண்களை கீற வேண்டாம்",
            "குடைச்சலுக்கு ஆன்டிஹிஸ்டமின்களை எடுத்துக்கொள்ளவும்",
            "புண்கள் உலரும் வரை தனிமைப்படுத்து",
            "அதிகம் தண்ணீர் குடித்து ஓய்வு எடு"
        ]
    },
    "eczema": {
        "English": [
            "Use fragrance-free moisturizers",
            "Avoid harsh soaps and detergents",
            "Wear cotton clothing",
            "Use antihistamines if itching is severe"
        ],
        "தமிழ்": [
            "வாசனை இல்லாத ஈரப்பதப்படுத்திகளை பயன்படுத்தவும்",
            "கடுமையான சோப்புகள் மற்றும் சவர்க்காரங்களை தவிர்க்கவும்",
            "பருத்தி உடைகளை அணியவும்",
            "குடைச்சல் அதிகமாக இருந்தால் ஆன்டிஹிஸ்டமின்கள் எடுத்துக் கொள்ளவும்"
        ]
    },
    "fungal infection": {
        "English": [
            "Keep affected area clean and dry",
            "Do not share personal items",
            "Use antifungal creams as prescribed",
            "Wash bedding and clothes regularly"
        ],
        "தமிழ்": [
            "பாதிக்கப்பட்ட பகுதியை சுத்தமாகவும் உலர்த்தவும்",
            "தனிப்பட்ட பொருட்களை பகிராதீர்கள்",
            "மருத்துவர் பரிந்துரைத்த பூஞ்சை எதிர்ப்பு கிரீம்களைப் பயன்படுத்தவும்",
            "படுக்கை மற்றும் ஆடைகளை அடிக்கடி துவைக்கவும்"
        ]
    },
    "scabies": {
        "English": [
            "Wash clothes and bedding in hot water",
            "Avoid close skin contact until treated",
            "Use prescribed scabicide lotions",
            "Consult doctor for family-wide treatment"
        ],
        "தமிழ்": [
            "ஆடைகள் மற்றும் படுக்கை வஸ்திரங்களை வெப்பநீரில் கழுவவும்",
            "சிகிச்சை முடியும் வரை நெருங்கிய தோல் தொடர்பை தவிர்க்கவும்",
            "மருத்துவர் பரிந்துரைத்த கிரீம்களை பயன்படுத்தவும்",
            "குடும்பத்தாருக்கும் சிகிச்சை தேவையாகும்"
        ]
    },
    "drug reaction": {
        "English": [
            "Stop the suspected medication immediately",
            "Consult a doctor urgently",
            "Avoid self-medicating",
            "Monitor for severe allergic reactions"
        ],
        "தமிழ்": [
            "சந்தேகப்படும் மருந்தை உடனடியாக நிறுத்தவும்",
            "உடனடி மருத்துவ ஆலோசனை பெறவும்",
            "சுயமருத்துவம் தவிர்க்கவும்",
            "கடுமையான ஒவ்வாமிகளை கண்காணிக்கவும்"
        ]
    },
    "acne (facial)": {
        "English": [
            "Cleanse face gently twice daily",
            "Avoid heavy makeup and oily products",
            "Do not pop or scratch pimples",
            "Consult a dermatologist if severe"
        ],
        "தமிழ்": [
            "முகத்தை மெதுவாக நாளில் இருமுறை சுத்தம் செய்யவும்",
            "கனமான மேக்கப் மற்றும் எண்ணெய் பொருட்களை தவிர்க்கவும்",
            "முகப்பருக்களை குத்தவோ பிடிக்கவோ வேண்டாம்",
            "கடுமையானிருந்தால் தோல் மருத்துவரை அணுகவும்"
        ]
    },
    "melasma": {
        "English": [
            "Apply sunscreen daily",
            "Avoid prolonged sun exposure",
            "Use prescribed skin-lightening creams",
            "Consult dermatologist for treatment"
        ],
        "தமிழ்": [
            "தினமும் சன்ஸ்கிரீன் பயன்படுத்தவும்",
            "நீண்ட நேரம் சூரிய ஒளியில் இருக்க வேண்டாம்",
            "மருத்துவர் பரிந்துரைத்த கிரீம்களை பயன்படுத்தவும்",
            "தோல் மருத்துவரை அணுகவும்"
        ]
    },
    "vitiligo": {
        "English": [
            "Protect skin from direct sunlight",
            "Avoid skin injuries or trauma",
            "Use prescribed creams or therapies",
            "Seek emotional support groups if needed"
        ],
        "தமிழ்": [
            "சூரியஒளியிலிருந்து தோலை பாதுகாக்கவும்",
            "தோல் காயங்கள் அல்லது குத்துகளை தவிர்க்கவும்",
            "மருத்துவர் கூறிய கிரீம்களை பயன்படுத்தவும்",
            "உளவியல் ஆதரவு தேவைப்பட்டால் குழுக்களை அணுகவும்"
        ]
    },
    "warts": {
        "English": [
            "Avoid scratching or picking warts",
            "Keep area clean and covered",
            "Do not share razors or towels",
            "Seek cryotherapy if persistent"
        ],
        "தமிழ்": [
            "முட்டைகளை கீறவோ எடுக்கவோ வேண்டாம்",
            "பகுதியை சுத்தமாகவும் மூடியும் வைத்திருங்கள்",
            "ரேசர் அல்லது துணிகளை பகிர வேண்டாம்",
            "தொடர்ந்தால் மருத்துவரை அணுகவும்"
        ]
    },
    "cellulitis": {
        "English": [
            "Take prescribed antibiotics",
            "Keep the infected area clean",
            "Elevate the affected limb",
            "Consult a doctor urgently if symptoms worsen"
        ],
        "தமிழ்": [
            "மருத்துவர் குறிப்பிட்ட ஆன்டிபயாடிக்களை எடுத்துக்கொள்ளவும்",
            "பாதிக்கப்பட்ட பகுதியை சுத்தமாக வைத்திருங்கள்",
            "பாதிக்கப்பட்ட அங்கத்தை உயர்த்தவும்",
            "அறிகுறிகள் மோசமாகினால் உடனடியாக மருத்துவரை அணுகவும்"
        ]
    },
    "seborrheic dermatitis": {
        "English": [
            "Use medicated shampoos regularly",
            "Avoid harsh hair or skin products",
            "Wash scalp and face gently",
            "Consult dermatologist if persistent"
        ],
        "தமிழ்": [
            "மருந்து கலந்த ஷாம்பூக்களை பயன்படுத்தவும்",
            "கடுமையான தலைமுடி/தோல் தயாரிப்புகளை தவிர்க்கவும்",
            "தலை மற்றும் முகத்தை மெதுவாக கழுவவும்",
            "தொடர்ந்தால் தோல் மருத்துவரை அணுகவும்"
        ]
    },
    "lichen planus": {
        "English": [
            "Avoid scratching lesions",
            "Apply soothing creams or ointments",
            "Take prescribed medications",
            "Consult doctor for oral/genital involvement"
        ],
        "தமிழ்": [
            "காயங்களை கீற வேண்டாம்",
            "தணிவு கிரீம்களைப் பயன்படுத்தவும்",
            "மருத்துவர் குறிப்பிட்ட மருந்துகளை எடுத்துக்கொள்ளவும்",
            "வாய்/பாலியல் பகுதியில் இருந்தால் மருத்துவரை அணுகவும்"
        ]
    },
    "herpes simplex": {
        "English": [
            "Avoid kissing or close contact during outbreak",
            "Use antiviral creams or medicines as prescribed",
            "Keep affected area clean and dry",
            "Avoid sharing utensils or towels"
        ],
        "தமிழ்": [
            "வெடிப்பு காலத்தில் நெருங்கிய தொடர்பைத் தவிர்க்கவும்",
            "மருத்துவர் பரிந்துரைத்த வைரஸ் எதிர்ப்பு மருந்துகளைப் பயன்படுத்தவும்",
            "பாதிக்கப்பட்ட பகுதியை சுத்தமாகவும் உலர்ந்தாகவும் வைத்திருங்கள்",
            "உபகரணங்களை பகிர வேண்டாம்"
        ]
    },
    "warts (facial)": {
        "English": [
            "Do not scratch or shave over warts",
            "Apply topical treatments as prescribed",
            "Avoid sharing personal grooming tools",
            "Consult dermatologist for removal options"
        ],
        "தமிழ்": [
            "முகத்தில் உள்ள முட்டைகளை கீறவோ குத்தவோ வேண்டாம்",
            "மருத்துவர் பரிந்துரைத்த கிரீம்களை பயன்படுத்தவும்",
            "பராமரிப்பு கருவிகளை பகிராதீர்கள்",
            "நீக்குவதற்காக தோல் மருத்துவர் அணுகவும்"
        ]
    },
    "urticaria": {
        "English": [
            "Avoid known triggers (foods, allergens, stress)",
            "Take antihistamines as prescribed",
            "Wear loose cotton clothing",
            "Consult doctor if breathing difficulty occurs"
        ],
        "தமிழ்": [
            "அறிந்த தூண்டுதல்களை (உணவு, ஒவ்வாமி, மனஅழுத்தம்) தவிர்க்கவும்",
            "ஆண்டிஹிஸ்டமின்களை மருத்துவர் சொல்லியபடி எடுத்துக்கொள்ளவும்",
            "பருத்தி உடைகளை அணியவும்",
            "சுவாச சிக்கல் ஏற்பட்டால் மருத்துவரை அணுகவும்"
        ]
    }
}

         
MEDICINES = {
    "acne": {
        "English": ["Benzoyl peroxide gel (topical)", "Clindamycin gel (topical)", "Doxycycline (oral, if severe)"],
        "தமிழ்": ["பென்சாயில் பெராக்சைடு ஜெல் (உட்புறம்)", "கிளின்டமைசின் ஜெல் (உட்புறம்)", "டாக்ஸிசைக்ளின் (வாய் மாத்திரை, கடுமையானால்)"]
    },
    "acne (facial)": {
        "English": ["Adapalene gel (topical)", "Azelaic acid cream", "Topical clindamycin"],
        "தமிழ்": ["அடப்பலீன் ஜெல்", "அசிலிக் அமில கிரீம்", "கிளின்டமைசின் டாபிக்கல்"]
    },
    "allergy": {
        "English": ["Antihistamines (Cetirizine, Loratadine)", "Topical hydrocortisone for rashes"],
        "தமிழ்": ["ஆண்டிஹிஸ்டமின்கள் (செடிரிசின், லோரடாடின்)", "தோல் சிரங்குக்கு ஹைட்ரோகோர்டிசோன் கிரீம்"]
    },
    "cellulitis": {
        "English": ["Oral antibiotics as prescribed (e.g., Amoxicillin-clavulanate, Cephalexin)"],
        "தமிழ்": ["மருத்துவர் சொல்லும் வாய்வழி ஆன்டிபயாட்டிக்கள் (உதாரணம்: அமோக்ஸிசில்லின்-கிளாவுலானேட், செபாலெக்சின்)"]
    },
    "chicken pox": {
        "English": ["Paracetamol for fever", "Antihistamines for itching", "Calamine lotion"],
        "தமிழ்": ["காய்ச்சலுக்கு பராசீட்டமால்", "குடைச்சலுக்கு ஆன்டிஹிஸ்டமின்கள்", "காலமின் லோஷன்"]
    },
    "contact dermatitis": {
        "English": ["Topical corticosteroids (hydrocortisone/mild steroid)", "Emollients/moisturizers"],
        "தமிழ்": ["உட்புற ஸ்டீராய்டு கிரீம்கள் (ஹைட்ரோகோர்டிசோன்)", "ஈரப்பதப்படுத்திகள்"]
    },
    "drug reaction": {
        "English": ["Stop suspected drug", "Antihistamines", "Systemic steroids in severe cases (doctor)"],
        "தமிழ்": ["சந்தேகப்படும் மருந்தை நிறுத்தவும்", "ஆண்டிஹிஸ்டமின்கள்", "கடுமையான நிலையில் டாக்டர் சொல்லும் ஸ்டீராய்டுகள்"]
    },
    "eczema": {
        "English": ["Emollients", "Topical corticosteroids (as advised)", "Antihistamines for itching"],
        "தமிழ்": ["ஈரப்பதப்படுத்திகள்", "டாபிக்கல் ஸ்டீராய்டு (மருத்துவர் கூறியபடி)", "குடைச்சலுக்கு ஆன்டிஹிஸ்டமின்கள்"]
    },
    "fungal infection": {
        "English": ["Clotrimazole or Miconazole cream", "Ketoconazole shampoo (if scalp)", "Oral Fluconazole if widespread"],
        "தமிழ்": ["க்ளோட்ரிமசோல்/மிக்கோநேசோல் கிரீம்", "தலைக்காக கெட்டோகனசோல் ஷாம்பு", "பெரிதாக இருந்தால் வாய்மார்தாங்கு ஃப்ளூகனசோல்"]
    },
    "herpes simplex": {
        "English": ["Acyclovir cream (topical)", "Oral Acyclovir/Valacyclovir for outbreaks"],
        "தமிழ்": ["அசிக்லோவிர் கிரீம்", "வெடிப்புகளுக்கு வாய்மார்தாங்கு அசிக்லோவிர்/வாலாசிக்லோவிர்"]
    },
    "impetigo": {
        "English": ["Topical mupirocin", "Oral antibiotics if extensive (as prescribed)"],
        "தமிழ்": ["உட்புற மியூபிரோசின் கிரீம்", "பரவலமாக இருந்தால் வாய்மார்தாங்கு ஆன்டிபயாடிக்கள்"]
    },
    "lichen planus": {
        "English": ["Topical corticosteroids", "Oral steroids or immunosuppressants for severe cases (doctor)"],
        "தமிழ்": ["டாபிக்கல் ஸ்டீராய்டுகள்", "கடுமையான நிலையில் வாய்மார்தாங்கு ஸ்டீராய்டுகள்/இம்யூனோஸப்ரஸன்ட்கள்"]
    },
    "melasma": {
        "English": ["Topical depigmenting creams (hydroquinone/azelaic acid)", " Use Sunscreen daily"],
        "தமிழ்": ["தோல் வெளிர்ப்புக் கிரீம்கள் (ஹைட்ரோகுயோனோன்/அசிலிக் அமிலம்)", "தினமும் சன்ஸ்கிரீன்"]
    },
    "perioral dermatitis": {
        "English": ["Topical metronidazole or ivermectin creams", "Oral doxycycline if needed"],
        "தமிழ்": ["டாபிக்கல் மெட்ரோனிடசோல் அல்லது இன்னொரு கிரீம்கள்", "தேவைப்பட்டால் வாய்மார்தாங்கு டாக்ஸிசைக்ளின்"]
    },
    "psoriasis": {
        "English": ["Topical corticosteroids", "Vitamin D analogues (calcipotriol)", "Systemic agents for severe cases (methotrexate)"],
        "தமிழ்": ["டாபிக்கல் ஸ்டீராய்டுகள்", "விடமின் D வடிவங்கள்", "கடுமையான போது வாய்மார்தாங்கு மருந்துகள் (மெதோட்ரெக்சேட்)"]
    },
    "ringworm": {
        "English": ["Topical antifungal (terbinafine/clotrimazole)", "Oral antifungal for widespread infection (terbinafine)"],
        "தமிழ்": ["டாபிக்கல் ஆன்டிஃபங்கல் (டெர்பினாஃபின்/க்ளோட்ரிமசோல்)", "பெரிதாக இருந்தால் வாய்மார்தாங்கு டெர்பினாஃபின்"]
    },
    "rosacea": {
        "English": ["Topical metronidazole or azelaic acid", "Oral doxycycline for inflammatory rosacea"],
        "தமிழ்": ["டாபிக்கல் மெட்ரோனிடசோல் அல்லது அசிலிக் அமிலம்", "ஈன்ஃபிளமேட்டரி ரோசேசியாவுக்கு வாய்மார்தாங்கு டாக்ஸிசைக்ளின்"]
    },
    "scabies": {
        "English": ["Permethrin 5% cream", "Oral ivermectin if indicated", "Treat contacts & wash bedding hot"],
        "தமிழ்": ["பெர்மெத்ரின் 5% கிரீம்", "தேவையானால் வாய்மார்தாங்கு ஐவர்மெக்டின்", "த்ரவைகளை சிகிச்சை மற்றும் வெப்பநீரில் துவைப்பு"]
    },
    "seborrheic dermatitis": {
        "English": ["Medicated shampoos (ketoconazole, selenium sulfide)", "Topical antifungal or mild steroids as advised"],
        "தமிழ்": ["மருந்து ஷாம்பூக்கள் (கெட்டோகனசோல், செலினியம் சல்ஃபைடு)", "டாபிக்கல் ஆன்டிஃபங்கல் அல்லது மிதமான ஸ்டீராய்டுகள்"]
    },
    "urticaria": {
        "English": ["Second-generation antihistamines (Cetirizine/Loratadine)", "Avoid triggers; short course steroids if severe"],
        "தமிழ்": ["இரண்டாம் தலைமுறை ஆன்டிஹிஸ்டமின்கள் (செடிரிசின்/லோரடாடின்)", "தூண்டுதல்களைத் தவிர்க்கவும்; கடுமையானதில் குறுகிய கால ஸ்டீராய்டு"]
    },
    "vitiligo": {
        "English": ["Topical corticosteroids or calcineurin inhibitors", "Phototherapy in specialist centers"],
        "தமிழ்": ["டாபிக்கல் ஸ்டீராய்டுகள் அல்லது கால்சின்யுரின் inhibitors", "ஸ்பெஷலிஸ்ட் மையங்களில் ஃபோட்டோதெரபி"]
    },
    "warts": {
        "English": ["Topical salicylic acid preparates", "Cryotherapy by doctor", "Imiquimod cream for certain types"],
        "தமிழ்": ["சாலிசிலிக் அமில அடிப்படை மருந்துகள்", "கிரியோதெரபி (மருத்துவர்)", "இமிகுவிமோட் கிரீம்"]
    },
    "warts (facial)": {
        "English": ["Topical salicylic acid (carefully)", "Imiquimod cream", "Cryotherapy in clinic"],
        "தமிழ்": ["முகத்துக்கு பாதுகாப்பாக சாலிசிலிக் அமிலம்", "இமிகுவிமோட் கிரீம்", "கிரியோதெரபி"]
    }
}
# -------------------- helper functions --------------------
def find_image_for_disease(disease_name):
    for ext in (".jpg", ".jpeg", ".png"):
        for name_variant in [disease_name, disease_name.replace(' ', '_'), disease_name.lower()]:
            p = os.path.join(IMAGE_FOLDER, f"{name_variant}{ext}")
            if os.path.exists(p):
                return p
    return None

def get_symptom_display_list(lang):
    if lang == "தமிழ்":
        return [SYMPTOM_TRANSLATIONS.get(s, s) for s in SYMPTOMS]
    return SYMPTOMS.copy()

def map_tamil_selected_to_english(selected_tamil):
    reverse = {tam: eng for eng, tam in SYMPTOM_TRANSLATIONS.items()}
    mapped = []
    for t in selected_tamil:
        if t in reverse:
            mapped.append(reverse[t])
        elif t in SYMPTOMS:
            mapped.append(t)
    return mapped

# -------------------- session state --------------------
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "user_choice" not in st.session_state:
    st.session_state.user_choice = None
if "confirmed_disease" not in st.session_state:
    st.session_state.confirmed_disease = None
if "show_buttons" not in st.session_state:
    st.session_state.show_buttons = False

# -------------------- symptom selection --------------------
symptoms_ui = get_symptom_display_list(language)
selected_symptoms_display = st.multiselect(UI_TEXT[language]["choose_symptoms"], symptoms_ui)
if language == "தமிழ்":
    selected_symptoms = map_tamil_selected_to_english(selected_symptoms_display)
else:
    selected_symptoms = selected_symptoms_display

# -------------------- predict --------------------
if st.button(UI_TEXT[language]["predict"]):
    if not selected_symptoms:
        st.warning(UI_TEXT[language]["warning"])
    else:
        input_df = pd.DataFrame([[1 if s in selected_symptoms else 0 for s in SYMPTOMS]], columns=SYMPTOMS)
        probs = model.predict_proba(input_df)[0]
        classes = model.classes_
        top3_idx = np.argsort(probs)[-3:][::-1]
        st.session_state.predictions = [(classes[i], float(probs[i])*100.0) for i in top3_idx]
        st.session_state.user_choice = None
        st.session_state.confirmed_disease = None
        st.session_state.show_buttons = False

# -------------------- show predictions --------------------
if st.session_state.predictions:
    st.subheader(UI_TEXT[language]["predictions"])
    for disease, conf in st.session_state.predictions:
        st.write(f"**{disease}** - {conf:.2f}%")
        img_path = find_image_for_disease(disease)
        if img_path:
            img = Image.open(img_path)
            img.thumbnail((300,300))
            st.image(img, caption=disease)
        else:
            st.warning(f"⚠️ No image found for {disease} in `{IMAGE_FOLDER}`")

    st.subheader(UI_TEXT[language]["confirm_question"])
    st.session_state.user_choice = st.radio("Select", [d for d, _ in st.session_state.predictions], index=0)

# -------------------- confirm disease --------------------
if st.button(UI_TEXT[language]["confirm_button"]):
    sel = st.session_state.user_choice
    if sel:
        st.session_state.confirmed_disease = sel
        st.session_state.show_buttons = True
        st.success(f"{UI_TEXT[language]['you_selected']}: **{sel}**")

# -------------------- show Precautions & Medicines --------------------
if st.session_state.show_buttons and st.session_state.confirmed_disease:
    normalized = st.session_state.confirmed_disease.strip().lower()
    col1, col2 = st.columns(2)
    
    if col1.button(UI_TEXT[language]["precautions"], key="prec"):
        if normalized in PRECAUTIONS:
            for p in PRECAUTIONS[normalized].get(language, PRECAUTIONS[normalized].get("English", [])):
                st.write(f"- {p}")
        else:
            st.info(UI_TEXT[language]["no_precautions"])

    if col2.button(UI_TEXT[language]["medicines"], key="meds"):
        if normalized in MEDICINES:
            for m in MEDICINES[normalized].get(language, MEDICINES[normalized].get("English", [])):
                st.write(f"- {m}")
        else:
            st.info(UI_TEXT[language]["no_meds"])

    st.info(UI_TEXT[language]["final_info"])

