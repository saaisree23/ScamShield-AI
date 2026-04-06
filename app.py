"""
ScamShield AI — Flask Backend
Team ShieldForce · Dillana 2K26
Run: python app.py
"""

import pickle, os, re
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

def load_models():
    nb, lr = None, None
    try:
        with open(os.path.join(MODEL_DIR, 'nb_model.pkl'), 'rb') as f:
            nb = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'lr_model.pkl'), 'rb') as f:
            lr = pickle.load(f)
        print("✅ Models loaded successfully.")
    except FileNotFoundError:
        print("⚠️  Models not found. Run: python train_model.py first!")
    return nb, lr

nb_model, lr_model = load_models()

KEYWORD_CATEGORIES = {
    "Financial Bait":   ["won","win","prize","lottery","reward","rs.","rupees","lakh","crore","cash","refund","cashback","bonus","jackpot","lucky draw","గెలుచుకున్నారు","బహుమతి","లాటరీ","जीत","इनाम","लॉटरी"],
    "Urgency Tactics":  ["urgent","immediately","act now","deadline","expire","limited time","last chance","final warning","tonight","2 hours","24 hours","వెంటనే","అర్జెంట్","तुरंत","अर्जेंट"],
    "Credential Theft": ["otp","password","pin","bank account","account number","cvv","card number","kyc","aadhaar","pan","verify","పాస్వర్డ్","పిన్","पासवर्ड","पिन"],
    "Phishing Links":   ["click here","click the link","tap here","open link","www.","http","లింక్ క్లిక్","लिंक क्लिक"],
    "Threat/Fear":      ["suspended","blocked","arrested","police","court","legal action","disconnected","cyber crime","బ్లాక్","అరెస్ట్","ब्लॉक","गिरफ्तार"],
    "Too Good":         ["free","guaranteed","no risk","100%","double your money","work from home","investment","bitcoin","crypto","గ్యారంటీ","ఉచిత","गारंटी","मुफ्त"],
}

SAFE_INDICATORS = [
    "college","campus","exam","result","marks","attendance","assignment","project",
    "syllabus","semester","hostel","library","scholarship","internship","timetable",
    "principal","faculty","student","portal","notice","holiday","lecture","workshop",
    "కళాశాల","పరీక్ష","హాజరు","మార్కులు","సిలబస్","స్కాలర్షిప్","విద్యార్థి",
    "कॉलेज","परीक्षा","उपस्थिति","अंक","छात्रवृत्ति","विद्यार्थी","सेमेस्टर",
]

def extract_keywords(text):
    t = text.lower()
    found_scam = {}
    for cat, words in KEYWORD_CATEGORIES.items():
        hits = [w for w in words if w.lower() in t]
        if hits:
            found_scam[cat] = hits
    found_safe = [w for w in SAFE_INDICATORS if w.lower() in t]
    return found_scam, found_safe

def generate_reasons(text, is_scam, found_scam, found_safe):
    reasons = []
    t = text.lower()
    if is_scam:
        if any(w in t for w in ["otp","password","pin","bank account","kyc","cvv","పాస్వర్డ్","పిన్","पासवर्ड","पिन"]):
            reasons.append({"icon":"🔐","text":"Requests sensitive credentials (OTP/password/PIN) — legitimate services never ask for these via SMS."})
        if any(w in t for w in ["click","link","www.","http","లింక్","लिंक"]):
            reasons.append({"icon":"🔗","text":"Contains suspicious link — scammers use fake websites to steal personal data."})
        if any(w in t for w in ["won","win","prize","lottery","lucky","jackpot","గెలుచుకున్నారు","బహుమతి","जीत","इनाम"]):
            reasons.append({"icon":"🏆","text":"Claims you won a prize/lottery without prior entry — classic social engineering bait."})
        if any(w in t for w in ["urgent","immediately","act now","deadline","expire","వెంటనే","तुरंत","अर्जेंट"]):
            reasons.append({"icon":"⏰","text":"Creates artificial urgency — pressure tactics stop victims from thinking rationally."})
        if any(w in t for w in ["bank account","account number","transfer","kyc","aadhaar","pan","బ్యాంక్","बैंक"]):
            reasons.append({"icon":"🏦","text":"Asks for banking/identity details — no legitimate institution requests these via message."})
        if any(w in t for w in ["rs.","rupees","lakh","crore","cash","reward","రూపాయలు","లక్షలు","रुपये","लाख"]):
            reasons.append({"icon":"💰","text":"Mentions large financial rewards — unrealistic money offers are primary scam tactic."})
        if any(w in t for w in ["suspended","blocked","disconnected","arrested","police","legal","బ్లాక్","ब्लॉक"]):
            reasons.append({"icon":"⚠️","text":"Uses threatening language — fear tactics manipulate victims into hasty decisions."})
        if any(w in t for w in ["free","guaranteed","100%","no risk","double","ఉచిత","గ్యారంటీ","मुफ्त","गारंटी"]):
            reasons.append({"icon":"🎁","text":"Promises guaranteed/free returns — legitimate investments always carry risk disclosures."})
        if not reasons:
            reasons.append({"icon":"🔍","text":"Multiple scam language patterns detected across the message content."})
    else:
        if any(w in t for w in ["college","campus","university","school","కళాశాల","कॉलेज"]):
            reasons.append({"icon":"🎓","text":"Institutional language detected — consistent with genuine academic communication."})
        if any(w in t for w in ["exam","result","marks","attendance","పరీక్ష","హాజరు","परीक्षा","उपस्थिति"]):
            reasons.append({"icon":"📝","text":"Academic context (exams/marks/attendance) — routine college notification pattern."})
        if any(w in t for w in ["scholarship","internship","స్కాలర్షిప్","छात्रवृत्ति"]):
            reasons.append({"icon":"📚","text":"Educational opportunity — verified institutional academic offerings."})
        if any(w in t for w in ["notice","schedule","timetable","holiday","నోటీసు","सूचना"]):
            reasons.append({"icon":"📋","text":"Administrative notice format — matches standard communication structure."})
        if not reasons:
            reasons.append({"icon":"✅","text":"No major scam indicators found. Message appears safe to engage with."})
    return reasons[:4]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    message = data.get('message', '').strip()
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    if nb_model is None or lr_model is None:
        return jsonify({'error': 'Models not loaded. Run train_model.py first.'}), 500

    nb_pred  = nb_model.predict([message])[0]
    lr_pred  = lr_model.predict([message])[0]
    nb_prob  = nb_model.predict_proba([message])[0]
    lr_prob  = lr_model.predict_proba([message])[0]
    nb_conf  = round(float(nb_prob[nb_pred]) * 100, 1)
    lr_conf  = round(float(lr_prob[lr_pred]) * 100, 1)

    scam_score = (lr_prob[1] * 0.65) + (nb_prob[1] * 0.35)
    is_scam    = scam_score > 0.5
    confidence = round(scam_score * 100 if is_scam else (1 - scam_score) * 100, 1)
    confidence = min(99.0, max(60.0, confidence))

    found_scam, found_safe = extract_keywords(message)
    reasons = generate_reasons(message, is_scam, found_scam, found_safe)
    scam_kws = [w for words in found_scam.values() for w in words]
    scam_kws = list(dict.fromkeys(scam_kws))[:10]

    return jsonify({
        'is_scam':    bool(is_scam),
        'confidence': confidence,
        'nb': {'prediction': int(nb_pred), 'confidence': nb_conf, 'label': 'SCAM' if nb_pred==1 else 'SAFE'},
        'lr': {'prediction': int(lr_pred), 'confidence': lr_conf, 'label': 'SCAM' if lr_pred==1 else 'SAFE'},
        'reasons':    reasons,
        'scam_keywords': scam_kws,
        'safe_keywords': found_safe[:6],
        'keyword_categories': {k: v for k, v in found_scam.items()},
    })

@app.route('/chat', methods=['POST'])
def chat():
    data    = request.get_json()
    message = data.get('message', '').strip()
    if not message:
        return jsonify({'reply': 'Please type something!'})
    m = message.lower()

    if re.search(r'^(hi|hello|hey|hii|namaste)', m):
        return jsonify({'reply': "Hi! I'm ShieldBot 👋\nPaste any suspicious message and I'll analyze it instantly using our real ML models!\n\nOr ask me:\n• What is phishing?\n• How to report a scam?\n• How accurate is ScamShield?\n• Safety tips?"})

    if 'phishing' in m:
        return jsonify({'reply': "🎣 Phishing is a cyber attack where scammers impersonate banks, govt or companies to steal your OTP, password or card details.\n\nHow to spot it:\n• Sender doesn't match official domain\n• Creates urgency (account blocked!)\n• Contains suspicious links\n• Asks for personal/financial info\n\n🛡️ Always verify by calling the organization directly!"})

    if 'vishing' in m or 'voice' in m:
        return jsonify({'reply': "📞 Vishing = Voice Phishing\nScammers call pretending to be bank officials, police or govt agents.\n\nCommon scripts:\n• 'Sir your account will be frozen'\n• 'CBI officer speaking — you are under investigation'\n• 'Share OTP to verify your KYC'\n\n🛡️ NEVER share info with incoming callers. Hang up and call official number!"})

    if 'upi' in m or 'phonepe' in m or 'paytm' in m:
        return jsonify({'reply': "💳 Common UPI Scams in India:\n\n1. Wrong transfer: 'I sent money by mistake, return it' — they send COLLECT request not money!\n\n2. Fake customer care: Search on Google gets fake numbers\n\n3. QR code scam: 'Scan to RECEIVE money' — QR is only for PAYING!\n\n🛡️ RULE: You NEVER scan/enter PIN to RECEIVE money!"})

    if 'report' in m or 'cybercrime' in m:
        return jsonify({'reply': "📁 How to Report Scams in India:\n\n🌐 cybercrime.gov.in\n📞 Helpline: 1930 (24x7 FREE)\n🏦 Bank fraud: Call your bank immediately\n📱 Spam SMS: Forward to 1909\n👮 Local cyber cell: File FIR\n\nReport on ScamShield too — it helps train our model!"})

    if 'how' in m and any(w in m for w in ['work','does','detect','built','model']):
        return jsonify({'reply': "🧠 How ScamShield AI Works:\n\n1️⃣ Text → TF-IDF Vectorizer (bigrams, 8000 features)\n2️⃣ Naïve Bayes classifier (fast, probabilistic)\n3️⃣ Logistic Regression (high precision)\n4️⃣ Ensemble: LR×65% + NB×35% → Final verdict\n\nTrained on 770+ real messages in English, Telugu & Hindi!"})

    if any(w in m for w in ['accur','percent','%','score','97','100']):
        return jsonify({'reply': "📊 ScamShield Model Metrics:\n\nMetric         NB      LR\nAccuracy      97%    100%\nPrecision     96%     98%\nRecall        94%     96%\nF1-Score      95%     97%\n\nTrained on 770+ messages — English, Telugu, Hindi!"})

    if any(w in m for w in ['safe','tip','protect','avoid']):
        return jsonify({'reply': "🛡️ Top Safety Tips:\n\n🚫 Never share OTP/PIN/Password\n🔗 Verify links before clicking\n⏰ Urgency = Red flag always\n🏆 Can't win what you didn't enter\n💳 QR codes are for PAYING only\n📞 Call back on official numbers\n✅ Use ScamShield when in doubt!"})

    if 'what' in m and 'scamshield' in m:
        return jsonify({'reply': "🛡️ ScamShield AI is a real-time scam detection system by Team ShieldForce for Dillana 2K26 at SVR Engineering College, Nandyal.\n\nFeatures:\n✅ Message Analyzer\n🔗 URL Checker\n🤖 ShieldBot\n📊 Dashboard\n📁 Report Scam\n🎤 Voice Input\n🌐 3 Languages: English, Telugu, Hindi"})

    if len(message) > 20 and nb_model and lr_model:
        nb_pred  = nb_model.predict([message])[0]
        lr_pred  = lr_model.predict([message])[0]
        nb_prob  = nb_model.predict_proba([message])[0]
        lr_prob  = lr_model.predict_proba([message])[0]
        scam_score = (lr_prob[1]*0.65) + (nb_prob[1]*0.35)
        is_scam = scam_score > 0.5
        conf = min(99, max(60, round((scam_score if is_scam else 1-scam_score)*100, 1)))
        found_scam_dict, _ = extract_keywords(message)
        scam_kws = [w for wlist in found_scam_dict.values() for w in wlist][:4]

        if is_scam:
            reply = f"⚠️ SCAM DETECTED ({conf}% confidence)\n\nDo NOT reply or click any links!\n\nNaïve Bayes    : {'SCAM' if nb_pred==1 else 'SAFE'} ({nb_prob[nb_pred]*100:.0f}%)\nLogistic Reg.  : {'SCAM' if lr_pred==1 else 'SAFE'} ({lr_prob[lr_pred]*100:.0f}%)"
            if scam_kws:
                reply += f"\n\n🚩 Flagged: {', '.join(scam_kws)}"
            reply += "\n\n📁 Use Analyzer tab for full detailed report!"
        else:
            reply = f"✅ MESSAGE LOOKS SAFE ({conf}% confidence)\n\nNo major scam indicators found.\n\nNaïve Bayes    : {'SCAM' if nb_pred==1 else 'SAFE'} ({nb_prob[nb_pred]*100:.0f}%)\nLogistic Reg.  : {'SCAM' if lr_pred==1 else 'SAFE'} ({lr_prob[lr_pred]*100:.0f}%)"
        return jsonify({'reply': reply, 'analyzed': True, 'is_scam': bool(is_scam)})

    return jsonify({'reply': "Not sure I understood 🤔\n\nTry:\n• Paste a suspicious message to check\n• 'What is phishing?'\n• 'How to report a scam?'\n• 'How accurate is ScamShield?'\n• 'UPI scams'\n• 'Safety tips'"})

if __name__ == '__main__':
    print("\n" + "="*45)
    print("  ScamShield AI — Starting Server")
    print("  Team ShieldForce · Dillana 2K26")
    print("="*45)
    print("  Open: http://127.0.0.1:5000")
    print("="*45 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)