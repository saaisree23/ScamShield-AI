"""
ScamShield AI — Model Training Script
Team ShieldForce · Dillana 2K26
Run: python train_model.py
"""

import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ════════════════════════════════════════════════════════════════
#  SCAM MESSAGES — English (500+), Telugu (20), Hindi (20)
# ════════════════════════════════════════════════════════════════
SCAM_MESSAGES = [

    # ── English Scam: Lottery / Prize (80) ──────────────────────
    "Congratulations! You have won Rs.50,000 in our lucky draw. Click here to claim now.",
    "You have been selected as the lucky winner of Rs.2 lakh prize. Contact immediately.",
    "WINNER! Your mobile number won Rs.10 lakh in the national lottery. Claim today.",
    "Dear winner, you have won an iPhone 15 Pro. Pay Rs.200 shipping to receive it now.",
    "Congratulations! You won a car in our lucky draw. Pay Rs.5,000 insurance to claim.",
    "Your number has been selected for Rs.25 lakh KBC prize. Call us immediately.",
    "You are the lucky winner of Rs.1 crore jackpot. Share bank details to receive funds.",
    "LUCKY DRAW WINNER! Claim your Rs.75,000 gift voucher. Limited time offer. Act now.",
    "You won a free laptop in our annual lucky draw. Click link and enter details to claim.",
    "Congratulations! Your SIM number selected for Rs.15 lakh prize. Contact within 24 hrs.",
    "You have been chosen for Rs.5 lakh reward. Share your bank account number immediately.",
    "JACKPOT WINNER! Rs.50 lakh waiting for you. Verify identity by sending OTP now.",
    "Your WhatsApp number won Rs.3 lakh in international lottery. Claim before expiry.",
    "You won gold coins worth Rs.2 lakh. Pay Rs.500 processing fee to receive delivery.",
    "Congratulations! You are selected for Rs.10 lakh government prize scheme. Register now.",
    "WINNER ALERT! Your number drawn for Rs.20 lakh cash prize. Call helpline immediately.",
    "You won free shopping worth Rs.50,000 at Amazon. Click link to redeem voucher today.",
    "Lucky customer! You won Rs.1 lakh cashback on your recent purchase. Claim now.",
    "Congratulations! Your email won Rs.5 lakh in international sweepstakes. Reply to claim.",
    "You are our 1 millionth customer! Claim your Rs.2 lakh prize by clicking this link.",
    "GRAND PRIZE WINNER! Rs.30 lakh cash prize in your name. Verify by sharing OTP.",
    "You won a diamond necklace worth Rs.1.5 lakh. Pay Rs.299 courier fee to receive.",
    "Your phone number selected in bumper lottery. Rs.45 lakh prize waiting. Call now.",
    "Congratulations from Google! You won Rs.8 lakh in Google lucky draw. Claim today.",
    "You won free recharge worth Rs.999. Click link and enter your mobile number now.",
    "LOTTERY WINNER! Rs.12 lakh prize in your name. Share account details to transfer.",
    "You won an international trip worth Rs.3 lakh. Pay Rs.1,000 tax to claim prize.",
    "Congratulations! Amazon lucky draw winner. Rs.1 lakh gift card. Click to redeem.",
    "Your lucky number won Rs.7 lakh cash. Send copy of ID to release funds today.",
    "You won Rs.18 lakh in TATA lucky draw. Processing fee Rs.2,000. Act immediately.",
    "Flipkart grand prize: You won Rs.25,000 shopping voucher. Claim within 2 hours.",
    "Your number is today's lucky winner! Rs.99,000 prize. Call to collect immediately.",
    "Congratulations! You won Rs.4 lakh in Reliance Jio lucky draw. Verify now.",
    "BUMPER PRIZE: Rs.55 lakh in your name. Pay 0.5% tax Rs.27,500 to receive now.",
    "You won a BMW car worth Rs.45 lakh. Pay registration Rs.10,000 to claim vehicle.",
    "International lottery winner! You won $10,000. Share bank details for wire transfer.",
    "Your Instagram account selected for Rs.2 lakh influencer prize. Claim immediately.",
    "You won 1 kg gold in Dhanteras lucky draw. Pay GST Rs.3,000 to receive gold.",
    "PRIZE ALERT! Rs.6 lakh cash prize waiting for you. Call within 1 hour to claim.",
    "Congratulations winner! Rs.11 lakh cheque ready. Share address to send by courier.",

    # ── English Scam: OTP / Bank / KYC (80) ─────────────────────
    "URGENT: Your bank account has been suspended. Verify your OTP immediately to restore.",
    "Dear customer, your KYC is expired. Update now or your account will be blocked in 24 hrs.",
    "Alert: Suspicious activity detected on your account. Share OTP to secure immediately.",
    "Your SBI account will be blocked. Call and share your password to prevent closure.",
    "URGENT: Rs.47,500 unauthorized debit from your account. Share OTP to stop transaction.",
    "Your HDFC account shows unusual login. Verify identity by entering OTP on this link.",
    "Account security alert! Someone trying to access your account. Share PIN to block them.",
    "Your net banking is temporarily suspended. Re-activate by clicking link and entering OTP.",
    "Dear user, complete KYC in 48 hours or your bank account will be permanently blocked.",
    "ALERT: Your debit card used at suspicious location. Block it by sharing card number now.",
    "Your Aadhaar linked account has been compromised. Call and share OTP immediately.",
    "UPI fraud alert! Unauthorized Rs.15,000 transfer attempted. Share PIN to cancel now.",
    "Your credit card is blocked due to suspicious transaction. Unblock by sharing CVV.",
    "BANK NOTICE: Rs.9,999 will be deducted as maintenance fee. Cancel by sharing OTP.",
    "Your account KYC documents are pending. Submit now or account closes within 24 hours.",
    "Security breach detected in your savings account. Verify by sharing account number.",
    "Your ATM card will expire tonight. Renew by clicking this link and entering PIN now.",
    "URGENT from RBI: Your account linked to money laundering. Share details to clear name.",
    "Your mobile banking is suspended. Reactivate by entering OTP on the official link.",
    "Credit card reward points expiring! Redeem Rs.45,000 points by sharing card number.",
    "Your bank account will be closed if KYC not updated. Click link to update in 2 hours.",
    "Suspicious login attempt on your account from unknown location. Verify OTP to secure.",
    "Dear account holder, your fixed deposit matures today. Renew by sharing account PIN.",
    "ALERT: Your PAN card linked to fraudulent transactions. Verify by calling immediately.",
    "Your ICICI bank account frozen due to suspicious activity. Share OTP to unfreeze now.",
    "Account update mandatory: Link Aadhaar to bank account by today or face deactivation.",
    "Your net banking password will expire in 2 hours. Reset by entering OTP on this page.",
    "Fraud alert from Axis Bank. Unknown device logged in. Share OTP to block immediately.",
    "Your bank is upgrading systems. Re-verify account by entering details on secure portal.",
    "IMMEDIATE ACTION: Deactivate unauthorized UPI ID linked to your account. Share PIN.",
    "Your Kotak bank account shows negative balance. Pay Rs.500 fee or account gets blocked.",
    "Dear customer, your bank mandate needs renewal. Share account details to continue.",
    "Your mobile number not linked to bank account. Update now or lose access to UPI.",
    "Bank security team: We detected your account cloning attempt. Share OTP to reverse.",
    "Your savings account interest rate is being reduced. Upgrade by sharing account details.",
    "Final notice: Update your bank account nominee details or account will be frozen.",
    "Your PNB account shows unauthorized transaction. Dispute by sharing account number now.",
    "SECURITY ALERT: Someone changed your UPI PIN. Revert change by sharing OTP immediately.",
    "Your bank FD worth Rs.2 lakh matures today. Renew online by entering account details.",
    "Account verification required: Submit bank statement and OTP within 6 hours.",

    # ── English Scam: Job / Work From Home (60) ─────────────────
    "Work from home! Earn Rs.50,000 per month. No experience needed. Register free today.",
    "Part time job: Earn Rs.2,000 daily by liking YouTube videos. Immediate start.",
    "Data entry jobs from home. Earn Rs.500 per hour. Registration fee Rs.299. Join now.",
    "HIRING NOW: Online survey jobs. Earn Rs.1,500 per survey. No experience needed.",
    "Earn Rs.3,000 daily from home by typing on your mobile. Free registration. Apply now.",
    "Work from home with Amazon! Earn Rs.45,000 monthly. Registration open. Limited seats.",
    "Online earning opportunity: Rs.10,000 per day guaranteed. No skills required. Join now.",
    "Part time online job: Copy paste work. Earn Rs.800 per hour. Registration Rs.199.",
    "URGENT HIRING: Social media managers work from home. Rs.35,000 monthly. Apply today.",
    "Earn money online: Watch ads and get paid Rs.50 per ad. Guaranteed daily payment.",
    "Home based packaging job. Earn Rs.15,000 weekly. Raw materials provided. Register now.",
    "Online teaching job: Teach from home. Earn Rs.60,000 monthly. No qualification needed.",
    "Earn Rs.5,000 daily by sharing referral links. Join our network. Registration free.",
    "Work from home: Image tagging work for AI company. Rs.400 per hour. Apply online.",
    "LEGITIMATE WORK: Earn Rs.25,000 per month testing apps from home. Join today free.",
    "Online reselling business: Earn Rs.1 lakh monthly. No investment. Products provided.",
    "Earn Rs.2,500 daily completing simple tasks on mobile. No experience required. Free join.",
    "BPO work from home: Rs.20,000 monthly. Voice and non-voice process. Apply immediately.",
    "Freelance data entry: Rs.30 per entry. Work anytime. Minimum payout Rs.500. Join now.",
    "MAKE MONEY ONLINE: Be your own boss. Earn Rs.70,000 monthly from home. Start today.",
    "Amazon affiliate program: Earn Rs.40,000 monthly sharing product links. Free to join.",
    "Earn by writing articles online. Rs.500 per article. No experience needed. Register now.",
    "Online tutoring: Teach students globally from home. Earn Rs.80,000 monthly easily.",
    "Stock market training: Learn and earn Rs.1 lakh monthly. Guaranteed profits. Join now.",
    "Earn Rs.7,000 daily with our proven system. Work 2 hours from home. Join free today.",
    "WORK FROM HOME: Insurance agent earn Rs.1 lakh monthly commission. No target. Apply.",
    "Digital marketing work from home: Rs.45,000 monthly. Training provided. Apply now.",
    "Earn Rs.15,000 weekly completing simple online tasks. Payment daily. Free registration.",
    "Home based manufacturing: Earn Rs.25,000 weekly. Materials supplied. Deposit Rs.2,000.",
    "Online business opportunity: Invest Rs.5,000 get Rs.50,000 monthly. Guaranteed income.",

    # ── English Scam: Investment / Crypto (50) ───────────────────
    "Crypto investment opportunity! Invest Rs.10,000 get Rs.1 lakh in 30 days. Guaranteed.",
    "Bitcoin trading: 300% returns in 7 days. Send Bitcoin to this wallet. Certified brokers.",
    "Forex trading signals: 100% accurate. Earn Rs.5 lakh monthly. Join our trading group.",
    "Stock market tips: Guaranteed 50% profit in 1 week. Subscribe Rs.999. Sure shot tips.",
    "Investment scheme: Double your money in 15 days. RBI approved. Limited slots available.",
    "Mutual fund with 40% monthly returns. Invest Rs.50,000 get Rs.70,000 next month.",
    "Gold investment scheme: Buy digital gold now. 25% guaranteed returns quarterly.",
    "Cryptocurrency arbitrage bot: Earn Rs.3,000 daily passively. Investment Rs.15,000.",
    "Multi-level marketing: Join our team. Earn Rs.2 lakh monthly with small investment.",
    "Fixed deposit scheme: 24% annual interest. Limited period offer. Invest before March.",
    "Ponzi alert: Earn Rs.500 for every Rs.1,000 invested weekly. 100% safe and legal.",
    "Real estate investment: Rs.50,000 investment gives Rs.5 lakh return in 6 months.",
    "Chit fund scheme: Invest Rs.1,000 monthly. Win Rs.1 lakh chit. Guaranteed bid.",
    "Stock tips today: Buy this penny stock. 10x returns guaranteed. Insider information.",
    "Binary options trading: Earn Rs.10,000 per trade. 95% accuracy guaranteed. Join now.",
    "NFT investment: Buy these tokens now. 1000% profit guaranteed next month. Act fast.",
    "Angel investment opportunity: Get 10x returns. Join elite investor group. Rs.25,000.",
    "Trading academy: Pay Rs.5,000. Learn secrets to earn Rs.50,000 daily. Enroll today.",
    "Agricultural investment: Invest Rs.1 lakh in farm. Get Rs.50,000 profit every season.",
    "Online poker earnings: Learn to earn Rs.1 lakh monthly. Foolproof strategy. Join now.",

    # ── English Scam: Government / Tax (50) ─────────────────────
    "Income tax refund of Rs.15,230 pending. Submit bank account details to receive.",
    "You qualify for PM housing scheme. Get Rs.2 lakh subsidy. Share Aadhaar and bank details.",
    "Government scheme: Get free Rs.6,000 per month. Register by sharing bank account now.",
    "URGENT tax notice: Pay Rs.2,000 fine or face arrest. Call immediately to avoid action.",
    "Your Aadhaar card is being misused. Call cyber cell and share OTP to protect yourself.",
    "Free solar panels under government scheme. Register now. Share Aadhaar to apply.",
    "PM Kisan scheme payment of Rs.2,000 pending. Update bank details to receive funds.",
    "Income tax department: Rs.45,000 refund approved. Submit bank details to receive.",
    "Free health insurance Rs.5 lakh under Ayushman Bharat. Register by sharing Aadhaar.",
    "Government laptop scheme for students. Register by paying Rs.200 registration fee.",
    "Your Aadhaar linked to suspicious account. Verify by calling and sharing details now.",
    "Tax raid on your property scheduled. Pay Rs.10,000 to cancel raid. Call immediately.",
    "EPFO notice: Your PF amount of Rs.1.2 lakh unclaimed. Share bank details to transfer.",
    "Voter ID correction: Update details or lose voting rights. Click link immediately.",
    "GST refund of Rs.25,000 approved for your business. Submit account details today.",
    "Cyber crime cell: Your IP address used in illegal downloads. Pay Rs.5,000 to clear case.",
    "ED notice: Your bank account linked to fraud case. Call immediately to give statement.",
    "Free LPG connection under government scheme. Register by sharing Aadhaar and address.",
    "Police case against your mobile number. Pay Rs.3,000 fine online to close complaint.",
    "Your passport renewal urgent. Share Aadhaar and pay Rs.500 processing fee online now.",

    # ── English Scam: Utility / Service (40) ────────────────────
    "Your electricity will be cut in 2 hours. Pay Rs.850 pending bill immediately now.",
    "BESCOM alert: Electricity disconnection tonight. Pay outstanding Rs.1,200 immediately.",
    "Your gas connection will be blocked. Pay pending amount Rs.650 on this number now.",
    "Internet disconnection notice: Pay Rs.750 overdue bill today or lose connection.",
    "Your mobile SIM will be blocked tonight. Update KYC by clicking link now.",
    "TRAI notice: Your number will be deactivated. Submit documents on link immediately.",
    "Water connection bill overdue. Pay Rs.450 today or supply will be disconnected.",
    "Your DTH subscription expired. Renew now by paying Rs.399 on this link. Act fast.",
    "IRCTC account blocked due to suspicious booking. Verify by sharing login details.",
    "Your Amazon account locked. Verify by clicking link and entering bank details now.",
    "Netflix payment failed. Update payment details by clicking link or lose subscription.",
    "Your Google account will be deleted. Verify phone number by entering OTP immediately.",
    "Facebook account compromised. Secure by entering password on this verification link.",
    "Your driving license will be suspended. Pay traffic challan Rs.2,500 online now.",
    "Vehicle insurance expired. Renew online immediately or face Rs.5,000 fine from police.",
    "Your ration card will be cancelled. Link Aadhaar by clicking this link immediately.",
    "Mobile tower installation: Earn Rs.40,000 monthly rent. Share property details now.",
    "Your property tax is overdue. Pay Rs.3,500 online immediately to avoid penalty.",
    "Insurance premium bounce: Pay Rs.1,200 today or policy will lapse permanently.",
    "Your EPF account will be frozen. Update KYC by clicking link and entering OTP now.",

    # ── English Scam: Phishing / Tech Support (40) ──────────────
    "Your computer has virus. Call Microsoft support immediately. Share remote access now.",
    "Google security alert: Your Gmail hacked. Verify by entering password on this link.",
    "Your iPhone iCloud storage full. Upgrade now by entering Apple ID and password here.",
    "Tech support: Your Windows license expired. Renew by paying Rs.1,500 on this link.",
    "Virus detected on your device! Install this app immediately to remove malware now.",
    "Your WhatsApp will be banned. Verify account by entering OTP on this official link.",
    "Instagram security: Someone logged into your account. Verify by sharing password now.",
    "Your Telegram account will be deleted. Verify by entering OTP on verification link.",
    "Amazon security: Unauthorized purchase Rs.24,999. Cancel by calling this number now.",
    "PayPal account suspended. Verify your identity by clicking link and entering details.",
    "Your email account will expire. Reactivate by entering password on this secure page.",
    "Antivirus expired! Your device at risk. Purchase renewal by clicking link now.",
    "Google has detected malware on your phone. Install security app from this link now.",
    "Your domain name is expiring. Renew now by clicking link and paying Rs.999 fee.",
    "Apple ID locked. Verify your identity immediately by entering details on this page.",
    "Suspicious download detected. Your files at risk. Call tech support immediately.",
    "Your smart TV is hacked. Factory reset now and call for recovery assistance Rs.2,000.",
    "Browser warning: This site is infected. Install cleaner app from this link now.",
    "Your Zoom account hacked. Change password immediately on this secure verification link.",
    "LinkedIn account activity suspicious. Verify by entering password on this page now.",

    # ── English Scam: Romance / Social (30) ─────────────────────
    "Hi! I am beautiful girl from Russia. Want to be friends? Send me gift card to chat.",
    "I am army officer deployed abroad. Need your help to transfer $50,000. Share details.",
    "I found your profile online. You seem kind. Can you help me with Western Union transfer?",
    "I am stuck abroad. My wallet stolen. Please send Rs.10,000 I will return double.",
    "Meet singles in your area! Pay Rs.500 to unlock premium matches. Limited time offer.",
    "Lonely housewife wants to meet you. Click link to see profile. Pay Rs.199 to chat.",
    "Your secret admirer sent you gift. Pay Rs.299 delivery fee to receive it now.",
    "Dating site: 100 girls waiting to chat with you. Subscribe Rs.499 to start chatting.",
    "I am orphan inheriting $1 million. Help me transfer. You keep 30%. Share account.",
    "Millionaire wants life partner. No dowry. Share details to get selected for marriage.",

    # ── English Scam: Health / Insurance (30) ───────────────────
    "FREE Covid insurance Rs.10 lakh. Register by giving Aadhaar and bank account now.",
    "Weight loss miracle drug! Lose 10 kg in 10 days. Order now pay Rs.1,500 on delivery.",
    "Cancer cure medicine available. Doctors hiding this secret. Order online Rs.2,000.",
    "Your health insurance claim of Rs.50,000 approved. Share bank details to receive.",
    "Corona positive detected in your area. Download this app to get free treatment now.",
    "Ayurvedic medicine cures diabetes permanently. Order now. Rs.800 home delivery.",
    "Hair regrowth guaranteed in 30 days. Rs.1,200 per bottle. Order 2 get 1 free now.",
    "Life insurance maturity amount Rs.3 lakh ready. Submit bank details for transfer.",
    "Medical emergency fund: Rs.2 lakh available for you. Apply now with Aadhaar details.",
    "Diabetes reversal program Rs.5,000. 100% guaranteed results or full refund. Join now.",

    # ── English Scam: Miscellaneous (60) ────────────────────────
    "Send Rs.100 to this UPI and get Rs.1,000 back. Trusted by 50,000 members. Join now.",
    "Wrong transfer: I accidentally sent Rs.5,000 to your account. Please return it now.",
    "Your nominee claimed your LIC policy Rs.5 lakh. Verify by sharing OTP and PIN.",
    "Free 5G SIM upgrade! Click link and enter OTP to upgrade to 5G network now.",
    "BEWARE: Your Aadhaar used to open fake accounts. Call and share OTP to block.",
    "Your child selected for free government coaching. Pay Rs.500 registration fee now.",
    "Earn by referring friends! Get Rs.500 per referral. Register free and start today.",
    "Chain message: Forward to 10 people and get Rs.1,000 recharge instantly. Share now.",
    "Your old currency notes can be exchanged. Contact us immediately with your bundle.",
    "Black money conversion: We convert Rs.10 lakh black to white. Pay 10% commission.",
    "URGENT: Someone using your PAN card for loans. Pay Rs.5,000 to block misuse now.",
    "Your education loan has been approved for Rs.5 lakh. Pay Rs.1,000 processing fee.",
    "Free laptop for BPL families under government scheme. Register with Aadhaar now.",
    "Your car loan has been pre-approved! Zero interest for 5 years. Apply immediately.",
    "Earn Rs.1,500 per hour as mystery shopper. No experience needed. Register free now.",
    "Your PF withdrawal of Rs.85,000 is stuck. Pay Rs.2,000 fee to release immediately.",
    "Lucky spin: Spin the wheel and win Rs.50,000. Click link to spin now. Free entry.",
    "Your deceased relative left Rs.10 lakh for you. Pay duty Rs.5,000 to claim inheritance.",
    "FLASH SALE: Buy 1 get 10 free! Only 100 products. Click link and order immediately.",
    "Your IPL team won contest! Claim Rs.25,000 prize by registering on this link now.",
    "Donate Rs.100 for flood victims and get Rs.1,000 cashback from our charity fund.",
    "Your home loan rate reduced to 3%. Refinance now. Limited period. Apply immediately.",
    "Dream home scheme: Book flat at Rs.1 lakh. Government subsidized. Register today.",
    "Your subscription to adult website was activated. Cancel by calling immediately.",
    "FINAL OFFER: Rs.2 lakh debt waived if you pay Rs.10,000 settlement fee today.",
    "Free MBA from abroad: Scholarship available. Pay Rs.5,000 processing fee to apply.",
    "Your account selected for Rs.1 lakh cashback promotion. Activate by sharing OTP now.",
    "URGENT: Your number used to send illegal messages. Pay fine Rs.8,000 immediately.",
    "Send Rs.500 to this number. It will multiply to Rs.5,000 in 24 hours. Guaranteed.",
    "Your property registered in your name without knowledge. Call and share details now.",
    "International parcel held at customs. Pay Rs.2,500 duty to release your package.",
    "Your online order Rs.15,000 cancelled. Refund to bank account. Share account details.",
    "You qualify for zero interest personal loan Rs.10 lakh. Share documents to process.",
    "ALERT: Pornographic content found linked to your IP. Pay Rs.3,000 to clear record.",
    "Free school kit for your child. Government scheme. Register with Aadhaar details.",
    "Your stock market demat account will be frozen. Update KYC by sharing details now.",
    "Earn Rs.500 per hour watching Netflix. Get paid to test streaming content from home.",
    "Your UPI showing transaction limit issue. Fix by sharing PIN on this official page.",
    "Exclusive club membership: Pay Rs.5,000 join fee. Earn Rs.50,000 per month returns.",
    "Your credit card limit doubled! Activate by calling and sharing card details now.",
    "BREAKING: RBI cancels your bank. Withdraw all money immediately. Urgent notice.",
    "Your WhatsApp Gold subscription is active. Disable by clicking link to avoid charges.",
    "FAKE MEDICINE ALERT: Buy genuine Viagra online Rs.500. Discrete home delivery.",
    "You have unused Jio points worth Rs.5,000. Redeem by clicking link and entering OTP.",
    "Personal loan approved in 5 minutes! Rs.5 lakh. No documents. Share Aadhaar only.",
    "Your building water connection will be cut. Pay maintenance Rs.1,500 on UPI now.",
    "Tax audit selected for your returns. Pay Rs.15,000 consultant fee to avoid scrutiny.",
    "Your employee ID is being used fraudulently. Share details to file complaint now.",
    "COURT SUMMONS: You must appear in court. Pay Rs.5,000 fine to settle case online.",
    "Paytm lottery: Your number won Rs.5 lakh. Claim by entering OTP on Paytm link now.",
    "Free recharge Rs.999 for Airtel users. Enter mobile number and OTP to receive now.",
    "Your car met with accident. Insurance claim Rs.2 lakh. Share bank details to process.",
    "Emergency: Your family member in hospital. Need Rs.10,000 urgently. Send on UPI.",
    "Your ancestral property worth Rs.50 lakh. Claim by sharing identity documents now.",
    "GST input credit of Rs.35,000 available. Claim by sharing GSTIN and bank details.",
    "Your gold loan interest overdue. Pay Rs.2,500 today or gold will be auctioned.",
    "Online casino: Win Rs.1 lakh tonight. Sign up free and get Rs.500 bonus to start.",
    "FLASH MOB SCAM: You seen in viral video. Pay Rs.2,000 to remove video from internet.",
    "Your sim card will be deactivated in 2 hours. Reactivate by clicking link and OTP.",

    # ── Telugu Scam Messages (20) ────────────────────────────────
    "అభినందనలు! మీరు Rs.50,000 లాటరీ గెలుచుకున్నారు. వెంటనే మీ బ్యాంక్ అకౌంట్ వివరాలు పంపించండి.",
    "మీ మొబైల్ నంబర్ లక్కీ డ్రా లో ఎంపికైంది. Rs.2 లక్షల బహుమతి పొందడానికి ఇప్పుడే క్లెయిమ్ చేయండి.",
    "అభినందనలు! మీరు KBC లో Rs.25 లక్షలు గెలుచుకున్నారు. OTP పంపించండి బహుమతి పొందడానికి.",
    "మీకు ఉచిత iPhone గెలుచుకున్నారు. లింక్ క్లిక్ చేసి మీ వివరాలు నమోదు చేయండి వెంటనే.",
    "Lucky winner! మీరు Rs.10 లక్షల కారు గెలుచుకున్నారు. Rs.5,000 ఇన్సూరెన్స్ ఫీజు చెల్లించండి.",
    "మీ SBI అకౌంట్ సస్పెండ్ చేయబడింది. వెంటనే OTP ద్వారా వెరిఫై చేయండి లేదా అకౌంట్ బ్లాక్ అవుతుంది.",
    "మీ KYC గడువు తీరింది. ఇప్పుడే అప్డేట్ చేయండి లేదా మీ బ్యాంక్ అకౌంట్ 24 గంటల్లో బ్లాక్ అవుతుంది.",
    "మీ పేటీఎం వాలెట్ బ్లాక్ అయింది. ఈ లింక్ పై క్లిక్ చేసి పాస్వర్డ్ మరియు OTP ఎంటర్ చేయండి.",
    "అర్జెంట్: మీ అకౌంట్ నుండి అనుమానాస్పద లావాదేవీ జరిగింది. OTP పంపించి సురక్షితం చేయండి.",
    "ఇంటి నుండి పని: రోజుకు Rs.3,000 సంపాదించండి. నైపుణ్యం అవసరం లేదు. ఇప్పుడే రిజిస్టర్ చేయండి.",
    "మీ విద్యుత్ కనెక్షన్ 2 గంటల్లో కట్ చేయబడుతుంది. Rs.850 ఇప్పుడే ఈ నంబర్ కు చెల్లించండి.",
    "సైబర్ క్రైమ్ సెల్: మీ నంబర్ పై కేసు నమోదైంది. Rs.5,000 చెల్లించి కేసు మూసివేయండి.",
    "Crypto పెట్టుబడి: Rs.10,000 పెట్టండి, 30 రోజుల్లో Rs.1 లక్ష పొందండి. 100% గ్యారంటీ.",
    "మీ UPI బ్లాక్ చేయబడింది. యాక్సెస్ పునరుద్ధరించడానికి ఈ లింక్ లో PIN ఎంటర్ చేయండి వెంటనే.",
    "ప్రభుత్వ స్కీమ్: Rs.6,000 నెలవారీ పొందండి. మీ ఆధార్ మరియు బ్యాంక్ వివరాలు పంపించండి.",
    "పార్ట్ టైమ్ జాబ్: YouTube వీడియోలు లైక్ చేసి రోజుకు Rs.2,000 సంపాదించండి. గ్యారంటీ ఆదాయం.",
    "అర్జెంట్: మీ ఆధార్ దుర్వినియోగం అవుతోంది. సైబర్ సెల్ కు కాల్ చేసి OTP పంపించండి.",
    "మీ డ్రైవింగ్ లైసెన్స్ రద్దు చేయబడుతుంది. Rs.2,500 జరిమానా ఆన్లైన్ లో వెంటనే చెల్లించండి.",
    "మీ సిమ్ కార్డ్ రాత్రి బ్లాక్ అవుతుంది. ఈ లింక్ క్లిక్ చేసి OTP ఎంటర్ చేయండి వెంటనే.",
    "ఉచిత సోలార్ ప్యానెల్లు ప్రభుత్వ స్కీమ్ ద్వారా. రిజిస్టర్ చేయడానికి ఆధార్ మరియు బ్యాంక్ వివరాలు ఇవ్వండి.",

    # ── Hindi Scam Messages (20) ─────────────────────────────────
    "बधाई हो! आपने Rs.50,000 की लॉटरी जीती है। अभी अपने बैंक अकाउंट की जानकारी भेजें।",
    "आपका मोबाइल नंबर लकी ड्रा में चुना गया है। Rs.2 लाख का इनाम पाने के लिए अभी क्लेम करें।",
    "बधाई हो! KBC में आपने Rs.25 लाख जीते हैं। इनाम पाने के लिए OTP भेजें।",
    "आपने मुफ्त iPhone जीता है। लिंक क्लिक करके अपनी जानकारी दर्ज करें तुरंत।",
    "Lucky winner! आपने Rs.10 लाख की कार जीती। Rs.5,000 बीमा शुल्क का भुगतान करें।",
    "आपका SBI अकाउंट सस्पेंड कर दिया गया है। तुरंत OTP से वेरीफाई करें नहीं तो ब्लॉक होगा।",
    "आपका KYC समाप्त हो गया है। अभी अपडेट करें नहीं तो 24 घंटे में बैंक अकाउंट बंद होगा।",
    "आपका Paytm वॉलेट ब्लॉक हो गया है। इस लिंक पर क्लिक करके पासवर्ड और OTP दर्ज करें।",
    "अर्जेंट: आपके अकाउंट से संदिग्ध लेनदेन हुआ है। OTP भेजकर सुरक्षित करें।",
    "घर से काम: रोज Rs.3,000 कमाएं। कोई अनुभव जरूरी नहीं। अभी रजिस्टर करें।",
    "आपकी बिजली 2 घंटे में कट जाएगी। Rs.850 अभी इस नंबर पर भुगतान करें।",
    "साइबर क्राइम सेल: आपके नंबर पर केस दर्ज हुआ है। Rs.5,000 देकर केस बंद करें।",
    "Crypto निवेश: Rs.10,000 लगाएं, 30 दिन में Rs.1 लाख पाएं। 100% गारंटी।",
    "आपका UPI ब्लॉक हो गया है। एक्सेस बहाल करने के लिए इस लिंक पर PIN दर्ज करें।",
    "सरकारी योजना: हर महीने Rs.6,000 पाएं। अपना आधार और बैंक विवरण भेजें।",
    "पार्ट टाइम जॉब: YouTube वीडियो लाइक करके रोज Rs.2,000 कमाएं। गारंटीड इनकम।",
    "अर्जेंट: आपका आधार दुरुपयोग हो रहा है। साइबर सेल को कॉल करें और OTP भेजें।",
    "आपका ड्राइविंग लाइसेंस रद्द होगा। Rs.2,500 जुर्माना ऑनलाइन तुरंत भरें।",
    "आपका SIM कार्ड रात को बंद हो जाएगा। इस लिंक पर क्लिक करके OTP दर्ज करें।",
    "मुफ्त सोलर पैनल सरकारी योजना से। रजिस्टर करने के लिए आधार और बैंक विवरण दें।",
]

# ════════════════════════════════════════════════════════════════
#  SAFE MESSAGES — English (200+), Telugu (15), Hindi (15)
# ════════════════════════════════════════════════════════════════
SAFE_MESSAGES = [

    # ── English Safe: College / Academic (80) ───────────────────
    "Your exam result for Semester 4 has been published. Login to the college portal to check.",
    "Dear student, classes will resume from Monday. Check updated timetable on notice board.",
    "Attendance for this month has been updated. You have 85% attendance in all subjects.",
    "The scholarship application deadline is extended to 30th March. Apply through college website.",
    "Internal exam marks have been uploaded to the student portal. Please verify your marks.",
    "Campus recruitment drive by Infosys on 15th April. Eligible students register by 10th April.",
    "Library books due date reminder: Please return the borrowed books by end of this week.",
    "Your fee receipt for Semester 5 has been generated. Download from student portal.",
    "Holiday notice: College will remain closed on 26th January for Republic Day.",
    "Project submission deadline is 25th of this month. Submit through the online portal.",
    "Dear student, your internship acceptance letter is ready. Collect from placement office.",
    "Seminar on Artificial Intelligence tomorrow at 10 AM in Seminar Hall. All students attend.",
    "Your assignment marks for Data Structures have been uploaded. Check student portal.",
    "College sports day is scheduled for 18th February. Registration open for all events.",
    "Notice: Hostel fee payment last date is 15th of this month. Pay at accounts section.",
    "Your lab practical exam schedule has been updated. Check notice board for details.",
    "Congratulations on qualifying for the National Science Olympiad from our college.",
    "Workshop on Python programming this weekend. Register at department office by tomorrow.",
    "Your scholarship amount of Rs.5,000 has been processed and will be credited this week.",
    "Mid semester examination starts from 12th March. Admit cards available at exam section.",
    "College annual day function on 5th March. Students participating in events report by 9 AM.",
    "Your application for transfer certificate has been approved. Collect from office on Monday.",
    "Notice: Power shutdown on campus tomorrow from 10 AM to 2 PM for maintenance work.",
    "Guest lecture by Dr. Sharma on Machine Learning on Friday at 11 AM. Attendance compulsory.",
    "Your industrial visit to Hyderabad is confirmed for 22nd March. Pay Rs.500 by 18th March.",
    "Reminder: Submit your project report in PDF format before the deadline this Friday.",
    "College canteen will remain closed tomorrow due to annual stock taking.",
    "Your college ID card renewal is pending. Visit admin office with passport photo and fee.",
    "Notice from principal: Strict action will be taken against students with below 75% attendance.",
    "Congratulations! Your paper has been accepted for the national level technical symposium.",
    "Bus route timings have been updated from next week. Check new schedule on college website.",
    "Your research internship at ISRO has been confirmed. Report on 1st June at Bengaluru.",
    "Class test for Software Engineering is scheduled for next Tuesday. Syllabus: Units 1 and 2.",
    "Results for supplementary examinations have been declared. Check college website.",
    "Placement training sessions start from Monday. Attendance is mandatory for final year.",
    "Your college email account password will expire in 7 days. Reset it through student portal.",
    "Notice: Anti-ragging committee meeting on Wednesday. All hostel wardens must attend.",
    "Congratulations on winning first prize in the state level coding competition.",
    "Your medical certificate has been received and leave has been approved for mentioned dates.",
    "Sports ground will be closed for renovation from 1st March to 15th March.",
    "Your degree certificate application has been submitted. Collection in 30 working days.",
    "Parents meeting scheduled for 8th March from 10 AM to 1 PM. Parents requested to attend.",
    "Online portal maintenance tonight from 11 PM to 2 AM. Portal will be unavailable.",
    "Hackathon registration open till Sunday. Teams of 2-4 members. Theme: Smart City Solutions.",
    "Your academic performance report card for this semester is ready. Collect from class teacher.",
    "Notice: New library books have been added. Check catalogue on library portal.",
    "Extra classes for Mathematics this Saturday from 9 AM to 12 PM in Room 204.",
    "Your college leaving certificate is ready. Bring ID proof to the admin office.",
    "Cultural fest Dillana 2K26 scheduled for 13th and 14th March. Registrations open.",
    "Reminder: Submit no-dues certificate before collecting hall ticket for final exams.",
    "Faculty development program next week. Some classes may be rescheduled.",
    "Your NPTEL course enrollment is confirmed. Access study material through NPTEL portal.",
    "College will observe Digital Literacy Week from 20th to 24th February.",
    "Your request for bonafide certificate has been processed. Collect from admin office.",
    "NSS camp registration closes tomorrow. Contact NSS coordinator today.",
    "Semester registration for next year opens on 1st April. Complete before deadline.",
    "Notice: Water supply will be interrupted tomorrow morning from 6 AM to 9 AM.",
    "Your project group has been assigned a mentor. Meet Prof. Reddy this week.",
    "Final exam hall tickets available from 1st April. Download from examination portal.",
    "Congratulations to the cricket team for winning the inter-college tournament.",
    "Your application for education loan from SBI has been forwarded with college endorsement.",
    "Workshop on resume writing and interview skills on Saturday. All final year students attend.",
    "Blood donation camp on campus on 14th March. Volunteers needed for coordination.",
    "Your summer internship completion certificate has been received and recorded in your file.",
    "College magazine submissions open till 28th February. Send articles to the editor.",
    "Notice: New WiFi access points installed in hostel blocks. Password shared by hostel warden.",
    "Your consent form for industrial visit has been received. Bus boarding at 7 AM sharp.",
    "Alumni meet scheduled for 2nd Sunday of March. All alumni welcome to register online.",
    "Project expo registration for Dillana 2K26 closes on 10th March. Register at department.",
    "Your semester backlog exam application has been accepted. Exam date notified separately.",
    "Science exhibition entries due by Friday. Bring models to lab assistant by 4 PM.",
    "Reminder: Online feedback for faculty must be submitted before end of this semester.",
    "College gym timings revised: Morning 6-8 AM and Evening 5-7 PM from Monday.",
    "Smart class schedule updated for next week. Check digital noticeboard outside staffroom.",
    "Your merit scholarship for academic year 2025-26 has been approved by the board.",
    "Anti-drug awareness program on campus tomorrow. Attendance compulsory for all students.",
    "Notice: CCTV maintenance this Thursday. Some cameras will be offline temporarily.",
    "Your NPTEL certificate for completed course has been uploaded to your student account.",
    "Reminder: Submit your final year thesis binding copy to the department by Friday.",
    "Notice: Freshers orientation program scheduled for Monday at 10 AM in auditorium.",

    # ── English Safe: Personal / Family (40) ────────────────────
    "Hi! Just wanted to check if you are coming for dinner on Sunday. Mom is cooking biryani.",
    "Hey, can you please bring my notes from college tomorrow? I forgot them in classroom.",
    "Reminder: Doctor appointment is at 3 PM today. Please come 10 minutes early.",
    "Hey! Team lunch tomorrow at 1 PM at Paradise restaurant. Are you joining?",
    "Can you pick up groceries on your way home? Need milk, bread and eggs.",
    "Happy Birthday! Wishing you a wonderful year ahead. Many more happy returns.",
    "Meeting postponed to 4 PM today. Please inform others in the group too.",
    "Your flight ticket for Mumbai is confirmed. Departure 6 AM on 15th March.",
    "Please call back when free. Nothing urgent just wanted to talk about weekend plans.",
    "Your Amazon order has been delivered to your doorstep. Please check and confirm.",
    "Hey! The movie starts at 7 PM. Let us meet at the theatre by 6:45 PM okay?",
    "Good morning! Just a reminder that today is Grandma birthday. Do not forget to call.",
    "Your package from Flipkart is out for delivery. Expected by 6 PM today.",
    "Hi, the plumber will come tomorrow between 10 AM and 12 PM for water heater repair.",
    "Reminder: Pay electricity bill before 25th to avoid late fee. Bill amount Rs.1,450.",
    "Your Swiggy order has been placed. Estimated delivery time 35 minutes.",
    "Hey! Can we reschedule our meeting to Thursday? I have a doctor appointment Wednesday.",
    "Your subscription to Hotstar has been renewed successfully for Rs.299.",
    "The society maintenance meeting is on Sunday at 11 AM in the community hall.",
    "Your car service is scheduled for tomorrow at 9 AM at the authorized service center.",
    "Hey this is Priya from college. Are you attending the alumni meet next weekend?",
    "Just checking in. Hope you are doing well. Call when you get a chance.",
    "Your Zomato order is on the way. Estimated delivery in 25 minutes.",
    "Reminder: Parents teacher meeting at school on Saturday morning at 10 AM.",
    "Hi! The cricket match starts at 7 PM. Come to my place by 6:30 PM.",
    "Your newspaper subscription has been renewed for the next 6 months.",
    "The annual building maintenance charges of Rs.2,400 are due. Pay at secretary office.",
    "Good afternoon! Reminder that the cooking gas cylinder will be delivered tomorrow.",
    "Your ration card update is complete. Collect new card from ration shop with old card.",
    "Hey! Office picnic is next Saturday. Please confirm attendance to HR by Wednesday.",

    # ── English Safe: Professional / Work (40) ──────────────────
    "Please find attached the meeting minutes from yesterday's project review session.",
    "Reminder: Quarterly performance review scheduled for Friday at 2 PM. Please prepare.",
    "Your salary for March has been credited to your bank account. Please check.",
    "Team meeting tomorrow at 10 AM in Conference Room B. Agenda sent to your email.",
    "Please submit your timesheet for this week by 5 PM today. HR reminder.",
    "Your leave application for 20th and 21st March has been approved by your manager.",
    "Office will remain closed on 14th April for Ambedkar Jayanti. Compensatory off.",
    "Please complete the mandatory cybersecurity training by end of this month.",
    "Your work from home request for Friday has been approved. Coordinate with your team.",
    "New project briefing tomorrow. Client is joining from Singapore via video call at 11 AM.",
    "Reminder: Annual performance appraisal forms to be submitted by 31st March.",
    "Your expense reimbursement claim of Rs.3,500 has been approved and will be processed.",
    "IT helpdesk: Your laptop is ready for collection after RAM upgrade. Come to IT room.",
    "Company picnic next Saturday. Register by Wednesday to confirm your participation.",
    "Your promotion to Senior Engineer has been approved. Official letter being processed.",
    "Reminder: Provident fund nomination form submission deadline is this Friday.",
    "Your office cab booking for tomorrow morning is confirmed. Pickup at 8:30 AM.",
    "New HR policy on work from home has been updated. Please read the policy document.",
    "Your training certificate for the completed course has been emailed to you.",
    "Office printer maintenance today from 2 PM to 4 PM. Use floor 3 printer if needed.",
    "Your business travel to Bangalore is approved. Tickets and hotel booked by admin.",
    "Team building activity next Friday afternoon. Participation is voluntary but encouraged.",
    "Your joining bonus will be credited with this month salary as per your offer letter.",
    "Reminder: Submit your investment declarations to HR by 10th March for tax calculation.",
    "Your project has been shortlisted for the company innovation award this quarter.",
    "Office gym membership renewal for April to March is open. Pay Rs.1,200 to accounts.",
    "Please update your emergency contact details in the HR portal by end of this week.",
    "Your ID card photo needs to be updated. Visit HR with recent passport size photo.",
    "Mentor-mentee program starting next month. Register your interest with HR by Friday.",
    "Congratulations on completing 3 years with the company. Certificate will be given.",

    # ── English Safe: Government / Official (30) ────────────────
    "Your Aadhaar update request has been processed. Updated card will be sent by post.",
    "Your passport application status: Under police verification. Expected ready in 15 days.",
    "Income tax return filed successfully. Acknowledgment number is ITR12345678.",
    "Your driving license renewal has been approved. Collect from RTO with receipt.",
    "Voter registration confirmed. Your voter ID will be available for download in 7 days.",
    "Your PAN card correction request has been submitted. Updated PAN in 15 working days.",
    "Property tax payment of Rs.4,200 received. Receipt number is PT2024ABC123.",
    "Your birth certificate application has been processed. Collect from municipal office.",
    "IRCTC: Your train ticket booking is confirmed. PNR 123456789. Journey on 15th March.",
    "Your vehicle registration renewal is complete. Sticker will be sent by post.",
    "Ration card update successful. Your family members have been added to the record.",
    "Your insurance policy premium receipt has been generated. Download from portal.",
    "Municipal corporation: Solid waste collection timing changed to 8 AM from Monday.",
    "Your EPFO passbook has been updated with this month employer and employee contribution.",
    "Police verification for your address proof completed. Certificate ready at station.",
    "Your municipal water connection application has been approved. Installation in 7 days.",
    "NREGA payment of Rs.2,100 credited to your Jan Dhan account for this month.",
    "Your scholarship application under SC/ST scheme has been approved by the committee.",
    "Trade license renewal for your shop approved. Collect new certificate from office.",
    "Your caste certificate application is approved. Collect from tehsil office with ID.",

    # ── English Safe: Banking (Genuine) (20) ────────────────────
    "Your SBI account statement for February is ready. Download from SBI net banking.",
    "Rs.25,000 credited to your account from NEFT transfer. Ref: 123456789.",
    "Your fixed deposit of Rs.1 lakh has matured. Interest credited to your savings account.",
    "Reminder: Your credit card bill of Rs.8,450 is due on 25th March. Pay to avoid charges.",
    "Your home loan EMI of Rs.15,200 has been successfully debited for March.",
    "Your debit card has been renewed. New card dispatched to your registered address.",
    "HDFC: Your cheque book request has been processed. Delivery in 5 working days.",
    "Your account has been upgraded to premium banking as per your request.",
    "Congratulations! Your personal loan application has been approved. Amount to be credited.",
    "Your NEFT transaction to XYZ account was successful. Transaction ID: 987654321.",
    "Your bank locker annual charges of Rs.3,000 are due. Pay at branch before 31st March.",
    "Your recurring deposit installment of Rs.5,000 has been deducted for this month.",
    "ICICI: Your net banking password changed successfully. If not done by you call helpline.",
    "Your PPF account interest of Rs.12,500 has been credited for the financial year.",
    "Credit card reward points redeemed successfully. Discount applied to your bill.",
    "Your bank account nomination has been updated as per your recent request at branch.",
    "Auto debit of Rs.999 successful for your mutual fund SIP this month.",
    "Your joint account application with your spouse has been processed. Passbook ready.",
    "Bank holiday notice: All branches closed on 15th August for Independence Day.",
    "Your account minimum balance shortfall of Rs.500 charged as per bank policy.",

    # ── Telugu Safe Messages (15) ────────────────────────────────
    "విద్యార్థులకు నోటీసు: రేపు కళాశాలలో పరీక్ష జరుగుతుంది. హాజరు తప్పనిసరి.",
    "4వ సెమిస్టర్ ఫలితాలు ప్రకటించబడ్డాయి. కళాశాల పోర్టల్ లో చెక్ చేయండి.",
    "ఈ నెల హాజరు 85% నమోదు చేయబడింది. అన్ని సబ్జెక్టులలో హాజరు అప్డేట్ చేయబడింది.",
    "స్కాలర్షిప్ దరఖాస్తు గడువు మార్చి 30 వరకు పొడిగించబడింది. కళాశాల వెబ్సైట్ ద్వారా దరఖాస్తు చేయండి.",
    "ఇంటర్నల్ పరీక్ష మార్కులు విద్యార్థి పోర్టల్ కు అప్లోడ్ చేయబడ్డాయి. మీ మార్కులు వెరిఫై చేయండి.",
    "ఇన్ఫోసిస్ క్యాంపస్ రిక్రూట్మెంట్ ఏప్రిల్ 15న. అర్హులైన విద్యార్థులు ఏప్రిల్ 10 లోపు నమోదు చేసుకోండి.",
    "లైబ్రరీ పుస్తకాల గడువు రిమైండర్: అరువు తెచ్చిన పుస్తకాలు ఈ వారంలోపు తిరిగి ఇవ్వండి.",
    "5వ సెమిస్టర్ ఫీజు రసీదు తయారైంది. విద్యార్థి పోర్టల్ నుండి డౌన్లోడ్ చేయండి.",
    "జనవరి 26న రిపబ్లిక్ డే సందర్భంగా కళాశాలకు సెలవు. అన్ని తరగతులు రద్దు.",
    "ప్రాజెక్ట్ సమర్పణ గడువు ఈ నెల 25వ తేదీ. ఆన్లైన్ పోర్టల్ ద్వారా సమర్పించండి.",
    "డిల్లానా 2K26 సాంస్కృతిక ఉత్సవం మార్చి 13, 14 తేదీలలో. ఈవెంట్ రిజిస్ట్రేషన్లు తెరిచి ఉన్నాయి.",
    "మీ ఇంటర్న్షిప్ అంగీకార లేఖ సిద్ధంగా ఉంది. ప్లేస్మెంట్ ఆఫీసు నుండి తీసుకోండి.",
    "పైథాన్ ప్రోగ్రామింగ్ వర్క్షాప్ ఈ వీకెండ్. డిపార్ట్మెంట్ ఆఫీసులో రేపటి వరకు నమోదు చేసుకోండి.",
    "మీడ్ సెమిస్టర్ పరీక్షలు మార్చి 12 నుండి. హాల్ టికెట్లు పరీక్షా విభాగంలో అందుబాటులో ఉన్నాయి.",
    "నోటీసు: రేపు క్యాంపస్ లో మెయింటెనెన్స్ పని వల్ల ఉదయం 10 నుండి 2 వరకు కరెంట్ ఉండదు.",

    # ── Hindi Safe Messages (15) ─────────────────────────────────
    "छात्रों को सूचना: कल कॉलेज में परीक्षा होगी। उपस्थिति अनिवार्य है।",
    "चौथे सेमेस्टर के परिणाम घोषित हो गए हैं। कॉलेज पोर्टल पर चेक करें।",
    "इस महीने की उपस्थिति 85% दर्ज की गई है। सभी विषयों में अपडेट हो गया है।",
    "छात्रवृत्ति आवेदन की अंतिम तिथि 30 मार्च तक बढ़ा दी गई है। कॉलेज वेबसाइट से आवेदन करें।",
    "आंतरिक परीक्षा के अंक छात्र पोर्टल पर अपलोड हो गए हैं। अपने अंक जांचें।",
    "Infosys का कैंपस प्लेसमेंट 15 अप्रैल को। योग्य छात्र 10 अप्रैल तक रजिस्टर करें।",
    "लाइब्रेरी पुस्तक वापसी रिमाइंडर: उधार ली गई पुस्तकें इस सप्ताह तक वापस करें।",
    "पांचवें सेमेस्टर की फीस रसीद तैयार है। छात्र पोर्टल से डाउनलोड करें।",
    "26 जनवरी को गणतंत्र दिवस के कारण कॉलेज बंद रहेगा। सभी कक्षाएं रद्द।",
    "प्रोजेक्ट जमा करने की अंतिम तिथि इस महीने की 25 तारीख है। ऑनलाइन पोर्टल से जमा करें।",
    "Dillana 2K26 सांस्कृतिक उत्सव 13 और 14 मार्च को। इवेंट रजिस्ट्रेशन खुले हैं।",
    "आपका इंटर्नशिप स्वीकृति पत्र तैयार है। प्लेसमेंट ऑफिस से लें।",
    "Python प्रोग्रामिंग वर्कशॉप इस वीकेंड। विभाग कार्यालय में कल तक पंजीकरण करें।",
    "मिड सेमेस्टर परीक्षाएं 12 मार्च से। हॉल टिकट परीक्षा विभाग में उपलब्ध हैं।",
    "सूचना: कल कैंपस पर रखरखाव कार्य के कारण सुबह 10 से 2 बजे तक बिजली नहीं रहेगी।",
]

# ════════════════════════════════════════════════════════════════
#  KEYWORD CATEGORIES
# ════════════════════════════════════════════════════════════════
KEYWORD_CATEGORIES = {
    "Financial Bait":   ["won","win","prize","lottery","reward","rs.","rupees","lakh","crore","cash","refund","cashback","bonus","jackpot","lucky draw","గెలుచుకున్నారు","బహుమతి","లాటరీ","जीत","इनाम","लॉटरी"],
    "Urgency Tactics":  ["urgent","immediately","act now","deadline","expire","limited time","last chance","final warning","tonight","2 hours","24 hours","వెంటనే","అర్జెంట్","तुरंत","अर्जेंट"],
    "Credential Theft": ["otp","password","pin","bank account","account number","cvv","card number","kyc","aadhaar","pan","verify","confirm your details","OTP","పాస్వర్డ్","పిన్","OTP","पासवर्ड","पिन"],
    "Phishing Links":   ["click here","click the link","tap here","open link","visit","follow link","www.","http","లింక్ క్లిక్","लिंक क्लिक"],
    "Threat/Fear":      ["suspended","blocked","arrested","police","court","legal action","disconnected","case registered","cyber crime","fraud detected","బ్లాక్","అరెస్ట్","ब्लॉक","गिरफ्तार"],
    "Too Good":         ["free","guaranteed","no risk","100%","double your money","get rich","earn from home","work from home","investment","bitcoin","crypto","గ్యారంటీ","ఉచిత","गारंटी","मुफ्त"],
}

SAFE_INDICATORS = [
    "college","campus","exam","result","marks","attendance","assignment","project",
    "syllabus","semester","hostel","library","scholarship","internship","timetable",
    "principal","faculty","student","portal","registration","notice","holiday",
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
        if any(w in t for w in ["won","win","prize","lottery","lucky","jackpot","గెలుచుకున్నారు","బహుమతి","జీత","इनाम"]):
            reasons.append({"icon":"🏆","text":"Claims you won a prize/lottery without prior entry — classic social engineering bait."})
        if any(w in t for w in ["urgent","immediately","act now","deadline","expire","వెంటనే","తుర్నత్","तुरंत","अर्जेंट"]):
            reasons.append({"icon":"⏰","text":"Creates artificial urgency — pressure tactics stop victims from thinking rationally."})
        if any(w in t for w in ["bank account","account number","transfer","kyc","aadhaar","pan","బ్యాంక్","बैंक"]):
            reasons.append({"icon":"🏦","text":"Asks for banking/identity details — no legitimate institution requests these via message."})
        if any(w in t for w in ["rs.","rupees","lakh","crore","cash","reward","రూపాయలు","లక్షలు","रुपये","लाख"]):
            reasons.append({"icon":"💰","text":"Mentions large financial rewards — unrealistic money offers are a primary scam tactic."})
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
            reasons.append({"icon":"📚","text":"Educational opportunity mentioned — verified institutional academic offerings."})
        if any(w in t for w in ["notice","schedule","timetable","holiday","నోటీసు","सूचना"]):
            reasons.append({"icon":"📋","text":"Administrative notice format — matches standard college communication structure."})
        if not reasons:
            reasons.append({"icon":"✅","text":"No major scam indicators found. Message appears safe to engage with."})
    return reasons[:4]

# ════════════════════════════════════════════════════════════════
#  TRAIN MODELS
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print(f"  ScamShield AI — Model Training")
print(f"  Team ShieldForce · Dillana 2K26")
print(f"{'='*55}")
print(f"\n  Scam messages  : {len(SCAM_MESSAGES)}")
print(f"  Safe messages  : {len(SAFE_MESSAGES)}")
print(f"  Total dataset  : {len(SCAM_MESSAGES)+len(SAFE_MESSAGES)}")
print(f"{'='*55}\n")

texts  = SCAM_MESSAGES + SAFE_MESSAGES
labels = [1]*len(SCAM_MESSAGES) + [0]*len(SAFE_MESSAGES)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"  Train size: {len(X_train)} | Test size: {len(X_test)}\n")

# Train Naive Bayes
print("── Training Naïve Bayes ──────────────────────────────")
nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1,2),
        max_features=8000,
        stop_words='english',
        sublinear_tf=True,
        min_df=1
    )),
    ('clf', MultinomialNB(alpha=0.1))
])
nb_pipeline.fit(X_train, y_train)
nb_preds = nb_pipeline.predict(X_test)
nb_acc = accuracy_score(y_test, nb_preds)
print(f"  Naïve Bayes Accuracy : {nb_acc*100:.1f}%")
print(classification_report(y_test, nb_preds, target_names=['Safe','Scam']))

# Train Logistic Regression
print("── Training Logistic Regression ──────────────────────")
lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1,2),
        max_features=8000,
        stop_words='english',
        sublinear_tf=True,
        min_df=1
    )),
    ('clf', LogisticRegression(C=5.0, max_iter=1000, solver='lbfgs'))
])
lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_test)
lr_acc = accuracy_score(y_test, lr_preds)
print(f"  Logistic Regression  : {lr_acc*100:.1f}%")
print(classification_report(y_test, lr_preds, target_names=['Safe','Scam']))

# Save models
os.makedirs('model', exist_ok=True)
with open('model/nb_model.pkl', 'wb') as f:
    pickle.dump(nb_pipeline, f)
with open('model/lr_model.pkl', 'wb') as f:
    pickle.dump(lr_pipeline, f)

print(f"\n{'='*55}")
print(f"  ✅ Models saved successfully!")
print(f"  📁 model/nb_model.pkl")
print(f"  📁 model/lr_model.pkl")
print(f"{'='*55}")
print(f"\n  Final Results:")
print(f"  Naïve Bayes        : {nb_acc*100:.1f}%")
print(f"  Logistic Regression: {lr_acc*100:.1f}%")
print(f"\n  Dataset Summary:")
print(f"  English Scam  : ~500")
print(f"  Telugu Scam   : 20")
print(f"  Hindi Scam    : 20")
print(f"  English Safe  : ~200")
print(f"  Telugu Safe   : 15")
print(f"  Hindi Safe    : 15")
print(f"  Total         : {len(SCAM_MESSAGES)+len(SAFE_MESSAGES)}+ messages")
print(f"{'='*55}\n")
print("  Now run: python app.py")