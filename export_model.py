import pandas as pd
import json
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_dir, "Disease prediction and prescription generator", "Training.csv")

train = pd.read_csv(train_path)
if 'Unnamed: 133' in train.columns:
    train = train.drop(columns=['Unnamed: 133'])

cols = [c.strip() for c in train.columns if c != 'prognosis']
diseases = sorted(train['prognosis'].unique())

alpha = 0.1
cond_probs = {}

for d in diseases:
    sub = train[train['prognosis'] == d]
    n = len(sub)
    cond_probs[d] = {}
    for c in cols:
        k = sub[c].sum()
        p = (k + alpha) / (n + 2 * alpha)
        cond_probs[d][c] = round(float(p), 6)

friendly_names = {col: col.replace('_', ' ').strip().title() for col in cols}

# Comprehensive clinical prescriptions and guidance
CLINICAL_DATABASE = {
    'Fungal infection': {
        'display': 'Fungal Infection (Mycosis)',
        'icon': '🦠',
        'overview': "A fungal infection occurs when fungi overcome the body's defenses and infect tissue, commonly affecting the skin, nails, or mucosal surfaces (athlete's foot, ringworm, candidiasis).",
        'prescription': [
            {'name': 'Clotrimazole / Terbinafine 1% Cream', 'instruction': 'Apply thin layer twice daily to cleaned, dried affected areas', 'duration': '2-4 weeks'},
            {'name': 'Fluconazole 150mg (Oral)', 'instruction': 'Single weekly dose if prescribed for extensive cutaneous infection', 'duration': 'As prescribed'},
            {'name': 'Antifungal Dusting Powder', 'instruction': 'Dust inside shoes and skin folds daily to keep dry', 'duration': 'Daily'}
        ],
        'care': [
            'Keep skin clean, thoroughly dried, and exposed to air when possible.',
            'Wear loose-fitting, breathable cotton clothing and change daily.',
            'Avoid sharing towels, bed linens, or personal grooming items.'
        ],
        'diet': [
            'Reduce refined sugar intake, which may promote yeast proliferation.',
            'Incorporate probiotic-rich foods like plain yogurt and fermented kefir.',
            'Stay hydrated with 2.5L water daily to bolster skin barrier recovery.'
        ]
    },
    'Allergy': {
        'display': 'Allergic Rhinitis & Hypersensitivity',
        'icon': '🌿',
        'overview': "An exaggerated immune response to harmless environmental allergens (pollen, dust mites, pet dander), causing mucosal inflammation, sneezing, and ocular irritation.",
        'prescription': [
            {'name': 'Cetirizine 10mg / Levocetirizine 5mg', 'instruction': 'Take 1 tablet once daily at bedtime', 'duration': '5-7 days'},
            {'name': 'Fluticasone Propionate Nasal Spray', 'instruction': '1-2 sprays per nostril once daily morning', 'duration': '1-2 weeks'},
            {'name': 'Saline Nasal Irrigation', 'instruction': 'Rinse nasal cavities twice daily with isotonic saline', 'duration': 'Daily'}
        ],
        'care': [
            'Keep windows closed during high pollen or windy conditions.',
            'Use HEPA air filtration in living and sleeping quarters.',
            'Wash bedding weekly in hot water (>130°F / 55°C).'
        ],
        'diet': [
            'Consume foods rich in quercetin (apples, red onions, berries).',
            'Incorporate anti-inflammatory omega-3 fatty acids (flaxseeds, walnuts).',
            'Avoid known trigger foods or excessive histaminic foods during flare-ups.'
        ]
    },
    'GERD': {
        'display': 'Gastroesophageal Reflux Disease (GERD)',
        'icon': '🔥',
        'overview': "A digestive disorder where stomach acid persistently flows back into the esophagus, irritating the lining and causing heartburn, regurgitation, and chest discomfort.",
        'prescription': [
            {'name': 'Pantoprazole 40mg / Omeprazole 20mg', 'instruction': 'Take 1 tablet 30 minutes before breakfast on an empty stomach', 'duration': '14 days'},
            {'name': 'Antacid Suspension (Magaldrate + Simethicone)', 'instruction': '10-15 ml after meals and before bedtime as needed', 'duration': 'As needed'},
            {'name': 'Famotidine 20mg', 'instruction': 'Take 1 tablet before dinner for nocturnal symptoms', 'duration': '7-10 days'}
        ],
        'care': [
            'Elevate the head of your bed by 6-8 inches (15-20 cm).',
            'Wait at least 3 hours after eating before lying down or going to sleep.',
            'Eat smaller, more frequent meals rather than large heavy dinners.'
        ],
        'diet': [
            'Avoid acidic citrus, tomatoes, caffeine, chocolate, and carbonated beverages.',
            'Minimize deep-fried, fatty, spicy foods and mint.',
            'Incorporate oatmeal, non-citrus fruits (bananas, melons), and lean poultry.'
        ]
    },
    'Chronic cholestasis': {
        'display': 'Chronic Cholestasis',
        'icon': '🟡',
        'overview': "A reduction or stagnation of bile flow from the liver to the duodenum, leading to bile acid accumulation, jaundice, and intense pruritus.",
        'prescription': [
            {'name': 'Ursodeoxycholic Acid (UDCA) 300mg', 'instruction': 'Take 1 tablet 2-3 times daily with meals under specialist guidance', 'duration': 'Ongoing'},
            {'name': 'Cholestyramine 4g', 'instruction': 'Mix with water/juice taken 1 hour after breakfast for itching', 'duration': 'As directed'},
            {'name': 'Fat-Soluble Vitamin Supplement (A, D, E, K)', 'instruction': 'Daily formulation to counteract biliary malabsorption', 'duration': 'Daily'}
        ],
        'care': [
            'Regular hepatic liver panel monitoring (ALT, AST, Bilirubin, Alk Phos).',
            'Avoid all alcohol consumption and hepatotoxic medications.',
            'Maintain cool baths and gentle moisturizers for bile salt skin irritation.'
        ],
        'diet': [
            'Adopt a low-fat, nutrient-dense Mediterranean diet.',
            'Incorporate medium-chain triglyceride (MCT) fats for easier absorption.',
            'Stay well-hydrated to assist renal clearance of circulating bile acids.'
        ]
    },
    'Drug Reaction': {
        'display': 'Adverse Cutaneous Drug Reaction',
        'icon': '💊',
        'overview': "An immunological or non-immunological adverse cutaneous reaction to an ingested or administered pharmaceutical agent, producing rashes or urticaria.",
        'prescription': [
            {'name': 'IMMEDIATE DISCONTINUATION of Suspected Drug', 'instruction': 'Stop offending drug immediately under physician supervision', 'duration': 'Immediate'},
            {'name': 'Fexofenadine 180mg / Bilastine 20mg', 'instruction': 'Take 1 tablet once daily morning for histamine relief', 'duration': '5-7 days'},
            {'name': 'Hydrocortisone 1% Topical Cream / Calamine', 'instruction': 'Apply gently to itchy patches 2 times daily', 'duration': '5 days'}
        ],
        'care': [
            'Document the offending medication and carry an allergy alert card.',
            'Seek immediate emergency care if facial swelling, lips or breathing are involved.',
            'Avoid hot showers and harsh soaps that exacerbate skin inflammation.'
        ],
        'diet': [
            'Drink plentiful cool fluids and electrolytes.',
            'Eat soothing, non-spicy foods to prevent secondary mucosal irritation.',
            'Avoid artificial dyes and preservatives during the recovery phase.'
        ]
    },
    'Peptic ulcer disease': {
        'display': 'Peptic Ulcer Disease',
        'icon': '🩹',
        'overview': "Sores that develop on the inside lining of the stomach or duodenum, usually caused by H. pylori infection or long-term NSAID painkiller use.",
        'prescription': [
            {'name': 'Esomeprazole 40mg', 'instruction': 'Take 1 tablet once daily 30 minutes before breakfast', 'duration': '4-8 weeks'},
            {'name': 'Sucralfate 1g Suspension', 'instruction': '10ml 4 times daily, 1 hour before meals and at bedtime', 'duration': '4 weeks'},
            {'name': 'H. Pylori Eradication Regimen (if positive)', 'instruction': 'Triple/quadruple antibiotic therapy under physician direction', 'duration': '14 days'}
        ],
        'care': [
            'Completely avoid NSAID painkillers like ibuprofen, naproxen, and aspirin.',
            'Stop smoking and eliminate alcoholic beverages which erode mucosal barriers.',
            'Do not skip meals; keep regular consistent eating schedules.'
        ],
        'diet': [
            'Eat fiber-rich foods, particularly soluble fiber (oats, legumes, vegetables).',
            'Include probiotic foods (plain yogurt, kefir) to inhibit pathogenic bacteria.',
            'Avoid highly spiced chili peppers, black pepper, and excessive coffee.'
        ]
    },
    'AIDS': {
        'display': 'Acquired Immunodeficiency Syndrome (AIDS)',
        'icon': '🧬',
        'overview': "Advanced retroviral immunocompromise caused by Human Immunodeficiency Virus (HIV), resulting in depletion of CD4+ T lymphocytes.",
        'prescription': [
            {'name': 'Antiretroviral Therapy (ART)', 'instruction': 'Combination antiretroviral regimen prescribed by infectious disease specialist', 'duration': 'Lifelong'},
            {'name': 'Trimethoprim-Sulfamethoxazole (Bactrim DS)', 'instruction': '1 tablet daily for opportunistic infection prophylaxis', 'duration': 'Continuous'},
            {'name': 'Multivitamin with Zinc and Selenium', 'instruction': 'Take 1 tablet daily with food', 'duration': 'Daily'}
        ],
        'care': [
            'Strict adherence to ART timing without missing doses.',
            'Regular CD4 cell counts and HIV viral load monitoring every 3-6 months.',
            'Practice stringent food hygiene to avoid waterborne and foodborne pathogens.'
        ],
        'diet': [
            'High-calorie, high-protein diet (eggs, legumes, fish, nuts) to prevent wasting.',
            'Only consume thoroughly cooked foods and clean boiled/filtered water.',
            'Eat frequent small nutrient-dense meals throughout the day.'
        ]
    },
    'Diabetes ': {
        'display': 'Type 2 Diabetes Mellitus',
        'icon': '🩺',
        'overview': "A chronic metabolic condition characterized by insulin resistance and relative insulin deficiency, leading to persistently elevated blood glucose.",
        'prescription': [
            {'name': 'Metformin 500mg (Doctor Prescribed)', 'instruction': 'Take 1 tablet with or after evening meal', 'duration': 'Ongoing'},
            {'name': 'Capillary Blood Glucose Log', 'instruction': 'Test fasting and 2-hour postprandial glucose 3 times a week', 'duration': 'Continuous'},
            {'name': 'HbA1c Glycated Hemoglobin Test', 'instruction': 'Comprehensive laboratory assessment every 3 months', 'duration': 'Quarterly'}
        ],
        'care': [
            'Engage in 30 minutes of moderate aerobic exercise (brisk walking) 5 days/week.',
            'Perform daily foot examinations to identify any blisters, cuts, or sores early.',
            'Maintain strict blood pressure and lipid panel control.'
        ],
        'diet': [
            'Eliminate refined sugars, sugary beverages, pastries, and white flour.',
            'Base meals on low-glycemic complex carbohydrates (quinoa, barley, vegetables).',
            'Incorporate lean proteins and healthy fats to blunt post-meal glucose spikes.'
        ]
    },
    'Gastroenteritis': {
        'display': 'Acute Gastroenteritis (Stomach Flu)',
        'icon': '🤢',
        'overview': "Inflammation of the stomach and intestines typically caused by viral or bacterial infection, resulting in diarrhea, vomiting, and cramps.",
        'prescription': [
            {'name': 'Oral Rehydration Salts (ORS)', 'instruction': 'Dissolve 1 packet in 1 liter clean water; sip 200ml after each loose stool', 'duration': '3-5 days'},
            {'name': 'Ondansetron 4mg', 'instruction': 'Take 1 tablet 15 minutes before oral intake if nausea prevents hydration', 'duration': '2-3 days'},
            {'name': 'Probiotic S. boulardii 250mg', 'instruction': 'Take 1 capsule twice daily with lukewarm water', 'duration': '5 days'}
        ],
        'care': [
            'Prioritize active electrolyte replenishment to avoid dehydration.',
            'Rest comfortably and avoid intense physical exertion.',
            'Monitor urine output (should remain pale yellow, not dark or absent).'
        ],
        'diet': [
            'Follow BRAT diet during acute recovery: Bananas, Rice, Applesauce, Toast.',
            'Drink clear broths, diluted coconut water, and herbal chamomile tea.',
            'Avoid dairy products, greasy foods, caffeine, and alcohol for 5-7 days.'
        ]
    },
    'Bronchial Asthma': {
        'display': 'Bronchial Asthma',
        'icon': '🫁',
        'overview': "A chronic inflammatory disorder of the airways causing episodic airway constriction, hyperresponsiveness, wheezing, and breathlessness.",
        'prescription': [
            {'name': 'Salbutamol / Albuterol Inhaler (100mcg)', 'instruction': 'Inhale 1-2 puffs with spacer as needed for acute shortness of breath', 'duration': 'As needed'},
            {'name': 'Budesonide 200mcg / Formoterol Inhaler', 'instruction': 'Inhale 1 puff twice daily; rinse mouth thoroughly with water afterward', 'duration': 'Maintenance'},
            {'name': 'Peak Expiratory Flow Meter', 'instruction': 'Record morning and evening PEF readings to monitor airway control', 'duration': 'Daily'}
        ],
        'care': [
            'Always carry your rescue bronchodilator inhaler wherever you go.',
            'Identify and eliminate environmental asthma triggers (smoke, cold air, pet dander).',
            'Establish an Asthma Action Plan with your pulmonologist.'
        ],
        'diet': [
            'Eat antioxidant-rich colorful vegetables and fruits (vitamin C and E).',
            'Avoid foods containing sulfites or artificial preservatives (dried fruits, wine).',
            'Maintain a healthy body weight to reduce respiratory workload.'
        ]
    },
    'Hypertension ': {
        'display': 'Essential Systemic Hypertension',
        'icon': '💓',
        'overview': "A long-term medical condition in which systemic arterial blood pressure is persistently elevated (>130/80 mmHg), increasing cardiovascular risk.",
        'prescription': [
            {'name': 'Amlodipine 5mg / Telmisartan 40mg', 'instruction': 'Take 1 tablet once daily morning under physician prescription', 'duration': 'Ongoing'},
            {'name': 'Digital Home Blood Pressure Monitor', 'instruction': 'Record morning and evening readings twice weekly seated quietly', 'duration': 'Continuous'},
            {'name': 'Routine Serum Electrolytes & Renal Panel', 'instruction': 'Monitor creatinine and potassium every 6 months', 'duration': 'Bi-annual'}
        ],
        'care': [
            'Strictly limit dietary sodium intake to under 1,500 - 2,000 mg per day.',
            'Engage in 150 minutes of moderate aerobic cardiovascular exercise weekly.',
            'Practice stress reduction techniques (mindfulness, diaphragmatic breathing).'
        ],
        'diet': [
            'Adhere to the DASH Diet (Dietary Approaches to Stop Hypertension).',
            'Increase potassium-rich whole foods (bananas, sweet potatoes, spinach).',
            'Minimize processed foods, canned soups, cured meats, and alcohol.'
        ]
    },
    'Migraine': {
        'display': 'Migraine Headache Disorder',
        'icon': '🤕',
        'overview': "A neurological disorder characterized by recurrent, severe throbbing unilateral headaches often accompanied by nausea, photophobia, and sensory aura.",
        'prescription': [
            {'name': 'Sumatriptan 50mg / Zolmitriptan 2.5mg', 'instruction': 'Take 1 tablet at the earliest onset of migraine attack (max 2/day)', 'duration': 'Per attack'},
            {'name': 'Naproxen Sodium 500mg', 'instruction': 'Take 1 tablet with food for moderate headache pain', 'duration': 'As needed'},
            {'name': 'Propranolol / Topiramate (Preventive)', 'instruction': 'Daily prescription if attacks occur >4 days per month', 'duration': 'Preventive'}
        ],
        'care': [
            'Rest in a quiet, dark, cool room with a cold compress across forehead.',
            'Maintain consistent sleep schedules; avoid sleep deprivation or oversleeping.',
            'Keep a headache diary to pinpoint individual dietary and sensory triggers.'
        ],
        'diet': [
            'Stay steadily hydrated throughout the day; dehydration triggers attacks.',
            'Limit aged cheeses, red wine, chocolate, cured meats containing nitrates, and MSG.',
            'Ensure regular meal timing to prevent hypoglycemia-induced migraines.'
        ]
    },
    'Cervical spondylosis': {
        'display': 'Cervical Spondylosis (Neck Arthritis)',
        'icon': '🦴',
        'overview': "Age-related wear and tear affecting cervical spinal disks and vertebrae, causing neck stiffness, local pain, and occasional radicular numbness.",
        'prescription': [
            {'name': 'Paracetamol 650mg / Ibuprofen 400mg', 'instruction': 'Take 1 tablet with meals twice daily for acute neck discomfort', 'duration': '5-7 days'},
            {'name': 'Thiocolchicoside / Baclofen 10mg', 'instruction': 'Muscle relaxant taken at bedtime for acute neck spasm', 'duration': '5 days'},
            {'name': 'Topical Diclofenac 1.16% Gel', 'instruction': 'Gently massage onto posterior cervical region 3 times daily', 'duration': '7-10 days'}
        ],
        'care': [
            'Perform gentle cervical isometric and range-of-motion physical therapy exercises.',
            'Use an ergonomic contoured cervical pillow to maintain neutral spinal alignment.',
            'Avoid prolonged forward-head posture when viewing computer screens or mobile phones.'
        ],
        'diet': [
            'Ensure adequate dietary calcium (1,000-1,200 mg/day) and Vitamin D3 (1,000 IU/day).',
            'Eat anti-inflammatory foods (fatty fish, olive oil, turmeric, leafy greens).',
            'Maintain optimal hydration to nourish intervertebral disk hydration.'
        ]
    },
    'Paralysis (brain hemorrhage)': {
        'display': 'Acute Hemorrhagic Stroke / Neurological Deficit',
        'icon': '🚨',
        'overview': "A critical emergency resulting from rupture of a cerebral blood vessel, causing localized brain tissue compression and sudden motor/sensory paralysis.",
        'prescription': [
            {'name': 'IMMEDIATE EMERGENCY AMBULANCE TRANSPORT (911 / 108)', 'instruction': 'Immediate transfer to a certified Comprehensive Stroke Center', 'duration': 'IMMEDIATE'},
            {'name': 'Strict Hospital Neurosurgical Management', 'instruction': 'Blood pressure regulation, neurocritical monitoring, surgical decompression if indicated', 'duration': 'Inpatient'},
            {'name': 'Multidisciplinary Neuro-Rehabilitation', 'instruction': 'Physical, occupational, and speech-language therapy initiated upon stabilization', 'duration': 'Long-term'}
        ],
        'care': [
            'Call emergency services immediately. Note the exact time symptoms started.',
            'Do NOT give the patient any food, drink, or oral medications (choking risk).',
            'Keep patient lying down on their side with head slightly elevated if vomiting.'
        ],
        'diet': [
            'Strict medically supervised clinical enteral or pureed nutrition in hospital.',
            'Lifelong low-sodium, cardiovascular-protective Mediterranean diet after stabilization.',
            'Complete cessation of tobacco and alcohol.'
        ]
    },
    'Jaundice': {
        'display': 'Hepatobiliary Jaundice (Hyperbilirubinemia)',
        'icon': '🟡',
        'overview': "Yellowish discoloration of the skin, sclera, and mucous membranes caused by elevated bilirubin levels from hepatic dysfunction, hemolysis, or biliary obstruction.",
        'prescription': [
            {'name': 'Silymarin / Milk Thistle Extract 140mg', 'instruction': 'Take 1 capsule twice daily with meals as hepatic support', 'duration': '4 weeks'},
            {'name': 'Comprehensive Liver Function Test (LFT)', 'instruction': 'Measure total/direct bilirubin, AST, ALT, ALP and viral hepatitis markers', 'duration': 'Immediate'},
            {'name': 'Vitamin B-Complex & Vitamin C', 'instruction': 'Daily therapeutic formulation to support cellular hepatocyte repair', 'duration': '30 days'}
        ],
        'care': [
            'Strictly refrain from all alcohol and hepatotoxic substances (e.g. high-dose acetaminophen).',
            'Get plenty of bed rest to optimize hepatic blood circulation.',
            'Seek immediate hospital attention if accompanied by confusion, lethargy, or dark stools.'
        ],
        'diet': [
            'Consume light, easily digestible carbohydrate-rich foods (porridge, boiled rice, papayas).',
            'Drink fresh sugarcane juice, coconut water, and clean boiled water throughout the day.',
            'Strictly avoid deep-fried, oily, fatty meats, and spicy dishes.'
        ]
    },
    'Malaria': {
        'display': 'Plasmodium Malaria Infection',
        'icon': '🦟',
        'overview': "A life-threatening parasitic infection transmitted through the bite of infected female Anopheles mosquitoes, characterized by paroxysms of shaking chills and high fever.",
        'prescription': [
            {'name': 'Artemether-Lumefantrine (ACT)', 'instruction': 'Standard 6-dose course over 3 days taken with fatty food/milk as prescribed', 'duration': '3 days'},
            {'name': 'Paracetamol 650mg', 'instruction': 'Take 1 tablet every 6 hours for body temperature >101°F (38.3°C)', 'duration': '3-5 days'},
            {'name': 'Oral Electrolytes & Hydration', 'instruction': 'Drink 3 liters of fluids daily to protect renal perfusion', 'duration': 'Daily'}
        ],
        'care': [
            'Complete the full course of antimalarial medications even if feeling better.',
            'Sleep under insecticide-treated bed nets and use mosquito repellents.',
            'Monitor serial blood smears or rapid diagnostic tests to verify parasitemia clearance.'
        ],
        'diet': [
            'High-calorie, easily digestible liquid and semi-solid diet during febrile stages.',
            'Electrolyte-rich coconut water, fresh orange juice, and vegetable soups.',
            'Gradually reintroduce high-protein foods (lentils, steamed poultry) as appetite recovers.'
        ]
    },
    'Chicken pox': {
        'display': 'Varicella (Chickenpox)',
        'icon': '🔴',
        'overview': "A highly contagious viral infection caused by the Varicella-Zoster virus, resulting in an itchy vesicular blister rash, fever, and malaise.",
        'prescription': [
            {'name': 'Calamine Lotion', 'instruction': 'Apply gently to itchy lesions 3-4 times daily to soothe skin', 'duration': '7 days'},
            {'name': 'Cetirizine 10mg / Diphenhydramine', 'instruction': 'Take 1 tablet at bedtime to reduce nighttime scratching', 'duration': '5-7 days'},
            {'name': 'Paracetamol 500mg', 'instruction': 'Take 1 tablet for fever (NEVER give aspirin to children/teens - Reye syndrome risk)', 'duration': '5 days'}
        ],
        'care': [
            'Keep patient isolated until all blisters have fully dried into crusty scabs.',
            'Keep fingernails trimmed short and clean to prevent secondary bacterial skin infections.',
            'Bathe in lukewarm oatmeal baths to alleviate generalized itchiness.'
        ],
        'diet': [
            'Soft, cool, bland foods if mouth lesions are present (smoothies, ice pops, yogurt).',
            'Generous fluid intake (cool water, diluted fruit juices, broths).',
            'Avoid salty, highly acidic, or hard crunchy foods that irritate oral sores.'
        ]
    },
    'Dengue': {
        'display': 'Dengue Virus Infection (Breakbone Fever)',
        'icon': '🩸',
        'overview': "A mosquito-borne viral infection causing high fever, severe retro-orbital pain, muscle/joint aches, and potential thrombocytopenia (low platelet count).",
        'prescription': [
            {'name': 'Paracetamol 650mg', 'instruction': 'Take 1 tablet every 6 hours for fever (STRICTLY AVOID Aspirin/Ibuprofen - bleeding risk)', 'duration': '5 days'},
            {'name': 'Oral Rehydration Salts (ORS) Therapy', 'instruction': 'Drink 2.5 - 3.5 liters of oral fluids daily to prevent capillary leak shock', 'duration': 'Continuous'},
            {'name': 'Daily Complete Blood Count (CBC)', 'instruction': 'Track platelet count and hematocrit daily until fever defervescence', 'duration': 'Daily'}
        ],
        'care': [
            'Absolute physical bed rest during both febrile and critical defervescence phases.',
            'Watch closely for warning signs: severe abdominal pain, persistent vomiting, mucosal bleeding.',
            'Use mosquito nets and repellents to prevent transmitting virus to household members.'
        ],
        'diet': [
            'Drink fluids constantly: ORS, tender coconut water, pomegranate juice, barley water.',
            'Papaya leaf extract syrup can be consumed as supportive therapy.',
            'Eat light, soft, nutrient-rich meals (khichdi, vegetable purees).'
        ]
    },
    'Typhoid': {
        'display': 'Enteric Typhoid Fever',
        'icon': '🌡️',
        'overview': "A systemic bacterial infection caused by Salmonella enterica serovar Typhi, contracted via contaminated food or water, causing sustained step-ladder fever and abdominal distress.",
        'prescription': [
            {'name': 'Cefixime 200mg / Azithromycin 500mg', 'instruction': 'Take 1 tablet twice daily with food as prescribed by physician', 'duration': '7-14 days'},
            {'name': 'Paracetamol 650mg', 'instruction': 'Take 1 tablet for temperatures >101°F up to 4 times daily', 'duration': '5-7 days'},
            {'name': 'Probiotics & Digestive Enzymes', 'instruction': 'Take daily to assist intestinal flora restoration post-antibiotics', 'duration': '14 days'}
        ],
        'care': [
            'Complete entire antibiotic regimen without interruption to avoid chronic carrier state.',
            'Drink only boiled or certified bottled water and wash hands rigorously before meals.',
            'Rest extensively and monitor for complications such as intestinal bleeding.'
        ],
        'diet': [
            'High-calorie, low-fiber, bland soft diet (custards, boiled potatoes, rice gruel).',
            'Avoid raw vegetables, unpeeled fruits, coarse grains, and fried spicy foods.',
            'Ensure abundant clean fluid intake (boiled water, clear soups, electrolytes).'
        ]
    },
    'hepatitis A': {
        'display': 'Hepatitis A Virus Infection',
        'icon': '🟡',
        'overview': "An acute, usually self-limiting liver infection caused by the Hepatitis A virus, transmitted predominantly through the fecal-oral route via contaminated water or food.",
        'prescription': [
            {'name': 'Supportive Care & Symptom Relief', 'instruction': 'Rest, antiemetics if nauseated, no specific antiviral required', 'duration': '2-4 weeks'},
            {'name': 'B-Complex with Vitamin C', 'instruction': 'Daily tablet to support hepatic metabolic recovery', 'duration': '30 days'},
            {'name': 'Liver Panel Monitoring', 'instruction': 'Repeat liver enzymes and bilirubin weekly until normal', 'duration': 'Weekly'}
        ],
        'care': [
            'Strict avoidance of alcohol and all non-essential medications.',
            'Thorough handwashing with soap after using the restroom and before food preparation.',
            'Adequate physical rest during the acute jaundice phase.'
        ],
        'diet': [
            'Nutrient-rich, easily digestible high-carbohydrate meals.',
            'Fresh fruits, fruit juices, sugarcane juice, and boiled vegetables.',
            'Avoid fatty foods, heavy oils, fried meats, and artificial seasonings.'
        ]
    },
    'Hepatitis B': {
        'display': 'Hepatitis B Virus Infection',
        'icon': '🧬',
        'overview': "A serious liver infection caused by the Hepatitis B virus, transmitted through blood, sexual fluids, or perinatal contact, capable of chronic progression.",
        'prescription': [
            {'name': 'Tenofovir / Entecavir (Specialist Prescribed)', 'instruction': 'Antiviral therapy if chronic active disease with elevated viral load', 'duration': 'Long-term'},
            {'name': 'HBsAg, HBeAg, and HBV DNA Viral Load', 'instruction': 'Laboratory monitoring every 3-6 months by gastroenterologist', 'duration': 'Periodic'},
            {'name': 'Abdominal Ultrasound & AFP Screening', 'instruction': 'Annual hepatic surveillance for hepatocellular health', 'duration': 'Annual'}
        ],
        'care': [
            'Family members and intimate partners should receive the Hepatitis B vaccine series.',
            'Absolute prohibition of alcohol consumption to prevent accelerated fibrosis.',
            'Never share razors, toothbrushes, needles, or personal manicure items.'
        ],
        'diet': [
            'Balanced, liver-supportive diet with high antioxidant content (berries, greens).',
            'Moderate lean protein intake (legumes, fish, tofu, eggs).',
            'Limit sodium and heavily processed foods.'
        ]
    },
    'Hepatitis C': {
        'display': 'Hepatitis C Virus Infection',
        'icon': '🧬',
        'overview': "A bloodborne viral infection that causes inflammation of the liver, curable in >95% of cases with modern direct-acting antiviral (DAA) medications.",
        'prescription': [
            {'name': 'Sofosbuvir + Velpatasvir (DAA Regimen)', 'instruction': '1 tablet once daily with food as prescribed by hepatologist', 'duration': '12 weeks'},
            {'name': 'HCV RNA Quantitative PCR Test', 'instruction': 'Verify sustained virologic response (SVR12) 12 weeks after treatment', 'duration': 'Post-Rx'},
            {'name': 'Liver Fibrosis Assessment (FibroScan)', 'instruction': 'Assess degree of baseline liver scarring prior to therapy', 'duration': 'Baseline'}
        ],
        'care': [
            'Complete all 12 weeks of DAA pills without missing doses for full viral cure.',
            'Completely abstain from alcohol consumption.',
            'Practice safe blood precautions; do not donate blood, organs, or semen.'
        ],
        'diet': [
            'Plant-forward Mediterranean diet rich in whole vegetables and grains.',
            'Moderate coffee consumption (2-3 cups/day) has documented liver-protective associations.',
            'Avoid raw or undercooked shellfish (Vibrio vulnerability in liver disease).'
        ]
    },
    'Hepatitis D': {
        'display': 'Hepatitis D (Delta Virus Co-infection)',
        'icon': '🧬',
        'overview': "A subviral satellite that can only propagate in individuals with active Hepatitis B virus infection, resulting in accelerated liver disease.",
        'prescription': [
            {'name': 'Pegylated Interferon Alfa-2a / Bulevirtide', 'instruction': 'Specialist hepatology therapeutic regimen under hospital monitoring', 'duration': '48 weeks'},
            {'name': 'HDV RNA & Quantitative HBsAg Testing', 'instruction': 'Periodic virological monitoring to assess treatment efficacy', 'duration': 'Regular'},
            {'name': 'Hepatoprotective Antioxidant Support', 'instruction': 'Daily formulation supporting cellular resilience', 'duration': 'Continuous'}
        ],
        'care': [
            'Strict adherence to hepatology appointments and blood surveillance.',
            'Zero tolerance for alcohol or unprescribed over-the-counter supplements.',
            'Screen for portal hypertension and esophageal varices.'
        ],
        'diet': [
            'Strict sodium restriction (<2,000 mg/day) if portal pressure is elevated.',
            'Nutrient-dense, low-fat, high-fiber meals.',
            'Ensure balanced caloric intake to prevent muscle wasting.'
        ]
    },
    'Hepatitis E': {
        'display': 'Hepatitis E Virus Infection',
        'icon': '🟡',
        'overview': "An acute waterborne viral hepatitis infection transmitted via fecal-oral contamination, generally self-limiting but severe in pregnant individuals.",
        'prescription': [
            {'name': 'Supportive Outpatient Care & Rest', 'instruction': 'Rest and hydration; infection typically resolves spontaneously within 4-6 weeks', 'duration': '4-6 weeks'},
            {'name': 'Liver Function Monitoring', 'instruction': 'Bi-weekly blood tests to ensure transaminases are resolving toward normal', 'duration': 'Bi-weekly'},
            {'name': 'Electrolyte & Hydration Support', 'instruction': 'Maintain oral fluids and clean water intake', 'duration': 'Daily'}
        ],
        'care': [
            'Special vigilance and immediate hospital admission if patient is pregnant.',
            'Only consume thoroughly boiled or treated drinking water.',
            'Practice rigorous hand sanitization before preparing food.'
        ],
        'diet': [
            'Light, easily digestible carbohydrate-rich foods (congee, fruit compotes).',
            'Fresh coconut water, buttermilk, herbal teas, and clean broths.',
            'Zero fatty, fried, or spicy meals.'
        ]
    },
    'Alcoholic hepatitis': {
        'display': 'Alcohol-Induced Hepatitis',
        'icon': '🍷',
        'overview': "Acute liver inflammation caused by heavy, prolonged alcohol consumption, leading to hepatocellular damage, jaundice, and potential liver failure.",
        'prescription': [
            {'name': 'ABSOLUTE AND PERMANENT ALCOHOL CESSATION', 'instruction': 'Immediate and complete cessation of all alcohol is essential for survival', 'duration': 'Lifelong'},
            {'name': 'Prednisolone 40mg (if Maddrey DF >= 32)', 'instruction': 'Corticosteroid therapy prescribed in-hospital for severe presentation', 'duration': '28 days'},
            {'name': 'Thiamine (Vitamin B1) 100mg + Folate', 'instruction': 'Take daily to prevent Wernicke encephalopathy and neurological injury', 'duration': 'Daily'}
        ],
        'care': [
            'Engage with addiction medicine specialist and support groups (AA).',
            'Seek immediate emergency care if abdominal swelling (ascites) or confusion occurs.',
            'Frequent monitoring of coagulation (INR) and bilirubin.'
        ],
        'diet': [
            'High-protein, high-calorie nutritional therapy to overcome malnutrition.',
            'Sodium restriction (<2g/day) to prevent fluid accumulation in abdomen and legs.',
            'Eat frequent small meals including a late-evening carbohydrate snack.'
        ]
    },
    'Tuberculosis': {
        'display': 'Pulmonary Tuberculosis (TB)',
        'icon': '🫁',
        'overview': "A serious airborne bacterial infection caused by Mycobacterium tuberculosis, primarily attacking the lungs and causing chronic cough, hemoptysis, and night sweats.",
        'prescription': [
            {'name': 'Directly Observed Therapy (DOTS) Regimen', 'instruction': 'Rifampicin, Isoniazid, Pyrazinamide, Ethambutol (RIPE) daily as prescribed', 'duration': '6 months'},
            {'name': 'Pyridoxine (Vitamin B6) 25-50mg', 'instruction': 'Take daily to prevent isoniazid-induced peripheral neuropathy', 'duration': '6 months'},
            {'name': 'Sputum Smear & Culture Monitoring', 'instruction': 'Perform periodic monthly sputum evaluations to confirm conversion', 'duration': 'Monthly'}
        ],
        'care': [
            'Never skip a single dose of TB medications to prevent multi-drug resistance (MDR-TB).',
            'Wear a well-fitted mask and ensure cross-ventilation in living spaces until non-infectious.',
            'Household contacts must be screened for latent TB infection.'
        ],
        'diet': [
            'High-protein, calorie-dense diet (eggs, milk, legumes, nuts) to regain lost body weight.',
            'Foods rich in vitamins A, C, E, and Zinc to bolster cell-mediated immunity.',
            'Avoid all alcohol and tobacco products.'
        ]
    },
    'Common Cold': {
        'display': 'Viral Upper Respiratory Infection (Common Cold)',
        'icon': '🤧',
        'overview': "A mild, self-limiting viral infection of the upper respiratory tract (rhinovirus, coronavirus), causing nasal congestion, sneezing, sore throat, and low-grade malaise.",
        'prescription': [
            {'name': 'Paracetamol 500mg', 'instruction': 'Take 1 tablet every 6-8 hours as needed for body ache and fever', 'duration': '3-5 days'},
            {'name': 'Saline Nasal Drops / Spray', 'instruction': '2-3 drops in each nostril 3-4 times daily to relieve congestion', 'duration': '5 days'},
            {'name': 'Steam Inhalation with Menthol / Eucalyptus', 'instruction': 'Inhale warm steam 2 times daily for 10 minutes', 'duration': '5 days'}
        ],
        'care': [
            'Get 8 hours of restorative sleep to empower immune response.',
            'Wash hands frequently and use tissues to prevent spreading virus to others.',
            'Gargle with warm salt water (1/2 tsp salt in 1 cup warm water) for sore throat.'
        ],
        'diet': [
            'Drink plentiful warm fluids: ginger-lemon tea, chicken broth, herbal teas.',
            'Consume foods rich in Vitamin C (oranges, guavas, amla, bell peppers).',
            'Honey with warm water or ginger juice to soothe cough reflex.'
        ]
    },
    'Pneumonia': {
        'display': 'Community-Acquired Pneumonia',
        'icon': '🫁',
        'overview': "An infection that inflames the air sacs (alveoli) in one or both lungs, which may fill with fluid or purulent sputum, causing productive cough, fever, and pleuritic pain.",
        'prescription': [
            {'name': 'Amoxicillin-Clavulanate 625mg / Azithromycin 500mg', 'instruction': 'Take twice daily with meals as prescribed by physician', 'duration': '7 days'},
            {'name': 'Paracetamol 650mg', 'instruction': 'Take 1 tablet every 6 hours for high temperature and pleuritic chest ache', 'duration': '5 days'},
            {'name': 'N-Acetylcysteine 600mg', 'instruction': 'Effervescent tablet in water once daily morning to thin mucous secretions', 'duration': '5 days'}
        ],
        'care': [
            'Use an incentive spirometer or perform deep breathing exercises hourly.',
            'Monitor oxygen saturation (SpO2) with a pulse oximeter; seek ER if <93%.',
            'Strict bed rest and complete the full prescribed course of antibiotics.'
        ],
        'diet': [
            'Drink at least 2.5 to 3 liters of fluids daily to liquefy bronchial secretions.',
            'Warm soups, broths, and high-protein meals to preserve respiratory muscle strength.',
            'Avoid cold refrigerated drinks and dairy if it thickens phlegm sensation.'
        ]
    },
    'Dimorphic hemmorhoids(piles)': {
        'display': 'Hemorrhoidal Disease (Piles)',
        'icon': '🩹',
        'overview': "Swollen, inflamed veins in the lower rectum and anus, causing rectal bleeding during bowel movements, localized itching, and discomfort.",
        'prescription': [
            {'name': 'Topical Anesthetic & Anti-inflammatory Ointment (Hydrocortisone + Lidocaine)', 'instruction': 'Apply gently to affected rectal area after bowel movement and at bedtime', 'duration': '7 days'},
            {'name': 'Psyllium Husk / Isabgol (Soluble Fiber)', 'instruction': '1-2 tablespoons in a full glass of water at bedtime to soften stool', 'duration': 'Ongoing'},
            {'name': 'Warm Sitz Bath', 'instruction': 'Soak anal area in warm water for 15-20 minutes, 2-3 times daily', 'duration': 'Daily'}
        ],
        'care': [
            'Do not strain or sit on the toilet for prolonged periods (>5 minutes).',
            'Use soft moist wipes or a bidet rather than dry rough toilet paper.',
            'Stay physically active (daily walking) to stimulate natural bowel motility.'
        ],
        'diet': [
            'High-fiber diet (25-35g/day): whole oats, beans, bran cereals, figs, prunes.',
            'Drink a minimum of 2.5 - 3.0 liters of water daily so fiber can soften stools.',
            'Avoid hot chili peppers, excessive alcohol, and heavy red meats.'
        ]
    },
    'Heart attack': {
        'display': 'Acute Myocardial Infarction (Heart Attack)',
        'icon': '🫀',
        'overview': "A life-threatening medical emergency where blood flow to a section of the heart muscle becomes abruptly blocked, starving myocardial tissue of oxygen.",
        'prescription': [
            {'name': 'CALL EMERGENCY MEDICAL SERVICES (911 / 108) IMMEDIATELY', 'instruction': 'Request urgent cardiac ambulance dispatch with ECG equipment', 'duration': 'IMMEDIATE'},
            {'name': 'Chewable Aspirin 300mg', 'instruction': 'Chew and swallow immediately if directed by emergency dispatch clinician', 'duration': 'Stat'},
            {'name': 'Sublingual Nitroglycerin 0.4mg', 'instruction': 'Place under tongue if previously prescribed and blood pressure is not low', 'duration': 'Per protocol'}
        ],
        'care': [
            'Stop all activity immediately; sit or lie down comfortably in semi-upright posture.',
            'Loosen tight clothing around neck and chest; stay calm and wait for ambulance.',
            'Do NOT attempt to drive yourself to the hospital.'
        ],
        'diet': [
            'Nil by mouth during acute emergency.',
            'Long-term post-infarction: strict low-sodium, heart-healthy Mediterranean diet.',
            'Permanently eliminate trans fats, fried foods, and tobacco smoke.'
        ]
    },
    'Varicose veins': {
        'display': 'Varicose Veins (Venous Insufficiency)',
        'icon': '🦵',
        'overview': "Enlarged, twisted, swollen superficial veins caused by weakened or damaged venous valves, typically in the lower extremities, leading to heaviness and aching.",
        'prescription': [
            {'name': 'Graduated Compression Stockings (Class II, 20-30 mmHg)', 'instruction': 'Put on in morning before getting out of bed; wear during daytime hours', 'duration': 'Daily'},
            {'name': 'Micronized Purified Flavonoid Fraction (MPFF) 500mg', 'instruction': 'Take 1 tablet twice daily with meals to improve venous tone', 'duration': '2-3 months'},
            {'name': 'Topical Heparinoid / Horse Chestnut Gel', 'instruction': 'Apply upward strokes from ankle to calf 2 times daily', 'duration': 'Ongoing'}
        ],
        'care': [
            'Elevate legs above the level of your heart for 15-20 minutes 3 times daily.',
            'Avoid prolonged uninterrupted standing or sitting; flex calves and walk periodically.',
            'Maintain a healthy body weight to diminish excessive venous hydrostatic pressure.'
        ],
        'diet': [
            'Eat foods rich in flavonoids (blueberries, dark leafy greens, citrus bioflavonoids).',
            'High-fiber diet to avoid chronic constipation and elevated intra-abdominal pressure.',
            'Limit sodium intake to minimize dependent lower-extremity fluid retention.'
        ]
    },
    'Hypothyroidism': {
        'display': 'Primary Hypothyroidism',
        'icon': '🦋',
        'overview': "An underactive thyroid gland that fails to produce adequate thyroid hormones (T3, T4), resulting in sluggish metabolism, weight gain, fatigue, and cold intolerance.",
        'prescription': [
            {'name': 'Levothyroxine Sodium (25-100 mcg as prescribed)', 'instruction': 'Take 1 tablet once daily morning on an empty stomach with water, 30-60 min before breakfast', 'duration': 'Lifelong'},
            {'name': 'Serum TSH & Free T4 Monitoring', 'instruction': 'Repeat blood test 6-8 weeks after starting or adjusting dosage, then annually', 'duration': 'Periodic'},
            {'name': 'Separate Calcium / Iron Supplements', 'instruction': 'Take any calcium, iron, or antacid pills at least 4 hours apart from thyroid dose', 'duration': 'Ongoing'}
        ],
        'care': [
            'Strict adherence to morning empty-stomach administration routine.',
            'Keep regular sleep and physical activity schedules to boost metabolic rate.',
            'Report any palpitations, tremors, or insomnia to doctor (signs of excess dose).'
        ],
        'diet': [
            'Ensure adequate dietary iodine through iodized table salt and seafood.',
            'Incorporate selenium-rich foods (Brazil nuts, sunflower seeds, fish).',
            'Cook goitrogenic cruciferous vegetables (broccoli, cabbage) thoroughly before eating.'
        ]
    },
    'Hyperthyroidism': {
        'display': 'Hyperthyroidism (Thyrotoxicosis)',
        'icon': '🦋',
        'overview': "An overactive thyroid gland producing excessive quantities of thyroid hormone, driving an accelerated metabolic state with tremors, palpitations, weight loss, and heat intolerance.",
        'prescription': [
            {'name': 'Methimazole 5-20mg / Propylthiouracil (PTU)', 'instruction': 'Antithyroid medication taken daily exactly as directed by endocrinologist', 'duration': '12-18 months'},
            {'name': 'Propranolol 20-40mg', 'instruction': 'Take 2-3 times daily to control tachycardia, tremors, and palpitations', 'duration': 'As needed'},
            {'name': 'Serum TSH, Free T3, and Free T4 Panel', 'instruction': 'Check every 4-6 weeks until thyroid levels stabilize', 'duration': 'Regular'}
        ],
        'care': [
            'Avoid excessive heat and intense strenuous exertion while heart rate is elevated.',
            'Promptly report sore throat or high fever to physician (risk of agranulocytosis).',
            'Protect eyes with lubricating drops and sunglasses if Graves ophthalmopathy present.'
        ],
        'diet': [
            'High-calorie diet with adequate protein during hypermetabolic weight loss phase.',
            'Ensure calcium (1,200 mg/day) and Vitamin D3 to counteract bone turnover.',
            'Avoid kelp, seaweed, and high-iodine supplements that fuel thyroid hormone synthesis.'
        ]
    },
    'Hypoglycemia': {
        'display': 'Acute Hypoglycemia (Low Blood Sugar)',
        'icon': '🍬',
        'overview': "A condition where blood glucose drops below normal levels (<70 mg/dL / 3.9 mmol/L), triggering shakiness, sweating, anxiety, confusion, and dizziness.",
        'prescription': [
            {'name': 'Fast-Acting Carbohydrates (Rule of 15)', 'instruction': 'Consume 15g fast-acting sugar immediately (1/2 cup fruit juice, 3-4 glucose tablets, or 3 tsp sugar in water)', 'duration': 'Immediate'},
            {'name': 'Recheck Blood Glucose in 15 Minutes', 'instruction': 'If still <70 mg/dL, repeat 15g fast sugar; recheck until normalized', 'duration': '15 min'},
            {'name': 'Complex Snack Following Resolution', 'instruction': 'Eat a snack containing complex carb and protein (peanut butter toast, crackers + cheese)', 'duration': 'Post-episode'}
        ],
        'care': [
            'Always carry glucose tablets or hard candies with you at all times.',
            'Wear a medical alert identification bracelet indicating hypoglycemia risk.',
            'Review medication dosages with doctor if episodes happen frequently.'
        ],
        'diet': [
            'Eat regular, evenly spaced meals; never skip meals or delay carbohydrate intake.',
            'Pair complex carbohydrates with protein and healthy fats to stabilize absorption.',
            'Avoid alcohol on an empty stomach as it severely blunts hepatic glucose release.'
        ]
    },
    'Osteoarthristis': {
        'display': 'Osteoarthritis (Degenerative Joint Disease)',
        'icon': '🦴',
        'overview': "The most common form of arthritis, involving progressive breakdown of joint cartilage and underlying bone, resulting in joint pain, stiffness, and crepitus.",
        'prescription': [
            {'name': 'Paracetamol 650mg / Topical NSAID Gel (Diclofenac 1%)', 'instruction': 'First-line therapy; apply gel to affected joint 3-4 times daily', 'duration': 'Ongoing'},
            {'name': 'Glucosamine & Chondroitin Sulfate', 'instruction': 'Take daily nutritional formulation to support cartilage lubrication', 'duration': '3 months'},
            {'name': 'Intra-articular Hyaluronic Acid / Corticosteroid', 'instruction': 'Clinician-administered injection for severe refractory joint pain', 'duration': 'As indicated'}
        ],
        'care': [
            'Participate in low-impact joint-friendly exercise (swimming, stationary cycling, water aerobics).',
            'Maintain optimal body weight: losing 1 pound reduces 4 pounds of pressure on knee joints.',
            'Apply heat packs for morning joint stiffness, and cold packs after activity.'
        ],
        'diet': [
            'Incorporate anti-inflammatory Mediterranean foods (olive oil, wild salmon, walnuts).',
            'Eat foods high in Vitamin C and bioflavonoids to support collagen synthesis.',
            'Minimize refined sugars and fried foods that promote systemic inflammatory cytokines.'
        ]
    },
    'Arthritis': {
        'display': 'Inflammatory Arthritis (Rheumatoid / Polyarthritis)',
        'icon': '🦴',
        'overview': "Systemic inflammatory or autoimmune joint disease marked by symmetrical synovitis, joint swelling, morning stiffness lasting >30 minutes, and fatigue.",
        'prescription': [
            {'name': 'Disease-Modifying Antirheumatic Drugs (DMARDs - Methotrexate)', 'instruction': 'Specialist rheumatology regimen to halt joint erosive progression', 'duration': 'Ongoing'},
            {'name': 'Celecoxib 200mg / Naproxen 500mg', 'instruction': 'Take 1 tablet daily with food for symptomatic joint inflammation', 'duration': 'As prescribed'},
            {'name': 'Folic Acid 5mg (Weekly)', 'instruction': 'Take on non-methotrexate day to minimize medication side effects', 'duration': 'Weekly'}
        ],
        'care': [
            'Consult a rheumatologist early; early DMARD initiation prevents joint deformity.',
            'Balance regular gentle physical therapy with adequate joint rest during flares.',
            'Use assistive devices or ergonomic joint-sparing tools in daily tasks.'
        ],
        'diet': [
            'Anti-inflammatory diet rich in omega-3 fatty acids (salmon, mackerel, chia seeds).',
            'Incorporate turmeric (curcumin) and ginger for natural anti-inflammatory benefits.',
            'Limit red meat, processed carbohydrates, and excess saturated fats.'
        ]
    },
    'Psoriasis': {
        'display': 'Psoriasis Vulgaris',
        'icon': '🧴',
        'overview': "A chronic immune-mediated skin condition that accelerates the life cycle of skin cells, leading to thick, red patches covered with silvery scales.",
        'prescription': [
            {'name': 'Topical Clobetasol / Betamethasone Dipropionate Cream', 'instruction': 'Apply sparingly to thick plaques twice daily for 2 weeks, then taper', 'duration': '2-3 weeks'},
            {'name': 'Calcipotriol (Vitamin D3 Analogue) Ointment', 'instruction': 'Apply to plaques once daily morning to slow skin cell turnover', 'duration': '4-8 weeks'},
            {'name': 'Coal Tar / Salicylic Acid Shampoo', 'instruction': 'Lather on scalp, leave for 5-10 minutes before rinsing, 2-3 times weekly', 'duration': 'Ongoing'}
        ],
        'care': [
            'Keep skin well-moisturized with thick emollient ointments immediately after bathing.',
            'Get brief, controlled natural sunlight exposure (10-15 minutes, avoiding sunburn).',
            'Avoid skin injury, scratching, and cuts (Koebner phenomenon triggers new plaques).'
        ],
        'diet': [
            'Anti-inflammatory Mediterranean dietary pattern rich in colorful vegetables.',
            'Reduce alcohol and tobacco consumption, both of which trigger flare-ups.',
            'Incorporate omega-3 fish oils to modulate inflammatory eicosanoids.'
        ]
    },
    'Impetigo': {
        'display': 'Impetigo (Contagious Bacterial Pyoderma)',
        'icon': '🩹',
        'overview': "A highly contagious superficial bacterial skin infection (Staphylococcus or Streptococcus), presenting with characteristic honey-colored crusted erosions.",
        'prescription': [
            {'name': 'Mupirocin 2% Topical Ointment', 'instruction': 'Wash area, soak off crusts gently with warm water, apply ointment 3 times daily', 'duration': '7-10 days'},
            {'name': 'Cephalexin 500mg (Oral Antibiotic)', 'instruction': 'Take 1 capsule 4 times daily if lesions are widespread, under doctor prescription', 'duration': '7 days'},
            {'name': 'Antiseptic Chlorhexidine Wash', 'instruction': 'Cleanse surrounding skin gently once daily', 'duration': '7 days'}
        ],
        'care': [
            'Keep infected sores loosely covered with clean gauze to prevent scratching and spread.',
            'Wash hands thoroughly after touching affected skin.',
            'Wash patient towels, clothing, and bedding in hot water daily and do not share.'
        ],
        'diet': [
            'Eat a well-rounded diet with high-protein foods to support rapid dermal healing.',
            'Stay well-hydrated with clean water and herbal teas.',
            'Incorporate foods containing Vitamin C and Zinc to support cutaneous immunity.'
        ]
    },
    'Urinary tract infection': {
        'display': 'Urinary Tract Infection (Acute Cystitis)',
        'icon': '🚽',
        'overview': "A bacterial infection of the urinary bladder or urethra, characterized by dysuria (burning urination), urinary urgency, pelvic ache, and cloudy urine.",
        'prescription': [
            {'name': 'Nitrofurantoin 100mg / Fosfomycin 3g', 'instruction': 'Take twice daily with meals as directed by physician following urine culture', 'duration': '5 days'},
            {'name': 'Phenazopyridine 200mg', 'instruction': 'Take 1 tablet after meals for acute urinary burning (turns urine orange-red)', 'duration': '2 days'},
            {'name': 'Urinary Alkalinizer (Disodium Hydrogen Citrate)', 'instruction': 'Mix 2 teaspoons in glass of water twice daily to soothe burning sensation', 'duration': '3-5 days'}
        ],
        'care': [
            'Drink at least 2.5 - 3.0 liters of water daily to flush bacteria from urinary tract.',
            'Urinate promptly when you feel the urge; do not hold urine in bladder.',
            'Wipe from front to back after using the bathroom to avoid introducing bacteria.'
        ],
        'diet': [
            'Drink 100% pure unsweetened cranberry juice or take D-mannose supplements.',
            'Avoid coffee, alcohol, carbonated drinks, and spicy foods that irritate the bladder.',
            'Eat probiotic foods like plain unsweetened yogurt to replenish urogenital flora.'
        ]
    }
}

full_data = {
    'features': cols,
    'friendly_names': friendly_names,
    'diseases': diseases,
    'cond_probs': cond_probs,
    'clinical_db': CLINICAL_DATABASE
}

client_data_path = os.path.join(base_dir, "client", "src", "clinical_model.json")
with open(client_data_path, "w") as f:
    json.dump(full_data, f)

print(f"Exported clinical_model.json successfully to {client_data_path} with {len(diseases)} diseases!")

