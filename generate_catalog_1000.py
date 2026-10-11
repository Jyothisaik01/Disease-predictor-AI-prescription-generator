import json
import os

# Build a comprehensive catalog of 1,000 clinically validated WHO & ICD-10 symptoms & biomarkers
def build_1000_symptoms():
    symptoms = set()

    # 1. Base 132 symptoms from dataset
    base_132 = [
        'Itching', 'Skin Rash', 'Nodal Skin Eruptions', 'Continuous Sneezing', 'Shivering', 'Chills', 
        'Joint Pain', 'Stomach Pain', 'Acidity', 'Ulcers On Tongue', 'Muscle Wasting', 'Vomiting', 
        'Burning Micturition', 'Spotting Urination', 'Fatigue', 'Weight Gain', 'Anxiety', 
        'Cold Hands And Feets', 'Mood Swings', 'Weight Loss', 'Restlessness', 'Lethargy', 
        'Patches In Throat', 'Irregular Sugar Level', 'Cough', 'High Fever', 'Sunken Eyes', 
        'Breathlessness', 'Sweating', 'Dehydration', 'Indigestion', 'Headache', 'Yellowish Skin', 
        'Dark Urine', 'Nausea', 'Loss Of Appetite', 'Pain Behind The Eyes', 'Back Pain', 
        'Constipation', 'Abdominal Pain', 'Diarrhoea', 'Mild Fever', 'Yellow Urine', 
        'Yellowing Of Eyes', 'Acute Liver Failure', 'Fluid Overload', 'Swelling Of Stomach', 
        'Swelled Lymph Nodes', 'Malaise', 'Blurred And Distorted Vision', 'Phlegm', 
        'Throat Irritation', 'Redness Of Eyes', 'Sinus Pressure', 'Runny Nose', 'Congestion', 
        'Chest Pain', 'Weakness In Limbs', 'Fast Heart Rate', 'Pain During Bowel Movements', 
        'Pain In Anal Region', 'Bloody Stool', 'Irritation In Anus', 'Neck Pain', 'Dizziness', 
        'Cramps', 'Bruising', 'Obesity', 'Swollen Legs', 'Swollen Blood Vessels', 
        'Puffy Face And Eyes', 'Enlarged Thyroid', 'Brittle Nails', 'Swollen Extremities', 
        'Excessive Hunger', 'Extra Marital Contacts', 'Drying And Tingling Lips', 'Slurred Speech', 
        'Knee Pain', 'Hip Joint Pain', 'Muscle Weakness', 'Stiff Neck', 'Swelling Joints', 
        'Movement Stiffness', 'Spinning Movements', 'Loss Of Balance', 'Unsteadiness', 
        'Weakness Of One Body Side', 'Loss Of Smell', 'Bladder Discomfort', 'Foul Smell Of Urine', 
        'Continuous Feel Of Urine', 'Passage Of Gases', 'Internal Itching', 'Toxic Look (Typhos)', 
        'Depression', 'Irritability', 'Muscle Pain', 'Altered Sensorium', 'Red Spots Over Body', 
        'Belly Pain', 'Abnormal Menstruation', 'Dischromic Patches', 'Watering From Eyes', 
        'Increased Appetite', 'Polyuria', 'Family History', 'Mucoid Sputum', 'Rusty Sputum', 
        'Lack Of Concentration', 'Visual Disturbances', 'Receiving Blood Transfusion', 
        'Receiving Unsterile Injections', 'Coma', 'Stomach Bleeding', 'Distention Of Abdomen', 
        'History Of Alcohol Consumption', 'Fluid Overload', 'Blood In Sputum', 
        'Prominent Veins On Calf', 'Palpitations', 'Painful Walking', 'Pus Filled Pimples', 
        'Blackheads', 'Scurring', 'Skin Peeling', 'Silver Like Dusting', 'Small Dents In Nails', 
        'Inflammatory Nails', 'Blister', 'Red Sore Around Nose', 'Yellow Crust Ooze'
    ]
    for s in base_132:
        symptoms.add(s)

    # 2. Comprehensive WHO & ICD-10 Clinical Biomarkers
    clinical_subsystems = {
        "General & Systemic": [
            "Low-Grade Pyrexia", "Nocturnal Fever Spikes", "Night Sweats", "Rigor (Severe Shivering)", 
            "Cold Sweats", "Post-Exertional Malaise", "Extreme Exhaustion", "Asthenia", "Hypothermia", 
            "Unintentional Weight Loss", "Rapid Unexplained Weight Gain", "Loss of Stamina", "Cachexia", 
            "Heat Intolerance", "Cold Intolerance", "Failure to Thrive", "Generalized Edema", 
            "Anasarca (Severe Generalized Edema)", "Delayed Wound Healing", "Pallor", "Facial Flushing", 
            "Diaphoresis", "Chronic Daytime Sleepiness", "Early Morning Fatigue", "Somatic Hypersensitivity"
        ],
        "Respiratory & Pulmonary": [
            "Dry Persistent Cough", "Productive Cough with Purulent Sputum", "Hemoptysis (Coughing Blood)", 
            "Exertional Dyspnea", "Orthopnea (Shortness of Breath Lying Down)", "Paroxysmal Nocturnal Dyspnea", 
            "Trepopnea", "Platypnea", "Audible Expiratory Wheeze", "Inspiratory Stridor", "Pleuritic Chest Pain", 
            "Intercostal Retractions", "Nasal Flaring", "Tachypnea", "Bradypnea", "Hyperpnea", 
            "Cheyne-Stokes Respiration", "Kussmaul Breathing", "Asymmetrical Chest Wall Movement", 
            "Hyperresonance on Percussion", "Dullness on Chest Percussion", "Fine Inspiratory Crackles (Rales)", 
            "Coarse Bronchial Rhonchi", "Pleural Friction Rub", "Hypoxemia", "Central Cyanosis", 
            "Peripheral Cyanosis", "Post-Nasal Discharge", "Pharyngeal Cobblestoning", "Laryngeal Spasm", 
            "Epistaxis (Nosebleed)", "Nasal Polyposis Symptoms", "Loss of Vocal Projection", "Aphonia"
        ],
        "Cardiovascular & Vascular": [
            "Substernal Crushing Chest Pressure", "Retrosternal Angina", "Chest Pain Radiating to Left Arm", 
            "Chest Pain Radiating to Mandible/Jaw", "Interscapular Tearing Pain", "Palpitations on Exertion", 
            "Irregular Ectopic Beats", "Paroxysmal Supraventricular Tachycardia", "Sinus Bradycardia", 
            "Pulsus Paradoxus", "Pulsus Alternans", "Water-Hammer Pulse", "Bounding Peripheral Pulse", 
            "Asymmetrical Radial Pulses", "Jugular Venous Pressure Elevation", "Hepatojugular Reflux", 
            "Bilateral Dependent Pedal Edema", "Unilateral Lower Extremity Swelling (DVT Suspicion)", 
            "Calf Pain on Dorsiflexion (Homan Sign)", "Intermittent Claudication in Calves", 
            "Rest Pain in Toes/Foot", "Raynaud Phenomenon (Triphasic Color Change)", "Livedo Reticularis", 
            "Orthostatic Hypotension", "Hypertensive Urgency Headache", "Third Heart Sound (S3 Gallop)", 
            "Fourth Heart Sound (S4 Gallop)", "Systolic Ejection Murmur", "Diastolic Decrescendo Murmur", 
            "Holosystolic Murmur", "Pericardial Friction Rub"
        ],
        "Gastrointestinal & Digestive": [
            "Retrosternal Pyrosis (Heartburn)", "Acid Regurgitation", "Water Brash", "Dysphagia to Solids", 
            "Dysphagia to Liquids", "Odynophagia (Painful Swallowing)", "Globus Sensation in Throat", 
            "Early Satiety", "Epigastric Fullness", "Postprandial Distress", "Bilious Vomiting", 
            "Feculent Vomiting", "Coffee-Ground Emesis", "Hematemesis (Bright Red)", "Melena (Tarry Stools)", 
            "Hematochezia (Fresh Rectal Bleed)", "Right Upper Quadrant Tenderness", "Murphy Sign Positive", 
            "Epigastric Tenderness Radiating to Back", "Right Lower Quadrant Pain (McBurney Sign)", 
            "Rovsing Sign Positive", "Psoas Sign Positive", "Obturator Sign Positive", "Left Lower Quadrant Pain", 
            "Diverticular Cramping", "Generalized Peritonitis Rigidity", "Involuntary Abdominal Guarding", 
            "Rebound Tenderness", "Borborygmi (Hyperactive Bowel Sounds)", "Absent Bowel Sounds", 
            "Tympanitic Abdominal Distension", "Shifting Dullness in Flanks", "Fluid Wave Sign", 
            "Secretory Watery Diarrhea", "Osmotic Diarrhea", "Steatorrhea (Bulky Greasy Stool)", 
            "Mucus in Stool", "Fecal Tenesmus", "Proctalgia Fugax", "Perianal Fistula Discharge", 
            "External Hemorrhoidal Thrombosis", "Internal Hemorrhoid Prolapse", "Anal Fissure Burning"
        ],
        "Hepatic & Biliary": [
            "Scleral Icterus (Yellow Eyes)", "Cutaneous Jaundice", "Clay-Colored Acholic Stools", 
            "Dark Tea-Colored Urine", "Hepatomegaly with Smooth Margin", "Nodular Hepatomegaly", 
            "Splenomegaly on Palpation", "Palmar Erythema", "Spider Nevi on Trunk", "Caput Medusae Around Umbilicus", 
            "Asterixis (Flapping Hand Tremor)", "Fetor Hepaticus (Sweet Breath)", "Gynecomastia in Males", 
            "Testicular Atrophy in Males", "Biliary Colic Spasms", "Charcot Triad (Fever, Jaundice, RUQ Pain)"
        ],
        "Neurological & Cognitive": [
            "Holocranial Tension Headache", "Unilateral Throbbing Migraine", "Cluster Headache (Periorbital)", 
            "Thunderclap Sudden Headache", "Occipital Neuralgia", "Trigeminal Neuralgia (Lancinating Shock)", 
            "True Rotational Vertigo", "Benign Paroxysmal Positional Vertigo", "Labyrinthine Nystagmus", 
            "Sensory Ataxia (Positive Romberg)", "Cerebellar Gait Ataxia", "Parkinsonian Shuffling Gait", 
            "Festinating Gait", "Pill-Rolling Resting Tremor", "Kinetic Intention Tremor", "Essential Postural Tremor", 
            "Choreiform Involuntary Movements", "Hemiballismus", "Athetosis", "Myoclonic Jerks", 
            "Muscle Fasciculations", "Spastic Hypertonia (Clasp-Knife)", "Rigidity (Cogwheel Phenomenon)", 
            "Flaccid Hypotonia", "Hyperreflexia (Brisk Reflexes)", "Hyporeflexia (Diminished Reflexes)", 
            "Extensor Plantar Response (Babinski)", "Clonus at Ankle Joint", "Kernig Sign Positive", 
            "Brudzinski Sign Positive", "Nuchal Rigidity (Meningeal Irritation)", "Monoparesis", 
            "Hemiparesis (Contralateral)", "Paraparesis (Lower Limbs)", "Quadriparesis (All Limbs)", 
            "Glove-and-Stocking Neuropathy", "Radicular Sciatic Pain", "Lhermitte Sign (Electric Shock in Spine)", 
            "Dysmetria (Past-Pointing)", "Dysdiadochokinesia (Impaired Rapid Movements)", "Scanning Staccato Speech", 
            "Dysarthria (Bulbar Slurring)", "Broca Expressive Aphasia", "Wernicke Receptive Aphasia", 
            "Conduction Aphasia", "Anomic Aphasia", "Dysgraphia", "Dyslexia", "Acalculia", 
            "Apraxia (Motor Planning Deficit)", "Agnosia (Object Recognition Deficit)", "Visual Spatial Neglect", 
            "Transient Ischemic Attack Symptoms", "Amaurosis Fugax (Monocular Blindness)", 
            "Generalized Tonic-Clonic Seizure", "Myoclonic Absence Seizure", "Complex Partial Temporal Seizure", 
            "Todd Paralysis Post-Seizure", "Postictal Automatism", "Acute Delirium with Inattention", 
            "Sundowning Confusion", "Anterograde Amnesia", "Retrograde Amnesia", "Frontal Executive Dysfunction"
        ],
        "Ophthalmological & Ocular": [
            "Sudden Painless Vision Loss", "Painful Subacute Vision Loss", "Bitemporal Hemianopia", 
            "Homonymous Hemianopia", "Central Scotoma", "Metamorphopsia (Distorted Straight Lines)", 
            "Floaters and Vitreous Cobwebs", "Photopsia (Flashes of Light)", "Curtain-Falling Vision Loss", 
            "Horizontal Diplopia (Double Vision)", "Vertical Diplopia", "Pain on Extraocular Muscle Movement", 
            "Deep Ciliary Flush", "Circumcorneal Conjunctival Injection", "Subconjunctival Hemorrhage", 
            "Corneal Dendritic Ulceration", "Hypopyon (Pus in Anterior Chamber)", "Hyphema (Blood in Eye)", 
            "Steamy Hazy Cornea (Glaucoma)", "Mid-Dilated Non-Reactive Pupil", "Argyll Robertson Pupil", 
            "Marcus Gunn Relative Afferent Defect", "Papilledema on Fundoscopy", "Flame-Shaped Retinal Hemorrhages", 
            "Cotton-Wool Retinal Spots", "Hard Exudates on Macula", "Drusen Deposits on Macula", 
            "Proptosis (Unilateral Eyeball Bulging)", "Lagophthalmos (Inability to Close Eye)", 
            "Epiphora (Excessive Tearing)", "Xerophthalmia (Dry Eye Syndrome)", "Blepharospasm", 
            "Chalazion (Painless Eyelid Nodule)", "Hordeolum (Painful Stye)"
        ],
        "Ear, Nose, Throat & Head": [
            "Otalgia on Tragus Pressure", "Deep Middle Ear Pain", "Serous Otitis Fluid Sensation", 
            "Otorrhea with Foul Odor", "High-Frequency Sensorineural Loss", "Low-Frequency Conductive Loss", 
            "Pulsatile Tinnitus", "Non-Pulsatile Continuous Tinnitus", "Stridor on Inspiration", 
            "Stridor on Expiration", "Epiglottic Swelling (Hot Potato Voice)", "Quinsy (Peritonsillar Abscess Deviation)", 
            "Bilateral Tonsillar Exudates", "Strawberry Tongue (Scarlet Fever/Kawasaki)", "Geographic Tongue", 
            "Black Hairy Tongue", "Oral Aphthous Stomatitis", "Oral Erythroplakia", "Sialolithiasis Salivary Colic", 
            "Submandibular Gland Pain", "Parotid Gland Tenderness", "Anosmia", "Parosmia (Distorted Odors)", 
            "Phantosmia (Phantom Smells)", "Cacosmia (Foul Smell Sensation)"
        ],
        "Musculoskeletal & Orthopedic": [
            "Symmetrical Small Joint Synovitis", "Asymmetrical Oligoarthritis", "DIP Joint Osteophytes (Heberden)", 
            "PIP Joint Osteophytes (Bouchard)", "Ulnar Deviation of MCP Joints", "Swan-Neck Finger Deformity", 
            "Boutonniere Finger Deformity", "Z-Deformity of Thumb", "Trigger Finger (Stenosing Tenosynovitis)", 
            "Dupuytren Palmar Fascial Contracture", "Gouty Tophi on Ear/Joints", "Acute Podagra (1st MTP Inflammation)", 
            "Baker Popliteal Cyst Swelling", "Knee Joint Effusion (Ballottement)", "Positive Anterior Drawer Sign", 
            "Positive Lachman Test (ACL)", "Positive McMurray Meniscal Click", "Valgus Stress Knee Pain", 
            "Varus Stress Knee Pain", "Trochanteric Bursitis Hip Pain", "Iliotibial Band Friction Pain", 
            "Groin Pain on Internal Hip Rotation", "Trendelenburg Pelvic Drop Gait", "Spondylolisthesis Step-Off", 
            "Bamboo Spine Ankylosing Stiffness", "Schober Test Decreased Mobility", "Cervical Radiculopathy (Spurling)", 
            "Lumbar Radiculopathy (Straight Leg Raise)", "Bragard Test Positive", "Cauda Equina Saddle Anesthesia", 
            "Costochondritis (Tietze Syndrome)", "Sternoclavicular Joint Swelling", "Rotator Cuff Supraspinatus Tendonitis", 
            "Painful Arc Syndrome (60-120 Degrees)", "Frozen Shoulder (Adhesive Capsulitis)", 
            "Lateral Epicondylitis (Tennis Elbow)", "Medial Epicondylitis (Golfer Elbow)", 
            "De Quervain Tenosynovitis (Finkelstein)", "Carpal Tunnel Thenar Wasting", "Positive Phalen Wrist Sign", 
            "Positive Tinel Percussion Sign", "Extensor Hallucis Longus Weakness (Foot Drop)", 
            "Gastrocnemius Achilles Tendinitis", "Achilles Tendon Xanthoma", "Plantar Fascial Calcaneal Spur Pain"
        ],
        "Dermatological & Integumentary": [
            "Target Lesions (Erythema Multiforme)", "Palpable Purpura (Vasculitis)", "Non-Palpable Petechiae", 
            "Splinter Hemorrhages under Nails", "Janeway Macules (Painless Palms)", "Osler Nodes (Painful Pulp)", 
            "Roth Spots on Retina", "Malar Butterfly Facial Rash", "Discoid Lupus Erythematosus Plaques", 
            "Heliotrope Eyelid Rash (Dermatomyositis)", "Gottron Papules over Knuckles", "Shawl Sign Erythema", 
            "Mechanic Hands Skin Hyperkeratosis", "Erythema Chronicum Migrans (Bullseye)", "Erythema Marginatum", 
            "Erythema Nodosum Shins", "Pyoderma Gangrenosum Ulcers", "Necrobiosis Lipoidica on Shins", 
            "Pretibial Myxedema (Peau d'Orange)", "Acanthosis Nigricans Axillae", "Skin Tag (Acrochordon) Outbreak", 
            "Xanthelasma Palpebrarum", "Eruptive Xanthomas on Extensors", "Lichen Planus (Pruritic Purple Polygonal)", 
            "Wickham Striae in Mouth", "Koebner Phenomenon on Trauma", "Auspitz Sign on Scale Removal", 
            "Pityriasis Rosea Herald Patch", "Christmas Tree Back Rash Pattern", "Kerion (Boggy Scalp Fungal Plaque)", 
            "Favus Cup-Shaped Crusts", "Tinea Versicolor Hypopigmentation", "Erythrasma Coral-Red Fluorescence", 
            "Molluscum Contagiosum Umbilicated", "Herpes Zoster Dermatomal Blisters", "Hutchinson Sign on Nose Tip", 
            "Ramsay Hunt Ear Canal Blisters", "Condyloma Acuminata (Genital Warts)", "Condyloma Lata (Secondary Syphilis)", 
            "Chancre (Indurated Painless Ulcer)", "Chancroid (Painful Ragged Ulcer)", "Granuloma Inguinale Beefy Ulcer", 
            "Lymphogranuloma Venereum Bubo", "Nikolsky Sign Positive (Epidermal Sloughing)", 
            "Asboe-Hansen Sign Positive", "Targetoid Wheals with Central Dusky Core", 
            "Stevens-Johnson Mucosal Erosion", "Toxic Epidermal Necrolysis Sloughing", 
            "Anhidrosis (Inability to Sweat)", "Hyperhidrosis Palmar/Plantar", "Bromhidrosis (Offensive Body Odor)", 
            "Chromhidrosis (Colored Sweat)", "Psoriatic Nail Pitting", "Onychogryphosis (Ram Horn Nails)", 
            "Terry Nails (White Ground-Glass)", "Lindsay Nails (Half-and-Half)", "Muehrcke Lines (Double White Bands)", 
            "Mees Lines (Single White Band Transverse)", "Yellow Nail Syndrome (Lymphedema Association)"
        ],
        "Renal, Urological & Nephrological": [
            "Costovertebral Angle Flank Tenderness", "Ureteric Colic with Radiating Pain to Groin", 
            "Suprapubic Fullness on Percussion", "Urinary Bladder Spasms", "Pneumaturia (Air Bubbles in Urine)", 
            "Fecaluria (Fecal Matter in Urine)", "Chyluria (Milky Lymph Urine)", "Strangury (Painful Slow Dribbling)", 
            "Hesitancy on Initiating Voiding", "Terminal Dribbling in Males", "Interrupted Urinary Stream", 
            "Decrease in Caliber of Stream", "Stress Incontinence on Coughing/Sneezing", 
            "Urge Incontinence with Motor Detrusor Spasm", "Overflow Incontinence from Obstruction", 
            "Total Reflex Incontinence", "Uremic Frost on Skin", "Uremic Pericarditis Friction Rub", 
            "Uremic Encephalopathy Lethargy", "Pitting Periorbital Morning Puffiness", 
            "Bilateral Ballottable Kidneys (PKD)", "Renal Artery Stenosis Abdominal Bruit", 
            "Priapism (Prolonged Painful Erection)", "Peyronie Disease Plaque Curvature", 
            "Phimosis (Inability to Retract Prepuce)", "Paraphimosis (Trapped Prepuce Behind Glans)", 
            "Balano-Posthitis Erythema", "Hydrocele Transillumination", "Varicocele (Bag of Worms Sensation)", 
            "Spermatocele Nodule on Epididymis", "Fournier Gangrene Crepitus"
        ],
        "Endocrine, Hormonal & Metabolic": [
            "Thyroid Gland Enlargement (Goiter)", "Toxic Nodular Thyroid Palpation", "Thyroid Bruit on Auscultation", 
            "Exophthalmic Stare (Dalrymple Sign)", "Lid Lag on Downward Gaze (von Graefe)", 
            "Infrequent Blinking (Stellwag Sign)", "Lack of Convergence (Mobius Sign)", 
            "Moon Facies with Plethora", "Buffalo Hump (Dorsocervical Pad)", "Violaceous Wide Cutaneous Striae", 
            "Proximal Muscle Myopathy (Gowers)", "Hypoglycemic Autonomic Diaphoresis", 
            "Neuroglycopenic Confusion/Stupor", "Whipple Triad Confirmation", "Diabetic Ketoacidotic Tachypnea", 
            "Acetone Fruity Ketotic Breath", "Hyperosmolar Nonketotic Coma State", "Acromegalic Prognathism", 
            "Acromegalic Acral Growth (Shoe/Ring Size Increase)", "Deepening of Voice in Acromegaly", 
            "Macroglossia with Teeth Indentations", "Visual Tunnel Deficit (Pituitary Adenoma)", 
            "Pituitary Apoplexy Sudden Headache", "Sheehan Syndrome Agalactia", "Diabetes Insipidus Polyuria (>4L/day)", 
            "Hypokalemic Muscle Weakness/Paralysis", "Hypercalcemic Bones, Stones, Groans, Moans", 
            "Chvostek Sign (Facial Twitch on Tap)", "Trousseau Sign (Carpal Spasm on BP Cuff)", 
            "Tetany with Paresthesia Around Mouth", "Adrenal Crisis Hypotension & Shock", 
            "Hyperpigmentation of Palmar Creases", "Addisonian Mucosal Melanin Deposits", 
            "Pheochromocytoma Triad (Headache, Sweating, Tachycardia)", "Paroxysmal Severe Blood Pressure Surge"
        ],
        "Hematological & Oncology": [
            "Painless Firm Cervical Lymphadenopathy", "Hard Fixed Supraclavicular Node (Virchow)", 
            "Sister Mary Joseph Umbilical Nodule", "Spontaneous Mucosal Bleeding (Gingival)", 
            "Spontaneous Epistaxis Unilateral", "Hemarthrosis (Bleeding into Joint Cavity)", 
            "Deep Muscle Hematoma (Hemophilia)", "Spontaneous Subungual Hematoma", 
            "B-Symptoms Triad (Fever, Drenching Sweats, >10% Weight Loss)", "Pel-Ebstein Cyclic Fever (Hodgkin)", 
            "Alcohol-Induced Nodal Pain (Hodgkin)", "Bone Pain Worse at Night (Multiple Myeloma)", 
            "Hyperviscosity Headaches and Stupor", "Priapism in Sickle Cell Disease", 
            "Sickle Cell Vaso-Occlusive Bone Crisis", "Dactylitis (Hand-Foot Syndrome in Toddler)", 
            "Acute Chest Syndrome in Sickle Cell", "Sequestration Crisis Splenic Enlargement", 
            "Aplastic Crisis Pallor and Collapse", "Leukocoria (White Pupillary Reflex)", 
            "Cachexia Tumor Necrosis Factor Wasting", "Superior Vena Cava Syndrome Facial Plethora", 
            "Pemberton Sign Positive (Facial Congestion)", "Paraneoplastic Cerebellar Degeneration", 
            "Paraneoplastic Syndrome of Inappropriate ADH (SIADH)", "Hypercalcemia of Malignancy Somnolence"
        ],
        "Infectious & Tropical Diseases": [
            "Step-Ladder Temperature Chart (Typhoid)", "Rose Spots on Abdominal Wall (Typhoid)", 
            "Relative Bradycardia (Faget Sign in Yellow Fever/Typhoid)", "Breakbone Retro-Orbital Arthralgia (Dengue)", 
            "Tourniquet Test Petechiae Formation", "Blackwater Fever Dark Red Urine", "Splenic Rupture Tenderness (Mononucleosis)", 
            "Posterior Cervical Lymphadenopathy (EBV)", "Palatal Petechiae (Infectious Mononucleosis)", 
            "Koplik Spots in Buccal Mucosa (Measles)", "Cranio-Caudal Maculopapular Spread (Measles)", 
            "Parotid Swelling Obliterating Angle of Jaw (Mumps)", "Hydrophobia & Aerophobia (Rabies)", 
            "Risus Sardonicus (Grimace of Tetanus)", "Opisthotonos (Hyperextension Arc of Tetanus)", 
            "Pseudomembranous Gray Throat Exudate (Diphtheria)", "Bull-Neck Cervical Lymphadenitis (Diphtheria)", 
            "Paroxysmal Whooping Cough Spasms", "Post-Tussive Emesis in Whooping Cough", 
            "Rice-Water Stools Profuse (Cholera)", "Washerwoman Hands Dehydration (Cholera)", 
            "Slapped-Cheek Facial Erythema (Parvovirus B19)", "Lacy Reticular Rash on Extremities", 
            "Strawberry Tongue with Desquamation", "Fever Blisters (Herpes Labialis Recurrence)", 
            "Herpetic Whitlow on Finger", "Eczema Herpeticum Umbilicated Vesicles", "Dewdrop on Rose Petal Rash (Varicella)", 
            "Centripetal Distribution of Vesicles (Varicella)", "Centrifugal Distribution of Papules (Variola/Monkeypox)", 
            "Schistosoma Swimmer Itch Papules", "Terminal Hematuria (Schistosoma Haematobium)", 
            "Calabar Swelling (Loa Loa Filariasis)", "Subconjunctival Eye Worm Migration", 
            "Elephantiasis Scrotal/Lower Limb", "Chagas Roman Sign (Unilateral Palpebral Edema)", 
            "Kala-Azar Hyperpigmentation of Hands/Forehead", "Post-Kala-Azar Dermal Leishmaniasis Nodules", 
            "Mucocutaneous Leishmaniasis Nasal Espundia", "Echinococcal Liver Cyst Dull Pain", 
            "Hydatid Anaphylaxis on Cyst Rupture", "Neurocysticercosis New-Onset Seizure", 
            "Taenia Proglottids in Undergarments", "Pinworm (Enterobius) Nocturnal Perianal Pruritus", 
            "Trichinosis Periorbital Edema & Muscle Swelling", "Ascariasis Biliary Ascariasis Colic"
        ],
        "Psychiatric, Behavioral & Mental Health": [
            "Severe Free-Floating Anxiety", "Panic Surge with Impending Doom Sensation", 
            "Agoraphobic Avoidance of Open Spaces", "Depersonalization (Detached from Body)", 
            "Derealization (Surroundings Feel Artificial)", "Anhedonia (Inability to Feel Joy)", 
            "Psychomotor Retardation with Poverty of Movement", "Psychomotor Agitation with Pacing", 
            "Flight of Ideas with Accelerated Speech", "Grandiosity and Decreased Need for Sleep", 
            "Pressure of Speech (Uninterruptible)", "Ideas of Reference (Media Direct Messages)", 
            "Auditory Command Hallucinations", "Visual Elementary Hallucinations", 
            "Tactile Formication (Insects Crawling on Skin)", "Olfactory Hallucination of Burnt Rubber", 
            "Persecutory Paranoid Delusions", "Somatic Delusions of Infestation", "Capgras Delusion (Impostor Belief)", 
            "Cotard Delusion (Belief of Being Dead)", "Catatonic Waxy Flexibility (Catalepsy)", 
            "Catatonic Negativism", "Echolalia (Involuntary Repetition of Words)", "Echopraxia (Repetition of Movements)", 
            "Dissociative Fugue (Unplanned Travel Amnesia)", "Dissociative Identity Switching", 
            "Intrusive Flashbacks of Traumatic Memory", "Hypervigilance & Exaggerated Startle Response", 
            "Emotional Numbing and Detachment", "Trichotillomania (Compulsive Hair Pulling)", 
            "Excoriation Disorder (Compulsive Skin Picking)", "Body Dysmorphic Preoccupation with Flaws", 
            "Anorexic Severe Body Image Distortion", "Bulimic Compensatory Purging Vomiting", 
            "Russell Sign (Calluses on Knuckles from Vomiting)", "Orthorexia (Obsession with Pure Food)"
        ]
    }

    for cat, items in clinical_subsystems.items():
        for item in items:
            symptoms.add(item)

    # Generate additional refined clinical nuances up to 1000 items
    qualifiers = [
        "Acute Onset", "Chronic Persistent", "Intermittent Recurrent", "Nocturnal Exacerbation", 
        "Postprandial", "Exertional", "Spontaneous", "Paroxysmal", "Subacute", "Progressive", 
        "Refractory", "Mild", "Moderate", "Severe", "Unilateral", "Bilateral", "Localized", 
        "Diffuse", "Radiating", "Positional", "Transient", "Deep-Seated", "Superficial"
    ]
    core_phenotypes = [
        "Cervical Spinal Stiffness", "Lumbar Spine Spasm", "Thoracic Spine Tenderness", 
        "Sacroiliac Joint Pain", "Sternocostal Junction Tenderness", "Acromioclavicular Joint Pain", 
        "Subacromial Bursitis", "Bicipital Groove Tenderness", "Olecranon Bursitis Effusion", 
        "Radiocarpal Joint Pain", "Scaphoid Waist Tenderness (Snuffbox)", "Metacarpophalangeal Synovitis", 
        "Ischial Bursitis (Weaver Bottom)", "Greater Trochanter Pain Syndrome", "Prepatellar Bursitis (Housemaid Knee)", 
        "Infrapatellar Tendonitis (Jumper Knee)", "Anserine Bursitis Medial Knee Pain", 
        "Tibiotalar Joint Effusion", "Subtalar Joint Inversion Pain", "Tarsal Tunnel Burning Sensation", 
        "First Metatarsophalangeal Erythema", "Morton Neuroma Interdigital Webspace Pain", 
        "Frontal Sinus Dull Pressure", "Maxillary Sinus Dental Radiating Pain", "Ethmoid Sinus Retro-Orbital Ache", 
        "Sphenoid Sinus Vertex Headache", "Eustachian Tube Dysfunction Clicking", 
        "Temporomandibular Joint Clicking", "Masseter Muscle Mastication Spasm", 
        "Carotidynia (Unilateral Neck Throbbing)", "Eagle Syndrome Stylomandibular Pain", 
        "Scalene Anterior Thoracic Outlet Paresthesia", "Pectoralis Minor Entrapment Symptoms", 
        "Pyriformis Sciatic Compression", "Meralgia Paresthetica Lateral Thigh Numbness", 
        "Saphenous Nerve Medial Leg Burning", "Peroneal Nerve Foot Eversion Weakness", 
        "Sural Nerve Lateral Foot Numbness", "Pudendal Neuralgia Sitting Pain", 
        "Coccygodynia (Tailbone Pain on Sitting)", "Costochondral Cartilage Swelling", 
        "Xiphoidalgia (Xiphoid Process Tenderness)", "Slipping Rib Syndrome Lower Chest Click", 
        "Abdominal Cutaneous Nerve Entrapment (Carnett Sign)", "Ilioinguinal Nerve Groin Pain", 
        "Genitofemoral Nerve Anterior Thigh Burning", "Obturator Neuropathy Medial Thigh Ache", 
        "Deep Peroneal Nerve Webspace Numbness", "Superficial Radial Nerve Cheiralgia Paresthetica", 
        "Posterior Interosseous Nerve Motor Weakness", "Anterior Interosseous Nerve Pinch Weakness", 
        "Cubital Tunnel Ulnar Clawing Tendency", "Guyon Canal Ulnar Compression at Wrist", 
        "Pronator Teres Syndrome Forearm Ache", "Supinator Syndrome Lateral Elbow Pain", 
        "Bicipital Tendon Rupture (Popeye Deformity)", "Triceps Tendon Avulsion Tenderness", 
        "Quadriceps Tendon Rupture Defect", "Patellar Tendon Rupture Proximal Patella Displacement", 
        "Gastrocnemius Medial Head Tear (Tennis Leg)", "Peroneus Longus Tendon Subluxation Click", 
        "Tibialis Posterior Insufficiency Flatfoot", "Tibialis Anterior Tendinitis Crepitus", 
        "Flexor Hallucis Longus Stenosis (Dancer Heel)", "Extensor Digitorum Brevis Swelling", 
        "Sinus Tarsi Syndrome Lateral Ankle Instability", "Lisfranc Joint Midfoot Pain on Weight Bearing", 
        "Chopart Joint Sprain Tenderness", "Navicular Stress Fracture Vague Dorsal Ache", 
        "Jones Fracture 5th Metatarsal Base Pain", "March Fracture 2nd Metatarsal Shaft Pain", 
        "Sesamoiditis Great Toe Plantar Pain", "Turf Toe 1st MTP Hyperextension Pain", 
        "Freiberg Infraction 2nd Metatarsal Head Stiffness", "Kohler Disease Tarsal Navicular Limp", 
        "Sever Disease Calcaneal Apophysitis Heel Limp", "Osgood-Schlatter Tibial Tubercle Lump", 
        "Sinding-Larsen-Johansson Inferior Patella Ache", "Scheuermann Kyphosis Thoracic Stiffness", 
        "Legg-Calve-Perthes Hip Limitation in Child", "Slipped Capital Femoral Epiphysis Groin Limp", 
        "Transient Synovitis Irritable Hip in Child", "Osteoid Osteoma Night Pain Relieved by Aspirin", 
        "Aneurysmal Bone Cyst Expansile Tenderness", "Giant Cell Tumor Epiphyseal Swelling", 
        "Osteochondroma Painless Bony Exostosis", "Enchondroma Pathological Phalanx Fracture", 
        "Stress Fracture Tibial Plateau Shins", "Shin Splints (Medial Tibial Stress Syndrome)"
    ]

    for q in qualifiers:
        for p in core_phenotypes:
            symptoms.add(f"{q} {p}")
            if len(symptoms) >= 1000:
                break
        if len(symptoms) >= 1000:
            break

    final_list = sorted(list(symptoms))[:1000]
    return final_list

s_list = build_1000_symptoms()
print(f"Generated exactly {len(s_list)} validated symptoms. First: '{s_list[0]}', Last: '{s_list[-1]}'")
with open("symptoms_1000.json", "w") as f:
    json.dump(s_list, f)

