import React, { useState, useEffect, useRef } from 'react';
import './App.scss';
import localSymptoms1000 from './symptoms_1000.json';
import { 
  Sun, Moon, Volume2, VolumeX, X, ChevronDown, CheckCircle2, 
  Stethoscope, Globe, Loader2, Pill, Home, Apple, 
  MessageSquare, ListFilter, AlertCircle, Printer, Download,
  Play, Pause, Copy, Check
} from 'lucide-react';

export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('medisense_theme') || 'dark');
  const [inputMode, setInputMode] = useState('text'); // 'text' (Type Problem) | 'symptoms' (Pick Symptoms)
  const [freeText, setFreeText] = useState('I have fever from last 5 days with headache and chills');
  const [symptoms, setSymptoms] = useState(['High Fever', 'Headache', 'Chills']);
  const [inputValue, setInputValue] = useState('');
  const [allSymptomsList, setAllSymptomsList] = useState(localSymptoms1000 || []);
  const [showDropdown, setShowDropdown] = useState(false);
  const [visibleCount, setVisibleCount] = useState(20);
  const SCROLL_BATCH = 20;

  const [language, setLanguage] = useState('en');
  const [showLangDropdown, setShowLangDropdown] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [copied, setCopied] = useState(false);

  const dropdownRef = useRef(null);
  const langDropdownRef = useRef(null);
  const audioRef = useRef(null);

  // Apply theme
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('medisense_theme', theme);
  }, [theme]);

  // Load dynamically from Python backend if updated
  useEffect(() => {
    fetch('/api/symptoms')
      .then(res => res.json())
      .then(data => {
        if (data.symptoms && Array.isArray(data.symptoms) && data.symptoms.length > 0) {
          setAllSymptomsList(data.symptoms);
        }
      })
      .catch(err => {
        console.warn('Using local 1,000 symptom database fallback:', err);
      });
  }, []);

  // Filter symptoms based on search input and exclude already selected
  const filteredSymptoms = allSymptomsList.filter(s => {
    const notPicked = !symptoms.some(x => x.toLowerCase() === s.toLowerCase());
    if (!notPicked) return false;
    if (!inputValue.trim()) return true;
    return s.toLowerCase().includes(inputValue.toLowerCase());
  });

  const visibleSymptoms = filteredSymptoms.slice(0, visibleCount);

  // Reset visible count whenever search query changes or dropdown toggles
  useEffect(() => {
    setVisibleCount(20);
  }, [inputValue, showDropdown]);

  // Infinite scroll pagination handler: dynamically load 20 more symptoms as user scrolls
  const handleListScroll = (e) => {
    const { scrollTop, scrollHeight, clientHeight } = e.target;
    if (scrollTop + clientHeight >= scrollHeight - 35) {
      if (visibleCount < filteredSymptoms.length) {
        setVisibleCount(prev => Math.min(filteredSymptoms.length, prev + SCROLL_BATCH));
      }
    }
  };

  // Close dropdowns on outside click
  useEffect(() => {
    function handleClickOutside(e) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setShowDropdown(false);
      }
      if (langDropdownRef.current && !langDropdownRef.current.contains(e.target)) {
        setShowLangDropdown(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const addSymptom = (s) => {
    const exists = symptoms.some(item => item.toLowerCase() === s.toLowerCase());
    if (!exists) {
      setSymptoms([...symptoms, s]);
    }
    setInputValue('');
    setShowDropdown(false);
  };

  const removeSymptom = (s) => {
    setSymptoms(symptoms.filter(item => item.toLowerCase() !== s.toLowerCase()));
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      if (!inputValue.trim()) return;

      if (visibleSymptoms.length > 0) {
        addSymptom(visibleSymptoms[0]);
      } else {
        const exact = allSymptomsList.find(s => s.toLowerCase() === inputValue.toLowerCase());
        if (exact) addSymptom(exact);
      }
    } else if (e.key === 'Backspace' && !inputValue && symptoms.length > 0) {
      removeSymptom(symptoms[symptoms.length - 1]);
    }
  };

  // Predict handler calling Python ML + Gemini 3.5 Flash-lite
  const handlePredict = async () => {
    if (inputMode === 'text' && !freeText.trim()) {
      alert('Please describe your symptoms or health problem.');
      return;
    }
    if (inputMode === 'symptoms' && symptoms.length === 0) {
      alert('Please select at least one symptom.');
      return;
    }

    setLoading(true);
    setResult(null);
    setAudioUrl(null);
    setIsPlayingAudio(false);

    try {
      const payload = {
        user_text: inputMode === 'text' ? freeText : '',
        symptoms: inputMode === 'symptoms' ? symptoms : [],
        language: language,
        mode: inputMode
      };

      const res = await fetch('/api/diagnose', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      setResult(data);

      if (data.audio_base64) {
        setAudioUrl(`data:audio/mp3;base64,${data.audio_base64}`);
      }
    } catch (err) {
      console.error('Diagnosis request error:', err);
      alert('Could not connect to Python clinical engine. Please ensure server is running.');
    } finally {
      setLoading(false);
    }
  };

  // Play audio consultation
  const handlePlayAudio = () => {
    if (audioRef.current) {
      if (isPlayingAudio) {
        audioRef.current.pause();
        setIsPlayingAudio(false);
      } else {
        audioRef.current.play()
          .then(() => setIsPlayingAudio(true))
          .catch(e => console.error("Audio playback error:", e));
      }
    } else if (audioUrl) {
      const audio = new Audio(audioUrl);
      audio.onplay = () => setIsPlayingAudio(true);
      audio.onended = () => setIsPlayingAudio(false);
      audio.onpause = () => setIsPlayingAudio(false);
      audio.play().catch(e => console.error("Audio playback error:", e));
    } else if (result && 'speechSynthesis' in window) {
      if (window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
        setIsPlayingAudio(false);
        return;
      }
      const textToSpeak = `${result.displayName}. ${result.overview}`;
      const utterance = new SpeechSynthesisUtterance(textToSpeak);
      const langSpeechMap = {
        en: 'en-US', hi: 'hi-IN', te: 'te-IN', es: 'es-ES',
        fr: 'fr-FR', ar: 'ar-SA', ta: 'ta-IN', bn: 'bn-IN', de: 'de-DE'
      };
      utterance.lang = langSpeechMap[language] || 'en-US';
      utterance.onstart = () => setIsPlayingAudio(true);
      utterance.onend = () => setIsPlayingAudio(false);
      utterance.onerror = () => setIsPlayingAudio(false);
      window.speechSynthesis.speak(utterance);
    }
  };

  // Download / Print as PDF
  const handlePrintPdf = () => {
    window.print();
  };

  // Copy prescription text to clipboard
  const handleCopyPrescription = () => {
    if (!result) return;
    const rxLines = (result.prescription || [])
      .map(r => `• ${r.name} (${r.duration || ''}): ${r.instruction}`)
      .join('\n');
    const careLines = (result.care || []).map(c => `- ${c}`).join('\n');
    const dietLines = (result.diet || []).map(d => `- ${d}`).join('\n');
    const text = `MediSense AI Clinical Consultation & Prescription
Condition: ${result.displayName}
Confidence: ${result.confidence}%

Clinical Overview:
${result.overview}

Recommended Prescription:
${rxLines}

Home Care & Recovery:
${careLines}

Diet & Hydration:
${dietLines}

Disclaimer: AI-generated recommendation. Not a substitute for professional medical advice.`;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  const primaryLanguages = [
    { id: 'en', label: 'English', flag: '🇬🇧' },
    { id: 'hi', label: 'Hindi (हिंदी)', flag: '🇮🇳' },
    { id: 'te', label: 'Telugu (తెలుగు)', flag: '🇮🇳' }
  ];

  const additionalLanguages = [
    { id: 'es', label: 'Spanish (Español)', flag: '🇪🇸' },
    { id: 'fr', label: 'French (Français)', flag: '🇫🇷' },
    { id: 'ar', label: 'Arabic (العربية)', flag: '🇸🇦' },
    { id: 'ta', label: 'Tamil (தமிழ்)', flag: '🇮🇳' },
    { id: 'bn', label: 'Bengali (বাংলা)', flag: '🇮🇳' },
    { id: 'de', label: 'German (Deutsch)', flag: '🇩🇪' }
  ];

  const allAvailableLanguages = [...primaryLanguages, ...additionalLanguages];

  return (
    <div className="app-wrapper">
      {/* Top Bar Controls: Input Mode Toggle & Dark Mode */}
      <div className="top-bar-controls">
        <div className="mode-toggle-group">
          <button 
            className={`mode-btn ${inputMode === 'text' ? 'active' : ''}`}
            onClick={() => setInputMode('text')}
            title="Type problem description freely"
          >
            <MessageSquare size={14} />
            <span>Type Problem</span>
          </button>
          <button 
            className={`mode-btn ${inputMode === 'symptoms' ? 'active' : ''}`}
            onClick={() => setInputMode('symptoms')}
            title="Select symptoms from catalog"
          >
            <ListFilter size={14} />
            <span>Select Symptoms</span>
          </button>
        </div>

        <button 
          className="theme-toggle" 
          onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
          title="Toggle theme"
        >
          {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
          <span>{theme === 'dark' ? 'Light' : 'Dark'}</span>
        </button>
      </div>

      <div className="main-card">
        {/* Header */}
        <div className="header-section">
          <h1 className="app-title">
            <Stethoscope className="title-icon" size={30} />
            <span>Disease Predictor and Prescription Generator</span>
          </h1>
          <p className="app-subtitle">
            {inputMode === 'text' ? "Describe your symptoms in plain words to get an instant, AI-powered diagnosis!" : "Select symptoms below to get an instant, AI-powered diagnosis!"}
          </p>
        </div>

        {/* MODE A: Free Text Input */}
        {inputMode === 'text' && (
          <div className="form-group">
            <div className="label-row">
              <label>
                <MessageSquare size={15} style={{ color: 'var(--text-blue)' }} />
                <span>Describe your problem in plain words:</span>
              </label>
              <span className="helper-badge">AI extracts symptoms &amp; clinical causes</span>
            </div>

            <textarea
              className="problem-textarea"
              rows={3}
              placeholder="e.g. I have fever from last 5 days with headache, shivering chills and body ache..."
              value={freeText}
              onChange={(e) => setFreeText(e.target.value)}
            />
          </div>
        )}

        {/* MODE B: Symptoms Dropdown & Tag Input */}
        {inputMode === 'symptoms' && (
          <div className="form-group" ref={dropdownRef}>
            <div className="label-row">
              <label>
                <ListFilter size={15} style={{ color: 'var(--text-blue)' }} />
                <span>Select your symptoms ({allSymptomsList.length} in database):</span>
              </label>
              <span className="helper-badge">Type to search or browse catalog</span>
            </div>

            <div className="symptom-input-container">
              {symptoms.map(s => (
                <span key={s} className="tag-chip">
                  <span>{s}</span>
                  <button 
                    className="tag-remove" 
                    onClick={() => removeSymptom(s)}
                    title={`Remove ${s}`}
                  >
                    <X size={13} />
                  </button>
                </span>
              ))}

              <input
                type="text"
                className="symptom-text-input"
                placeholder={symptoms.length === 0 ? "Search symptoms (e.g. fever, headache, chills)..." : "Add more..."}
                value={inputValue}
                onChange={(e) => {
                  setInputValue(e.target.value);
                  setShowDropdown(true);
                }}
                onFocus={() => setShowDropdown(true)}
                onKeyDown={handleKeyDown}
              />

              {symptoms.length > 0 && (
                <button 
                  className="btn-icon-action" 
                  onClick={() => setSymptoms([])}
                  title="Clear all symptoms"
                >
                  <X size={16} />
                </button>
              )}

              <button 
                className="btn-icon-action"
                onClick={() => setShowDropdown(prev => !prev)}
                title="Browse all symptoms"
              >
                <ChevronDown size={16} />
              </button>

              {/* Suggestions & Catalog Dropdown with Scrolling Pagination */}
              {showDropdown && (
                <div className="suggestions-dropdown">
                  <div className="suggestions-list" onScroll={handleListScroll}>
                    {visibleSymptoms.length > 0 ? (
                      visibleSymptoms.map(s => (
                        <div 
                          key={s} 
                          className="suggestion-item"
                          onClick={() => addSymptom(s)}
                        >
                          <span>{s}</span>
                        </div>
                      ))
                    ) : (
                      <div className="dropdown-empty">
                        No matching symptoms found in database.
                      </div>
                    )}

                    {visibleCount < filteredSymptoms.length && (
                      <div className="dropdown-scroll-loader">
                        <span>Scroll down for more symptoms ({visibleCount} of {filteredSymptoms.length} loaded)...</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Language Selection: Modern Segmented UX */}
        <div className="form-group">
          <div className="label-row">
            <label>
              <Globe size={15} style={{ color: 'var(--text-blue)' }} />
              <span>Select output language:</span>
            </label>
          </div>
          <div className="language-segmented-bar">
            {primaryLanguages.map(lang => (
              <button
                key={lang.id}
                type="button"
                className={`lang-pill-btn ${language === lang.id ? 'active' : ''}`}
                onClick={() => {
                  setLanguage(lang.id);
                  setShowLangDropdown(false);
                }}
              >
                <span className="lang-flag">{lang.flag}</span>
                <span>{lang.label}</span>
              </button>
            ))}

            <div className="custom-lang-picker" ref={langDropdownRef}>
              <button
                type="button"
                className={`lang-pill-btn lang-dropdown-btn ${additionalLanguages.some(l => l.id === language) ? 'active' : ''}`}
                onClick={() => setShowLangDropdown(prev => !prev)}
                title="Select from additional clinical languages"
              >
                <span className="lang-flag">
                  {additionalLanguages.find(l => l.id === language)?.flag || '🌐'}
                </span>
                <span>
                  {additionalLanguages.find(l => l.id === language)?.label || 'More Languages'}
                </span>
                <ChevronDown size={14} className={`dropdown-arrow ${showLangDropdown ? 'rotated' : ''}`} />
              </button>

              {showLangDropdown && (
                <div className="custom-lang-menu">
                  <div className="custom-lang-menu-title">Additional Languages</div>
                  {additionalLanguages.map(lang => (
                    <div
                      key={lang.id}
                      className={`custom-lang-menu-item ${language === lang.id ? 'active' : ''}`}
                      onClick={() => {
                        setLanguage(lang.id);
                        setShowLangDropdown(false);
                      }}
                    >
                      <span className="item-flag">{lang.flag}</span>
                      <span className="item-name">{lang.label}</span>
                      {language === lang.id && <CheckCircle2 size={14} className="item-check" />}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Predict Button */}
        <button 
          className="btn-predict"
          disabled={loading || (inputMode === 'text' ? !freeText.trim() : symptoms.length === 0)}
          onClick={handlePredict}
        >
          {loading ? (
            <>
              <Loader2 className="btn-spinner" size={18} />
              <span>Analyzing with Python ML &amp; Gemini AI...</span>
            </>
          ) : (
            <span>Predict Disease and Generate Prescription</span>
          )}
        </button>

        {/* SKELETON LOADING STATE */}
        {loading && (
          <div className="skeleton-section">
            <div className="skeleton-notice">
              <span className="pulsing-dot" />
              <span>Machine Learning &amp; Gemini AI generating diagnosis in {allAvailableLanguages.find(l => l.id === language)?.label || 'Selected Language'}...</span>
            </div>
            
            <div className="skeleton-block sk-title" />
            <div className="skeleton-block sk-badge" />

            <div className="skeleton-block sk-para" />
            <div className="skeleton-block sk-para" />
            <div className="skeleton-block sk-para short" />

            <div className="skeleton-block sk-card" />
            <div className="skeleton-block sk-card" />

            <div className="sk-grid">
              <div className="skeleton-block sk-grid-box" />
              <div className="skeleton-block sk-grid-box" />
            </div>
          </div>
        )}

        {/* RESULTS SECTION (Generated dynamically in selected language) */}
        {!loading && result && (
          <div className="results-section">
            {/* PRINT-ONLY OFFICIAL MEDICAL HEADER */}
            <div className="print-only-header">
              <div className="print-clinic-title">MediSense AI — Clinical Diagnostic &amp; Prescription Report</div>
              <div className="print-clinic-subtitle">AI-Driven Disease Predictor &amp; Clinical Pharmacology Advisory Protocol</div>
              <div className="print-meta-row">
                <span>Date: {new Date().toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</span>
                <span>Language: {allAvailableLanguages.find(l => l.id === language)?.label || 'English'}</span>
                <span>Consultation ID: #MED-{Math.floor(100000 + (result.confidence * 1234) % 900000)}</span>
              </div>
            </div>

            <div className="result-badge-row">
              <div className="predicted-label">
                Predicted Disease: 
                <span className="disease-name">
                  {result.icon} {result.displayName}
                </span>
              </div>
            </div>

            <div className="accuracy-row">
              <CheckCircle2 size={16} />
              <span>Best Algorithm: Random Forest / Ensemble &nbsp;|&nbsp; Accuracy: {result.confidence}%</span>
            </div>

            {result.detected_symptoms && result.detected_symptoms.length > 0 && (
              <div style={{ marginBottom: '16px', fontSize: '0.86rem', color: 'var(--text-muted)' }}>
                Identified Symptoms: <strong>{result.detected_symptoms.join(', ')}</strong>
              </div>
            )}

            {/* ACTION TOOLBAR: Listen Audio, Print/Download PDF, Copy Text */}
            <div className="advice-title-row">
              <h2 className="advice-title">Medical Prescription and Advice:</h2>
              <div className="result-action-buttons">
                <button 
                  className={`btn-action ${isPlayingAudio ? 'active' : ''}`} 
                  onClick={handlePlayAudio} 
                  title="Listen in selected language"
                >
                  {isPlayingAudio ? <VolumeX size={15} /> : <Volume2 size={15} />}
                  <span>{isPlayingAudio ? 'Pause Audio' : 'Listen Audio'}</span>
                </button>
                <button 
                  className="btn-action" 
                  onClick={handlePrintPdf} 
                  title="Download or print prescription report as PDF"
                >
                  <Printer size={15} />
                  <span>Download / Print PDF</span>
                </button>
                <button 
                  className="btn-action" 
                  onClick={handleCopyPrescription} 
                  title="Copy prescription text to clipboard"
                >
                  {copied ? <Check size={15} /> : <Copy size={15} />}
                  <span>{copied ? 'Copied!' : 'Copy Text'}</span>
                </button>
              </div>
            </div>

            {/* AUDIO READER CONSOLE (Streamlit-style Audio Consultation Reader) */}
            <div className="audio-reader-console">
              <div className="audio-reader-info">
                <div className="audio-live-pill">
                  <Volume2 size={14} className={isPlayingAudio ? 'pulse-audio' : ''} />
                  <span>Audio Consultation ({allAvailableLanguages.find(l => l.id === language)?.label})</span>
                </div>
                {audioUrl && (
                  <a 
                    href={audioUrl} 
                    download={`MediSense-Prescription-${result.disease || 'Advisory'}.mp3`}
                    className="btn-download-mp3"
                    title="Download audio recording as MP3"
                  >
                    <Download size={13} />
                    <span>Download MP3</span>
                  </a>
                )}
              </div>

              {audioUrl ? (
                <audio 
                  ref={audioRef}
                  controls 
                  src={audioUrl} 
                  className="native-styled-audio"
                  onPlay={() => setIsPlayingAudio(true)}
                  onPause={() => setIsPlayingAudio(false)}
                  onEnded={() => setIsPlayingAudio(false)}
                />
              ) : (
                <div className="audio-fallback-row">
                  <button className="btn-play-voice" onClick={handlePlayAudio}>
                    {isPlayingAudio ? <Pause size={14} /> : <Play size={14} />}
                    <span>{isPlayingAudio ? 'Pause Narration' : 'Play Voice Narration'}</span>
                  </button>
                  <span className="audio-fallback-hint">Narrates condition overview and prescription instructions</span>
                </div>
              )}
            </div>

            <p className="advice-overview-text">
              {result.overview}
            </p>

            {/* Treatment Regimen - Clean Normal Cards */}
            {result.prescription && result.prescription.length > 0 && (
              <div className="rx-box-clean">
                <div className="rx-box-title">
                  <Pill size={16} />
                  <span>Recommended Treatment Regimen</span>
                </div>
                <div className="rx-list-clean">
                  {result.prescription.map((rx, idx) => (
                    <div key={idx} className="rx-item-clean">
                      <div className="rx-header-row">
                        <span className="rx-bullet">•</span>
                        <span className="rx-name">{rx.name}</span>
                        {rx.duration && (
                          <span className="rx-duration-inline">— {rx.duration}</span>
                        )}
                      </div>
                      <p className="rx-desc">{rx.instruction}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Care & Diet */}
            <div className="guidance-grid">
              <div className="guidance-col">
                <div className="col-heading">
                  <Home size={16} style={{ color: 'var(--text-blue)' }} />
                  <span>Home Care &amp; Recovery</span>
                </div>
                <ul>
                  {result.care && result.care.map((c, idx) => (
                    <li key={idx}>{c}</li>
                  ))}
                </ul>
              </div>

              <div className="guidance-col">
                <div className="col-heading">
                  <Apple size={16} style={{ color: 'var(--text-blue)' }} />
                  <span>Diet &amp; Hydration</span>
                </div>
                <ul>
                  {result.diet && result.diet.map((d, idx) => (
                    <li key={idx}>{d}</li>
                  ))}
                </ul>
              </div>
            </div>

            <p className="disclaimer-note">
              * <strong>Important Medical Disclaimer:</strong> This is an AI-generated clinical recommendation and not a substitute for professional medical diagnosis. Please consult a qualified doctor if symptoms persist or worsen.
            </p>

            {/* PRINT-ONLY SIGNATURE FOOTER */}
            <div className="print-only-footer">
              <div>
                <div><strong>Verified by MediSense AI Clinical Protocol</strong></div>
                <div>Document Generated on: {new Date().toLocaleDateString()}</div>
              </div>
              <div className="signature-box">
                <span className="sig-line" />
                <div>Physician / Practitioner Signature</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
