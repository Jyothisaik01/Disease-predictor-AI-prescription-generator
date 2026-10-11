import React, { useState, useMemo } from 'react';
import { 
  Sparkles, Search, Trash2, ArrowRight, AlertTriangle, 
  Check, X, FileText, LayoutGrid, HeartPulse, Stethoscope
} from 'lucide-react';

export default function SymptomIntake({ 
  symptomsList, 
  setSymptomsList, 
  catalog, 
  onRunDiagnosis, 
  onOpenModal 
}) {
  const [activeTab, setActiveTab] = useState('text'); // 'text' | 'catalog'
  const [userText, setUserText] = useState('');
  const [isExtracting, setIsExtracting] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [extractMessage, setExtractMessage] = useState(null);

  // Filtered categories based on search
  const filteredCategories = useMemo(() => {
    if (!catalog?.categories) return {};
    if (!searchQuery.trim()) return catalog.categories;

    const q = searchQuery.toLowerCase();
    const result = {};

    Object.entries(catalog.categories).forEach(([category, items]) => {
      const matched = items.filter(s => s.toLowerCase().includes(q));
      if (matched.length > 0) {
        result[category] = matched;
      }
    });

    return result;
  }, [catalog, searchQuery]);

  // Add symptom helper
  const addSymptom = (s) => {
    if (!symptomsList.includes(s)) {
      setSymptomsList([...symptomsList, s]);
    }
  };

  // Remove symptom helper
  const removeSymptom = (s) => {
    setSymptomsList(symptomsList.filter(item => item !== s));
  };

  // Preset click handler
  const handlePresetClick = (preset) => {
    const updated = [...symptomsList];
    preset.symptoms.forEach(s => {
      if (!updated.includes(s)) updated.push(s);
    });
    setSymptomsList(updated);
  };

  // Natural language extraction via FastAPI
  const handleExtractNLP = async () => {
    if (!userText.trim()) return;
    setIsExtracting(true);
    setExtractMessage(null);

    try {
      const res = await fetch('http://localhost:8000/api/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: userText })
      });
      const data = await res.json();

      if (data.extracted && data.extracted.length > 0) {
        const newlyAdded = [];
        const current = [...symptomsList];
        data.extracted.forEach(s => {
          if (!current.includes(s)) {
            current.push(s);
            newlyAdded.push(s);
          }
        });
        setSymptomsList(current);
        setExtractMessage({
          type: 'success',
          text: `Extracted ${data.extracted.length} clinical biomarker${data.extracted.length > 1 ? 's' : ''}: ${data.extracted.join(', ')}`
        });
      } else {
        setExtractMessage({
          type: 'info',
          text: 'No clinical matches found in your description. Try selecting from the catalog.'
        });
      }
    } catch (err) {
      console.error('Extraction error:', err);
      setExtractMessage({
        type: 'error',
        text: 'Error connecting to clinical extractor. Check backend server.'
      });
    } finally {
      setIsExtracting(false);
    }
  };

  return (
    <div className="intake-grid">
      {/* Left Column: Intake Workspace */}
      <div className="card intake-workspace">
        <div className="card-header">
          <div>
            <div className="card-title-group">
              <Stethoscope className="card-icon" size={24} />
              <h2 className="card-title">Patient Symptom Intake</h2>
            </div>
            <p className="card-subtitle">
              Describe your symptoms in natural language or select verified indicators from the catalog.
            </p>
          </div>
        </div>

        {/* Mode Switcher */}
        <div className="intake-mode-switcher">
          <button 
            className={`mode-tab-btn ${activeTab === 'text' ? 'active' : ''}`}
            onClick={() => setActiveTab('text')}
          >
            <FileText size={16} />
            <span>Natural Language Description</span>
          </button>
          <button 
            className={`mode-tab-btn ${activeTab === 'catalog' ? 'active' : ''}`}
            onClick={() => setActiveTab('catalog')}
          >
            <LayoutGrid size={16} />
            <span>Clinical Biomarkers Catalog</span>
          </button>
        </div>

        {/* Tab 1: Natural Language Intake */}
        {activeTab === 'text' && (
          <div className="tab-pane-nl">
            {/* Quick Presets */}
            {catalog?.presets && (
              <div className="preset-section">
                <div className="preset-label">
                  <Sparkles size={13} />
                  <span>Common Clinical Presentations</span>
                </div>
                <div className="preset-chips-wrap">
                  {catalog.presets.map((preset, idx) => (
                    <button 
                      key={idx}
                      className="preset-chip"
                      onClick={() => handlePresetClick(preset)}
                    >
                      <span>+ {preset.label}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Text Area */}
            <div className="nlp-textarea-box">
              <textarea
                value={userText}
                onChange={(e) => setUserText(e.target.value)}
                placeholder="Describe your health condition in your own words (e.g. I have a high fever with chills, throat irritation, dry cough and feeling exhausted for 2 days)..."
              />
              <div className="nlp-submit-row">
                <button 
                  className="btn-extract"
                  onClick={handleExtractNLP}
                  disabled={!userText.trim() || isExtracting}
                >
                  <Sparkles size={16} />
                  <span>{isExtracting ? 'Analyzing with Clinical AI...' : 'Analyze & Extract Biomarkers'}</span>
                </button>
              </div>
              {extractMessage && (
                <div style={{
                  marginTop: '10px',
                  padding: '10px 14px',
                  borderRadius: '10px',
                  fontSize: '0.86rem',
                  fontWeight: '600',
                  background: extractMessage.type === 'success' ? 'var(--primary-light)' : 'var(--bg-surface-elevated)',
                  color: extractMessage.type === 'success' ? 'var(--primary)' : 'var(--text-secondary)',
                  border: '1px solid var(--border-subtle)'
                }}>
                  {extractMessage.text}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Tab 2: Clinical Catalog */}
        {activeTab === 'catalog' && (
          <div className="tab-pane-catalog">
            <div className="catalog-search-box">
              <Search className="search-icon" size={16} />
              <input 
                type="text"
                placeholder="Filter 130+ clinical symptoms (e.g. cough, fever, nausea)..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>

            <div className="catalog-categories-accordion">
              {Object.keys(filteredCategories).length === 0 ? (
                <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.88rem' }}>
                  No symptoms match "{searchQuery}"
                </div>
              ) : (
                Object.entries(filteredCategories).map(([category, items]) => (
                  <div key={category} className="catalog-cat-item">
                    <div className="cat-header">{category}</div>
                    <div className="cat-chips">
                      {items.map(s => {
                        const isSelected = symptomsList.includes(s);
                        return (
                          <button
                            key={s}
                            className={`cat-chip ${isSelected ? 'selected' : ''}`}
                            onClick={() => isSelected ? removeSymptom(s) : addSymptom(s)}
                          >
                            <span>{isSelected ? '✓ ' : '+ '}{s}</span>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {/* Active Symptoms Tray */}
        <div className="active-symptoms-tray">
          <div className="tray-header">
            <div className="tray-title">
              <span>Active Presenting Symptoms</span>
              <span className="tray-count">{symptomsList.length} Active</span>
            </div>
            {symptomsList.length > 0 && (
              <button className="btn-clear" onClick={() => setSymptomsList([])}>
                <Trash2 size={14} />
                <span>Clear All</span>
              </button>
            )}
          </div>

          <div className="active-chips-container">
            {symptomsList.length === 0 ? (
              <div className="empty-tray-notice">
                <HeartPulse size={16} />
                <span>No symptoms selected yet. Describe symptoms above or pick from the catalog.</span>
              </div>
            ) : (
              symptomsList.map(s => (
                <div key={s} className="active-chip">
                  <span>{s}</span>
                  <button 
                    className="chip-remove-btn" 
                    onClick={() => removeSymptom(s)}
                    title={`Remove ${s}`}
                  >
                    <X size={14} />
                  </button>
                </div>
              ))
            )}
          </div>

          {/* Big Action Button */}
          <button 
            className="btn-run-diagnosis"
            disabled={symptomsList.length === 0}
            onClick={() => onRunDiagnosis(userText)}
          >
            <span>Run Multi-Model Diagnostic Engine</span>
            <ArrowRight size={20} />
          </button>
        </div>
      </div>

      {/* Right Column: Side Info & Realistic Medical Imagery */}
      <div className="side-info-col">
        {/* Step-by-Step Info */}
        <div className="how-it-works-card">
          <div className="card-header" style={{ marginBottom: '8px' }}>
            <div className="card-title" style={{ fontSize: '1.08rem' }}>Clinical Workflow</div>
            <button 
              className="nav-btn-link" 
              style={{ padding: '4px 10px', fontSize: '0.78rem' }}
              onClick={() => onOpenModal('howItWorks')}
            >
              Learn More
            </button>
          </div>

          <div className="step-list">
            <div className="step-item">
              <div className="item-num">1</div>
              <div className="item-text">
                <div className="item-heading">Symptom Biomarker Intake</div>
                <div className="item-desc">Patient presents colloquial or clinical observations for NLP normalization.</div>
              </div>
            </div>
            <div className="step-item">
              <div className="item-num">2</div>
              <div className="item-text">
                <div className="item-heading">Multi-Model Classification</div>
                <div className="item-desc">Calibrated ensemble predicts primary diagnosis and differential probabilities.</div>
              </div>
            </div>
            <div className="step-item">
              <div className="item-num">3</div>
              <div className="item-text">
                <div className="item-heading">Evidence-Based Treatment</div>
                <div className="item-desc">Formulates ICD-10 aligned pharmacotherapy, lifestyle, and alert parameters.</div>
              </div>
            </div>
          </div>
        </div>

        {/* Realistic Clinic Photo Card */}
        <div className="clinic-facility-card">
          <img 
            src="/images/clinic.jpg" 
            alt="State-of-the-Art Medical Diagnostics Facility" 
            className="clinic-photo"
          />
          <div className="clinic-overlay">
            <div className="clinic-tag">Diagnostic Facility</div>
            <div className="clinic-name">Certified Clinical Decision Architecture</div>
          </div>
        </div>

        {/* Important Clinical Disclaimer */}
        <div className="alert-warning-card">
          <AlertTriangle className="alert-icon" size={22} />
          <div className="alert-text">
            <div className="alert-title">Clinical Triage Disclaimer</div>
            <div className="alert-desc">
              MediSense AI is a diagnostic support instrument. For acute chest pain, severe dyspnea, or anaphylaxis, call emergency services (911 / 108) immediately.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

