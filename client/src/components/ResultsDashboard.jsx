import React, { useState } from 'react';
import { 
  Pill, Brain, Apple, AlertOctagon, Headphones, 
  RotateCcw, Printer, FileText, CheckCircle2, ShieldCheck,
  Building2, Tag, Activity
} from 'lucide-react';

export default function ResultsDashboard({ 
  result, 
  symptomsList, 
  onReset, 
  onOpenModal 
}) {
  const [activeTab, setActiveTab] = useState('rx'); // 'rx' | 'reason' | 'diet' | 'flags' | 'audio'

  if (!result) return null;

  const {
    primary_disease,
    confidence,
    specialty,
    icd10,
    urgency,
    assessment,
    rx_regimen,
    differentials,
    clinical_rationale,
    diet_lifestyle,
    red_flags,
    audio_base64
  } = result;

  const isUrgent = urgency.toLowerCase().includes('critical') || urgency.toLowerCase().includes('high');

  return (
    <div className="results-dashboard">
      {/* 1. Primary Diagnosis Hero Card */}
      <div className="diagnosis-hero-card">
        <div className="hero-info">
          <div className="hero-tag">Primary Clinical Assessment</div>
          <h2 className="condition-title">{primary_disease}</h2>

          <div className="condition-pills">
            <span className="pill pill-specialty">
              <Building2 size={14} />
              <span>{specialty}</span>
            </span>
            <span className="pill pill-icd10">
              <Tag size={14} />
              <span>{icd10}</span>
            </span>
            <span className={`pill ${isUrgent ? 'pill-urgency' : 'pill-urgency'}`} style={{
              background: isUrgent ? 'var(--accent-rose-light)' : 'var(--accent-emerald-light)',
              color: isUrgent ? 'var(--accent-rose)' : 'var(--accent-emerald)'
            }}>
              <ShieldCheck size={14} />
              <span>{urgency}</span>
            </span>
          </div>

          <p className="condition-summary">{assessment}</p>
        </div>

        <div className="hero-confidence-gauge">
          <div className="gauge-val">{confidence}%</div>
          <div className="gauge-label">Confidence Index</div>
        </div>
      </div>

      {/* 2. Results Navigation Tabs */}
      <div className="results-tabs-nav">
        <button 
          className={`res-tab-btn ${activeTab === 'rx' ? 'active' : ''}`}
          onClick={() => setActiveTab('rx')}
        >
          <Pill size={16} />
          <span>Prescription &amp; Plan</span>
        </button>
        <button 
          className={`res-tab-btn ${activeTab === 'reason' ? 'active' : ''}`}
          onClick={() => setActiveTab('reason')}
        >
          <Brain size={16} />
          <span>Clinical Reasoning</span>
        </button>
        <button 
          className={`res-tab-btn ${activeTab === 'diet' ? 'active' : ''}`}
          onClick={() => setActiveTab('diet')}
        >
          <Apple size={16} />
          <span>Diet &amp; Lifestyle</span>
        </button>
        <button 
          className={`res-tab-btn ${activeTab === 'flags' ? 'active' : ''}`}
          onClick={() => setActiveTab('flags')}
        >
          <AlertOctagon size={16} />
          <span>Critical Red Flags</span>
        </button>
        <button 
          className={`res-tab-btn ${activeTab === 'audio' ? 'active' : ''}`}
          onClick={() => setActiveTab('audio')}
        >
          <Headphones size={16} />
          <span>Audio Consultation</span>
        </button>
      </div>

      {/* 3. Tab Pane Contents */}
      <div className="tab-pane-content">
        {/* Tab 1: Prescription & Plan */}
        {activeTab === 'rx' && (
          <div>
            <h3 style={{ fontSize: '1.15rem', fontWeight: 800, marginBottom: '14px', color: 'var(--text-primary)' }}>
              ℞ Evidence-Based Pharmacotherapy Regimen
            </h3>
            <div className="rx-grid">
              {rx_regimen.map((med, idx) => (
                <div key={idx} className="rx-card-row">
                  <div className="rx-med-info">
                    <div className="med-icon-box">💊</div>
                    <div>
                      <div className="med-name">{med.name}</div>
                      <div className="med-instruction">{med.instruction}</div>
                    </div>
                  </div>
                  <div className="rx-duration-badge">{med.duration}</div>
                </div>
              ))}
            </div>

            <div style={{ 
              marginTop: '18px', 
              fontSize: '0.82rem', 
              color: 'var(--text-muted)',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}>
              <CheckCircle2 size={14} style={{ color: 'var(--accent-emerald)' }} />
              <span>Formulated following WHO Model List of Essential Medicines &amp; CDC Standard Practice Guidelines.</span>
            </div>
          </div>
        )}

        {/* Tab 2: Clinical Reasoning & Differentials */}
        {activeTab === 'reason' && (
          <div>
            <h3 style={{ fontSize: '1.15rem', fontWeight: 800, marginBottom: '10px', color: 'var(--text-primary)' }}>
              Pathophysiology &amp; Diagnostic Rationale
            </h3>
            <div style={{
              background: 'var(--bg-surface-elevated)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '12px',
              padding: '16px 20px',
              lineHeight: '1.6',
              fontSize: '0.94rem',
              color: 'var(--text-secondary)',
              marginBottom: '24px'
            }}>
              {clinical_rationale}
            </div>

            <h4 style={{ fontSize: '1rem', fontWeight: 750, marginBottom: '12px', color: 'var(--text-primary)' }}>
              Differential Diagnoses (Ensemble Stratification)
            </h4>
            <div className="differentials-list">
              {differentials.map((diff, idx) => (
                <div key={idx} className="diff-row">
                  <div>
                    <span className="diff-name">{diff.name}</span>
                    <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginLeft: '10px' }}>
                      ({diff.specialty})
                    </span>
                  </div>
                  <div className="diff-pct-bar">
                    <div className="diff-bar-track">
                      <div className="diff-bar-fill" style={{ width: `${Math.min(100, diff.probability)}%` }} />
                    </div>
                    <span className="diff-pct-text">{diff.probability}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tab 3: Diet & Lifestyle */}
        {activeTab === 'diet' && (
          <div>
            <h3 style={{ fontSize: '1.15rem', fontWeight: 800, marginBottom: '14px', color: 'var(--text-primary)' }}>
              🥗 Clinical Dietary &amp; Lifestyle Management
            </h3>
            <div className="bullet-list">
              {diet_lifestyle.map((item, idx) => (
                <div key={idx} className="bullet-item">
                  <Apple className="bullet-icon" size={18} />
                  <div className="bullet-text">{item}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tab 4: Critical Red Flags */}
        {activeTab === 'flags' && (
          <div>
            <h3 style={{ fontSize: '1.15rem', fontWeight: 800, marginBottom: '14px', color: 'var(--accent-rose)' }}>
              🚨 Emergency Warning Signs (Red Flags)
            </h3>
            <p style={{ fontSize: '0.88rem', color: 'var(--text-muted)', marginBottom: '16px' }}>
              If you or the patient experience any of the following acute symptoms, seek immediate emergency care (911 / 108):
            </p>
            <div className="redflag-list">
              {red_flags.map((item, idx) => (
                <div key={idx} className="redflag-item">
                  <AlertOctagon className="redflag-icon" size={20} />
                  <div className="redflag-text">{item}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tab 5: Audio Consultation */}
        {activeTab === 'audio' && (
          <div>
            <h3 style={{ fontSize: '1.15rem', fontWeight: 800, marginBottom: '14px', color: 'var(--text-primary)' }}>
              🎧 Audio Clinical Consultation
            </h3>
            <div className="audio-player-box">
              <Headphones size={42} style={{ color: 'var(--primary)' }} />
              <div>
                <div style={{ fontSize: '1.05rem', fontWeight: 700 }}>Listen to Diagnostic Overview</div>
                <div style={{ fontSize: '0.84rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                  Synthesized clinical audio summary of {primary_disease}
                </div>
              </div>

              {audio_base64 ? (
                <audio controls src={`data:audio/mp3;base64,${audio_base64}`} autoPlay={false}>
                  Your browser does not support the audio element.
                </audio>
              ) : (
                <div style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
                  Audio playback unavailable for this consultation.
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* 4. Action Bar */}
      <div className="dashboard-action-bar">
        <div style={{ display: 'flex', gap: '12px' }}>
          <button className="btn-secondary" onClick={() => onOpenModal('discharge')}>
            <FileText size={16} />
            <span>Discharge Summary</span>
          </button>
          <button className="btn-secondary" onClick={() => window.print()}>
            <Printer size={16} />
            <span>Print Report</span>
          </button>
        </div>

        <button className="btn-primary-action" onClick={onReset}>
          <RotateCcw size={16} />
          <span>New Assessment</span>
        </button>
      </div>
    </div>
  );
}

