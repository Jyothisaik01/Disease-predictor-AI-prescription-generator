import React from 'react';
import { X, HelpCircle, ShieldCheck, FileText, CheckCircle2 } from 'lucide-react';

export default function Modals({ modalType, onClose, result, symptomsList }) {
  if (!modalType) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        {/* MODAL 1: HOW IT WORKS */}
        {modalType === 'howItWorks' && (
          <>
            <div className="modal-header">
              <div className="modal-title">
                <HelpCircle size={20} style={{ color: 'var(--primary)' }} />
                <span>How MediSense AI Operates</span>
              </div>
              <button className="btn-close" onClick={onClose}>
                <X size={18} />
              </button>
            </div>
            <div className="modal-body">
              <h4 style={{ fontWeight: 800, color: 'var(--text-primary)', marginBottom: '6px' }}>
                1. Clinical NLP Extraction
              </h4>
              <p style={{ marginBottom: '16px' }}>
                Converts colloquial patient health descriptions into standardized clinical biomarkers (e.g., <em>"shivering with high temp"</em> &rarr; <code>High Fever</code>, <code>Chills</code>) using regex tokenization and Gemini 3.8 Flash semantic mapping.
              </p>

              <h4 style={{ fontWeight: 800, color: 'var(--text-primary)', marginBottom: '6px' }}>
                2. Multi-Model Voting Classifier Ensemble
              </h4>
              <p style={{ marginBottom: '16px' }}>
                Evaluates symptom vectors across 4,960+ verified medical records using a soft-voting ensemble of <strong>Multinomial Naive Bayes</strong>, <strong>Random Forest (100 trees)</strong>, and <strong>Extra Trees</strong> to output probability distributions and differential diagnoses.
              </p>

              <h4 style={{ fontWeight: 800, color: 'var(--text-primary)', marginBottom: '6px' }}>
                3. WHO &amp; ICD-10 Aligned Regimen
              </h4>
              <p style={{ marginBottom: '16px' }}>
                Maps conditions to official ICD-10 diagnostic codes, specialty referrals, first-line supportive pharmacotherapy regimens, and evidence-based lifestyle interventions.
              </p>

              <h4 style={{ fontWeight: 800, color: 'var(--text-primary)', marginBottom: '6px' }}>
                4. Patient Safety &amp; Emergency Escalation
              </h4>
              <p>
                Strict triage safeguards detect life-threatening presentations (e.g. myocardial infarction, acute pneumonia) and present urgent emergency warnings.
              </p>
            </div>
            <div className="modal-footer">
              <button className="btn-primary-action" onClick={onClose}>
                Understood
              </button>
            </div>
          </>
        )}

        {/* MODAL 2: DISCHARGE SUMMARY */}
        {modalType === 'discharge' && (
          <>
            <div className="modal-header">
              <div className="modal-title">
                <FileText size={20} style={{ color: 'var(--primary)' }} />
                <span>Clinical Assessment &amp; Discharge Summary</span>
              </div>
              <button className="btn-close" onClick={onClose}>
                <X size={18} />
              </button>
            </div>
            <div className="modal-body">
              {result ? (
                <div>
                  <div style={{ 
                    borderBottom: '1px solid var(--border-subtle)', 
                    paddingBottom: '14px', 
                    marginBottom: '16px' 
                  }}>
                    <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                      Timestamp: {result.timestamp || new Date().toLocaleString()}
                    </div>
                    <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-primary)', marginTop: '4px' }}>
                      {result.primary_disease}
                    </div>
                    <div style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                      Specialty: <strong>{result.specialty}</strong> | Code: <strong>{result.icd10}</strong> | Confidence: <strong>{result.confidence}%</strong>
                    </div>
                  </div>

                  <h5 style={{ fontWeight: 750, color: 'var(--text-primary)', marginBottom: '8px' }}>
                    Reported Patient Symptoms:
                  </h5>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '16px' }}>
                    {symptomsList.map(s => (
                      <span key={s} style={{
                        background: 'var(--bg-surface-elevated)',
                        border: '1px solid var(--border-subtle)',
                        padding: '3px 10px',
                        borderRadius: '12px',
                        fontSize: '0.82rem'
                      }}>
                        {s}
                      </span>
                    ))}
                  </div>

                  <h5 style={{ fontWeight: 750, color: 'var(--text-primary)', marginBottom: '8px' }}>
                    Recommended Pharmacotherapy:
                  </h5>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '18px' }}>
                    {result.rx_regimen.map((rx, idx) => (
                      <div key={idx} style={{
                        background: 'var(--bg-surface-elevated)',
                        padding: '10px 14px',
                        borderRadius: '8px',
                        fontSize: '0.88rem'
                      }}>
                        <strong>{rx.name}</strong>: {rx.instruction} <em>({rx.duration})</em>
                      </div>
                    ))}
                  </div>

                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                    * Disclaimer: This AI Clinical Summary is prepared for decision support and does not replace in-person physician authorization.
                  </div>
                </div>
              ) : (
                <div>No assessment data available.</div>
              )}
            </div>
            <div className="modal-footer">
              <button className="btn-secondary" onClick={() => window.print()}>
                Print Summary
              </button>
              <button className="btn-primary-action" onClick={onClose}>
                Close
              </button>
            </div>
          </>
        )}

        {/* MODAL 3: SAFETY POLICY */}
        {modalType === 'safety' && (
          <>
            <div className="modal-header">
              <div className="modal-title">
                <ShieldCheck size={20} style={{ color: 'var(--accent-emerald)' }} />
                <span>Patient Safety &amp; Triage Policy</span>
              </div>
              <button className="btn-close" onClick={onClose}>
                <X size={18} />
              </button>
            </div>
            <div className="modal-body">
              <p style={{ marginBottom: '14px' }}>
                MediSense AI is developed to assist patients and clinicians with triage stratification and evidence-based differential diagnosis.
              </p>

              <div style={{
                background: 'var(--accent-amber-light)',
                border: '1px solid rgba(245, 158, 11, 0.3)',
                padding: '14px 18px',
                borderRadius: '12px',
                marginBottom: '16px',
                color: 'var(--text-primary)'
              }}>
                <strong>Emergency Situations:</strong> In cases of sudden crushing chest pain, dyspnea, slurred speech, acute trauma, or loss of consciousness, dial emergency services (911 / 108) immediately.
              </div>

              <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)' }}>
                All prescription suggestions must be formally validated by a registered medical practitioner prior to pharmaceutical administration.
              </p>
            </div>
            <div className="modal-footer">
              <button className="btn-primary-action" onClick={onClose}>
                I Understand
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

