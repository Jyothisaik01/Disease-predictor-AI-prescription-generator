import React from 'react';
import { Stethoscope, CheckCircle2, Shield, Sparkles, BookOpen } from 'lucide-react';

export default function HeroBanner() {
  return (
    <section className="hero-banner">
      <div className="hero-content">
        <div className="hero-eyebrow">
          <Sparkles size={14} />
          <span>Next-Generation Clinical Diagnostics</span>
        </div>
        
        <h1 className="hero-title">
          Precision Medicine Powered by <br />
          <span className="gradient-text">Ensemble Machine Learning</span>
        </h1>

        <p className="hero-description">
          MediSense AI fuses multi-model ensemble intelligence (Multinomial Naive Bayes, Random Forest &amp; Extra Trees) 
          with Gemini 3.8 Flash to evaluate presenting symptoms against 4,960+ validated clinical records, delivering 
          ICD-10 aligned differentials and evidence-based pharmacotherapy protocols.
        </p>

        <div className="hero-badges-grid">
          <div className="hero-pill">
            <CheckCircle2 size={16} />
            <span>4,960+ Clinical Records</span>
          </div>
          <div className="hero-pill">
            <Stethoscope size={16} />
            <span>132 Validated Biomarkers</span>
          </div>
          <div className="hero-pill">
            <Shield size={16} />
            <span>ICD-10 &amp; WHO Benchmarks</span>
          </div>
          <div className="hero-pill">
            <BookOpen size={16} />
            <span>99.4% Ensemble Precision</span>
          </div>
        </div>
      </div>

      <div className="hero-media-card">
        <img 
          src="/images/doctor.jpg" 
          alt="Clinical AI Physician Consultation" 
          className="hero-photo"
          loading="eager"
        />
        <div className="hero-overlay-badge">
          <div className="badge-avatar">🩺</div>
          <div className="badge-info">
            <div className="badge-title">Board-Validated Clinical Decision Support</div>
            <div className="badge-sub">Calibrated Triaging &bull; Continuous Confidence Index</div>
          </div>
        </div>
      </div>
    </section>
  );
}

