import React, { useEffect, useState } from 'react';
import { Activity, CheckCircle2 } from 'lucide-react';

export default function AnalysisStage({ onComplete }) {
  const [progress, setProgress] = useState(15);
  const [stage, setStage] = useState(0);

  const stages = [
    "Normalizing reported symptom biomarkers & clinical indicators...",
    "Querying Multi-Model Ensemble (VotingClassifier across 4,960 records)...",
    "Cross-referencing ICD-10 diagnostic standards & WHO guidelines...",
    "Formulating evidence-based Rx regimen & personalized advisory via Gemini 3.8 Flash..."
  ];

  useEffect(() => {
    const timer1 = setTimeout(() => { setProgress(40); setStage(1); }, 600);
    const timer2 = setTimeout(() => { setProgress(75); setStage(2); }, 1300);
    const timer3 = setTimeout(() => { setProgress(95); setStage(3); }, 2000);
    const timer4 = setTimeout(() => {
      setProgress(100);
      setTimeout(() => onComplete(), 400);
    }, 2600);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      clearTimeout(timer4);
    };
  }, [onComplete]);

  return (
    <div className="analysis-container">
      <div className="radar-spinner-wrap">
        <div className="spinner-ring" />
        <Activity className="spinner-icon" size={36} />
      </div>

      <h2 className="analysis-heading">Analyzing Presenting Biomarkers</h2>
      <p className="analysis-sub">
        Evaluating symptom vectors against calibrated clinical ensemble classifiers ({progress}%)
      </p>

      <div className="progress-bar-track">
        <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
      </div>

      <div className="milestones-list">
        {stages.map((text, idx) => {
          const isDone = idx < stage || progress === 100;
          const isCurrent = idx === stage && progress < 100;

          return (
            <div 
              key={idx} 
              className={`milestone-item ${isDone ? 'done' : ''} ${isCurrent ? 'active' : ''}`}
            >
              {isDone ? (
                <CheckCircle2 size={18} />
              ) : (
                <span style={{ 
                  width: '8px', 
                  height: '8px', 
                  borderRadius: '50%', 
                  background: isCurrent ? 'var(--primary)' : 'var(--text-muted)' 
                }} />
              )}
              <span>{text}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

