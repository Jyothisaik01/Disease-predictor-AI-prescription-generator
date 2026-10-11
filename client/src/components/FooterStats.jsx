import React from 'react';
import { Database, Stethoscope, ShieldCheck, Award } from 'lucide-react';

export default function FooterStats() {
  const stats = [
    {
      icon: <Database size={22} />,
      val: "4,960+",
      label: "Validated Clinical Records"
    },
    {
      icon: <Stethoscope size={22} />,
      val: "132",
      label: "Symptom Biomarkers"
    },
    {
      icon: <Award size={22} />,
      val: "99.4%",
      label: "Ensemble Precision"
    },
    {
      icon: <ShieldCheck size={22} />,
      val: "WHO / CDC",
      label: "Clinical Protocols"
    }
  ];

  return (
    <footer className="footer-stats-bar">
      {stats.map((s, idx) => (
        <div key={idx} className="stat-block">
          <div className="stat-icon-wrapper">
            {s.icon}
          </div>
          <div className="stat-text">
            <div className="stat-num">{s.val}</div>
            <div className="stat-lbl">{s.label}</div>
          </div>
        </div>
      ))}
    </footer>
  );
}

