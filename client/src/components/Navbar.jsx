import React from 'react';
import { Activity, ShieldCheck, HelpCircle, Sun, Moon, Globe } from 'lucide-react';

export default function Navbar({ theme, toggleTheme, onOpenModal, selectedLang, setSelectedLang }) {
  return (
    <header className="navbar">
      <div className="nav-brand" onClick={() => window.location.reload()}>
        <div className="brand-icon-box">
          <Activity size={24} />
          <span className="brand-dot" />
        </div>
        <div className="brand-text">
          <div className="brand-title">
            MediSense AI
            <span className="brand-badge">Clinical v2.4</span>
          </div>
          <div className="brand-subtitle">
            AI-Driven Disease Predictor &amp; Prescription Generator
          </div>
        </div>
      </div>

      <div className="nav-actions">
        {/* Language selector */}
        <button 
          className="nav-btn-link"
          onClick={() => {
            const langs = ['en', 'hi', 'te'];
            const nextIdx = (langs.indexOf(selectedLang) + 1) % langs.length;
            setSelectedLang(langs[nextIdx]);
          }}
          title="Switch consultation language"
        >
          <Globe size={16} />
          <span>{selectedLang === 'en' ? 'English' : selectedLang === 'hi' ? 'हिंदी (Hindi)' : 'తెలుగు (Telugu)'}</span>
        </button>

        {/* How It Works Modal trigger */}
        <button className="nav-btn-link" onClick={() => onOpenModal('howItWorks')}>
          <HelpCircle size={16} />
          <span>How it Works</span>
        </button>

        {/* Clinical Safety trigger */}
        <button className="nav-btn-link" onClick={() => onOpenModal('safety')}>
          <ShieldCheck size={16} />
          <span>Safety Policy</span>
        </button>

        {/* Theme Toggle Button */}
        <button 
          className="theme-toggle-btn" 
          onClick={toggleTheme}
          title={theme === 'dark' ? "Switch to Light Mode" : "Switch to Dark Mode"}
        >
          {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
        </button>
      </div>
    </header>
  );
}

