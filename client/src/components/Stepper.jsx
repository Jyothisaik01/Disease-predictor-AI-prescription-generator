import React from 'react';
import { ChevronRight } from 'lucide-react';

export default function Stepper({ currentStep, onStepClick }) {
  const steps = [
    { id: 1, label: 'Symptom Intake' },
    { id: 2, label: 'AI Diagnostic Analysis' },
    { id: 3, label: 'Clinical Results & Rx' }
  ];

  return (
    <nav className="stepper-nav" aria-label="Progress Stepper">
      {steps.map((step, index) => {
        const isActive = currentStep === step.id;
        const isCompleted = currentStep > step.id;
        
        return (
          <React.Fragment key={step.id}>
            <div 
              className={`step-node ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}`}
              onClick={() => isCompleted && onStepClick(step.id)}
              style={{ cursor: isCompleted ? 'pointer' : 'default' }}
            >
              <span className="step-num">{isCompleted ? '✓' : step.id}</span>
              <span>{step.label}</span>
            </div>
            {index < steps.length - 1 && (
              <span className="step-connector">
                <ChevronRight size={18} />
              </span>
            )}
          </React.Fragment>
        );
      })}
    </nav>
  );
}

