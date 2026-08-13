import { Database, Radio, Sparkles } from "lucide-react";

export function Onboarding({
  onFinish,
}: {
  onFinish: () => void;
}): JSX.Element {
  return (
    <div className="onboarding">
      <div className="onboarding-card">
        <div className="hero-mark">
          <Sparkles />
        </div>
        <span className="eyebrow">INITIALIZE DESKTOP LINK</span>
        <h1>Welcome to ChudGPT Desktop</h1>
        <p className="onboarding-lead">
          A native command center powered by ChudGPT-Public.
        </p>
        <div className="onboarding-points">
          <div>
            <Sparkles />
            <span>
              <strong>Experimental intelligence</strong>Responses can be
              chaotic, inconsistent, and confidently wrong.
            </span>
          </div>
          <div>
            <Database />
            <span>
              <strong>Local chat storage</strong>Your saved conversations remain
              on this device.
            </span>
          </div>
          <div>
            <Radio />
            <span>
              <strong>Owner-hosted API</strong>Availability depends on the
              ChudGPT-Public inference server.
            </span>
          </div>
        </div>
        <p className="warning-text">
          ChudGPT-Public is an experimental small language model. Do not rely on
          it for important decisions.
        </p>
        <button className="primary-action" onClick={onFinish}>
          Start Chatting
        </button>
      </div>
    </div>
  );
}
