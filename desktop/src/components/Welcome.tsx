import { Beaker, Braces, HelpCircle, Lightbulb } from "lucide-react";

const starters = [
  [HelpCircle, "Ask something", "Ask ChudGPT anything..."],
  [Braces, "Write some code", "Write a small program that..."],
  [Lightbulb, "Explain something", "Explain this in simple terms..."],
  [Beaker, "Test ChudGPT", "Let's test your experimental brain: "],
] as const;

export function Welcome({
  onStarter,
  music = false,
}: {
  onStarter: (text: string) => void;
  music?: boolean;
}): JSX.Element {
  return (
    <section className="welcome">
      <div className="welcome-orb">
        <span />
        <span />
        <span />
        <strong>CHUD</strong>
      </div>
      <span className="eyebrow">
        {music ? "MUSIC V1 DESKTOP" : "PUBLIC DESKTOP"}
      </span>
      <h1>
        {music ? "What are we writing today?" : "What are we building today?"}
      </h1>
      <p>
        {music
          ? "Write a song, forge a hook, or request certified lyrical nonsense."
          : "Ask a question, test the chaos, or start creating."}
      </p>
      <div className="starter-grid">
        {starters.map(([Icon, title, prompt]) => (
          <button key={title} onClick={() => onStarter(prompt)}>
            <Icon />
            <span>
              <strong>{title}</strong>
              <small>{prompt}</small>
            </span>
          </button>
        ))}
      </div>
    </section>
  );
}
