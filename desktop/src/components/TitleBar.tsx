import { Bot, Maximize2, Minus, Square, X } from "lucide-react";
import { useEffect, useState } from "react";

export function TitleBar(): JSX.Element {
  const [maximized, setMaximized] = useState(false);
  useEffect(() => {
    void window.chudDesktop.isMaximized().then(setMaximized);
    return window.chudDesktop.onMaximized(setMaximized);
  }, []);
  return (
    <header className="titlebar">
      <div className="titlebar-brand">
        <span className="mini-mark">
          <Bot size={16} />
        </span>
        <span>ChudGPT Desktop</span>
        <span className="version-chip">PUBLIC</span>
      </div>
      <div className="window-controls">
        <button
          aria-label="Minimize"
          onClick={() => void window.chudDesktop.minimize()}
        >
          <Minus size={17} />
        </button>
        <button
          aria-label={maximized ? "Restore" : "Maximize"}
          onClick={() => void window.chudDesktop.toggleMaximize()}
        >
          {maximized ? <Square size={13} /> : <Maximize2 size={15} />}
        </button>
        <button
          className="close-control"
          aria-label="Close"
          onClick={() => void window.chudDesktop.close()}
        >
          <X size={17} />
        </button>
      </div>
    </header>
  );
}
