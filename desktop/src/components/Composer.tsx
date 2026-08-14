import { Eraser, Send, Square } from "lucide-react";
import { useCallback, useEffect, useLayoutEffect, useRef } from "react";

interface Props {
  value: string;
  busy: boolean;
  sendWithEnter: boolean;
  focusKey: string | null;
  onChange: (value: string) => void;
  onSend: () => void;
  onStop: () => void;
}

export function Composer({
  value,
  busy,
  sendWithEnter,
  focusKey,
  onChange,
  onSend,
  onStop,
}: Props): JSX.Element {
  const ref = useRef<HTMLTextAreaElement>(null);
  const focusComposer = useCallback(() => {
    window.requestAnimationFrame(() => {
      const composer = ref.current;
      if (!composer) return;
      composer.focus({ preventScroll: true });
      composer.setSelectionRange(composer.value.length, composer.value.length);
    });
  }, []);
  useEffect(() => {
    window.addEventListener("chud:focus-composer", focusComposer);
    return () =>
      window.removeEventListener("chud:focus-composer", focusComposer);
  }, [focusComposer]);
  useLayoutEffect(() => focusComposer(), [focusKey, focusComposer]);
  useEffect(() => {
    if (!ref.current) return;
    ref.current.style.height = "0";
    ref.current.style.height = `${Math.min(ref.current.scrollHeight, 180)}px`;
  }, [value]);
  return (
    <div className="composer-wrap">
      <div className="composer">
        <textarea
          id="message-composer"
          ref={ref}
          autoFocus
          value={value}
          maxLength={4000}
          rows={1}
          placeholder="Message ChudGPT..."
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (sendWithEnter && event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              if (!busy) onSend();
            }
          }}
        />
        {value && !busy && (
          <button
            className="composer-clear"
            title="Clear input"
            onClick={() => onChange("")}
          >
            <Eraser size={17} />
          </button>
        )}
        <button
          className={`send-button ${busy ? "stop" : ""}`}
          disabled={!busy && !value.trim()}
          onClick={busy ? onStop : onSend}
        >
          {busy ? <Square size={16} fill="currentColor" /> : <Send size={17} />}
        </button>
      </div>
      <div className="composer-foot">
        <span>{value.length.toLocaleString()} / 4,000</span>
        <span>Enter to send · Shift + Enter for newline</span>
      </div>
    </div>
  );
}
