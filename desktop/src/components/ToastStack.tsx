export interface Toast {
  id: string;
  message: string;
}
export function ToastStack({ toasts }: { toasts: Toast[] }): JSX.Element {
  return (
    <div className="toast-stack" aria-live="polite">
      {toasts.map((toast) => (
        <div className="toast" key={toast.id}>
          {toast.message}
        </div>
      ))}
    </div>
  );
}
