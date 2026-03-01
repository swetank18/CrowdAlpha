import { Component, type ErrorInfo, type ReactNode } from "react";

type Props = {
  children: ReactNode;
};

type State = {
  hasError: boolean;
  message: string;
};

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, message: "" };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, message: error?.message ?? "Unknown frontend error" };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Keep this log for production debugging from browser console.
    // eslint-disable-next-line no-console
    console.error("CrowdAlpha frontend crashed:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <main className="min-h-screen bg-surface text-slate-100 p-6">
          <div className="mx-auto max-w-3xl rounded-xl border border-rose-900/60 bg-rose-950/40 p-5">
            <h1 className="text-lg font-semibold text-rose-200">Dashboard runtime error</h1>
            <p className="mt-2 text-sm text-rose-100/90">
              The app hit an unexpected frontend exception. Open browser devtools and inspect Console/Network for the root cause.
            </p>
            <p className="mt-3 rounded bg-surface-1 px-3 py-2 font-mono text-xs text-slate-300 break-all">
              {this.state.message}
            </p>
          </div>
        </main>
      );
    }
    return this.props.children;
  }
}
