import { Component, type ReactNode } from "react";
import { Link } from "react-router-dom";

interface ErrorBoundaryProps {
  children: ReactNode;
  backTo?: string;
}

interface ErrorBoundaryState {
  hasError: boolean;
  message: string;
}

/*
 * Catches render-time exceptions in its subtree and shows a fallback
 * instead of unmounting the entire app (blank page).
 */
export class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = {
    hasError: false,
    message: "",
  };

  static getDerivedStateFromError(
    error: unknown
  ): ErrorBoundaryState {
    return {
      hasError: true,
      message:
        error instanceof Error
          ? error.message
          : "Something went wrong.",
    };
  }

  componentDidCatch(error: unknown) {
    console.error(
      "ErrorBoundary caught:",
      error
    );
  }

  render() {
    if (this.state.hasError) {
      return (
        <div>
          {this.props.backTo && (
            <Link
              to={this.props.backTo}
              style={{
                display: "inline-block",
                marginBottom: "20px",
                color: "#15803d",
                textDecoration: "none",
              }}
            >
              ← Back
            </Link>
          )}

          <div
            style={{
              padding: "15px",
              marginTop: "20px",
              borderRadius: "8px",
              background: "#fee2e2",
              color: "#991b1b",
              maxWidth: "800px",
            }}
          >
            <strong>
              Could not display this section.
            </strong>
            <br />
            {this.state.message}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
