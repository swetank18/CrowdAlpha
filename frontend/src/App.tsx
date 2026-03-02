import "./App.css";
import { useEffect, useMemo, useState } from "react";
import { useAnalyticsSnapshot } from "./hooks/useAnalyticsSnapshot";
import { useMarketWS } from "./hooks/useMarketWS";
import { DashboardPage } from "./pages/DashboardPage";
import { DeployPage } from "./pages/DeployPage";
import { DocsPage } from "./pages/DocsPage";
import { LandingPage } from "./pages/LandingPage";
import { LeaderboardPage } from "./pages/LeaderboardPage";
import { ResearchPage } from "./pages/ResearchPage";
import { StrategyPage } from "./pages/StrategyPage";

function normalizePath(path: string) {
  if (!path) return "/";
  const clean = path.replace(/\/+$/, "");
  return clean === "" ? "/" : clean;
}

function usePathname() {
  const [pathname, setPathname] = useState(() => normalizePath(window.location.pathname));

  useEffect(() => {
    const onPop = () => setPathname(normalizePath(window.location.pathname));
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const go = (to: string) => {
    const target = normalizePath(to);
    if (target === pathname) return;
    window.history.pushState({}, "", target);
    setPathname(target);
  };

  return { pathname, go };
}

type NavLinkProps = {
  href: string;
  label: string;
  pathname: string;
  go: (path: string) => void;
};

function NavLink({ href, label, pathname, go }: NavLinkProps) {
  const active = pathname === href || (href !== "/" && pathname.startsWith(`${href}/`));
  return (
    <a
      href={href}
      onClick={(e) => {
        e.preventDefault();
        go(href);
      }}
      className={`rounded-md px-3 py-1.5 text-xs font-medium uppercase tracking-wider transition ${
        active ? "bg-sky-900/40 text-sky-200" : "text-slate-300 hover:bg-slate-800/50"
      }`}
    >
      {label}
    </a>
  );
}

function NotFound() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-8 md:px-6 md:py-10">
      <div className="card">
        <h1 className="text-xl font-semibold text-slate-100">Route Not Found</h1>
        <p className="mt-2 text-sm text-slate-400">This page does not exist in the current frontend shell.</p>
      </div>
    </div>
  );
}

function App() {
  useMarketWS();
  useAnalyticsSnapshot(2500);

  const { pathname, go } = usePathname();

  const page = useMemo(() => {
    if (pathname === "/") return <LandingPage go={go} />;
    if (pathname === "/dashboard") return <DashboardPage />;
    if (pathname === "/deploy") return <DeployPage />;
    if (pathname === "/leaderboard") return <LeaderboardPage />;
    if (pathname === "/research") return <ResearchPage />;
    if (pathname === "/docs") return <DocsPage />;

    const match = pathname.match(/^\/strategy\/([^/]+)$/);
    if (match) return <StrategyPage agentId={decodeURIComponent(match[1])} />;

    return <NotFound />;
  }, [pathname]);

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,#1f2937_0%,#0f1117_55%,#0a0c12_100%)] text-slate-100">
      <nav className="sticky top-0 z-30 border-b border-slate-800/70 bg-slate-950/75 backdrop-blur">
        <div className="mx-auto flex max-w-[1500px] flex-wrap items-center gap-1 px-4 py-2 md:px-6">
          <NavLink href="/" label="Home" pathname={pathname} go={go} />
          <NavLink href="/dashboard" label="Dashboard" pathname={pathname} go={go} />
          <NavLink href="/deploy" label="Deploy" pathname={pathname} go={go} />
          <NavLink href="/leaderboard" label="Leaderboard" pathname={pathname} go={go} />
          <NavLink href="/research" label="Research" pathname={pathname} go={go} />
          <NavLink href="/docs" label="Docs" pathname={pathname} go={go} />
        </div>
      </nav>
      {page}
    </main>
  );
}

export default App;
