import type { Metadata } from "next";
import { Inter, IBM_Plex_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "AppleSupport AI Agent",
  description:
    "An AI support copilot for AppleSupport: intent classification, grounded reply drafting, and escalation decisions, evaluated against real baselines.",
};

const NAV_LINKS = [
  { href: "/", label: "Overview" },
  { href: "/demo", label: "Live demo" },
  { href: "/eval", label: "Eval dashboard" },
];

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${inter.variable} ${plexMono.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col bg-bg text-text">
        <header className="border-b border-border">
          <div className="mx-auto max-w-[1120px] px-6 flex items-center justify-between h-14">
            <Link href="/" className="text-[14px] font-medium tracking-tight">
              AppleSupport <span className="text-text-muted font-normal">/ AI agent</span>
            </Link>
            <nav className="flex gap-6">
              {NAV_LINKS.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  className="text-[13px] text-text-secondary hover:text-text transition-colors"
                >
                  {link.label}
                </Link>
              ))}
            </nav>
          </div>
        </header>
        <main className="flex-1">{children}</main>
        <footer className="border-t border-border">
          <div className="mx-auto max-w-[1120px] px-6 py-6 text-[12px] text-text-muted font-mono">
            Local-only demo — no auth, no persistence beyond the artifacts checked into the repo.
          </div>
        </footer>
      </body>
    </html>
  );
}
