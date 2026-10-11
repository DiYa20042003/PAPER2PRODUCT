import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "PAPER2PRODUCT — Reproduction workspace",
  description:
    "An evidence-backed workspace for inspecting research papers, running experiments, and evaluating reproducibility.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
