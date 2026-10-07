import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PAPER2PRODUCT — Reproduction workspace",
  description:
    "An evidence-backed workspace for inspecting research papers, running experiments, and evaluating reproducibility.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
