import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Research Gap Finder",
  description:
    "Identify unexplored research opportunities from scientific paper abstracts using fine-tuned Qwen 2.5-14B with LoRA adapters, HDBSCAN clustering, and Semantic Scholar citation analysis.",
  openGraph: {
    title: "Research Gap Finder",
    description:
      "AI-powered research gap analysis from paper abstracts.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
