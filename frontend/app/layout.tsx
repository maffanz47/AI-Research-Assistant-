import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-sans" });

export const metadata: Metadata = {
  title: "Research Gap Finder",
  description:
    "Discover unexplored frontiers in any research domain using citation network analysis powered by Qwen 14B and HDBSCAN clustering.",
  openGraph: {
    title: "Research Gap Finder",
    description: "AI-powered research gap analysis from paper abstracts.",
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
      <body className="min-h-screen bg-grid antialiased">{children}</body>
    </html>
  );
}
