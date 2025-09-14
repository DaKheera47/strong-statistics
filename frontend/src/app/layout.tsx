import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "Strong Statistics - Your Training Analytics Dashboard",
    template: "%s | Strong Statistics"
  },
  description: "Track and analyze your lifting progress with comprehensive workout analytics, volume tracking, and performance insights. Your personal training dashboard.",
  keywords: ["lifting", "workout", "analytics", "fitness", "strength training", "gym tracker", "progressive overload", "training log"],
  authors: [{ name: "Strong Statistics" }],
  creator: "Strong Statistics",
  publisher: "Strong Statistics",
  openGraph: {
    type: "website",
    locale: "en_US",
    title: "Strong Statistics - Your Training Analytics Dashboard",
    description: "Track and analyze your lifting progress with comprehensive workout analytics, volume tracking, and performance insights.",
    siteName: "Strong Statistics",
    images: [{
      url: "/og-image.jpg",
      width: 1200,
      height: 630,
      alt: "Strong Statistics Training Analytics Dashboard"
    }]
  },
  twitter: {
    card: "summary_large_image",
    title: "Strong Statistics - Your Training Analytics Dashboard",
    description: "Track and analyze your lifting progress with comprehensive workout analytics.",
    images: ["/og-image.jpg"]
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  verification: {
    google: 'verification-code-here',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang='en'
      suppressHydrationWarning
    >
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased container mx-auto px-4 max-w-7xl`}
      >
        <ThemeProvider
          attribute='class'
          defaultTheme='system'
          enableSystem
          disableTransitionOnChange
        >
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
