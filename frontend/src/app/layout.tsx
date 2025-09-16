import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";
import Script from "next/script";

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
    template: "%s | Strong Statistics",
  },
  description:
    "Track and analyze your lifting progress with comprehensive workout analytics, volume tracking, and performance insights. Your personal training dashboard.",
  keywords: [
    "lifting",
    "workout",
    "analytics",
    "fitness",
    "strength training",
    "gym tracker",
    "progressive overload",
    "training log",
  ],
  authors: [{ name: "Strong Statistics" }],
  creator: "Strong Statistics",
  publisher: "Strong Statistics",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
    apple: "/favicon.svg",
  },
  openGraph: {
    type: "website",
    locale: "en_US",
    title: "Strong Statistics - Your Training Analytics Dashboard",
    description:
      "Track and analyze your lifting progress with comprehensive workout analytics, volume tracking, and performance insights.",
    siteName: "Strong Statistics",
    images: [
      {
        url: "/og-image.jpg",
        width: 1200,
        height: 630,
        alt: "Strong Statistics Training Analytics Dashboard",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Strong Statistics - Your Training Analytics Dashboard",
    description:
      "Track and analyze your lifting progress with comprehensive workout analytics.",
    images: ["/og-image.jpg"],
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
  verification: {
    google: "verification-code-here",
  },
  themeColor: "#0A0A0A",
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
      <head>
        <meta name='theme-color' content='#0A0A0A' />
        <meta name='apple-mobile-web-app-status-bar-style' content='#0A0A0A' />
        <Script
          src='https://umami.dakheera47.com/script.js'
          data-website-id='ac4af018-5e53-408a-9f7a-8102a6618065'
          strategy='afterInteractive'
        />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <ThemeProvider
          attribute='class'
          defaultTheme='system'
          enableSystem
          disableTransitionOnChange
        >
          <main className='container mx-auto px-4 max-w-7xl'>{children}</main>
        </ThemeProvider>
      </body>
    </html>
  );
}
