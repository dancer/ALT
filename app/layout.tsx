"use client";

import { useEffect } from "react";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

function DisableCopyPaste({ children }: { children: React.ReactNode }) {
  useEffect(() => {
    const preventDefaultAction = (e: Event) => {
      e.preventDefault();
      return false;
    };

    const preventCopyPaste = (e: KeyboardEvent) => {
      if (e.ctrlKey && (e.key === "c" || e.key === "v" || e.key === "x")) {
        e.preventDefault();
        return false;
      }
    };

    document.addEventListener("copy", preventDefaultAction);
    document.addEventListener("paste", preventDefaultAction);
    document.addEventListener("cut", preventDefaultAction);
    document.addEventListener("keydown", preventCopyPaste);

    return () => {
      document.removeEventListener("copy", preventDefaultAction);
      document.removeEventListener("paste", preventDefaultAction);
      document.removeEventListener("cut", preventDefaultAction);
      document.removeEventListener("keydown", preventCopyPaste);
    };
  }, []);

  return <>{children}</>;
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <title>Alt</title>
        <meta
          name="description"
          content="Exploring the frontiers of AI, quantum computing, and the digital realm to shape a better tomorrow."
        />
        <meta property="og:image" content="/AI.png" />
        <meta property="og:image:alt" content="AI Logo" />
        <meta property="og:image:width" content="1200" />
        <meta property="og:image:height" content="630" />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:image" content="/AI.png" />
        <link rel="icon" href="/AI.png" type="image/png" />
      </head>
      <body
        className={`${inter.className} bg-black text-white min-h-screen flex flex-col`}
      >
        <DisableCopyPaste>
          <main className="flex-grow container mx-auto px-4 py-8 max-w-lg">
            {children}
          </main>
        </DisableCopyPaste>
      </body>
    </html>
  );
}
