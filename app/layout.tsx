import type { Metadata, Viewport } from 'next'
import { Instrument_Serif } from 'next/font/google'
import './globals.css'

const instrumentSerif = Instrument_Serif({
  subsets: ['latin'],
  weight: '400',
  display: 'swap',
  variable: '--font-serif',
})

export const metadata: Metadata = {
  title: 'after a long time',
}

export const viewport: Viewport = {
  themeColor: '#000000',
  colorScheme: 'dark',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={instrumentSerif.variable}>{children}</body>
    </html>
  )
}
