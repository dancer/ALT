import { Space_Mono } from 'next/font/google'
import './globals.css'
import { InitialLoading } from '@/components/initial-loading'
import { Footer } from '@/components/footer'

const spaceMono = Space_Mono({
  subsets: ['latin'],
  weight: ['400', '700'],
  variable: '--font-mono'
})

export const metadata = {
  title: 'Anywho',
  description: 'Software Development Studio in London, UK',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={`${spaceMono.variable} font-mono bg-[#f5f5f0] text-neutral-800 flex flex-col min-h-screen`}>
        <InitialLoading>
          <div className="flex-grow">
            {children}
          </div>
          <Footer />
        </InitialLoading>
      </body>
    </html>
  )
}

