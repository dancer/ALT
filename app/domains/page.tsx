import { Header } from '@/components/header'
import Link from 'next/link'

const domains = [
  "shiemi.com", "gnar.ai", "afterima.ge", "spiritu.al", "behzin.ga", "chord.cat",
  "convert.cat", "cryba.by", "disguisedtoa.st", "hecar.im", "hidden.cat",
  "kazema.ru", "league.cat", "line.al", "lovehe.art", "nidal.ee", "nish.im",
  "pictori.al", "reng.ar", "shahz.am", "supership.it", "termtohome.com",
  "tre.ad", "uou.cat", "vikkst.ar", "yctrainer.com"
]

export default function DomainsPage() {
  return (
    <>
      <Header />
      <main className="pt-20 pb-16">
        <div className="max-w-[800px] mx-auto px-6">
          <div className="flex items-center justify-between mb-8">
            <h1 className="text-xl font-bold">Domains Acquired in 2024</h1>
            <Link 
              href="/"
              className="text-sm hover:underline text-neutral-600 hover:text-neutral-800 transition-colors duration-200 ease-in-out"
            >
              ← Back to Projects
            </Link>
          </div>
          <ul className="space-y-2 font-mono text-lg">
            {domains.map((domain, index) => (
              <li key={index} className="flex items-center">
                <span className="mr-4">{'>'}</span>
                <Link 
                  href={`https://${domain}`} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="hover:text-[#C1E1C1] transition-colors duration-200 ease-in-out"
                >
                  {domain}
                </Link>
              </li>
            ))}
          </ul>
        </div>
      </main>
    </>
  )
}

