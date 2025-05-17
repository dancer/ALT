import Link from 'next/link'

export function Header() {
  return (
    <header className="fixed top-0 left-0 right-0 z-40 bg-[#f5f5f0]">
      <div className="max-w-[2000px] mx-auto px-6 py-4 flex justify-between items-center font-mono text-sm">
        <Link
          href="/domains"
          className="tracking-tight hover:text-[#C1E1C1] transition-colors duration-200 ease-in-out"
        >
          Domains
        </Link>
        <Link
          href="/"
          className="tracking-tighter hover:text-neutral-600 transition-colors duration-200 ease-in-out"
        >
          Anywho
        </Link>
        <div className="tracking-tight">London, UK</div>
      </div>
    </header>
  )
}