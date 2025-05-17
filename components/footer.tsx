import Link from 'next/link'

export function Footer() {
  return (
    <footer className="bg-[#f5f5f0] py-4 px-6 flex justify-between items-center text-xs text-neutral-600 font-mono">
      <div className="flex items-center">
        <span className="mr-2">©</span>
        <span>Anywho 2024</span>
      </div>
    </footer>
  )
}

