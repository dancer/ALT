import Image from 'next/image'
import Link from 'next/link'

export default function LoveheartPage() {
  return (
    <div className="min-h-screen bg-[#f5f5f0]">
      <div className="fixed top-8 left-8 z-10">
        <Link 
          href="/"
          className="text-sm hover:text-[#C1E1C1] transition-colors duration-200 ease-in-out"
        >
          ← Back to Projects
        </Link>
      </div>

      {/* Hero Section */}
      <div className="w-full min-h-screen relative flex flex-col items-center justify-center p-6">
        <div className="max-w-[1200px] w-full text-center">
          <Image
            src="/loveheart.png"
            alt="Loveheart Interface"
            width={1200}
            height={800}
            className="mb-8"
          />
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-[2000px] mx-auto px-6 py-12">
        {/* Project Header */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12 mb-16">
          <div>
            <h1 className="text-4xl font-bold mb-2 text-[#ff69b4]">LOVEHE.ART</h1>
            <p className="text-[#ff69b4] uppercase text-sm">PAYMENT PLATFORM</p>
          </div>
          <div className="md:col-span-2">
            <p className="text-lg mb-6">
              A revolutionary payment platform that connects stablecoins with empathy.
              Lovehe.art enables seamless, global transactions with a focus on user experience
              and emotional connection.
            </p>
          </div>
        </div>

        {/* Project Details */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12 mb-16">
          <div>
            <h2 className="text-sm text-[#ff69b4] uppercase mb-2">Role</h2>
            <p>Lead Developer</p>
          </div>
          <div>
            <h2 className="text-sm text-[#ff69b4] uppercase mb-2">Version</h2>
            <p>v0.9.5</p>
          </div>
          <div>
            <h2 className="text-sm text-[#ff69b4] uppercase mb-2">Tools</h2>
            <ul>
              <li>Next.js</li>
              <li>Solidity</li>
              <li>Web3.js</li>
            </ul>
          </div>
        </div>

        {/* Features Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
          <div className="bg-white p-6 rounded border border-pink-200">
            <h3 className="text-lg font-semibold mb-4 text-[#ff69b4]">Key Features</h3>
            <ul className="space-y-2">
              <li>• Seamless stablecoin transfers</li>
              <li>• Real-time global settlements</li>
              <li>• Multi-chain support</li>
              <li>• Empathy-driven user experience</li>
            </ul>
          </div>
          <div className="bg-white p-6 rounded border border-pink-200">
            <h3 className="text-lg font-semibold mb-4 text-[#ff69b4]">Coming Soon</h3>
            <ul className="space-y-2">
              <li>• AI-powered financial advice</li>
              <li>• Charitable giving integration</li>
              <li>• Cross-chain liquidity pools</li>
              <li>• Decentralized identity solutions</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

