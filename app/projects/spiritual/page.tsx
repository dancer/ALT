import Image from 'next/image'
import Link from 'next/link'

export default function SpiritualPage() {
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
            src="/spiritual.png"
            alt="Spiritu.al Interface"
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
            <h1 className="text-4xl font-bold mb-2 text-emerald-800">SPIRITU.AL</h1>
            <p className="text-emerald-700 uppercase text-sm">PROMPT ENHANCER</p>
          </div>
          <div className="md:col-span-2">
            <p className="text-lg mb-6">
              A pioneering prompt engineering platform designed to enhance AI interactions.
              Spiritu.al transforms simple prompts into powerful, detailed instructions for more
              effective and nuanced AI conversations.
            </p>
          </div>
        </div>

        {/* Project Details */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12 mb-16">
          <div>
            <h2 className="text-sm text-emerald-700 uppercase mb-2">Role</h2>
            <p>Lead Developer</p>
          </div>
          <div>
            <h2 className="text-sm text-emerald-700 uppercase mb-2">Version</h2>
            <p>v1.2.0</p>
          </div>
          <div>
            <h2 className="text-sm text-emerald-700 uppercase mb-2">Tools</h2>
            <ul>
              <li>Next.js</li>
              <li>Anthropic</li>
              <li>Vercel AI SDK</li>
            </ul>
          </div>
        </div>

        {/* Features Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
          <div className="bg-white p-6 rounded border border-emerald-200">
            <h3 className="text-lg font-semibold mb-4 text-emerald-800">Key Features</h3>
            <ul className="space-y-2">
              <li>• Intelligent prompt enhancement</li>
              <li>• Context-aware suggestions</li>
              <li>• Real-time preview</li>
              <li>• Multiple AI model support</li>
            </ul>
          </div>
          <div className="bg-white p-6 rounded border border-emerald-200">
            <h3 className="text-lg font-semibold mb-4 text-emerald-800">Coming Soon</h3>
            <ul className="space-y-2">
              <li>• Custom prompt templates</li>
              <li>• Collaborative workspaces</li>
              <li>• Advanced analytics</li>
              <li>• API integration</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

