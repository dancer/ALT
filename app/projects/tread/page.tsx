import Image from 'next/image'
import Link from 'next/link'

export default function TreadPage() {
  return (
    <div className="min-h-screen bg-[#f5f5f0] text-neutral-800">
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
            src="/tread.png"
            alt="Tread Interface"
            width={1200}
            height={800}
            className="mb-8 rounded-lg shadow-2xl"
          />
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-[2000px] mx-auto px-6 py-12">
        {/* Project Header */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12 mb-16">
          <div>
            <h1 className="text-4xl font-bold mb-2 text-purple-700">TRE.AD</h1>
            <p className="text-purple-600 uppercase text-sm">CODING SPEED TEST</p>
          </div>
          <div className="md:col-span-2">
            <p className="text-lg mb-6">
              A sophisticated coding speed test platform designed to help developers measure and improve
              their typing velocity and accuracy. Tre.ad provides real-time metrics and personalized
              practice sessions across multiple programming languages.
            </p>
          </div>
        </div>

        {/* Project Details */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12 mb-16">
          <div>
            <h2 className="text-sm text-purple-600 uppercase mb-2">Role</h2>
            <p>Lead Developer</p>
          </div>
          <div>
            <h2 className="text-sm text-purple-600 uppercase mb-2">Version</h2>
            <p>v0.8.2</p>
          </div>
          <div>
            <h2 className="text-sm text-purple-600 uppercase mb-2">Tools</h2>
            <ul>
              <li>Next.js</li>
              <li>TypeScript</li>
              <li>Anthropic</li>
            </ul>
          </div>
        </div>

        {/* Features Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
          <div className="bg-white p-6 rounded border border-purple-200">
            <h3 className="text-lg font-semibold mb-4 text-purple-700">Key Features</h3>
            <ul className="space-y-2">
              <li>• Real-time WPM tracking</li>
              <li>• Multi-language support</li>
              <li>• Code snippet library</li>
              <li>• Accuracy metrics</li>
            </ul>
          </div>
          <div className="bg-white p-6 rounded border border-purple-200">
            <h3 className="text-lg font-semibold mb-4 text-purple-700">Coming Soon</h3>
            <ul className="space-y-2">
              <li>• Global leaderboards</li>
              <li>• Custom practice modes</li>
              <li>• Team competitions</li>
              <li>• Progress analytics</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

