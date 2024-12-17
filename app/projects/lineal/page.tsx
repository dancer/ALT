import Image from 'next/image'
import Link from 'next/link'

export default function LinealPage() {
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
            src="/lineal.png"
            alt="Line.al Terminal UI Components"
            width={1200}
            height={800}
            className="mb-8 rounded-lg shadow-lg"
          />
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-[2000px] mx-auto px-6 py-12">
        {/* Project Header */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12 mb-16">
          <div>
            <h1 className="text-4xl font-bold mb-2 text-green-600">LINE.AL</h1>
            <p className="text-green-500 uppercase text-sm">TERMINAL UI LIBRARY</p>
          </div>
          <div className="md:col-span-2">
            <p className="text-lg mb-6">
              Beautiful, responsive components with a terminal aesthetic. Line.al is a comprehensive
              UI library that brings the classic command-line interface feel to modern web applications,
              built with Tailwind CSS and available as open source.
            </p>
          </div>
        </div>

        {/* Project Details */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12 mb-16">
          <div>
            <h2 className="text-sm text-green-500 uppercase mb-2">Role</h2>
            <p>Lead Developer</p>
          </div>
          <div>
            <h2 className="text-sm text-green-500 uppercase mb-2">Version</h2>
            <p>v1.3.0</p>
          </div>
          <div>
            <h2 className="text-sm text-green-500 uppercase mb-2">Tools</h2>
            <ul>
              <li>React</li>
              <li>Tailwind CSS</li>
              <li>TypeScript</li>
            </ul>
          </div>
        </div>

        {/* Features Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
          <div className="bg-white p-6 rounded border border-green-200">
            <h3 className="text-lg font-semibold mb-4 text-green-600">Key Features</h3>
            <ul className="space-y-2">
              <li>• Terminal-inspired components</li>
              <li>• Command palette system</li>
              <li>• Interactive chat interface</li>
              <li>• Data tables and forms</li>
            </ul>
          </div>
          <div className="bg-white p-6 rounded border border-green-200">
            <h3 className="text-lg font-semibold mb-4 text-green-600">Coming Soon</h3>
            <ul className="space-y-2">
              <li>• Custom themes support</li>
              <li>• Animation presets</li>
              <li>• CLI integration</li>
              <li>• Interactive tutorials</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

