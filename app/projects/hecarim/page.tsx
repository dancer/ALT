import Image from 'next/image'
import Link from 'next/link'

export default function HecarimPage() {
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
            src="/hecarim.png"
            alt="Hecarim VS Code Theme Interface"
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
            <h1 className="text-4xl font-bold mb-2 text-blue-600">HECAR.IM</h1>
            <p className="text-blue-500 uppercase text-sm">VS CODE THEME</p>
          </div>
          <div className="md:col-span-2">
            <p className="text-lg mb-6">
              Elevate your coding experience with our sleek and modern theme for Visual Studio Code.
              Carefully crafted color schemes and thoughtful design choices make Hecar.im the perfect
              companion for long coding sessions.
            </p>
          </div>
        </div>

        {/* Project Details */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12 mb-16">
          <div>
            <h2 className="text-sm text-blue-500 uppercase mb-2">Role</h2>
            <p>Lead Designer</p>
          </div>
          <div>
            <h2 className="text-sm text-blue-500 uppercase mb-2">Version</h2>
            <p>v2.1.0</p>
          </div>
          <div>
            <h2 className="text-sm text-blue-500 uppercase mb-2">Tools</h2>
            <ul>
              <li>VS Code API</li>
              <li>TypeScript</li>
              <li>Node.js</li>
            </ul>
          </div>
        </div>

        {/* Features Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
          <div className="bg-white p-6 rounded border border-blue-200">
            <h3 className="text-lg font-semibold mb-4 text-blue-600">Key Features</h3>
            <ul className="space-y-2">
              <li>• Carefully crafted syntax highlighting</li>
              <li>• Performance optimized theme engine</li>
              <li>• Eye-strain reduction color palette</li>
              <li>• Multiple language support</li>
            </ul>
          </div>
          <div className="bg-white p-6 rounded border border-blue-200">
            <h3 className="text-lg font-semibold mb-4 text-blue-600">Coming Soon</h3>
            <ul className="space-y-2">
              <li>• Custom syntax configurations</li>
              <li>• Light theme variant</li>
              <li>• Language-specific tweaks</li>
              <li>• Semantic highlighting</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

