import Image from 'next/image'
import Link from 'next/link'

export default function PictorialPage() {
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
            src="/pictorial.png"
            alt="Pictori.al Interface"
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
            <h1 className="text-4xl font-bold mb-2 text-neutral-900">PICTORI.AL</h1>
            <p className="text-neutral-600 uppercase text-sm">PROFILE MATCHING</p>
          </div>
          <div className="md:col-span-2">
            <p className="text-lg mb-6">
              A visual harmony-based profile picture matching platform that helps users find
              perfectly coordinated profile and banner combinations. Pictori.al uses advanced
              image analysis to create aesthetically pleasing social media presence.
            </p>
          </div>
        </div>

        {/* Project Details */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12 mb-16">
          <div>
            <h2 className="text-sm text-neutral-600 uppercase mb-2">Role</h2>
            <p>Lead Developer</p>
          </div>
          <div>
            <h2 className="text-sm text-neutral-600 uppercase mb-2">Version</h2>
            <p>v0.9.0</p>
          </div>
          <div>
            <h2 className="text-sm text-neutral-600 uppercase mb-2">Tools</h2>
            <ul>
              <li>Next.js</li>
              <li>TensorFlow.js</li>
              <li>Sharp</li>
            </ul>
          </div>
        </div>

        {/* Features Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
          <div className="bg-white p-6 rounded border border-neutral-200">
            <h3 className="text-lg font-semibold mb-4 text-neutral-900">Key Features</h3>
            <ul className="space-y-2">
              <li>• Smart image matching algorithm</li>
              <li>• Couple profile coordination</li>
              <li>• Custom galleries and collections</li>
              <li>• Advanced search filters</li>
            </ul>
          </div>
          <div className="bg-white p-6 rounded border border-neutral-200">
            <h3 className="text-lg font-semibold mb-4 text-neutral-900">Coming Soon</h3>
            <ul className="space-y-2">
              <li>• AI-powered style suggestions</li>
              <li>• Automatic banner generation</li>
              <li>• Social platform integration</li>
              <li>• Custom theme creator</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

