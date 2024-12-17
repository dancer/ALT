import Image from 'next/image'
import Link from 'next/link'

export default function NidaleePage() {
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
            src="/nidalee.png"
            alt="Nidalee Interface"
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
            <h1 className="text-4xl font-bold mb-2 text-[#ff6b6b]">NIDAL.EE</h1>
            <p className="text-neutral-600 uppercase text-sm">RIOT GAMES LAUNCHER</p>
          </div>
          <div className="md:col-span-2">
            <p className="text-lg mb-6">
              A streamlined account manager for Riot Games launchers. Nidal.ee simplifies the process
              of managing multiple accounts and provides quick access to your favorite games.
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
            <p>v0.1.1</p>
          </div>
          <div>
            <h2 className="text-sm text-neutral-600 uppercase mb-2">Tools</h2>
            <ul>
              <li>Tauri</li>
              <li>Rust</li>
              <li>TypeScript</li>
            </ul>
          </div>
        </div>

        {/* Features Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
          <div className="bg-white p-6 rounded border border-neutral-200">
            <h3 className="text-lg font-semibold mb-4">Key Features</h3>
            <ul className="space-y-2">
              <li>• Quick account switching</li>
              <li>• Secure credential storage</li>
              <li>• Auto-launch support</li>
              <li>• Multiple game profiles</li>
            </ul>
          </div>
          <div className="bg-white p-6 rounded border border-neutral-200">
            <h3 className="text-lg font-semibold mb-4">Coming Soon</h3>
            <ul className="space-y-2">
              <li>• Cloud sync</li>
              <li>• Custom themes</li>
              <li>• Stats tracking</li>
              <li>• Friend lists</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

