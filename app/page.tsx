import Link from "next/link";
import { Github, Twitter } from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="max-w-lg w-full px-4 space-y-4 text-xs text-white">
        <header>
          <h1 className="text-sm font-semibold">josh</h1>
          <p className="text-gray-400">
            <a href="mailto:josh@afterima.ge" className="hover:underline">
              email me
            </a>
          </p>
        </header>

        <p>Software engineer passionate about AI and Web development.</p>

        <section>
          <h2 className="font-semibold">Currently:</h2>
          <ul className="list-disc list-inside">
            <li>Exploring the latest in AI and machine learning.</li>
            <li>Building web applications with Next.js and React.</li>
            <li>Experimenting with PyTorch for deep learning projects.</li>
            <li>Sharing knowledge through blog posts and tutorials.</li>
          </ul>
        </section>

        <section>
          <h2 className="font-semibold">Posts:</h2>
          <ul>
            <li>
              <Link href="/posts/rag" className="text-blue-400 hover:underline">
                {"@"} Local Agentic RAG: Revolutionizing LLMs with Private
                Knowledge
              </Link>
            </li>
            <li>
              <Link
                href="/posts/models"
                className="text-blue-400 hover:underline"
              >
                {"@"} Diffusion Models: Turning Noise into Art with AI Magic
              </Link>
            </li>
            <li>
              <Link
                href="/posts/pytorch"
                className="text-blue-400 hover:underline"
              >
                {"@"} Creating an AI with PyTorch: A Beginner's Guide
              </Link>
            </li>
            <li>
              <Link href="/posts/ai" className="text-blue-400 hover:underline">
                {"@"} Revolutionizing AI Compilers: The Path to Efficient
                Hardware Utilization
              </Link>
            </li>
            <li>
              <Link
                href="/posts/quantum"
                className="text-blue-400 hover:underline"
              >
                {"@"} The Future of Quantum Computing
              </Link>
            </li>
          </ul>
        </section>

        <footer className="pt-4">
          <div className="flex space-x-4">
            <a
              href="https://github.com/dancer"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-400 hover:text-white"
            >
              <Github size={16} />
              <span className="sr-only">GitHub</span>
            </a>
            <a
              href="https://twitter.com/dxd"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-400 hover:text-white"
            >
              <Twitter size={16} />
              <span className="sr-only">Twitter</span>
            </a>
          </div>
          <div className="container mx-auto py-2 text-[10px] text-gray-500 italic">
            <p>Be the best you can be in silent.</p>
          </div>
        </footer>
      </div>
    </div>
  );
}
