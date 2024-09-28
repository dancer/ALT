"use client";

"use client";

import Link from "next/link";
import { useState, useEffect } from "react";
import { ChevronUp, Clock, Share2, Check } from "lucide-react";

export default function FutureOfQuantumComputing() {
  const [scrollProgress, setScrollProgress] = useState(0);
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const totalHeight =
        document.documentElement.scrollHeight -
        document.documentElement.clientHeight;
      const progress = (window.scrollY / totalHeight) * 100;
      setScrollProgress(progress);
      setShowScrollTop(window.scrollY > 300);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleShare = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy: ", err);
    }
  };

  return (
    <article className="relative space-y-6 text-xs">
      <div className="fixed top-0 left-0 w-full h-1 bg-gray-200">
        <div
          className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
          style={{ width: `${scrollProgress}%` }}
        ></div>
      </div>

      <Link
        href="/"
        className="text-blue-400 hover:underline inline-block mb-4"
      >
        {"<"} Back to home
      </Link>

      <header className="space-y-2">
        <h1 className="text-lg font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-purple-500">
          The Future of Quantum Computing
        </h1>
        <p className="text-gray-400">
          Exploring the revolutionary potential of quantum technologies
        </p>
        <div className="flex items-center space-x-4 text-gray-500">
          <span className="flex items-center">
            <Clock size={12} className="mr-1" /> 10 min read
          </span>
          <button
            onClick={handleShare}
            className="flex items-center hover:text-gray-300 transition-colors"
          >
            {copied ? <Check size={12} className="mr-1" /> : <Share2 size={12} className="mr-1" />}
            {copied ? "Copied!" : "Share"}
          </button>
        </div>
      </header>

      <nav className="border border-gray-800 rounded p-4">
        <h2 className="font-semibold mb-2">Table of Contents</h2>
        <ul className="space-y-1">
          <li>
            <a
              href="#introduction"
              className="hover:text-blue-400 transition-colors"
            >
              Introduction
            </a>
          </li>
          <li>
            <a
              href="#quantum-principles"
              className="hover:text-blue-400 transition-colors"
            >
              Quantum Principles
            </a>
          </li>
          <li>
            <a
              href="#applications"
              className="hover:text-blue-400 transition-colors"
            >
              Potential Applications
            </a>
          </li>
          <li>
            <a
              href="#challenges"
              className="hover:text-blue-400 transition-colors"
            >
              Challenges and Limitations
            </a>
          </li>
          <li>
            <a
              href="#future-outlook"
              className="hover:text-blue-400 transition-colors"
            >
              Future Outlook
            </a>
          </li>
        </ul>
      </nav>

      <section id="introduction" className="space-y-4">
        <h2 className="text-sm font-semibold">Introduction</h2>
        <p>
          Quantum computing stands at the frontier of technological innovation,
          promising to revolutionize fields ranging from cryptography to drug
          discovery. Unlike classical computers that operate on bits, quantum
          computers leverage the principles of quantum mechanics to process
          information using quantum bits, or qubits.
        </p>
        <blockquote className="border-l-2 border-purple-500 pl-4 italic text-gray-400">
          "I think I can safely say that nobody understands quantum mechanics." - Richard Feynman
        </blockquote>
        <p>
          Despite its complexity, quantum computing has the potential to solve problems that are intractable for classical computers, opening up new frontiers in science and technology.
        </p>
      </section>

      <section id="quantum-principles" className="space-y-4">
        <h2 className="text-sm font-semibold">Quantum Principles</h2>
        <p>At the heart of quantum computing lie two key principles:</p>
        <ul className="list-disc list-inside space-y-2">
          <li className="flex items-start">
            <span className="inline-block w-2 h-2 rounded-full bg-blue-500 mt-1.5 mr-2"></span>
            <span>
              <strong>Superposition:</strong> Qubits can exist in multiple
              states simultaneously, allowing quantum computers to process vast amounts of information in parallel.
            </span>
          </li>
          <li className="flex items-start">
            <span className="inline-block w-2 h-2 rounded-full bg-purple-500 mt-1.5 mr-2"></span>
            <span>
              <strong>Entanglement:</strong> Qubits can be correlated in ways
              that have no classical counterpart, enabling quantum computers to perform certain calculations exponentially faster than classical computers.
            </span>
          </li>
        </ul>
        <p>
          These principles form the foundation of quantum algorithms, which exploit the unique properties of quantum systems to solve complex problems more efficiently than classical algorithms.
        </p>
      </section>

      <section id="applications" className="space-y-4">
        <h2 className="text-sm font-semibold">Potential Applications</h2>
        <p>
          The potential applications of quantum computing are vast and
          transformative, spanning multiple industries and scientific disciplines:
        </p>
        <div className="grid grid-cols-2 gap-4">
          <div className="border border-gray-800 rounded p-3 hover:bg-gray-900 transition-colors">
            <h3 className="font-semibold mb-1">Cryptography</h3>
            <p>Breaking and creating unbreakable encryption systems, revolutionizing data security</p>
          </div>
          <div className="border border-gray-800 rounded p-3 hover:bg-gray-900 transition-colors">
            <h3 className="font-semibold mb-1">Drug Discovery</h3>
            <p>Simulating molecular interactions to accelerate the development of new medicines</p>
          </div>
          <div className="border border-gray-800 rounded p-3 hover:bg-gray-900 transition-colors">
            <h3 className="font-semibold mb-1">Financial Modeling</h3>
            <p>Optimizing investment strategies and risk assessment in complex markets</p>
          </div>
          <div className="border border-gray-800 rounded p-3 hover:bg-gray-900 transition-colors">
            <h3 className="font-semibold mb-1">Climate Modeling</h3>
            <p>Simulating complex environmental systems for more accurate climate predictions</p>
          </div>
        </div>
        <p>
          These applications have the potential to drive significant advancements in science, technology, and society as a whole.
        </p>
      </section>

      <section id="challenges" className="space-y-4">
        <h2 className="text-sm font-semibold">Challenges and Limitations</h2>
        <p>
          Despite its promise, quantum computing faces significant challenges:
        </p>
        <ul className="list-disc list-inside space-y-2">
          <li>Maintaining quantum coherence: Qubits are extremely sensitive to environmental disturbances, making it difficult to maintain their quantum states.</li>
          <li>Scaling up to useful numbers of qubits: Current quantum computers have limited numbers of qubits, and scaling up while maintaining coherence is a major challenge.</li>
          <li>Developing quantum-resistant encryption: As quantum computers threaten current encryption methods, new quantum-resistant cryptography needs to be developed.</li>
          <li>Creating practical quantum algorithms: Designing algorithms that can effectively leverage quantum properties for real-world problems is an ongoing area of research.</li>
        </ul>
        <p>
          Overcoming these challenges requires interdisciplinary collaboration between physicists, computer scientists, mathematicians, and engineers.
        </p>
      </section>

      <section id="future-outlook" className="space-y-4">
        <h2 className="text-sm font-semibold">Future Outlook</h2>
        <p>
          The future of quantum computing is both exciting and uncertain. As
          researchers continue to push the boundaries of what's possible, we can
          expect to see:
        </p>
        <ul className="list-disc list-inside space-y-2">
          <li>Increased qubit counts and improved coherence times, leading to more powerful quantum computers</li>
          <li>
            Development of quantum-specific programming languages and tools to make quantum computing more accessible
          </li>
          <li>Integration of quantum and classical computing systems, creating hybrid solutions for complex problems</li>
          <li>
            Emergence of new industries and job roles centered around quantum
            technologies, driving economic growth and innovation
          </li>
        </ul>
        <p>
          As quantum computing continues to evolve, it has the potential to reshape our technological landscape and solve some of humanity's most pressing challenges.
        </p>
      </section>

      <div className="mt-8 pt-4 border-t border-gray-800">
        <h2 className="text-sm font-semibold mb-2">Further Reading</h2>
        <ul className="space-y-2">
          <li>
            <a href="https://www.ibm.com/quantum/learn/what-is-quantum-computing" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">
              IBM: What is Quantum Computing?
            </a>
          </li>
          <li>
            <a href="https://www.nature.com/articles/d41586-019-02602-8" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">
              Nature: The Race for Quantum Supremacy
            </a>
          </li>
          <li>
            <a href="https://www.scientificamerican.com/article/ethical-implications-of-quantum-computing/" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">
              Scientific American: Ethical Implications of Quantum Computing
            </a>
          </li>
          <li>
            <a href="https://quantum.country/qcvc" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">
              Quantum Country: Quantum Computing for the Very Curious
            </a>
          </li>
        </ul>
      </div>

      {showScrollTop && (
        <button
          onClick={scrollToTop}
          className="fixed bottom-4 right-4 bg-gray-800 p-2 rounded-full hover:bg-gray-700 transition-colors"
          aria-label="Scroll to top"
        >
          <ChevronUp size={20} />
        </button>
      )}
    </article>
  );
}