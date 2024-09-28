"use client";

import Link from "next/link";
import { useState, useEffect } from "react";
import { ChevronUp, Clock, Share2 } from "lucide-react";

export default function FutureOfQuantumComputing() {
  const [scrollProgress, setScrollProgress] = useState(0);
  const [showScrollTop, setShowScrollTop] = useState(false);

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
            <Clock size={12} className="mr-1" /> 5 min read
          </span>
          <button className="flex items-center hover:text-gray-300 transition-colors">
            <Share2 size={12} className="mr-1" /> Share
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
          "Quantum computing is to classical computing what a warp drive is to a
          bicycle." - Unknown
        </blockquote>
      </section>

      <section id="quantum-principles" className="space-y-4">
        <h2 className="text-sm font-semibold">Quantum Principles</h2>
        <p>At the heart of quantum computing lie two key principles:</p>
        <ul className="list-disc list-inside space-y-2">
          <li className="flex items-start">
            <span className="inline-block w-2 h-2 rounded-full bg-blue-500 mt-1.5 mr-2"></span>
            <span>
              <strong>Superposition:</strong> Qubits can exist in multiple
              states simultaneously
            </span>
          </li>
          <li className="flex items-start">
            <span className="inline-block w-2 h-2 rounded-full bg-purple-500 mt-1.5 mr-2"></span>
            <span>
              <strong>Entanglement:</strong> Qubits can be correlated in ways
              that have no classical counterpart
            </span>
          </li>
        </ul>
      </section>

      <section id="applications" className="space-y-4">
        <h2 className="text-sm font-semibold">Potential Applications</h2>
        <p>
          The potential applications of quantum computing are vast and
          transformative:
        </p>
        <div className="grid grid-cols-2 gap-4">
          <div className="border border-gray-800 rounded p-3 hover:bg-gray-900 transition-colors">
            <h3 className="font-semibold mb-1">Cryptography</h3>
            <p>Breaking and creating unbreakable encryption</p>
          </div>
          <div className="border border-gray-800 rounded p-3 hover:bg-gray-900 transition-colors">
            <h3 className="font-semibold mb-1">Drug Discovery</h3>
            <p>Simulating molecular interactions</p>
          </div>
          <div className="border border-gray-800 rounded p-3 hover:bg-gray-900 transition-colors">
            <h3 className="font-semibold mb-1">Financial Modeling</h3>
            <p>Optimizing investment strategies</p>
          </div>
          <div className="border border-gray-800 rounded p-3 hover:bg-gray-900 transition-colors">
            <h3 className="font-semibold mb-1">Climate Modeling</h3>
            <p>Simulating complex environmental systems</p>
          </div>
        </div>
      </section>

      <section id="challenges" className="space-y-4">
        <h2 className="text-sm font-semibold">Challenges and Limitations</h2>
        <p>
          Despite its promise, quantum computing faces significant challenges:
        </p>
        <ul className="list-disc list-inside space-y-2">
          <li>Maintaining quantum coherence</li>
          <li>Scaling up to useful numbers of qubits</li>
          <li>Developing quantum-resistant encryption</li>
          <li>Creating practical quantum algorithms</li>
        </ul>
      </section>

      <section id="future-outlook" className="space-y-4">
        <h2 className="text-sm font-semibold">Future Outlook</h2>
        <p>
          The future of quantum computing is both exciting and uncertain. As
          researchers continue to push the boundaries of what's possible, we can
          expect to see:
        </p>
        <ul className="list-disc list-inside space-y-2">
          <li>Increased qubit counts and improved coherence times</li>
          <li>
            Development of quantum-specific programming languages and tools
          </li>
          <li>Integration of quantum and classical computing systems</li>
          <li>
            Emergence of new industries and job roles centered around quantum
            technologies
          </li>
        </ul>
      </section>

      <div className="mt-8 pt-4 border-t border-gray-800">
        <h2 className="text-sm font-semibold mb-2">Further Reading</h2>
        <ul className="space-y-2">
          <li>
            <a href="#" className="text-blue-400 hover:underline">
              Quantum Computing: A Gentle Introduction
            </a>
          </li>
          <li>
            <a href="#" className="text-blue-400 hover:underline">
              The Race for Quantum Supremacy
            </a>
          </li>
          <li>
            <a href="#" className="text-blue-400 hover:underline">
              Ethical Implications of Quantum Technologies
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
