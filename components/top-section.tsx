export function TopSection() {
  return (
    <section className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
      <div>
        <h2 className="text-xs uppercase mb-4 text-neutral-600">About</h2>
        <p className="text-sm leading-relaxed">
          Anywho is a software development studio specializing in innovative digital solutions.
          We thrive in complex, ambiguous problem spaces focused around interactive media,
          digital tooling, and multimodal interaction. Our team's expertise spans cutting-edge
          technologies and best practices in software development.
        </p>
        <div className="mt-4">
          <span className="text-sm text-neutral-600 mr-2">Email:</span>
          <a
            href="mailto:contact@afterima.ge"
            className="text-sm hover:text-[#C1E1C1] transition-colors duration-200 ease-in-out"
          >
            contact@afterima.ge
          </a>
        </div>
      </div>
      <div>
        <h2 className="text-xs uppercase mb-4 text-neutral-600">Team</h2>
        <ul className="space-y-4">
            <h3 className="text-sm font-semibold">Josh</h3>
            <p className="text-xs text-neutral-600">AIO</p>
            <p className="text-xs text-neutral-600">2024 - Present</p>
            <p className="text-xs text-neutral-600">Undergrad | Intern at Vercel</p>
        </ul>
      </div>
      <div>
        <h2 className="text-xs uppercase mb-4 text-neutral-600">Description</h2>
        <p className="text-sm leading-relaxed">
          At Anywho, we're building cutting-edge software solutions and interactive experiences.
          Our projects span various domains including AI tools, gaming, fintech, developer tools,
          and social platforms. We're actively expanding our team and portfolio of innovative projects.
        </p>
        <p className="text-sm leading-relaxed mt-4">
          Our expertise extends to design consulting for emerging AI and productivity companies.
          We specialize in creating intuitive interfaces and robust backend systems that push the
          boundaries of what's possible in software development.
        </p>
      </div>
    </section>
  )
}

