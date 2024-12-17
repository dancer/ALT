interface ContentSectionProps {
  title: string
  children: React.ReactNode
}

export function ContentSection({ title, children }: ContentSectionProps) {
  return (
    <section className="space-y-4">
      <h2 className="text-xs uppercase tracking-wider text-neutral-500">{title}</h2>
      <div className="space-y-4 text-sm leading-relaxed">{children}</div>
    </section>
  )
}

