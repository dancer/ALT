'use client'

import { useEffect, useRef } from 'react'

export function AnimatedLogo() {
  const pathRef = useRef<SVGPathElement>(null)

  useEffect(() => {
    if (pathRef.current) {
      const length = pathRef.current.getTotalLength()
      pathRef.current.style.strokeDasharray = `${length} ${length}`
      pathRef.current.style.strokeDashoffset = `${length}`
      pathRef.current.style.animation = 'dash 2s ease-in-out forwards'
    }
  }, [])

  return (
    <svg width="200" height="200" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path
        ref={pathRef}
        d="M10 50 Q 25 25, 50 50 T 90 50"
        stroke="white"
        strokeWidth="2"
        fill="none"
      />
      <text x="50" y="70" textAnchor="middle" fill="white" fontSize="12">
        Anywho
      </text>
    </svg>
  )
}

