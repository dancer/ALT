'use client'

import { useEffect, useState } from 'react'

export function LoadingAnimation() {
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress((prevProgress) => {
        if (prevProgress >= 100) {
          clearInterval(interval)
          return 100
        }
        return prevProgress + 1
      })
    }, 20)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="w-64 h-1 bg-neutral-800 rounded-full overflow-hidden">
      <div
        className="h-full bg-white transition-all duration-100 ease-out"
        style={{ width: `${progress}%` }}
      />
    </div>
  )
}

